"""The LLM layer of the voice pipeline: one logical call = retries + validation + a context-size check + a log record.

Ported from TCM-Meridian-mini (`mini/llm/calls.py`, `client.py`). Differences: the request goes through the `openai` SDK
(like every other agent of this project) in a worker thread instead of an httpx async client, and there is no scheduler:
the corrector is a single serial stream (one window at a time) and the LLM server handles concurrent requests, so the
transcript correction and the agents simply run side by side. `PRIORITY_*` stay as names so the callers read the same.
"""
from __future__ import annotations

import asyncio
import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import Endpoint
from .textutil import clean_think, estimate_messages_tokens

PRIORITY_USER = 0
PRIORITY_BACKGROUND = 1

CONTEXT_HINT = ('若伺服器實際可容納更多，請在「模型設定」調高「逐字稿校稿」的「context 上限」；'
                '否則請縮短輸入。')
_OVERFLOW_WORDS = ('exceed', 'too long', 'too large', 'overflow', 'n_ctx', 'maximum context', 'context length',
                   'context window', 'context size')


class LLMError(RuntimeError):
    pass


class ValidationError(Exception):
    """Raised by a validator when an LLM output is unusable; triggers a retry with the reason."""


class CallFailed(RuntimeError):
    def __init__(self, agent: str, last_error: str, call_ids: list[str]):
        super().__init__(f'{agent} 呼叫失敗：{last_error}')
        self.agent = agent
        self.last_error = last_error
        self.call_ids = call_ids


@dataclass
class LLMResult:
    text: str
    finish_reason: str = ''
    usage: dict = field(default_factory=dict)
    latency_ms: int = 0


@dataclass
class CallResult:
    text: str
    value: Any
    call_ids: list[str] = field(default_factory=list)
    attempts: int = 1


def prompt_too_long_message(endpoint: Endpoint, messages: list[dict], reserve: int | None) -> str:
    """Pre-flight check against the interface's context limit; '' when the prompt fits (or no limit is set)."""
    limit = endpoint.context_tokens
    if not limit:
        return ''
    estimate = estimate_messages_tokens(messages)
    reserve = reserve or 0
    if estimate + reserve <= limit:
        return ''
    return (f'提示詞估計約 {estimate} tokens，加上輸出預留 {reserve} tokens，超過此接口設定的 context 上限 {limit}。'
            + CONTEXT_HINT)


def is_context_overflow(text: str) -> bool:
    lowered = text.lower()
    return 'context' in lowered and any(word in lowered for word in _OVERFLOW_WORDS) or 'n_ctx' in lowered


class LLMClient:
    """OpenAI-compatible chat (non-streaming) through the `openai` SDK; one SDK client per (address, key)."""

    def __init__(self, transport=None):
        self._transport = transport             # an httpx transport: tests answer from memory instead of the network
        self._clients: dict[tuple[str, str], Any] = {}

    def _client(self, endpoint: Endpoint, timeout: float):
        key = (endpoint.api_url, endpoint.api_key)
        client = self._clients.get(key)
        if client is None:
            from openai import OpenAI
            options: dict[str, Any] = {}
            if self._transport is not None:
                import httpx
                options['http_client'] = httpx.Client(transport=self._transport)
            client = self._clients[key] = OpenAI(base_url=endpoint.api_url, api_key=endpoint.api_key or 'no-key',
                                                 max_retries=0, **options)
        return client.with_options(timeout=timeout)

    def _chat_sync(self, endpoint: Endpoint, messages: list[dict], timeout: float, max_tokens: int | None) -> LLMResult:
        kwargs: dict[str, Any] = {'model': endpoint.model_name, 'messages': messages, 'temperature': endpoint.temperature}
        limit = endpoint.max_tokens if max_tokens is None else max_tokens
        if limit and limit > 0:
            kwargs['max_completion_tokens'] = limit
        started = time.monotonic()
        try:
            response = self._client(endpoint, timeout).chat.completions.create(**kwargs)
        except Exception as exc:                # openai.APIError and friends, httpx errors, timeouts
            text = str(exc) or type(exc).__name__
            if endpoint.api_key:
                text = text.replace(endpoint.api_key, '***')
            if 'timed out' in text.lower() or type(exc).__name__ in ('APITimeoutError', 'ReadTimeout', 'TimeoutException'):
                raise LLMError(f'LLM 逾時（{timeout:g} 秒）。') from exc
            raise LLMError(f'LLM 呼叫失敗：{text[:500]}') from exc
        try:
            choice = response.choices[0]
            finish = choice.finish_reason or ''
            content = choice.message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMError('模型回應格式錯誤，缺少 choices/message/content。') from exc
        if finish not in ('', 'stop'):
            raise LLMError(f'模型未完整輸出（finish_reason={finish}，可能達到 token 上限）。')
        if not isinstance(content, str):
            raise LLMError('模型沒有回傳文字。')
        text = clean_think(content)
        if not text:
            raise LLMError('模型回傳空白內容。')
        usage = {}
        raw_usage = getattr(response, 'usage', None)
        if raw_usage is not None:
            usage = {k: getattr(raw_usage, k) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')
                     if getattr(raw_usage, k, None) is not None}
        return LLMResult(text, finish, usage, int((time.monotonic() - started) * 1000))

    async def chat(self, endpoint: Endpoint, messages: list[dict], *, timeout: float, max_tokens: int | None = None) -> LLMResult:
        return await asyncio.to_thread(self._chat_sync, endpoint, messages, timeout, max_tokens)


class LLMCaller:
    """Binds a client to a log sink. `logger(record)` receives one dict per attempt."""

    def __init__(self, client: LLMClient | None = None, *, timeout: Callable[[], float] = lambda: 180.0,
                 retries: Callable[[], int] = lambda: 3, logger: Callable[[dict], None] | None = None,
                 id_prefix: str = 't', backoff: float = 1.0):
        self.client = client or LLMClient()
        self._timeout = timeout
        self._retries = retries
        self.logger = logger
        self.id_prefix = id_prefix
        self.backoff = backoff
        self._ids = itertools.count(1)

    def next_id(self) -> str:
        return f'{self.id_prefix}{next(self._ids):04d}'

    async def call(self, *, agent: str, endpoint: Endpoint,
                   messages: list[dict] | Callable[[str | None], list[dict]],
                   job_id: str | None = None, priority: int = PRIORITY_USER,
                   validator: Callable[[str], Any] | None = None,
                   retries: int | None = None, max_tokens: int | None = None) -> CallResult:
        """Run the call, retrying up to `retries` times (default: the global setting).

        `messages` may be a callable receiving the previous attempt's error text so the retry can tell the model what was wrong.
        """
        limit = self._retries() if retries is None else retries
        last_error: str | None = None
        call_ids: list[str] = []
        for attempt in range(1, limit + 2):
            built = messages(last_error) if callable(messages) else messages
            call_id = self.next_id()
            call_ids.append(call_id)
            record = {'call_id': call_id, 'job_id': job_id, 'agent': agent, 'attempt': attempt,
                      'model': endpoint.model_name, 'url': endpoint.api_url,
                      'temperature': endpoint.temperature,
                      'max_tokens': endpoint.max_tokens if max_tokens is None else max_tokens,
                      'messages': built, 'response': None, 'finish_reason': None,
                      'latency_ms': None, 'usage': None, 'error': None}
            too_long = prompt_too_long_message(endpoint, built, record['max_tokens'])
            if too_long:                                    # retrying cannot help: fail now, clearly, with a log record
                record['error'] = too_long
                self._emit(record)
                raise CallFailed(agent, too_long, call_ids)
            try:
                result = await self.client.chat(endpoint, built, timeout=self._timeout(), max_tokens=max_tokens)
                record.update(response=result.text, finish_reason=result.finish_reason,
                              latency_ms=result.latency_ms, usage=result.usage)
                value = validator(result.text) if validator else None
            except asyncio.CancelledError:
                record['error'] = '已取消'
                self._emit(record)
                raise
            except (LLMError, ValidationError, ValueError) as exc:
                last_error = str(exc) or type(exc).__name__
                record['error'] = last_error
                self._emit(record)
                if isinstance(exc, LLMError) and is_context_overflow(last_error):
                    raise CallFailed(agent, f'伺服器回報輸入超過模型 context：{last_error[:200]}。{CONTEXT_HINT}',
                                     call_ids) from exc
                if attempt <= limit and self.backoff:
                    await asyncio.sleep(self.backoff)
                continue
            self._emit(record)
            return CallResult(result.text, value, call_ids, attempt)
        raise CallFailed(agent, last_error or '未知錯誤', call_ids)

    def _emit(self, record: dict):
        if self.logger is not None:
            try:
                self.logger(record)
            except Exception:  # logging must never break a job
                pass
