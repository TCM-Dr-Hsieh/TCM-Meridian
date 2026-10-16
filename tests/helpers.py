"""Test doubles: scripted OpenAI-compatible server, fake ASR, fake microphone.

Ported from TCM-Meridian-mini (`tests/helpers.py`). The voice pipeline reaches the LLM through the `openai` SDK, so the scripted
server is an httpx mock transport answering in the chat-completions shape.
"""
from __future__ import annotations

import json
import unicodedata

import httpx
import numpy as np

from voice.audio import AudioSource
from voice.config import Endpoint, VoiceSettings
from voice.wavfile import RATE

ROLE_MARKERS = [
    ('corrector', '三段滾動校字員'),
    ('speaker_roles', '診間逐字稿的角色判定員'),
    ('speaker_fill2', '說話者二階補標員'),
    ('speaker_fill', '說話者補標員'),
]


def roles_reply(payload: dict) -> dict:
    """The role layer's fake answer: the voice group whose lines start with 請問 is the doctor, every other group is 患者或家屬."""
    import re
    lines = '\n'.join(payload['transcript_by_voice_cluster'])
    doctors = set(re.findall(r'\[(S\d+)\] 請問', lines))
    return {'roles': {g: ('醫師' if g in doctors else '患者或家屬') for g in payload['groups']},
            'confidence': '高', 'reason': '假的角色判定'}


def jdump(value) -> str:
    return json.dumps(value, ensure_ascii=False)


class FakeLLM:
    """httpx.MockTransport handler. `script[role]` is a list consumed in order, then `default`.

    An item is a str (reply text), a dict (JSON reply), an Exception/'HTTP500' marker (failure),
    or a callable(messages) returning one of those.
    """

    def __init__(self):
        self.script: dict[str, list] = {}
        self.calls: list[tuple[str, list]] = []

    # -- script helpers --------------------------------------------------
    def queue(self, role: str, *items):
        self.script.setdefault(role, []).extend(items)

    def calls_for(self, role: str) -> list[list]:
        return [m for r, m in self.calls if r == role]

    @staticmethod
    def role_of(messages: list[dict]) -> str:
        system = messages[0]['content'] if messages else ''
        for role, marker in ROLE_MARKERS:
            if marker in system:
                return role
        return 'unknown'

    def default(self, role: str, messages: list[dict]):
        if role == 'corrector':
            payload = json.loads(messages[1]['content'])
            return {'segments': [{'index': r['index'], 'text': r['current']} for r in payload['segments']]}
        if role == 'speaker_roles':
            return roles_reply(json.loads(messages[1]['content']))
        if role in ('speaker_fill', 'speaker_fill2'):
            return {'answers': [{'id': i, 'role': '不明', 'basis': '其他'} for i in json.loads(messages[1]['content'])['ask_ids']]}
        return 'ok'

    # -- transport --------------------------------------------------------
    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        messages = body['messages']
        role = self.role_of(messages)
        self.calls.append((role, messages))
        queue = self.script.get(role, [])
        item = queue.pop(0) if queue else self.default(role, messages)
        if callable(item):
            item = item(messages)
        if item == 'HTTP500':
            return httpx.Response(500, text='boom')
        if isinstance(item, Exception):
            raise item
        text = item if isinstance(item, str) else jdump(item)
        return httpx.Response(200, json={
            'id': 'x', 'object': 'chat.completion', 'created': 0, 'model': body.get('model', 'fake'),
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}})

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def fake_items(text: str, duration: float) -> list[dict]:
    """One aligner item per alphanumeric character, evenly spread over `duration`."""
    chars = [c for c in text if unicodedata.normalize('NFKC', c).casefold().isalnum()]
    if not chars:
        return []
    step = duration / len(chars)
    return [{'text': c, 'start': i * step, 'end': (i + 1) * step} for i, c in enumerate(chars)]


class FakeASR:
    """Replaces LocalASR. `replies` maps the call number (0, 1, ...) -> text or Exception."""

    def __init__(self, replies: dict | None = None, default: str = ''):
        self.replies = replies or {}
        self.default = default
        self.calls: list[float] = []
        self.loaded = False
        self.info = {'type': 'ready', 'device': 'fake'}

    @property
    def ready(self) -> bool:
        return self.loaded

    async def load(self, settings):
        self.loaded = True
        return self.info

    async def recognize(self, settings, samples) -> dict:
        index = len(self.calls)
        self.calls.append(len(samples) / RATE)
        item = self.replies.get(index, self.default)
        if isinstance(item, Exception):
            raise item
        duration = len(samples) / RATE
        return {'text': item, 'items': fake_items(item, duration), 'language': 'Chinese'}

    async def close(self):
        self.loaded = False


def silent_source(settings, seconds: float, recording_path=None, block: float = 0.1, *, start_samples: int = 0,
                  append: bool = False):
    """A real AudioSource fed by a fast fake microphone producing `seconds` of low-level noise.

    `settings` is a VoiceSettings (or anything with `.asr`) or an ASRSettings."""
    asr = getattr(settings, 'asr', settings)
    rng = np.random.default_rng(0)

    def make_iterator(stop):
        total = int(seconds * RATE)
        step = int(block * RATE)
        produced = 0
        while produced < total and not stop.is_set():
            yield (rng.standard_normal(step) * 0.01).astype(np.float32)
            produced += step

    return AudioSource(make_iterator, window_seconds=asr.window_seconds, overlap_seconds=asr.overlap_seconds,
                       recording_path=recording_path, start_samples=start_samples, append=append)


def make_settings(tmp_path=None, **overrides) -> VoiceSettings:
    """Voice settings with a configured corrector (the fake LLM answers whatever address it is given)."""
    settings = VoiceSettings(corrector=Endpoint(api_url='http://fake.invalid/v1', api_key='k', model_name='fake-model'))
    settings.retries = 3
    for key, value in overrides.items():
        setattr(settings, key, value)
    settings.validate()
    return settings
