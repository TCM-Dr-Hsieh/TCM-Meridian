"""Small text helpers: joining, <think> stripping, JSON extraction, script conversion."""
from __future__ import annotations

import json
import re
import unicodedata


def is_content(char: str) -> bool:
    return not char.isspace() and not unicodedata.category(char).startswith('P')


def content_chars(text: str) -> str:
    return ''.join(c for c in text if is_content(c))


def join_text(left: str, right: str) -> str:
    if left and right and left[-1].isascii() and left[-1].isalnum() and right[0].isascii() and right[0].isalnum():
        return left + ' ' + right
    return left + right


def join_texts(parts) -> str:
    result = ''
    for part in parts:
        result = join_text(result, part)
    return result


_CJK_RE = re.compile('[⺀-鿿豈-﫿＀-￯]')


def estimate_tokens(text: str) -> int:
    """Deliberately pessimistic token estimate: 1 token per CJK character, ~0.4 per other character."""
    cjk = len(_CJK_RE.findall(text))
    return cjk + -(-(len(text) - cjk) * 2 // 5)


def estimate_messages_tokens(messages: list[dict]) -> int:
    return sum(estimate_tokens(str(m.get('content', ''))) + 8 for m in messages) + 3


def clean_think(text: str) -> str:
    """Remove inline <think> blocks that some servers expose in message content."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
    if '<think>' in text:
        text = text.split('<think>', 1)[0]
    return text.strip()


def extract_json(text: str):
    """Parse a JSON object from LLM output, tolerating code fences and surrounding prose.

    Raw tabs/newlines inside strings are accepted (`strict=False`): models copy tab-indented template lines and
    long reasoning verbatim into string values, and a strict parse would waste a whole call on that. Callers that
    care validate the content afterwards (e.g. an operation's `content` may not contain a newline).
    """
    text = clean_think(text).strip()
    fence = re.match(r'^```(?:json)?\s*(.*?)\s*```$', text, flags=re.S)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text, strict=False)
    except ValueError:
        pass
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end <= start:
        raise ValueError('回應中找不到 JSON 物件。')
    return json.loads(text[start:end + 1], strict=False)


class ScriptConverter:
    """Lazy OpenCC wrapper. Falls back to identity when OpenCC is unavailable."""

    def __init__(self):
        self._converters: dict[str, object] = {}
        self.available = True
        self.error = ''

    def _get(self, config: str):
        if config not in self._converters:
            try:
                from opencc import OpenCC
                self._converters[config] = OpenCC(config)
            except Exception as exc:  # pragma: no cover - depends on environment
                self.available = False
                self.error = f'OpenCC 無法使用：{exc}'
                self._converters[config] = None
        return self._converters[config]

    def convert(self, text: str, config: str) -> str:
        if not text:
            return text
        converter = self._get(config)
        return converter.convert(text) if converter is not None else text

    def to_traditional(self, text: str) -> str:
        """Simplified -> Traditional (character/variant level, no phrase substitution)."""
        return self.convert(text, 's2tw')

    def to_simplified(self, text: str) -> str:
        return self.convert(text, 'tw2s')


script = ScriptConverter()


def format_clock(seconds: float | None) -> str:
    if seconds is None:
        return '--:--'
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f'{hours}:{minutes:02d}:{secs:02d}' if hours else f'{minutes:02d}:{secs:02d}'
