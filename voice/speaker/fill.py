"""Text fill: an LLM answers who spoke the sentences the voice could not decide; the caller accepts an answer only when the
sentence's weak voice evidence agrees with it. Voice labels are never touched (the model is only asked about "?" lines)."""
from __future__ import annotations

import json

from ..config import PROMPTS_DIR
from ..llm import ValidationError
from ..textutil import extract_json, format_clock
from . import DOCTOR, OTHER, UNKNOWN

PROMPT_FILE = 'prompt_speaker_fill.txt'
PROMPT_FILE2 = 'prompt_speaker_fill2.txt'
CHUNK, BEFORE, AFTER = 25, 40, 10          # sentences asked per call; sentences shown before and after them
RETRY_NOTE = '\n上一次輸出無效，請只輸出符合格式的 JSON：answers 要為 ask_ids 裡的每個 id 各一筆（沒把握的 role 填「不明」）。'
ANSWER_NAMES = {'醫師': DOCTOR, '患者或家屬': OTHER, '患者家屬': OTHER, '不明': None}       # the old spelling is still understood
SHOWN_NAMES = {DOCTOR: '醫師', OTHER: '患者或家屬', UNKNOWN: '?'}


def load_prompt() -> str:
    return (PROMPTS_DIR / PROMPT_FILE).read_text(encoding='utf-8')


def load_prompt2() -> str:
    return (PROMPTS_DIR / PROMPT_FILE2).read_text(encoding='utf-8')


def chunks(ids: list[int], size: int = CHUNK) -> list[list[int]]:
    """Group sentence numbers into runs that start within `size` sentences of each other."""
    out: list[list[int]] = []
    for i in sorted(ids):
        if out and i < out[-1][0] + size:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def window(ask: list[int], decided: int, size: int = CHUNK, before: int = BEFORE, after: int = AFTER) -> range:
    """The sentences shown as context for one call: `before` before the first asked, `after` after the window, decided ones only."""
    start = ask[0]
    return range(max(0, start - before), min(decided, start + size + after))


def request_payload(ask: list[int], rows: list[dict]) -> str:
    return json.dumps({'ask_ids': ask, 'dialogue': rows}, ensure_ascii=False)


def row(index: int, start: float, previous_end: float | None, label: str, text: str, inferred: bool = False) -> dict:
    """One line of the dialogue shown to the model; `inferred` adds a star to a label that an earlier fill read from the text."""
    gap = 0.0 if previous_end is None else round(max(0.0, start - previous_end), 1)
    who = SHOWN_NAMES[label] + ('*' if inferred and label != UNKNOWN else '')
    return {'id': index, 't': format_clock(start), 'gap': gap, 'who': who, 'text': text}


def parse_answers(text: str, ask: list[int]) -> dict[int, str]:
    """{sentence: DOCTOR | OTHER} for the sentences the model dared to decide; raises ValidationError when ids are missing."""
    try:
        answers = {int(a['id']): a for a in extract_json(text)['answers']}
        missing = [i for i in ask if i not in answers]
        if missing:
            raise ValueError(f'缺少這些 id 的回答：{missing}')
        result = {}
        for i in ask:
            role = ANSWER_NAMES.get(answers[i].get('role'))
            if role is not None:
                result[i] = role
        return result
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ValidationError(f'補標回應無效：{exc}') from exc
