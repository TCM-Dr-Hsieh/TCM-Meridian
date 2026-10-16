"""Turn per-token speaker labels into readable runs of text, on the CURRENT (LLM-corrected) text of a segment.

Labels live on the timed ASR text (`Segment.added`). The corrector may have rewritten the segment since, and its prompt and
JSON contract do not carry speakers, so the label boundaries are projected onto the corrected text with a diff of the two.
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from . import BACKGROUND, LABEL_NAMES, TEXT_SOURCES, UNKNOWN

TEXT_MARK = '*'            # a label that came from reading the text, not from the voice (less reliable)


@dataclass(frozen=True)
class Run:
    label: str             # DOCTOR | OTHER | UNKNOWN | BACKGROUND
    source: str            # '' | 'voice' | 'text' | 'text2'
    text: str


def project(source: str, target: str, positions: list[int]) -> list[int]:
    """Where each (non-decreasing) position of `source` falls in `target`."""
    if source == target:
        return list(positions)
    opcodes = SequenceMatcher(None, source, target, autojunk=False).get_opcodes()
    result, k = [], 0
    for p in positions:
        while k < len(opcodes) - 1 and opcodes[k][2] < p:
            k += 1
        tag, i1, i2, j1, j2 = opcodes[k] if opcodes else ('equal', 0, 0, 0, 0)
        if p <= i1:
            result.append(j1)
        elif p >= i2:
            result.append(j2)
        elif tag == 'equal':
            result.append(j1 + (p - i1))
        else:
            result.append(j1 + round((p - i1) / (i2 - i1) * (j2 - j1)))
    return result


def segment_runs(added: str, corrected: str, labels: tuple | None) -> list[Run]:
    """The segment's text cut into runs of one label. `labels` is ((c0, c1, label, source), ...) over `added`, as the tracker publishes."""
    if not labels or not corrected.strip():
        return [Run(UNKNOWN, '', corrected.strip())] if corrected.strip() else []
    starts = [label[0] for label in labels] + [len(added)]
    cuts = project(added, corrected, starts)
    cuts[0], cuts[-1] = 0, len(corrected)
    runs: list[Run] = []
    for k, (_, _, label, source) in enumerate(labels):
        text = corrected[cuts[k]:cuts[k + 1]]
        if runs and runs[-1].label == label and runs[-1].source == source:
            runs[-1] = Run(label, source, runs[-1].text + text)
        else:
            runs.append(Run(label, source, text))
    return [Run(r.label, r.source, r.text.strip()) for r in runs if r.text.strip()]


def prompt_name(run: Run) -> str:
    return LABEL_NAMES[run.label] + (TEXT_MARK if run.source in TEXT_SOURCES and run.label != UNKNOWN else '')


def prompt_body(runs: list[Run], keep_background: bool = False) -> str:
    """`醫師: … -> 患者或家屬: … -> 不明: …` (one entry when the whole segment is one speaker). Background voices (another room) are left
    out, so they never reach the record writer; `keep_background` keeps them as `背景: …` (the saved transcript file)."""
    return ' -> '.join(f'{prompt_name(r)}: {r.text}' for r in runs if keep_background or r.label != BACKGROUND)
