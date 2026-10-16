"""Cut the aligned character stream into word-sized units (what gets a voiceprint) and into sentences (what gets a label).

The numbers come from the 2026-10-08 experiments (SPEC 4.3): a unit of about 0.8 s is short enough to hold one speaker and
long enough for a stable voiceprint; whole sentences or turns often hold two people.
"""
from __future__ import annotations

from dataclasses import dataclass

END_MARKS = '？。！?!'
TARGET = 0.8          # a unit ends once it spans this many seconds
PAUSE = 0.3           # ... or at a pause this long
MIN_UNIT = 0.4        # shorter fragments are merged into a neighbour, else dropped
MAX_MERGED = 1.3      # a merged unit may not exceed this
SENT_PAUSE = 0.5      # a pause this long ends a sentence


@dataclass(frozen=True)
class UnitParams:
    """The five numbers above, as one value a visit can change (SPEC 4.3, `tuning.py`); the defaults are the constants."""
    target: float = TARGET
    pause: float = PAUSE
    min_unit: float = MIN_UNIT
    max_merged: float = MAX_MERGED
    sent_pause: float = SENT_PAUSE


DEFAULT_PARAMS = UnitParams()


@dataclass(frozen=True)
class Token:
    """One aligned unit of ASR text: a character range of a segment's `added` text and its audio time."""
    seg: int
    c0: int
    c1: int
    start: float
    end: float
    sentence_end: bool = False       # carries sentence-final punctuation


def ends_sentence(text: str) -> bool:
    return any(ch in END_MARKS for ch in text)


def _span(tokens: list[Token], group: list[int]) -> tuple[float, float]:
    return tokens[group[0]].start, tokens[group[-1]].end


def groups_of(tokens: list[Token], params: UnitParams = DEFAULT_PARAMS) -> list[list[int]]:
    """Merged groups of token indexes, tiling `tokens` in order. Groups still shorter than MIN_UNIT are kept here (the caller drops them)."""
    groups: list[list[int]] = []
    current: list[int] = []
    for i, token in enumerate(tokens):
        if current:
            last = tokens[current[-1]]
            if last.sentence_end or token.start - last.end >= params.pause or last.end - tokens[current[0]].start >= params.target:
                groups.append(current)
                current = []
        current.append(i)
    if current:
        groups.append(current)
    pending, result, k = list(groups), [], 0
    while k < len(pending):
        group = pending[k]
        start, end = _span(tokens, group)
        if end - start < params.min_unit:
            previous = result[-1] if result else None
            following = pending[k + 1] if k + 1 < len(pending) else None
            if (previous is not None and not tokens[previous[-1]].sentence_end
                    and start - _span(tokens, previous)[1] <= params.pause and end - _span(tokens, previous)[0] <= params.max_merged):
                result[-1] = previous + group
                k += 1
                continue
            if (following is not None and not tokens[group[-1]].sentence_end
                    and _span(tokens, following)[0] - end <= params.pause and _span(tokens, following)[1] - start <= params.max_merged):
                pending[k + 1] = group + following
                k += 1
                continue
        result.append(group)
        k += 1
    return result


def build_units(tokens: list[Token], params: UnitParams = DEFAULT_PARAMS) -> list[list[int]]:
    """Units of a whole token list: the merged groups that last at least `min_unit`."""
    return [g for g in groups_of(tokens, params) if _span(tokens, g)[1] - _span(tokens, g)[0] >= params.min_unit]


class UnitStream:
    """Incremental `build_units`: tokens arrive in time order, units come out once nothing later can change them.

    The last two groups are held back (the newest one may still grow; the one before it may still merge into it or absorb it).
    The result is the same as `build_units` over all the tokens at once (tested on random streams).
    """

    def __init__(self, params: UnitParams = DEFAULT_PARAMS):
        self.params = params
        self.tokens: list[Token] = []
        self.settled = 0             # tokens before this index belong to finished groups: no later token can change them

    def extend(self, new: list[Token]):
        self.tokens.extend(new)

    def emit(self) -> list[list[int]]:
        """The units finished so far and not yet returned, as lists of global token indexes."""
        return self._emit(final=False)

    def add(self, new: list[Token]) -> list[list[int]]:
        self.extend(new)
        return self.emit()

    def finish(self) -> list[list[int]]:
        return self._emit(final=True)

    def _emit(self, *, final: bool) -> list[list[int]]:
        base = self.settled
        tail = self.tokens[base:]
        groups = groups_of(tail, self.params)
        done = groups if final else groups[:max(0, len(groups) - 2)]
        if final:
            self.settled = len(self.tokens)
        elif done:
            self.settled = base + groups[len(done)][0]
        return [[base + i for i in g] for g in done if _span(tail, g)[1] - _span(tail, g)[0] >= self.params.min_unit]
