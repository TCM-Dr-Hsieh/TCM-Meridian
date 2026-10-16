"""Map aligner units onto ASR source text; never time-align corrected text."""
from dataclasses import dataclass
import math
import unicodedata

# How far the aligner may place a word past the end of the audio it was given: where a word may START and where any word may END, 0.2 s.
# Only the END of the LAST word may go further, 0.8 s: a word that the end of the window cuts off is stretched past it (up to 0.52 s on
# the recordings it was measured on, 2 windows in 49 went past 0.2 s; none of this project's own 5230 windows did), though it starts no
# further out than the ordinary 0.2 s (a start may lie a little past the end of the audio: "system" began at 9.12 s of 9). Such a word is
# kept, its times held at the end of the audio; anything else further out means the times are broken, and the window is retried (and
# becomes a gap if it fails again). The same rule as the aligner of the voice_to_text project, where it was found.
SLACK_SECONDS = 0.2
EDGE_SECONDS = 0.8


class AlignmentError(RuntimeError):
    pass


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold() if c.isalnum())


@dataclass
class TimedText:
    text: str
    start: float
    end: float
    begin: int
    finish: int


def map_alignment(text, items, offset, duration):
    chars, positions = [], []
    for index, char in enumerate(text):
        for value in normalized(char):
            chars.append(value)
            positions.append(index)
    compact = ''.join(chars)
    if not compact:
        return []
    if not items:
        raise AlignmentError('原稿有文字但對齊結果為空，已暫停以避免漏字。')
    result, cursor, last_start, last_end = [], 0, -1.0, -1.0
    for index, item in enumerate(items, 1):
        try:
            token = normalized(item['text'])
            start, end = float(item['start']), float(item['end'])
        except (KeyError, TypeError, ValueError) as exc:
            raise AlignmentError('對齊欄位無效。') from exc
        if not token or not compact.startswith(token, cursor):
            raise AlignmentError('對齊文字與 ASR 原稿不一致，請重試該段。')
        end_limit = duration + (EDGE_SECONDS if index == len(items) else SLACK_SECONDS)
        if (not math.isfinite(start) or not math.isfinite(end) or
                start < 0 or end < start or start > duration + SLACK_SECONDS or end > end_limit or
                start < last_start or end < last_end):
            raise AlignmentError(
                '對齊時間超出音訊或不是遞增順序。'
                f'第 {index} 項 {str(item["text"])[:30]!r}：{start:g}–{end:g} 秒；'
                f'音訊長度 {duration:g} 秒，前一項 {last_start:g}–{last_end:g} 秒。')
        begin = positions[cursor]
        cursor += len(token)
        finish = positions[cursor - 1] + 1
        result.append(TimedText(text[begin:finish], offset + min(start, duration), offset + min(end, duration), begin, finish))
        last_start, last_end = start, end
    if cursor != len(compact):
        raise AlignmentError('對齊未覆蓋所有原稿，已保留音訊等待重試。')
    # Each unit owns adjacent punctuation/spacing through the next unit.
    for i, unit in enumerate(result):
        unit.finish = result[i + 1].begin if i + 1 < len(result) else len(text)
    result[0].begin = 0
    return result


def select_uncommitted(text, units, committed_end, commit_until=None, previous_unit=None):
    """Return a contiguous new prefix and the still-pending right context.

    The watermark is the end of the last accepted ASR unit, not wall clock.
    A boundary-straddling different token is retained with a warning.
    """
    first = 0
    warning = ''
    while first < len(units):
        unit = units[first]
        duplicate = unit.end <= committed_end
        if previous_unit and unit.start < committed_end < unit.end:
            intersection = min(unit.end, previous_unit.end) - max(unit.start, previous_unit.start)
            shortest = max(0.01, min(unit.end - unit.start, previous_unit.end - previous_unit.start))
            duplicate = normalized(unit.text) == normalized(previous_unit.text) and intersection / shortest >= 0.5
            if not duplicate:
                warning = '接縫時間有交疊，已保留邊界文字；請檢查是否重複。'
        if not duplicate:
            break
        first += 1
    stop = first
    while stop < len(units):
        unit = units[stop]
        if commit_until is not None and (unit.start + unit.end) / 2 >= commit_until:
            break
        stop += 1
    selected = units[first:stop]
    added = text[selected[0].begin:selected[-1].finish] if selected else ''
    pending = text[units[stop].begin:] if stop < len(units) else ''
    return added, pending, selected, warning


def select_new_window(text, units, covered_until):
    """Keep only ASR units whose midpoint is beyond the previous window end.

    The previous window owns its entire audio interval, including the overlap.
    The aligner is applied to the raw ASR text, never to LLM output.
    """
    first = next((index for index, unit in enumerate(units)
                  if (unit.start + unit.end) / 2 >= covered_until), len(units))
    selected = units[first:]
    return (text[selected[0].begin:] if selected else ''), selected
