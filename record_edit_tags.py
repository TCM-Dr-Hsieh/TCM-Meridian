"""Source tags and low-confidence marks of the daily record (NOTE, ASSESSMENT & TREATMENT) in the edit box.

The record carries two kinds of machine-made markup:

* source tags in square brackets -- `[問診紀錄_患者R2]`, `[語音#12]`, `[語音#12-14]`, `[語音#12, #14]`, `[人類醫師_T3]`,
  `[人類醫師_手動修改]`, `[上次病歷]`, `[歷史病歷]`, `[Forum_D4]`, and the file tags `[2026-06-02-Lab_data]`;
* the low-confidence marks the LC agent writes into the NOTE: `**phrase（reason）**` (NOTE only).

The edit box shows neither (`editable_text`): the physician cannot damage them and they do not get in the way of reading. On
save `merge_human_edit` puts them back by diffing the text the box was filled with against the edited text:

* a source tag stays where the words it covers are; it is REMOVED (and reported) when none of those words survive -- the only
  place where the program takes a source tag away, because the tag no longer describes anything. `[人類醫師_手動修改]` is never
  removed while its line exists;
* a low-confidence mark is put back only when its phrase is exactly as it was; if the phrase was edited at all, the mark is
  dropped (the physician looked at it and changed it; the next low-confidence scan can mark it again);
* every added or changed non-empty line ends with `[人類醫師_手動修改]` (`tag_human_edits`'s rule: not added twice at the end).

Source tags typed or pasted into the box are refused (`NoteEditError`): the record cannot tell a typed `[語音#9]` from a real one,
so it must not get in. The reference window (`ui_app.rendering.reference_html`) shows the version being edited with its tags.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

HUMAN_EDIT_TAG = "[人類醫師_手動修改]"
MANUAL_TAG = HUMAN_EDIT_TAG


class NoteEditError(RuntimeError):
    """A hand edit of the record that is refused; the message is for the physician."""


# A voice tag is one segment number followed by any further numbers: a range (`-4`) or a list (`, #4`). The prompts teach
# `[語音#N-M]` and `[語音#N, #M]`; hiding is deliberately lenient (`,#4`, `,4`, a full-width comma or 頓號, an en or em dash).
# Only the hiding is lenient; whether a tag is right is checked by the hallucination reviewer, not by this module.
_SEGMENTS = r"\d+(?:\s*[-–—,，、]\s*#?\d+)*"
_TAG_BODY = (
    r"(?:問診紀錄_(?:患者|家屬|實習醫師|病歷系統)R\d+"
    r"|人類醫師_(?:T\d+|手動修改)"
    r"|上次病歷|歷史病歷"
    r"|Forum_D\d+"
    rf"|語音#{_SEGMENTS}"
    r"|\d{4}-\d{2}-\d{2}-[^\[\]\n]+)"           # a patient file, named by its file name without the extension
)
_BARE_TAG_RE = re.compile(rf"\[{_TAG_BODY}\]")
_TAG_RE = re.compile(rf"[ \t　]*\[{_TAG_BODY}\]")       # a tag together with the blanks in front of it
# Every place a bold span could begin (`**`), with what lies up to the next `**` on the line. Overlapping on purpose: the closing
# `**` of a bold span that is not a mark may open the mark that follows it.
_BOLD_START_RE = re.compile(r"\*\*(?=([^*\n]+?)\*\*)")


def _reason_start(content: str) -> int | None:
    """Where the reason of a low-confidence mark begins in the text between its `**`: the full-width parenthesis that pairs with the
    last one. The LC agent writes `**phrase（reason）**` with the model's reason as it is, so the reason may hold parentheses of its own,
    and the phrase may too (`**血壓（坐姿）130（來源（R2）不明）**`). None when it is not a mark: no closing parenthesis at the end, the
    parentheses do not pair, or there is no phrase before them."""
    if not content.endswith("）"):
        return None
    depth = 0
    for index in range(len(content) - 1, -1, -1):
        if content[index] == "）":
            depth += 1
        elif content[index] == "（":
            depth -= 1
            if depth == 0:
                return index if index > 0 else None
    return None


def _mark_spans(text: str) -> list[tuple[int, int, str, str]]:
    """The low-confidence marks of `text` as (start, end, phrase, reason)."""
    spans: list[tuple[int, int, str, str]] = []
    taken_until = 0
    for match in _BOLD_START_RE.finditer(text):
        if match.start() < taken_until:
            continue
        content = match.group(1)
        split = _reason_start(content)
        if split is None:
            continue
        end = match.start() + len(content) + 4
        spans.append((match.start(), end, content[:split], content[split + 1:-1]))
        taken_until = end
    return spans

# A run of fewer than this many letters/digits that is left over after an edit is treated as chance (the physician typed
# "大便" again while rewriting "大便1天2次，偏軟"); it does not keep a tag alive. A span with fewer letters than this needs all
# of them (a two-letter span the physician only extended must keep its tag). Settings: `note_edit.min_surviving_run`.
MIN_SURVIVING_RUN = 3


def strip_source_tags(text: str) -> str:
    """The text without its source tags (and the blanks in front of them)."""
    return _TAG_RE.sub("", text or "")


def typed_tags(text: str) -> list[str]:
    """Source tags in `text` (what the physician typed or pasted into the edit box, which shows none), in order, once each."""
    return list(dict.fromkeys(_BARE_TAG_RE.findall(text or "")))


@dataclass(frozen=True)
class RemovedTag:
    tag: str            # e.g. '[語音#7]'
    text: str           # the words it covered (as they were before the edit), for the audit and the preview


@dataclass(frozen=True)
class DroppedMark:
    phrase: str         # the phrase a low-confidence mark was on
    reason: str


@dataclass(frozen=True)
class EditMerge:
    text: str                                   # the new record text, tags and marks included
    removed: tuple[RemovedTag, ...] = ()        # source tags that are gone ([人類醫師_手動修改] is not listed: it is added back anyway)
    dropped_marks: tuple[DroppedMark, ...] = () # low-confidence marks whose phrase was edited or deleted


@dataclass
class _Parsed:
    plain: str                                  # the text without tags and marks
    tags: list[tuple[int, str]]                 # (offset in `plain`, the tag with the blanks in front of it)
    marks: list[tuple[int, int, str, str]]      # (start, end in `plain`, phrase, reason)


def _events(text: str, marks: bool) -> list[tuple[int, int, str, object]]:
    """The markup of `text` in order, none overlapping another: (start, end, 'tag', token) and (start, end, 'mark', (phrase, reason))."""
    tag_spans = [(m.start(), m.end(), m.group()) for m in _TAG_RE.finditer(text)]
    mark_spans: list[tuple[int, int, str, str]] = []
    if marks:
        for mark_start, mark_end, phrase, reason in _mark_spans(text):
            # A mark with a source tag inside is not one the LC agent wrote (it never overlaps a tag): shown as it is.
            if any(start < mark_end and end > mark_start for start, end, _ in tag_spans):
                continue
            mark_spans.append((mark_start, mark_end, phrase, reason))
    return sorted(
        [(s, e, "tag", tok) for s, e, tok in tag_spans] + [(s, e, "mark", (phrase, reason)) for s, e, phrase, reason in mark_spans],
        key=lambda item: item[0],
    )


def _parse(text: str, marks: bool) -> _Parsed:
    """Split `text` into its plain words, its source tags and (with `marks`) its low-confidence marks."""
    plain: list[str] = []
    tags: list[tuple[int, str]] = []
    out_marks: list[tuple[int, int, str, str]] = []
    pos = length = 0
    for start, end, kind, payload in _events(text, marks):
        if start < pos:                          # overlapping markup: leave it as text
            continue
        chunk = text[pos:start]
        plain.append(chunk)
        length += len(chunk)
        if kind == "tag":
            tags.append((length, payload))
        else:
            phrase, reason = payload
            plain.append(phrase)
            out_marks.append((length, length + len(phrase), phrase, reason))
            length += len(phrase)
        pos = end
    plain.append(text[pos:])
    return _Parsed("".join(plain), tags, out_marks)


def render_pieces(text: str, marks: bool = False) -> list[tuple[str, str, str]]:
    """`text` as [(kind, text, reason)] for the reference window: kind 'text' | 'tag' | 'mark' (`text` is the phrase of a mark,
    `reason` its reason; both empty otherwise). A tag keeps the blanks in front of it as text."""
    out: list[tuple[str, str, str]] = []
    pos = 0
    for start, end, kind, payload in _events(text or "", marks):
        if start < pos:
            continue
        if start > pos:
            out.append(("text", text[pos:start], ""))
        if kind == "tag":
            token = payload
            stripped = token.lstrip(" \t　")
            if len(token) > len(stripped):
                out.append(("text", token[: len(token) - len(stripped)], ""))
            out.append(("tag", stripped, ""))
        else:
            out.append(("mark", payload[0], payload[1]))
        pos = end
    if pos < len(text or ""):
        out.append(("text", text[pos:], ""))
    return out


def editable_text(record: str, *, marks: bool = False) -> str:
    """What the edit box shows: the record without its source tags (and, for the NOTE, without its low-confidence marks)."""
    return _parse(record or "", marks).plain


def tokenize(text: str) -> list[tuple[str, bool]]:
    """`text` as [(piece, is_tag)]: the source tags (as written) and the text between them."""
    pieces, pos = [], 0
    for match in _BARE_TAG_RE.finditer(text):
        if match.start() > pos:
            pieces.append((text[pos:match.start()], False))
        pieces.append((match.group(), True))
        pos = match.end()
    if pos < len(text):
        pieces.append((text[pos:], False))
    return pieces


def _append_manual(line: str) -> str:
    # A tag already at the END of the line is not added again. So consecutive manual edits leave ONE tag; a second one appears
    # whenever there is anything after the first tag when the line is edited again -- an AI-appended source or text the physician
    # typed after it. A repeated tag therefore means "edited while the tag was not last", NOT a count of edits: the
    # 病歷修改 diff 過程 has the full history.
    return line if line.rstrip().endswith(HUMAN_EDIT_TAG) else line + HUMAN_EDIT_TAG


def tag_human_edits(old_text: str, new_text: str) -> str:
    """Append [人類醫師_手動修改] to every added/changed non-empty line unless the line already ends with it."""
    old_lines = (old_text or "").split("\n")
    new_lines = (new_text or "").split("\n")
    result: list[str] = []
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    for op, _i1, _i2, j1, j2 in matcher.get_opcodes():
        for line in new_lines[j1:j2]:
            result.append(_append_manual(line) if op != "equal" and line.strip() else line)
    return "\n".join(result)


# ---------------------------------------------------------------------------------------------------------------------
# The edit box shows the record WITHOUT its tags and marks; on save they are put back (and some taken away) by the program.
# ---------------------------------------------------------------------------------------------------------------------
def _content_mask(plain: str) -> list[bool]:
    """Per character: does it carry the source's words? Letters (CJK included) and digits do; punctuation and blanks do not.
    Templates differ from physician to physician, so no part of a line is assumed to be a field name: one that is left
    standing counts as words, and a tag may then stay when its sentence was rewritten (kept too long, never lost)."""
    return [ch.isalnum() for ch in plain]


def _groups(tags: list[tuple[int, str]]) -> list[tuple[int, int, list[str]]]:
    """[(start, offset, tags)]: tags written one after another (`…[語音#1][語音#2]`) are one group; it covers the text
    from the previous group to its own position."""
    grouped: list[tuple[int, list[str]]] = []
    for offset, token in tags:
        if grouped and grouped[-1][0] == offset:
            grouped[-1][1].append(token)
        else:
            grouped.append((offset, [token]))
    out, start = [], 0
    for offset, tokens in grouped:
        out.append((start, offset, tokens))
        start = offset
    return out


def _survives(ops, mask: list[bool], start: int, end: int, min_run: int = MIN_SURVIVING_RUN) -> bool:
    """Is the text of [start:end) of the old text still there after the edit (`ops`: old text -> new)? A span that had no
    words to begin with (only punctuation, or nothing: two tags side by side) has nothing to lose."""
    total = sum(mask[start:end])
    if total == 0:
        return True
    need = min(min_run, total)
    for op, i1, i2, _j1, _j2 in ops:
        if op != "equal":
            continue
        lo, hi = max(i1, start), min(i2, end)
        if lo < hi and sum(mask[lo:hi]) >= need:
            return True
    return False


def _new_offset(ops, offset: int, new_length: int) -> int:
    """Where a position of the old text is in the new one. A position at the end of unchanged text stays before text
    typed right after it, so a tag keeps covering only what it covered."""
    for op, i1, i2, j1, j2 in ops:
        if not i1 <= offset <= i2:
            continue
        if op == "equal":
            return j1 + offset - i1
        if offset == i1:
            return j1
        if offset < i2:                         # inside a rewritten stretch: after what replaced it
            return j2 if op == "replace" else j1
    return new_length


def _unchanged_range(ops, start: int, end: int) -> tuple[int, int] | None:
    """Where [start:end) of the old text is in the new one if all of it is unchanged (one 'equal' stretch covers it), else None."""
    for op, i1, i2, j1, _j2 in ops:
        if op == "equal" and i1 <= start and end <= i2:
            return j1 + start - i1, j1 + end - i1
    return None


def _snippet(plain: str, start: int, end: int) -> str:
    return re.sub(r"\s+", " ", plain[start:end]).strip()


def _merge_block(old_block: str, new_block: str, min_run: int, marks: bool) -> tuple[str, list[RemovedTag], list[DroppedMark]]:
    parsed = _parse(old_block, marks)
    if not parsed.tags and not parsed.marks:
        return new_block, [], []
    old_plain = parsed.plain
    ops = difflib.SequenceMatcher(None, old_plain, new_block, autojunk=False).get_opcodes()
    mask = _content_mask(old_plain)

    # (position in the new text, order at that position, text). At one position a mark's end comes first, then the tags,
    # then a mark's start: a tag stays outside the mark it follows or precedes, as the LC agent writes them.
    inserts: list[tuple[int, int, str]] = []
    removed: list[RemovedTag] = []
    floor = 0
    for start, offset, tokens in _groups(parsed.tags):
        alive = _survives(ops, mask, start, offset, min_run)
        # [人類醫師_手動修改] is never taken away (the prompts require every one of them to stay, and the audit keeps the history);
        # only the other source tags go with their words.
        stay = tokens if alive else [token for token in tokens if token.strip() == HUMAN_EDIT_TAG]
        if stay:
            floor = max(floor, _new_offset(ops, offset, len(new_block)))
            inserts.extend((floor, 1, token) for token in stay)
        if not alive:
            text = _snippet(old_plain, start, offset)
            removed.extend(RemovedTag(token.strip(), text) for token in tokens if token.strip() != HUMAN_EDIT_TAG)

    dropped: list[DroppedMark] = []
    for start, end, phrase, reason in parsed.marks:
        where = _unchanged_range(ops, start, end)
        if where is None:
            dropped.append(DroppedMark(phrase, reason))
            continue
        inserts.append((where[0], 2, "**"))
        inserts.append((where[1], 0, f"（{reason}）**"))

    inserts.sort(key=lambda item: (item[0], item[1]))
    pieces, prev = [], 0
    for pos, _order, token in inserts:
        pieces += [new_block[prev:pos], token]
        prev = pos
    pieces.append(new_block[prev:])
    return "".join(pieces), removed, dropped


def merge_human_edit(old_record: str, edited: str, min_run: int = MIN_SURVIVING_RUN, *, marks: bool = False) -> EditMerge:
    """The record after the physician edited `editable_text(old_record, marks=marks)` into `edited`.

    Unchanged lines are kept exactly as they were (tags and marks and all). In the lines that changed:
    * each tag stays where the words it covers are; a tag covers the words from the previous tag up to itself, and tags
      written one after another cover the same words;
    * a source tag is REMOVED when none of the words it covered survive (letters and digits count, punctuation does not; a left
      over run shorter than `min_run` is chance), whatever the physician typed instead: the new words are the physician's, and
      the old tag no longer describes them. Tags with no words in front of them to begin with (side by side) are not touched by
      this rule; `[人類醫師_手動修改]` tags are never removed while the line exists, however many there are;
    * (`marks`, the NOTE) a low-confidence mark is put back when its phrase is unchanged and dropped otherwise;
    * the line gets `[人類醫師_手動修改]` at its end, as `tag_human_edits` does.
    Tags and marks of lines deleted outright go with the lines. The edit box shows no tags, so tag syntax in `edited` was
    typed or pasted (the reference window can be copied from): `NoteEditError`.
    """
    typed = typed_tags(edited)
    if typed:
        raise NoteEditError(f"修改框裡有來源標籤（{'、'.join(typed)}）。來源標籤只能由系統加上，不能手打或貼上，"
                            "請先把它刪除（或按「放棄修改」）再儲存。")
    old_lines = (old_record or "").split("\n")
    new_lines = (edited or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    clean_old = [_parse(line, marks).plain for line in old_lines]
    out: list[str] = []
    removed: list[RemovedTag] = []
    dropped: list[DroppedMark] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, clean_old, new_lines, autojunk=False).get_opcodes():
        if op == "equal":
            out.extend(old_lines[i1:i2])
            continue
        if op == "delete":
            parsed = _parse("\n".join(old_lines[i1:i2]), marks)
            for start, offset, tokens in _groups(parsed.tags):
                removed.extend(RemovedTag(token.strip(), _snippet(parsed.plain, start, offset)) for token in tokens)
            dropped.extend(DroppedMark(phrase, reason) for _s, _e, phrase, reason in parsed.marks)
            continue
        if op == "replace":
            merged, gone, lost = _merge_block("\n".join(old_lines[i1:i2]), "\n".join(new_lines[j1:j2]), min_run, marks)
            removed.extend(gone)
            dropped.extend(lost)
            lines = merged.split("\n")
        else:                                   # insert
            lines = new_lines[j1:j2]
        out.extend(_append_manual(line) if line.strip() else line for line in lines)
    return EditMerge("\n".join(out), tuple(r for r in removed if r.tag != HUMAN_EDIT_TAG), tuple(dropped))
