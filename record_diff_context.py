from __future__ import annotations

import difflib
import re
from typing import Any

_ROUND_STEP_RE = re.compile(r"＜第\d+回合\(R(\d+)，Turn (\d+)-(\d+)\)")
_VERSION_STEP_RE = re.compile(r"Turn (\d+)-(\d+)")
_TRANSCRIPT_LINE_RE = re.compile(r"^語音#(\d+) (\d+(?::\d+){1,2})[–-](\d+(?::\d+){1,2})(?:\s|$)", re.M)


def _format_clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _recording_note(snapshot: dict[str, Any]) -> str:
    """Where this version sits on the recording clock, in the same `mm:ss` the voice transcript lines use."""
    rec_t = snapshot.get("rec_t")
    if rec_t is None:
        return "尚未錄音"
    if float(rec_t) <= 0:
        return "@錄音 00:00，當時還沒有任何逐字稿" + ("，錄音暫停中" if snapshot.get("rec_paused") else "")
    paused = "，錄音暫停中" if snapshot.get("rec_paused") else ""
    return f"@錄音 {_format_clock(float(rec_t))}{paused}"


def _snapshot_label(index: int, snapshot: dict[str, Any], with_recording: bool = False) -> str:
    source = snapshot.get("source") or "unknown"
    timestamp = snapshot.get("timestamp") or ""
    parts = [source]
    if timestamp:
        parts.append(timestamp)
    if snapshot.get("after_turn") is not None:
        after = int(snapshot["after_turn"])
        parts.append(f"第 {after} 主輪之後" if after > 0 else "尚無任何主輪")
    if with_recording:
        parts.append(_recording_note(snapshot))
    return f"版本 {index}（{', '.join(parts)}）"


def _interview_steps(dialogue: str) -> dict[int, tuple[int, int]]:
    """Interview round -> the Main Agent step (turn, step) that started it, from the round headers `＜第3回合(R3，Turn 2-3)…＞`."""
    rounds: dict[int, tuple[int, int]] = {}
    for match in _ROUND_STEP_RE.finditer(dialogue or ""):
        rounds.setdefault(int(match.group(1)), (int(match.group(2)), int(match.group(3))))
    return rounds


def _position(snapshot: dict[str, Any]) -> tuple[int, int] | None:
    """Where a version lies among the Main Agent's steps: an agent-written version is at its `Turn X-Y`; a physician's manual edit is
    after main turn N has ended (later than every step of N, earlier than any step of N+1). None when it cannot be told."""
    if snapshot.get("after_turn") is not None:
        return int(snapshot["after_turn"]), 10 ** 9
    match = _VERSION_STEP_RE.search(snapshot.get("source") or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def _round_list(numbers: list[int], prefix: str = "R") -> str:
    """[1, 2, 3, 5] -> 'R1~R3、R5'."""
    spans: list[list[int]] = []
    for number in sorted(numbers):
        if spans and number == spans[-1][1] + 1:
            spans[-1][1] = number
        else:
            spans.append([number, number])
    return "、".join(f"{prefix}{a}" if a == b else f"{prefix}{a}~{prefix}{b}" for a, b in spans)


def _clock_seconds(text: str) -> int:
    seconds = 0
    for part in text.split(":"):
        seconds = seconds * 60 + int(part)
    return seconds


def _transcript_ends(transcript: str) -> dict[int, int]:
    """{segment number: end in whole seconds} of the lines of a voice transcript as the prompts carry it (`語音#N mm:ss–mm:ss …`)."""
    return {int(match.group(1)): _clock_seconds(match.group(3)) for match in _TRANSCRIPT_LINE_RE.finditer(transcript or "")}


def _transcript_note(index: int, snapshot: dict[str, Any], ends: dict[int, int]) -> str:
    """Which transcript lines end at or before the point the transcript had reached when this version was made, and which after it, so the
    reviewer reads the order instead of comparing mm:ss itself. A rough boundary by whole seconds (the precision the lines show); it does
    not say that anyone read a line."""
    if not ends:
        return ""
    rec_t = snapshot.get("rec_t")
    if rec_t is None:
        return f"（版本 {index} 的逐字稿先後：全部逐字稿都晚於這個版本，它是錄音開始前做的。）"
    limit = int(float(rec_t))
    before = [number for number, end in ends.items() if end <= limit]
    placed = set(before)
    after = [number for number in ends if number not in placed]
    parts = []
    if before:
        parts.append(f"逐字稿 {_round_list(before, '#')} 約早於這個版本")
    if after:
        parts.append(f"逐字稿 {_round_list(after, '#')} 晚於這個版本")
    return f"（版本 {index} 的逐字稿先後（以時間粗略估計）：" + "；".join(parts) + "。）"


def _order_note(index: int, snapshot: dict[str, Any], rounds: dict[int, tuple[int, int]]) -> str:
    """Which interview rounds came before and which after this version, worked out from the step numbers, so the reviewer reads
    the order instead of comparing numbers itself."""
    position = _position(snapshot)
    if position is None or not rounds:
        return ""
    before = [number for number, step in rounds.items() if step < position]
    after = [number for number, step in rounds.items() if step >= position]
    parts = []
    if before:
        parts.append(f"問診回合 {_round_list(before)} 早於這個版本")
    if after:
        parts.append(f"問診回合 {_round_list(after)} 晚於這個版本")
    return f"（版本 {index} 的先後：" + "；".join(parts) + "。）"


def _edit_note(snapshot: dict[str, Any]) -> str:
    """What the physician's edit that made this version took away (source tags whose words were deleted or rewritten, low-confidence
    marks whose phrase was edited). Without it a `-` line with a tag and a `+` line without one could be read as a mistake."""
    meta = snapshot.get("meta")
    if not isinstance(meta, dict):
        return ""
    parts: list[str] = []
    removed = [item for item in meta.get("removed_tags") or [] if isinstance(item, dict) and item.get("tag")]
    if removed:
        shown = "；".join(
            f"{item.get('field') or ''}{'：' if item.get('field') else ''}{item['tag']}"
            + (f"（原涵蓋：{item['text']}）" if item.get("text") else "")
            for item in removed
        )
        parts.append(f"醫師改寫或刪除了標籤所涵蓋的文字，系統因此移除了來源標籤：{shown}")
    dropped = [item for item in meta.get("dropped_marks") or [] if isinstance(item, dict) and item.get("phrase")]
    if dropped:
        shown = "；".join(f"「{item['phrase']}」（{item.get('reason') or ''}）" for item in dropped)
        parts.append(f"醫師改了低信心標註的片段，系統因此移除了標註：{shown}")
    return "（" + "。".join(parts) + "。）" if parts else ""


def _unified_diff(old_text: str, new_text: str, old_label: str, new_label: str) -> str:
    old_lines = (old_text or "").splitlines()
    new_lines = (new_text or "").splitlines()
    return "\n".join(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=old_label,
            tofile=new_label,
            lineterm="",
            n=3,
        )
    )


def build_record_diff_context(
    snapshots: list[dict[str, Any]] | None,
    current_index: int | None = None,
    *,
    has_recording: bool = False,
    interview_dialogue: str = "",
    voice_transcript: str = "",
) -> str:
    """Format snapshot history up to current_index as plain-text diffs for LLM prompts.

    The recording clock is shown on the versions when this session has a recording: some version carries `rec_t`, or the caller says
    (`has_recording`) that a voice transcript exists. The second case matters when the recording began after the last version was
    made: no version has `rec_t` yet, but every one of them says 尚未錄音, which tells the reviewer that the whole transcript came after."""
    valid_snapshots = [snap for snap in (snapshots or []) if isinstance(snap, dict)]
    if current_index is None:
        current_index = len(valid_snapshots) - 1

    if current_index < 0 or not valid_snapshots:
        body = "（無病歷版本歷史）"
    else:
        current_index = max(0, min(current_index, len(valid_snapshots) - 1))
        active_snapshots = valid_snapshots[: current_index + 1]
        # The recording clock is shown only for a session that has a recording (some version carries it): a session without
        # one reads exactly as before. A version made before the recording began then says so.
        with_recording = has_recording or any(snap.get("rec_t") is not None for snap in active_snapshots)
        rounds = _interview_steps(interview_dialogue)
        ends = _transcript_ends(voice_transcript)
        if len(active_snapshots) < 2:
            body = "（目前只有一個版本，尚無版本間 diff）"
        else:
            parts: list[str] = []
            for idx in range(1, len(active_snapshots)):
                previous = active_snapshots[idx - 1]
                current = active_snapshots[idx]
                previous_label = _snapshot_label(idx, previous, with_recording)
                current_label = _snapshot_label(idx + 1, current, with_recording)

                note_diff = _unified_diff(
                    previous.get("note", ""),
                    current.get("note", ""),
                    f"{previous_label} NOTE",
                    f"{current_label} NOTE",
                )
                at_diff = _unified_diff(
                    previous.get("at", ""),
                    current.get("at", ""),
                    f"{previous_label} ASSESSMENT & TREATMENT",
                    f"{current_label} ASSESSMENT & TREATMENT",
                )

                changed_blocks: list[str] = []
                if note_diff:
                    changed_blocks.append(f"### 版本 {idx} -> 版本 {idx + 1}：NOTE\n```diff\n{note_diff}\n```")
                if at_diff:
                    changed_blocks.append(
                        f"### 版本 {idx} -> 版本 {idx + 1}：ASSESSMENT & TREATMENT\n"
                        f"```diff\n{at_diff}\n```"
                    )
                if changed_blocks:
                    parts.extend(changed_blocks)
                    note = _edit_note(current)
                    if note:
                        parts.append(note)
                    order = _order_note(idx + 1, current, rounds)
                    if order:
                        parts.append(order)
                    heard = _transcript_note(idx + 1, current, ends) if with_recording else ""
                    if heard:
                        parts.append(heard)

            body = "\n\n".join(parts) if parts else "（目前版本歷史中沒有內容差異）"

    return f"## 【病歷修改 diff 過程】\n{body}"
