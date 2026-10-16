from __future__ import annotations

import difflib
from typing import Any


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
    paused = "，錄音暫停中" if snapshot.get("rec_paused") else ""
    return f"@錄音 {_format_clock(float(rec_t))}{paused}"


def _snapshot_label(index: int, snapshot: dict[str, Any], with_recording: bool = False) -> str:
    source = snapshot.get("source") or "unknown"
    timestamp = snapshot.get("timestamp") or ""
    parts = [source]
    if timestamp:
        parts.append(timestamp)
    if with_recording:
        parts.append(_recording_note(snapshot))
    return f"版本 {index}（{', '.join(parts)}）"


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
) -> str:
    """Format snapshot history up to current_index as plain-text diffs for LLM prompts."""
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
        with_recording = any(snap.get("rec_t") is not None for snap in active_snapshots)
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

            body = "\n\n".join(parts) if parts else "（目前版本歷史中沒有內容差異）"

    return f"## 【病歷修改 diff 過程】\n{body}"
