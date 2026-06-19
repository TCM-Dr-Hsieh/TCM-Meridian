from __future__ import annotations

import difflib
from typing import Any


def _snapshot_label(index: int, snapshot: dict[str, Any]) -> str:
    source = snapshot.get("source") or "unknown"
    timestamp = snapshot.get("timestamp") or ""
    if timestamp:
        return f"版本 {index}（{source}, {timestamp}）"
    return f"版本 {index}（{source}）"


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
        if len(active_snapshots) < 2:
            body = "（目前只有一個版本，尚無版本間 diff）"
        else:
            parts: list[str] = []
            for idx in range(1, len(active_snapshots)):
                previous = active_snapshots[idx - 1]
                current = active_snapshots[idx]
                previous_label = _snapshot_label(idx, previous)
                current_label = _snapshot_label(idx + 1, current)

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

            body = "\n\n".join(parts) if parts else "（目前版本歷史中沒有內容差異）"

    return f"## 【病歷修改 diff 過程】\n{body}"
