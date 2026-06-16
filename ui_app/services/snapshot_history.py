from __future__ import annotations

from datetime import datetime


class SnapshotHistory:
    """Undo/redo history for NOTE and ASSESSMENT & TREATMENT snapshots."""

    def __init__(self):
        self.snapshots: list[dict] = []
        self.current_index: int = -1

    def push(self, note: str, at: str, source: str = "init"):
        self.snapshots = self.snapshots[: self.current_index + 1]
        self.snapshots.append({
            "note": note,
            "at": at,
            "source": source,
            "timestamp": datetime.now().isoformat(),
        })
        self.current_index = len(self.snapshots) - 1

    def can_undo(self) -> bool:
        return self.current_index > 0

    def can_redo(self) -> bool:
        return self.current_index < len(self.snapshots) - 1

    def undo(self) -> dict | None:
        if self.can_undo():
            self.current_index -= 1
            return self.snapshots[self.current_index]
        return None

    def redo(self) -> dict | None:
        if self.can_redo():
            self.current_index += 1
            return self.snapshots[self.current_index]
        return None

    def get_current(self) -> dict | None:
        if 0 <= self.current_index < len(self.snapshots):
            return self.snapshots[self.current_index]
        return None

    def get_previous(self) -> dict | None:
        if self.current_index > 0:
            return self.snapshots[self.current_index - 1]
        return None

    def reset(self):
        self.snapshots.clear()
        self.current_index = -1
