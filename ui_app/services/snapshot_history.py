from __future__ import annotations

from datetime import datetime
from uuid import uuid4


class SnapshotHistory:
    """Undo/redo history for NOTE and ASSESSMENT & TREATMENT snapshots."""

    def __init__(self):
        self.snapshots: list[dict] = []
        self.current_index: int = -1

    def push(self, note: str, at: str, source: str = "init") -> list[dict]:
        truncated = self.snapshots[self.current_index + 1 :]
        self.snapshots = self.snapshots[: self.current_index + 1]
        self.snapshots.append({
            "id": str(uuid4()),
            "note": note,
            "at": at,
            "source": source,
            "timestamp": datetime.now().isoformat(),
        })
        self.current_index = len(self.snapshots) - 1
        return truncated

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

    def to_dict(self) -> dict:
        return {
            "current_index": self.current_index,
            "snapshots": self.snapshots,
        }

    def restore(self, data: dict):
        snapshots = data.get("snapshots", [])
        if not isinstance(snapshots, list):
            self.reset()
            return

        self.snapshots = [snap for snap in snapshots if isinstance(snap, dict)]
        try:
            current_index = int(data.get("current_index", len(self.snapshots) - 1))
        except (TypeError, ValueError):
            current_index = len(self.snapshots) - 1

        if not self.snapshots:
            self.current_index = -1
        else:
            self.current_index = max(0, min(current_index, len(self.snapshots) - 1))
