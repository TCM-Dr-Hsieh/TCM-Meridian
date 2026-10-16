from __future__ import annotations

from datetime import datetime
from uuid import uuid4


class SnapshotHistory:
    """Undo/redo history for NOTE and ASSESSMENT & TREATMENT snapshots."""

    def __init__(self):
        self.snapshots: list[dict] = []
        self.current_index: int = -1

    def push(
        self,
        note: str,
        at: str,
        source: str = "init",
        *,
        rec_t: float | None = None,
        rec_paused: bool = False,
        meta: dict | None = None,
        after_turn: int | None = None,
    ) -> list[dict]:
        """Add a version. `rec_t` is the recording clock when the version was made: where the voice transcript that could be seen then ended
        (seconds of the recording), which orders the version against the transcript lines. 0 = a recording, but no transcript
        yet; None = this session had no recording yet.
        `after_turn`: how many Main Agent turns had been started when a physician edited by hand (0 = none yet): the edit lies after
        that turn and before the next one, which orders it against the interview rounds and the agents' own versions (`Turn X-Y`).
        `meta`: what a physician edit took away (`removed_tags`, `dropped_marks`), kept with the version for the audit and the diff."""
        truncated = self.snapshots[self.current_index + 1 :]
        self.snapshots = self.snapshots[: self.current_index + 1]
        snapshot = {
            "id": str(uuid4()),
            "note": note,
            "at": at,
            "source": source,
            "timestamp": datetime.now().isoformat(),
        }
        if rec_t is not None:
            snapshot["rec_t"] = round(float(rec_t), 2)
            if rec_paused:
                snapshot["rec_paused"] = True
        if meta:
            snapshot["meta"] = meta
        if after_turn is not None:
            snapshot["after_turn"] = int(after_turn)
        self.snapshots.append(snapshot)
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
