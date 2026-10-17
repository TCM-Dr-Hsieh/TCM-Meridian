"""The patient and visit the physician had open, kept in one small file so a page that is opened again comes back to it.

One file for the whole program: this is a single-physician workstation. Every browser that connects (the same computer, a phone,
a tunnel) is brought to the visit that was last opened; there is no per-browser memory.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


class LastSelectionStore:
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)

    def load(self) -> tuple[str | None, str | None]:
        """(patient folder, visit date) -- the date may be None (a patient without a visit open); (None, None) when nothing is stored
        or the file cannot be read."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None, None
        if not isinstance(data, dict):
            return None, None
        folder, date = data.get("folder"), data.get("date")
        if not isinstance(folder, str) or not folder:
            return None, None
        return folder, date if isinstance(date, str) and date else None

    def save(self, folder: str | None, date: str | None = None) -> None:
        """Remember this selection; no patient clears it. Never raises: failing to remember must not stop the physician working."""
        try:
            if not folder:
                self.path.unlink(missing_ok=True)
                return
            payload = json.dumps({"folder": folder, "date": date or None}, ensure_ascii=False)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                os.replace(tmp, self.path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        except OSError:
            pass
