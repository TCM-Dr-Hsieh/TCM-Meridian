from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from uuid import uuid4

from ui_app.services.file_io import atomic_write_json


class RecordSnapshotStore:
    """Persist linear NOTE/A&T snapshot history per patient session."""

    SCHEMA_VERSION = 1

    def _log_dir(self, folder_path: str, date_str: str) -> str:
        return os.path.join(folder_path, "log", f"{date_str}-log")

    def snapshot_path(self, folder_path: str, date_str: str) -> str:
        return os.path.join(self._log_dir(folder_path, date_str), f"{date_str}-record-snapshots.json")

    def audit_path(self, folder_path: str, date_str: str) -> str:
        return os.path.join(self._log_dir(folder_path, date_str), f"{date_str}-record-snapshot-events.jsonl")

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

    @classmethod
    def _combined_hash(cls, note: str, at: str) -> str:
        return cls._hash_text(f"{note or ''}\0{at or ''}")

    def _normalize_snapshot(self, snapshot: dict) -> dict:
        note = snapshot.get("note", "")
        at = snapshot.get("at", "")
        normalized = dict(snapshot)
        normalized.setdefault("id", str(uuid4()))
        normalized.setdefault("source", "")
        normalized.setdefault("timestamp", datetime.now().isoformat())
        normalized["note_sha256"] = self._hash_text(note)
        normalized["at_sha256"] = self._hash_text(at)
        normalized["combined_sha256"] = self._combined_hash(note, at)
        return normalized

    def save_history(self, folder_path: str, date_str: str, history) -> None:
        if not folder_path or not date_str:
            return
        log_dir = self._log_dir(folder_path, date_str)
        os.makedirs(log_dir, exist_ok=True)
        snapshots = [self._normalize_snapshot(snap) for snap in history.snapshots]
        data = {
            "schema_version": self.SCHEMA_VERSION,
            "current_index": history.current_index,
            "snapshots": snapshots,
        }
        current = history.get_current()
        if current:
            data["current_combined_sha256"] = self._combined_hash(
                current.get("note", ""),
                current.get("at", ""),
            )
        atomic_write_json(self.snapshot_path(folder_path, date_str), data)

    def load_history(self, folder_path: str, date_str: str) -> dict | None:
        path = self.snapshot_path(folder_path, date_str)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(data, dict) or not isinstance(data.get("snapshots"), list):
            return None
        if data.get("schema_version") != self.SCHEMA_VERSION:
            return None
        return data

    def append_audit_event(
        self,
        folder_path: str,
        date_str: str,
        *,
        event_type: str,
        source: str,
        current_index_before: int | None = None,
        current_index_after: int | None = None,
        snapshots: list[dict] | None = None,
        note: str = "",
    ) -> None:
        if not folder_path or not date_str:
            return
        log_dir = self._log_dir(folder_path, date_str)
        os.makedirs(log_dir, exist_ok=True)
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "source": source,
            "current_index_before": current_index_before,
            "current_index_after": current_index_after,
            "snapshots": [self._normalize_snapshot(snap) for snap in (snapshots or [])],
            "note": note,
        }
        with open(self.audit_path(folder_path, date_str), "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def restore_or_initialize(
        self,
        *,
        folder_path: str,
        date_str: str,
        history,
        note_text: str,
        at_text: str,
    ) -> dict:
        data = self.load_history(folder_path, date_str)
        if data is None:
            history.reset()
            history.push(note_text, at_text, source="init")
            self.save_history(folder_path, date_str, history)
            return {"restored": False, "external_state_added": False}

        history.restore(data)
        current = history.get_current()
        if current is None:
            history.reset()
            history.push(note_text, at_text, source="init")
            self.save_history(folder_path, date_str, history)
            return {"restored": False, "external_state_added": False}

        current_hash = self._combined_hash(current.get("note", ""), current.get("at", ""))
        disk_hash = self._combined_hash(note_text, at_text)
        if current_hash != disk_hash:
            before = history.current_index
            truncated = history.push(note_text, at_text, source="外部檔案狀態")
            self.append_audit_event(
                folder_path,
                date_str,
                event_type="external_file_state",
                source="外部檔案狀態",
                current_index_before=before,
                current_index_after=history.current_index,
                snapshots=truncated,
                note=(
                    "Disk NOTE/A&T differed from the persisted current snapshot; "
                    "added a new snapshot from disk."
                    + (
                        " Redo snapshots were truncated and preserved in this audit event."
                        if truncated
                        else ""
                    )
                ),
            )
            self.save_history(folder_path, date_str, history)
            return {"restored": True, "external_state_added": True}

        return {"restored": True, "external_state_added": False}
