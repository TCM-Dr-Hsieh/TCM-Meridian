from __future__ import annotations

import os

from ui_app.services.file_io import atomic_write_text


class TemplateFileService:
    """Read and write the standard medical record template file."""

    def __init__(self, path: str):
        self.path = path

    def load_record_template(self) -> str:
        if os.path.isfile(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def save_record_template(self, content: str):
        atomic_write_text(self.path, content)
