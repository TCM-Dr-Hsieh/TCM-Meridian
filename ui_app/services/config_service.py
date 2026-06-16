from __future__ import annotations

import json
import os
from typing import Callable

from ui_app.services.file_io import atomic_write_json


def load_json_config(config_path: str, default_factory: Callable[[], dict]) -> dict:
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            msg = (
                f"[Config] WARNING: 設定檔讀取失敗，已改用預設設定（模型與 API 設定可能不正確）："
                f"{config_path} ({e})"
            )
            try:
                print(msg)
            except UnicodeEncodeError:
                print(msg.encode("ascii", errors="replace").decode("ascii"))
    return default_factory()


def save_json_config(config_path: str, cfg: dict):
    atomic_write_json(config_path, cfg)
