from __future__ import annotations

import json
import os
import tempfile


def atomic_write_text(path: str, text: str):
    """先寫入同目錄暫存檔，再以 os.replace 原子置換目標檔。

    避免寫入中途崩潰或斷電時留下半截檔案（patient_info.json 等
    檔案是系統的單一事實來源，毀損會導致整位患者資料無法載入）。
    """
    dir_name = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", dir=dir_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_json(path: str, data):
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))
