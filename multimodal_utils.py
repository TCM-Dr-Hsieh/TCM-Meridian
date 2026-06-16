"""
multimodal_utils.py - 多模態圖片工具
將患者圖片檔案編碼為 base64，並構建 OpenAI Vision API 所需的 messages 格式。
"""
from __future__ import annotations

import base64
import mimetypes
import os
from typing import Optional


# 支援的圖片副檔名
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif"}


def encode_image_to_base64(file_path: str) -> Optional[str]:
    """讀取圖片檔案並回傳 base64 編碼字串。"""
    if not os.path.isfile(file_path):
        return None
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def get_image_media_type(file_path: str) -> str:
    """根據副檔名推斷圖片的 MIME 類型。"""
    mime, _ = mimetypes.guess_type(file_path)
    if mime and mime.startswith("image/"):
        return mime
    ext = os.path.splitext(file_path)[1].lower()
    mapping = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }
    return mapping.get(ext, "image/png")


def inject_images_into_messages(
    messages: list[dict],
    loaded_files: list[dict],
) -> list[dict]:
    """
    將 loaded_files 中的圖片注入到 messages 的最後一條 user 訊息。

    原始格式:
        [{"role": "user", "content": "text..."}]

    注入後（有圖片時）:
        [{"role": "user", "content": [
            {"type": "text", "text": "text..."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            ...
        ]}]

    若沒有圖片或所有圖片編碼失敗，則 messages 不變。
    """
    if not loaded_files:
        return messages

    # 收集圖片 content parts
    image_parts = []
    for lf in loaded_files:
        if lf.get("type") != "image":
            continue
        path = lf.get("path", "")
        if not path:
            continue
        b64 = encode_image_to_base64(path)
        if b64 is None:
            continue
        media_type = get_image_media_type(path)
        image_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{b64}",
            },
        })

    if not image_parts:
        return messages

    # 找到最後一條 user 訊息並轉換為 multimodal 格式
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            original_content = messages[i]["content"]
            # 已經是 list 格式的情況
            if isinstance(original_content, list):
                messages[i]["content"] = original_content + image_parts
            else:
                messages[i]["content"] = [
                    {"type": "text", "text": original_content},
                ] + image_parts
            break

    return messages
