from __future__ import annotations

from typing import Any
from uuid import uuid4


def normalize_quick_prompts(raw: Any) -> list[dict[str, str]]:
    """Return valid, UI-safe quick prompt records from config data."""
    if not isinstance(raw, list):
        return []

    prompts: list[dict[str, str]] = []
    used_ids: set[str] = set()
    used_names: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        content = str(item.get("content") or "").strip()
        if not name or not content:
            continue
        normalized_name = name.casefold()
        if normalized_name in used_names:
            continue

        prompt_id = str(item.get("id") or "").strip()
        while not prompt_id or prompt_id in used_ids:
            prompt_id = uuid4().hex
        used_ids.add(prompt_id)
        used_names.add(normalized_name)
        prompts.append({"id": prompt_id, "name": name, "content": content})
    return prompts


def load_quick_prompts(config: Any) -> list[dict[str, str]]:
    if not isinstance(config, dict):
        return []
    return normalize_quick_prompts(config.get("quick_prompts", []))
