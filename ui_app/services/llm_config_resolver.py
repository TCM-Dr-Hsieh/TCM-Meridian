from __future__ import annotations

from typing import Any


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def resolve_main_child_llm_config(
    main_cfg: dict,
    child_key: str,
    legacy_model_key: str,
    *,
    default_max_tokens: int,
    default_temperature: float,
) -> dict:
    """Resolve a Main Agent child LLM config with backwards-compatible fallbacks."""
    child_cfg = main_cfg.get(child_key, {})
    if not isinstance(child_cfg, dict):
        child_cfg = {}

    model_name = (
        (child_cfg.get("model_name") or "").strip()
        or (main_cfg.get(legacy_model_key) or "").strip()
        or (main_cfg.get("model_name", "") or "").strip()
    )

    return {
        "api_url": (child_cfg.get("api_url") or main_cfg.get("api_url") or "http://localhost:1234/v1").strip(),
        "api_key": child_cfg.get("api_key") or main_cfg.get("api_key") or "lm-studio",
        "model_name": model_name,
        "max_tokens": _to_int(child_cfg.get("max_tokens"), default_max_tokens),
        "temperature": _to_float(child_cfg.get("temperature"), default_temperature),
    }
