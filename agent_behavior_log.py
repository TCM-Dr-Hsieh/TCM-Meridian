from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any


AGENT_COLUMNS = [
    ("main_agent", "AI主治醫師 Agent"),
    ("record_subagent", "病歷登載 Subagent"),
    ("hallucination_subagent", "幻覺檢查 Subagent"),
    ("information_collection_subagent", "問診助理 Subagent"),
    ("low_confidence_subagent", "低信心標註 Subagent"),
    ("note_review_subagent", "病歷檢查員 Subagent"),
    ("professor_subagent", "醫學教授 Subagent"),
]


def behavior_log_path(folder_path: str, date_str: str) -> str:
    log_dir = os.path.join(folder_path, "log", f"{date_str}-log")
    return os.path.join(log_dir, f"{date_str}-agent-behavior.jsonl")


def append_behavior_event(
    folder_path: str | None,
    date_str: str | None,
    *,
    agent: str,
    event_type: str,
    label: str,
    title: str,
    content: str,
    content_type: str = "markdown",
    turn: int | None = None,
    sub_turn: str | None = None,
    severity: str = "normal",
    target_agent: str | None = None,
    meta: dict[str, Any] | None = None,
):
    if not folder_path or not date_str:
        return

    path = behavior_log_path(folder_path, date_str)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    event = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "agent": agent,
        "event_type": event_type,
        "label": label,
        "title": title,
        "content": content or "",
        "content_type": content_type,
        "turn": turn,
        "sub_turn": sub_turn,
        "severity": severity,
        "target_agent": target_agent,
        "meta": meta or {},
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def load_behavior_events(folder_path: str | None, date_str: str | None) -> list[dict[str, Any]]:
    if not folder_path or not date_str:
        return []
    path = behavior_log_path(folder_path, date_str)
    if not os.path.isfile(path):
        return []

    events: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                events.append({
                    "ts": "",
                    "agent": "main_agent",
                    "event_type": "log_parse_error",
                    "label": "log解析失敗",
                    "title": "agent-behavior.jsonl 解析失敗",
                    "content": line,
                    "content_type": "markdown",
                    "severity": "error",
                })
    return events
