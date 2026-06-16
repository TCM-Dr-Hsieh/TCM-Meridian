from __future__ import annotations

import json
import os
from datetime import datetime

from ui_app.services.file_io import atomic_write_json, atomic_write_text


class SessionArtifactService:
    """Read and write per-session logs and persisted UI state."""

    def _log_dir(self, folder_path: str, date_str: str) -> str:
        return os.path.join(folder_path, "log", f"{date_str}-log")

    def write_session_log(self, folder_path: str, date_str: str, entry: str):
        log_dir = self._log_dir(folder_path, date_str)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{date_str}-session.log")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {entry}\n")

    def save_chat_state(self, folder_path: str, date_str: str, chat_messages: list, agent_state: dict | None):
        log_dir = self._log_dir(folder_path, date_str)
        os.makedirs(log_dir, exist_ok=True)
        filepath = os.path.join(log_dir, f"{date_str}-chat-state.json")
        data = {
            "chat_messages": chat_messages,
            "agent_state": agent_state or {},
        }
        atomic_write_json(filepath, data)

    def load_chat_state(self, folder_path: str, date_str: str) -> dict | None:
        filepath = os.path.join(self._log_dir(folder_path, date_str), f"{date_str}-chat-state.json")
        if not os.path.isfile(filepath):
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def save_interview_state(self, folder_path: str, date_str: str, ic_state: dict):
        log_dir = self._log_dir(folder_path, date_str)
        os.makedirs(log_dir, exist_ok=True)
        filepath = os.path.join(log_dir, f"{date_str}-interview-state.json")
        atomic_write_json(filepath, ic_state)

    def load_interview_state(self, folder_path: str, date_str: str) -> dict | None:
        filepath = os.path.join(self._log_dir(folder_path, date_str), f"{date_str}-interview-state.json")
        if not os.path.isfile(filepath):
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def history_summary_path(self, folder_path: str, date_str: str) -> str:
        return os.path.join(self._log_dir(folder_path, date_str), f"{date_str}-History-Summary.md")

    def load_history_summary(self, folder_path: str, date_str: str) -> str:
        path = self.history_summary_path(folder_path, date_str)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def save_history_summary(self, folder_path: str, date_str: str, summary: str):
        path = self.history_summary_path(folder_path, date_str)
        atomic_write_text(path, summary)

    def save_forum_state(self, folder_path: str, date_str: str, forum_data: list[dict]):
        log_dir = self._log_dir(folder_path, date_str)
        os.makedirs(log_dir, exist_ok=True)
        state_path = os.path.join(log_dir, f"{date_str}-forum-state.json")
        atomic_write_json(state_path, {"forum_history": forum_data})
        log_path = os.path.join(log_dir, f"{date_str}-forum.txt")
        atomic_write_text(log_path, self._format_forum_log(forum_data))

    def load_forum_state(self, folder_path: str, date_str: str) -> list[dict]:
        state_path = os.path.join(self._log_dir(folder_path, date_str), f"{date_str}-forum-state.json")
        if os.path.isfile(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("forum_history", [])
            except Exception:
                pass
        return []

    @staticmethod
    def _format_forum_log(forum_data: list[dict]) -> str:
        parts = []
        for post in forum_data:
            post_id = post.get("post_id", "?")
            prof_id = post.get("professor_id", "?")
            prof_name = post.get("professor_name", "?")
            content = post.get("content", "")
            if post.get("role") == "main_agent":
                parts.append(f"<對話欄{post_id}，AI 主治醫師呼叫 {prof_id} ({prof_name})>\n{content}")
            else:
                parts.append(f"<對話欄{post_id}，{prof_id} ({prof_name})的回答>\n{content}")
        return "\n\n".join(parts) + ("\n" if parts else "")

    def save_conversation_file(self, folder_path: str, date_str: str, conversation_text: str):
        log_dir = self._log_dir(folder_path, date_str)
        os.makedirs(log_dir, exist_ok=True)
        filepath = os.path.join(log_dir, f"{date_str}-Human-Agent-Interaction.md")
        content = (
            "# 人類醫師 - AI 主治醫師 對話紀錄\n"
            f"# 日期: {date_str}\n\n"
            f"{conversation_text}"
        )
        atomic_write_text(filepath, content)
