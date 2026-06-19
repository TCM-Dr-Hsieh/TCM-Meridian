from __future__ import annotations

import asyncio
from typing import Any, Callable

from ui_app.controllers.agent_run_state_controller import get_agent_run_state
from ui_app.controllers.interview_state_controller import get_interview_state


class MainAgentTurnRunner:
    """Prepare context and run one MainAgent turn in the executor."""

    def __init__(
        self,
        *,
        app_state: dict[str, Any],
        history: Any,
        load_config: Callable[[], dict],
        agent_factory: Any,
        get_last_visit_content: Callable[[str, str], str],
        load_history_summary: Callable[[str, str], str],
        write_session_log: Callable[[str, str, str], None],
        save_chat_state: Callable[[str, str, list, dict | None], None],
        save_forum_state: Callable[[str, str, list[dict]], None],
    ):
        self.app_state = app_state
        self.history = history
        self.load_config = load_config
        self.agent_factory = agent_factory
        self.get_last_visit_content = get_last_visit_content
        self.load_history_summary = load_history_summary
        self.write_session_log = write_session_log
        self.save_chat_state = save_chat_state
        self.save_forum_state = save_forum_state
        self.agent_run_state = get_agent_run_state(app_state)

    async def run(self, *, user_msg: str, fp: str, dt: str, conversation_text: str) -> tuple[dict, int]:
        loop = asyncio.get_running_loop()
        cfg = self.load_config()

        agent = self.agent_run_state.ensure_agent(self.agent_factory, cfg)

        current = self.history.get_current()
        note_content = current["note"] if current else ""
        at_content = current["at"] if current else ""
        record_history_snapshots = [dict(snap) for snap in getattr(self.history, "snapshots", [])]
        record_history_current_index = getattr(self.history, "current_index", -1)
        patient_info = self.app_state.get("selected_patient_info")

        last_visit = self.get_last_visit_content(fp, dt)
        hist_summary = self.load_history_summary(fp, dt)
        interview_dialogue = get_interview_state(self.app_state).get_dialogue()
        generation = self.agent_run_state.current_generation()

        def log_cb(msg):
            self.write_session_log(fp, dt, msg)

        def on_step(_step: dict):
            ui_state = self.app_state.get("ui_state", {})
            self.save_chat_state(
                fp,
                dt,
                ui_state.get("chat_messages", []),
                agent.export_state(),
            )
            if getattr(agent, "forum_history", None):
                self.save_forum_state(fp, dt, agent.forum_history)
            behavior_render = self.app_state.get("render_agent_behavior")
            if behavior_render:
                loop.call_soon_threadsafe(behavior_render)

        result = await loop.run_in_executor(
            None,
            lambda: agent.process_message(
                user_message=user_msg,
                note_content=note_content,
                at_content=at_content,
                patient_folder=fp,
                patient_info=patient_info,
                session_date=dt,
                conversation_history=conversation_text,
                last_visit_block=last_visit,
                history_summary=hist_summary,
                interview_dialogue=interview_dialogue,
                record_history_snapshots=record_history_snapshots,
                record_history_current_index=record_history_current_index,
                on_step=on_step,
                log_callback=log_cb,
            ),
        )
        return result, generation
