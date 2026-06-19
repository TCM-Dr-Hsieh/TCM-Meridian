from __future__ import annotations

from typing import Any, Callable

from ui_app.rendering import build_conversation_text


class MainAgentResultProcessor:
    """Apply MainAgent results to record, chat, forum, and interview UI state."""

    def __init__(
        self,
        *,
        app_state: dict[str, Any],
        ui_state: dict[str, Any],
        history: Any,
        agent_run_state: Any,
        live_steps: Any,
        main_chat_input: Any,
        agent_status: Any,
        save_current_to_disk: Callable[[], None],
        update_display: Callable[[], None],
        update_buttons: Callable[[], None],
        render_chat: Callable[[], None],
        write_session_log: Callable[[str, str, str], None],
        save_chat_state: Callable[[str, str, list, dict | None], None],
        save_forum_state: Callable[[str, str, list[dict]], None],
        save_conversation_file: Callable[[str, str, str], None],
        record_snapshot_store: Any,
    ):
        self.app_state = app_state
        self.ui_state = ui_state
        self.history = history
        self.agent_run_state = agent_run_state
        self.live_steps = live_steps
        self.main_chat_input = main_chat_input
        self.agent_status = agent_status
        self.save_current_to_disk = save_current_to_disk
        self.update_display = update_display
        self.update_buttons = update_buttons
        self.render_chat = render_chat
        self.write_session_log = write_session_log
        self.save_chat_state = save_chat_state
        self.save_forum_state = save_forum_state
        self.save_conversation_file = save_conversation_file
        self.record_snapshot_store = record_snapshot_store

    async def process(self, result: dict, fp: str, dt: str, gen: int = -1):
        if gen >= 0 and not self.agent_run_state.is_current_generation(gen):
            print(
                f"[GUARD] session_generation 不符 (啟動時={gen}, "
                f"當前={self.agent_run_state.current_generation()}), 丟棄過時的 Agent 結果"
            )
            self.live_steps.stop(clear=False)
            self.agent_run_state.finish_run()
            self.main_chat_input.unlock("agent_running")
            self.agent_status.text = "⚠️ Session 已變更，已丟棄過時的結果"
            return

        steps = result.get("steps", [])
        self._apply_record_changes(result, fp, dt)

        if result.get("waiting_for_interview"):
            await self._handle_waiting_for_interview(result, fp, dt)
            return

        self._handle_normal_reply(result, steps, fp, dt)

    def _apply_record_changes(self, result: dict, fp: str, dt: str):
        if not (result.get("note_changed") or result.get("at_changed")):
            return

        pushed = 0
        for snapshot in result.get("record_snapshots") or []:
            if self._push_record_snapshot(
                snapshot.get("note", ""),
                snapshot.get("at", ""),
                snapshot.get("source", f"Agent Turn {result.get('turn_number', '?')} update_record"),
                fp,
                dt,
            ):
                pushed += 1

        final_source = f"Agent Turn {result.get('turn_number', '?')}"
        if self._push_record_snapshot(result["note"], result["at"], final_source, fp, dt):
            pushed += 1

        if pushed:
            self.save_current_to_disk()
            self.update_display()
            self.update_buttons()
            self.write_session_log(fp, dt, f"[AGENT_UPDATE] Agent 更新病歷 (版本 {self.history.current_index + 1}，新增 {pushed} 個快照)")

    def _push_record_snapshot(self, note: str, at: str, source: str, fp: str, dt: str) -> bool:
        current = self.history.get_current()
        if current and current.get("note", "") == note and current.get("at", "") == at:
            return False
        before_index = self.history.current_index
        truncated = self.history.push(note, at, source=source)
        if truncated:
            self.record_snapshot_store.append_audit_event(
                fp,
                dt,
                event_type="truncate_redo",
                source=source,
                current_index_before=before_index,
                current_index_after=self.history.current_index,
                snapshots=truncated,
                note="AI update created a new linear version and replaced redo snapshots.",
            )
        return True

    async def _handle_waiting_for_interview(self, result: dict, fp: str, dt: str):
        tab_auto = self.app_state.get("tab_auto")
        if tab_auto:
            tab_auto.style("color: #e53935; font-weight: bold;")
        self.agent_status.text = "⏳ 等待問診助理完成..."

        agent_inst = self.agent_run_state.get_agent()
        agent_st = agent_inst.export_state() if agent_inst else None
        self.save_chat_state(fp, dt, self.ui_state["chat_messages"], agent_st)
        self._save_and_render_forum(fp, dt, agent_inst)
        behavior_render = self.app_state.get("render_agent_behavior")
        if behavior_render:
            behavior_render()

        ic_guidelines = result.get("interview_guidelines", "")
        self.write_session_log(fp, dt, f"[IC_START] 啟動問診助理, 方針: {ic_guidelines[:500]}")
        start_interview_fn = self.app_state.get("_start_interview_from_agent")
        if start_interview_fn:
            await start_interview_fn(ic_guidelines, fp, dt)

    def _handle_normal_reply(self, result: dict, steps: list, fp: str, dt: str):
        reply = result.get("reply", "（無回覆）")
        self.ui_state["chat_messages"].append({
            "role": "agent",
            "content": reply,
            "steps": steps,
        })
        self.render_chat()

        conversation_text = build_conversation_text(self.ui_state["chat_messages"])
        self.save_conversation_file(fp, dt, conversation_text)

        self.write_session_log(fp, dt, f"[AGENT_REPLY] {reply}")
        self.agent_status.text = "⏹ 主 Agent 已中斷" if result.get("manual_stopped") else "✅ 回覆完成"

        agent_inst = self.agent_run_state.get_agent()
        agent_st = agent_inst.export_state() if agent_inst else None
        self.save_chat_state(fp, dt, self.ui_state["chat_messages"], agent_st)
        self._save_and_render_forum(fp, dt, agent_inst)
        behavior_render = self.app_state.get("render_agent_behavior")
        if behavior_render:
            behavior_render()

    def _save_and_render_forum(self, fp: str, dt: str, agent_inst: Any):
        if agent_inst and agent_inst.forum_history:
            self.save_forum_state(fp, dt, agent_inst.forum_history)
            render_fn = self.app_state.get("render_forum")
            if render_fn:
                render_fn()
