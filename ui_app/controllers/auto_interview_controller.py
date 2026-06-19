from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Any, Callable

from ui_app.controllers.agent_run_state_controller import get_agent_run_state
from ui_app.controllers.interview_state_controller import get_interview_state
from ui_app.rendering import build_conversation_text


class AutoInterviewController:
    """Owns the information-collection tab and MainAgent resume bridge."""

    def __init__(
        self,
        ui: Any,
        app_state: dict[str, Any],
        *,
        load_config: Callable[[], dict],
        write_session_log: Callable[[str, str, str], None],
        save_chat_state: Callable[[str, str, list, dict | None], None],
        save_interview_state: Callable[[str, str, dict], None],
        save_forum_state: Callable[[str, str, list[dict]], None],
        save_conversation_file: Callable[[str, str, str], None],
        get_last_visit_content: Callable[[str, str], str],
        load_history_summary: Callable[[str, str], str],
        agent_factory: Any,
    ):
        self.ui = ui
        self.app_state = app_state
        self.load_config = load_config
        self.write_session_log = write_session_log
        self.save_chat_state = save_chat_state
        self.save_interview_state = save_interview_state
        self.save_forum_state = save_forum_state
        self.save_conversation_file = save_conversation_file
        self.get_last_visit_content = get_last_visit_content
        self.load_history_summary = load_history_summary
        self.agent_factory = agent_factory
        self.agent_run_state = get_agent_run_state(app_state)
        self.interview_state = get_interview_state(app_state)
        self.refs: dict[str, Any] = {}

    def _next_ic_run_id(self) -> int:
        run_id = int(self.app_state.get("_ic_run_id", 0)) + 1
        self.app_state["_ic_run_id"] = run_id
        return run_id

    def _is_current_ic_run(self, run_id: int) -> bool:
        return self.app_state.get("_ic_run_id") == run_id and self.interview_state.is_active()

    def build_tab(self):
        ui = self.ui
        app_state = self.app_state

        with ui.column().classes("w-full").style("max-width: 900px; margin: 0 auto;"):
            ui.label("🏥 自動問診對話區").classes("section-title")
            ui.label(
                "此對話區由 AI 主治醫師在需要蒐集臨床資訊時自動啟動。"
                "問診助理將代為向患者/照顧者/實習醫師進行多回合問診。"
            ).style("color: #666; font-size: 14px; margin-bottom: 12px;")

            with ui.card().classes("w-full q-pa-sm q-mb-md").style(
                "border: 1px solid var(--border); border-radius: 12px; background: #f9f9fb;"
            ):
                ui.label("📋 資訊蒐集方針").style("font-weight: 700; font-size: 14px; color: var(--primary-dark);")
                guidelines_label = ui.label("（尚未啟動問診助理）").style(
                    "color: #888; font-size: 13px; white-space: pre-wrap; max-height: 120px; overflow-y: auto;"
                )

            with ui.card().classes("w-full q-pa-md q-mb-md ic-chat-scroll").style(
                "border: 1px solid var(--border); border-radius: 12px; "
                "min-height: 300px; max-height: 500px; overflow-y: auto;"
            ) as chat_card:
                chat_container = ui.column().classes("w-full gap-2")

            with ui.card().classes("w-full q-pa-md q-mb-md").style(
                "border: 1px solid var(--border); border-radius: 12px;"
            ):
                ui.label("💬 回覆問診助理").style("font-weight: 600; font-size: 14px; color: var(--primary-dark);")
                reply_input = ui.textarea(
                    label="輸入患者回覆 / 照顧者回覆 / 實習醫師回覆",
                    placeholder="在此輸入回覆內容...",
                ).classes("w-full").style("min-height: 100px;")

                with ui.row().classes("w-full gap-3 items-center"):
                    btn_send = ui.button("📤 送出回覆", color="primary").style("font-size: 14px;")
                    btn_stop = ui.button("⏹ 終止問診", color="red").props("outline").style("font-size: 14px;")
                    status = ui.label("").style("font-size: 13px; color: #666;")

        self.refs = {
            "guidelines_label": guidelines_label,
            "chat_container": chat_container,
            "chat_card": chat_card,
            "reply_input": reply_input,
            "btn_send": btn_send,
            "btn_stop": btn_stop,
            "status": status,
        }

        btn_send.disable()
        btn_stop.disable()

        app_state["_ic_add_message"] = self.add_message
        app_state["_clear_interview_ui"] = self.clear_ui
        app_state["_load_interview_ui"] = self.load_history_ui
        app_state["_start_interview_from_agent"] = self.start_interview_from_agent
        app_state["_resume_agent_after_interview"] = self.resume_agent_after_interview

        async def on_ic_send():
            answer = (reply_input.value or "").strip()
            if not answer:
                status.text = "⚠️ 請輸入回覆內容"
                return

            fp = app_state.get("selected_patient_folder")
            dt = app_state.get("selected_session_date")
            ic_sub = self.interview_state.get_subagent()
            if not ic_sub:
                status.text = "⚠️ 問診助理尚未啟動"
                return

            if fp and dt:
                ic_sub.behavior_context = {"folder_path": fp, "date_str": dt}

            run_id = self._next_ic_run_id()
            rollback_point = dict(app_state.get("_ic_rollback_point") or {})

            self.add_message("patient", answer, ic_sub.dialogue_round)
            reply_input.value = ""
            btn_send.disable()
            status.text = "⏳ 問診助理思考中..."

            hist = app_state.get("history")
            current = hist.get_current() if hist else None
            note_content = current["note"] if current else ""
            at_content = current["at"] if current else ""

            def log_cb(msg):
                if fp and dt:
                    self.write_session_log(fp, dt, msg)

            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: ic_sub.receive_answer(
                        patient_input=answer,
                        note_content=note_content,
                        at_content=at_content,
                        log_callback=log_cb,
                    ),
                )
                if not self._is_current_ic_run(run_id):
                    if not self.interview_state.is_active():
                        self.rollback_interrupted_interview(
                            fp,
                            dt,
                            rollback_point=rollback_point,
                            write_log=False,
                        )
                    if fp and dt:
                        self.write_session_log(
                            fp,
                            dt,
                            "[IC_STALE_RESULT] Discarded information-collection result after stop/cancel.",
                        )
                    return
                completed = self.handle_result(result, fp, dt)
                if completed:
                    await self.resume_agent_after_interview(fp, dt)
            except Exception as e:
                if not self._is_current_ic_run(run_id):
                    if not self.interview_state.is_active():
                        self.rollback_interrupted_interview(
                            fp,
                            dt,
                            rollback_point=rollback_point,
                            write_log=False,
                        )
                    if fp and dt:
                        self.write_session_log(
                            fp,
                            dt,
                            f"[IC_STALE_ERROR] Discarded information-collection error after stop/cancel: {e}",
                        )
                    return
                status.text = f"❌ 問診助理錯誤：{e}"
                btn_send.enable()
                if fp and dt:
                    self.write_session_log(fp, dt, f"[IC_ERROR] {e}")

        async def on_ic_stop():
            ic_sub = self.interview_state.get_subagent()
            fp = app_state.get("selected_patient_folder")
            dt = app_state.get("selected_session_date")
            self._next_ic_run_id()

            if ic_sub:
                self.rollback_interrupted_interview(fp, dt)
            else:
                self.interview_state.stop()
            status.text = "⏹ 問診已手動終止"
            btn_send.disable()
            btn_stop.disable()

            agent = self.agent_run_state.get_agent()
            hist = app_state.get("history")
            current = hist.get_current() if hist else None
            note_content = current["note"] if current else ""
            at_content = current["at"] if current else ""
            interrupted_result = None
            if agent and getattr(agent, "_suspended", None):
                interrupted_result = agent.finalize_suspended_turn(
                    reply_text="服務已中斷(使用者手動停止)",
                    interrupted_step_result=(
                        "啟動問診助理，但問診被使用者手動中斷，本次問診內容已丟棄"
                    ),
                    note_content=note_content,
                    at_content=at_content,
                )

            live_poll_timer = app_state.get("_live_poll_timer")
            if live_poll_timer:
                live_poll_timer.deactivate()
            live_steps = app_state.get("_live_steps_container")
            if live_steps:
                live_steps.set_visibility(False)
            main_ui_state = app_state.get("ui_state", {})
            self.agent_run_state.finish_run()
            main_chat_input = app_state.get("_main_chat_input")
            if main_chat_input:
                main_chat_input.unlock("agent_running")
                main_chat_input.unlock("interview_active")
            main_agent_status = app_state.get("_agent_status")
            if main_agent_status:
                main_agent_status.text = "⏹ 問診已中斷，可繼續對話"

            synthetic_reply = "服務已中斷(使用者手動停止)"
            main_ui_state.setdefault("chat_messages", []).append({
                "role": "agent",
                "content": synthetic_reply,
                "steps": (interrupted_result or {}).get("steps", []),
            })
            render_chat = app_state.get("_render_chat")
            if render_chat:
                render_chat()

            if fp and dt:
                from agent_behavior_log import append_behavior_event

                append_behavior_event(
                    fp,
                    dt,
                    agent="information_collection_subagent",
                    event_type="manual_stop",
                    label="手動中斷",
                    title="問診助理被使用者手動中斷",
                    content="本次中斷問診內容已丟棄，不寫入完整問診對話紀錄。",
                    severity="warning",
                )
                agent_st = agent.export_state() if agent else None
                self.save_chat_state(fp, dt, main_ui_state.get("chat_messages", []), agent_st)
                conversation_text = build_conversation_text(main_ui_state.get("chat_messages", []))
                self.save_conversation_file(fp, dt, conversation_text)
                if agent and getattr(agent, "forum_history", None):
                    self.save_forum_state(fp, dt, agent.forum_history)
                self.write_session_log(fp, dt, "[IC_STOP] 問診已手動中斷，Agent 未自動恢復")
                behavior_render = self.app_state.get("render_agent_behavior")
                if behavior_render:
                    behavior_render()

        btn_send.on_click(on_ic_send)
        btn_stop.on_click(on_ic_stop)

    def clear_ui(self):
        if self.refs.get("chat_container"):
            self.refs["chat_container"].clear()
        if self.refs.get("guidelines_label"):
            self.refs["guidelines_label"].text = "（尚未啟動問診助理）"
        if self.refs.get("status"):
            self.refs["status"].text = ""
        if self.refs.get("btn_send"):
            self.refs["btn_send"].disable()
        if self.refs.get("btn_stop"):
            self.refs["btn_stop"].disable()
        if self.refs.get("reply_input"):
            self.refs["reply_input"].value = ""

    def load_history_ui(
        self,
        guidelines: str,
        conversations: list[dict],
        total_rounds: int,
        *,
        active: bool = False,
    ):
        if self.refs.get("guidelines_label"):
            self.refs["guidelines_label"].text = guidelines or "（方針已載入）"
        if self.refs.get("chat_container"):
            self.refs["chat_container"].clear()
            for msg in conversations:
                self.add_message(msg["role"], msg["content"], msg.get("round", 0))
        if self.refs.get("status"):
            self.refs["status"].text = f"📂 已載入先前問診紀錄 (共 {total_rounds} 回合)。如需繼續問診請重新啟動。"
        if self.refs.get("btn_send"):
            self.refs["btn_send"].disable()
        if self.refs.get("btn_stop"):
            self.refs["btn_stop"].disable()
        if active:
            if self.refs.get("status"):
                self.refs["status"].text = f"問診進行中，已恢復至 R{total_rounds}，請繼續回答或手動停止"
            if self.refs.get("btn_send"):
                self.refs["btn_send"].enable()
            if self.refs.get("btn_stop"):
                self.refs["btn_stop"].enable()
        self._scroll_chat_to_bottom()

    def add_message(self, role: str, content: str, round_num: int):
        container = self.refs.get("chat_container")
        if not container:
            return

        if role == "subagent":
            bg = "#e8f5e9"
            label_prefix = f"🤖 問診助理 R{round_num}"
        else:
            bg = "#e3f2fd"
            label_prefix = f"👤 回覆 R{round_num}"

        with container:
            with self.ui.card().classes("w-full q-pa-sm").style(
                f"background: {bg}; border-radius: 8px; border: none;"
            ):
                self.ui.label(label_prefix).style("font-weight: 700; font-size: 13px; color: #333;")
                self.ui.markdown(content).style("font-size: 14px; color: #444;")
        self._scroll_chat_to_bottom()

    def _scroll_chat_to_bottom(self):
        self.ui.run_javascript(
            """
            setTimeout(() => {
              const el = document.querySelector('.ic-chat-scroll');
              if (el) el.scrollTop = el.scrollHeight;
            }, 0);
            """
        )

    def handle_result(self, result: dict, fp: str | None, dt: str | None) -> bool:
        action = result.get("action", "")
        message = result.get("message", "")
        round_num = result.get("round", 0)
        finished = result.get("finished", False)

        if action == "ask_patient" and not finished:
            self.add_message("subagent", message, round_num)
            self.refs["status"].text = f"📋 問診中 (R{round_num})"
            self.refs["btn_send"].enable()
            self.refs["btn_stop"].enable()

            ic_sub = self.interview_state.get_subagent()
            if ic_sub and fp and dt:
                self.save_interview_state(fp, dt, ic_sub.export_state())
            return False

        if finished or action == "finish_collection":
            self.refs["status"].text = f"✅ 問診完成 (共 {round_num} 回合)"
            self.refs["btn_send"].disable()
            self.refs["btn_stop"].disable()
            self.finalize_interview(fp, dt)
            return True

        self.refs["status"].text = f"⚠️ 未預期的動作: {action}"
        self.refs["btn_send"].disable()
        self.refs["btn_stop"].disable()
        self.finalize_interview(fp, dt)
        return True

    def finalize_interview(self, fp: str | None, dt: str | None):
        ic_sub = self.interview_state.get_subagent()
        if not ic_sub:
            return

        full_dialogue = ic_sub.get_full_dialogue()
        self.interview_state.finish(full_dialogue)
        self.app_state.pop("_ic_rollback_point", None)

        tab_auto = self.app_state.get("tab_auto")
        if tab_auto:
            tab_auto.style("color: inherit; font-weight: normal;")

        if fp and dt:
            log_dir = os.path.join(fp, "log", f"{dt}-log")
            os.makedirs(log_dir, exist_ok=True)
            dialogue_path = os.path.join(log_dir, f"{dt}-information-collection-dialogue.txt")

            with open(dialogue_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"問診時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*60}\n")
                f.write(full_dialogue)
                f.write("\n")

            self.write_session_log(fp, dt, f"[IC_FINISH] 問診完成，對話紀錄已存檔: {dialogue_path}")
            self.save_interview_state(fp, dt, ic_sub.export_state())

    def rollback_interrupted_interview(
        self,
        fp: str | None,
        dt: str | None,
        *,
        rollback_point: dict | None = None,
        write_log: bool = True,
    ):
        """丟棄本次中斷問診新增內容，保留先前已完成問診紀錄。"""
        ic_sub = self.interview_state.get_subagent()
        if not ic_sub:
            self.interview_state.stop()
            return

        rollback = rollback_point if rollback_point is not None else self.app_state.pop("_ic_rollback_point", None) or {}
        keep_len = int(rollback.get("all_conversations_len", len(getattr(ic_sub, "all_conversations", []))))
        keep_round = int(rollback.get("dialogue_round", getattr(ic_sub, "dialogue_round", 0)))

        ic_sub.all_conversations = getattr(ic_sub, "all_conversations", [])[:keep_len]
        ic_sub.conversation = []
        ic_sub.turn_history = []
        ic_sub.dialogue_round = keep_round
        ic_sub.finished = True

        full_dialogue = ic_sub.get_full_dialogue()
        self.interview_state.stop(full_dialogue)

        tab_auto = self.app_state.get("tab_auto")
        if tab_auto:
            tab_auto.style("color: inherit; font-weight: normal;")

        if fp and dt:
            self.save_interview_state(fp, dt, ic_sub.export_state())
            if write_log:
                self.write_session_log(
                    fp,
                    dt,
                    "[IC_STOP] 本次中斷問診內容已丟棄，interview-state 已回復至啟動前",
                )

    def _finalize_failed_interview_start(self, reason: str, fp: str, dt: str):
        """Finish the suspended MainAgent turn when IC cannot be started."""
        reason = reason or "unknown error"
        agent = self.agent_run_state.get_agent()

        ic_sub = self.interview_state.get_subagent()
        if ic_sub:
            rollback = self.app_state.pop("_ic_rollback_point", None) or {}
            keep_len = int(rollback.get("all_conversations_len", len(getattr(ic_sub, "all_conversations", []))))
            keep_round = int(rollback.get("dialogue_round", getattr(ic_sub, "dialogue_round", 0)))
            ic_sub.all_conversations = getattr(ic_sub, "all_conversations", [])[:keep_len]
            ic_sub.conversation = []
            ic_sub.turn_history = []
            ic_sub.dialogue_round = keep_round
            ic_sub.finished = True

        full_dialogue = ic_sub.get_full_dialogue() if ic_sub else ""
        self.interview_state.stop(full_dialogue)
        tab_auto = self.app_state.get("tab_auto")
        if tab_auto:
            tab_auto.style("color: inherit; font-weight: normal;")

        hist = self.app_state.get("history")
        current = hist.get_current() if hist else None
        note_content = current["note"] if current else ""
        at_content = current["at"] if current else ""
        record_history_snapshots = [dict(snap) for snap in getattr(hist, "snapshots", [])] if hist else []
        record_history_current_index = getattr(hist, "current_index", -1) if hist else -1

        reply_text = f"問診助理啟動失敗，本輪已停止。原因：{reason}"
        interrupted_result = None
        if agent and getattr(agent, "_suspended", None):
            interrupted_result = agent.finalize_suspended_turn(
                reply_text=reply_text,
                interrupted_step_result=f"information_collection_subagent 啟動失敗：{reason}",
                note_content=note_content,
                at_content=at_content,
            )

        btn_stop_agent = self.app_state.get("_btn_stop_agent")
        if btn_stop_agent:
            btn_stop_agent.disable()
        live_steps = self.app_state.get("_live_steps_controller")
        if live_steps:
            live_steps.stop()
        else:
            live_poll_timer = self.app_state.get("_live_poll_timer")
            if live_poll_timer:
                live_poll_timer.deactivate()
            live_steps_container = self.app_state.get("_live_steps_container")
            if live_steps_container:
                live_steps_container.set_visibility(False)

        self.agent_run_state.finish_run()
        main_chat_input = self.app_state.get("_main_chat_input")
        if main_chat_input:
            main_chat_input.unlock("agent_running")
            main_chat_input.unlock("interview_active")

        if self.refs.get("status"):
            self.refs["status"].text = f"問診助理無法啟動：{reason}"
        main_agent_status = self.app_state.get("_agent_status")
        if main_agent_status:
            main_agent_status.text = "問診助理無法啟動，本輪已中止"

        main_ui_state = self.app_state.get("ui_state", {})
        main_ui_state.setdefault("chat_messages", []).append({
            "role": "agent",
            "content": reply_text,
            "steps": (interrupted_result or {}).get("steps", []),
        })
        render_chat = self.app_state.get("_render_chat")
        if render_chat:
            render_chat()

        if fp and dt:
            from agent_behavior_log import append_behavior_event

            append_behavior_event(
                fp,
                dt,
                agent="information_collection_subagent",
                event_type="model_error",
                label="IC start failed",
                title="Information collection subagent start failed",
                content=reason,
                severity="error",
            )
            agent_st = agent.export_state() if agent else None
            self.save_chat_state(fp, dt, main_ui_state.get("chat_messages", []), agent_st)
            conversation_text = build_conversation_text(main_ui_state.get("chat_messages", []))
            self.save_conversation_file(fp, dt, conversation_text)
            if agent and getattr(agent, "forum_history", None):
                self.save_forum_state(fp, dt, agent.forum_history)
            if ic_sub:
                self.save_interview_state(fp, dt, ic_sub.export_state())
            self.write_session_log(fp, dt, f"[IC_START_FAILED] {reason}")

        behavior_render = self.app_state.get("render_agent_behavior")
        if behavior_render:
            behavior_render()

    async def resume_agent_after_interview(self, fp: str, dt: str):
        loop = asyncio.get_running_loop()
        agent = self.agent_run_state.get_agent()
        main_chat_input = self.app_state.get("_main_chat_input")
        if not agent or not getattr(agent, "_suspended", None):
            if main_chat_input:
                main_chat_input.unlock("agent_running")
                main_chat_input.unlock("interview_active")
            return

        if main_chat_input:
            main_chat_input.unlock("interview_active")
            main_chat_input.lock("agent_running")
        self.agent_run_state.start_run()
        btn_stop_agent = self.app_state.get("_btn_stop_agent")
        if btn_stop_agent:
            btn_stop_agent.enable()
        live_steps = self.app_state.get("_live_steps_controller")
        if live_steps:
            live_steps.start()

        ic_sub = self.interview_state.get_subagent()
        full_dialogue = ic_sub.get_full_dialogue() if ic_sub else ""
        summary = ic_sub.get_dialogue_summary_for_history() if ic_sub else "（無問診結果）"

        hist = self.app_state.get("history")
        current = hist.get_current() if hist else None
        note_content = current["note"] if current else ""
        at_content = current["at"] if current else ""
        record_history_snapshots = [dict(snap) for snap in getattr(hist, "snapshots", [])] if hist else []
        record_history_current_index = getattr(hist, "current_index", -1) if hist else -1

        def log_cb(msg):
            self.write_session_log(fp, dt, msg)

        def on_step(_step: dict):
            ui_state = self.app_state.get("ui_state", {})
            agent_st = agent.export_state() if agent else None
            self.save_chat_state(fp, dt, ui_state.get("chat_messages", []), agent_st)
            if agent and getattr(agent, "forum_history", None):
                self.save_forum_state(fp, dt, agent.forum_history)
            behavior_render = self.app_state.get("render_agent_behavior")
            if behavior_render:
                loop.call_soon_threadsafe(behavior_render)

        self.write_session_log(fp, dt, f"[IC_RESUME] 問診完成，恢復主 Agent 迴圈: {summary}")

        gen = self.agent_run_state.current_generation()

        try:
            result = await loop.run_in_executor(
                None,
                lambda: agent.continue_after_interview(
                    interview_dialogue=full_dialogue,
                    interview_summary=summary,
                    note_content=note_content,
                    at_content=at_content,
                    record_history_snapshots=record_history_snapshots,
                    record_history_current_index=record_history_current_index,
                    on_step=on_step,
                    log_callback=log_cb,
                ),
            )

            processor = self.app_state.get("_process_agent_result")
            if processor:
                await processor(result, fp, dt, gen)

        except Exception as e:
            self.write_session_log(fp, dt, f"[IC_RESUME_ERROR] {e}")
            ui_state = self.app_state.get("ui_state", {})
            if "chat_messages" in ui_state:
                ui_state["chat_messages"].append({
                    "role": "agent",
                    "content": f"❌ 問診完成後 Agent 恢復失敗：{e}",
                    "steps": [],
                })
                render_chat = self.app_state.get("_render_chat")
                if render_chat:
                    render_chat()

        finally:
            if btn_stop_agent:
                btn_stop_agent.disable()
            if live_steps:
                live_steps.stop()
            if not self.interview_state.is_active():
                self.agent_run_state.finish_run()
                if main_chat_input:
                    main_chat_input.unlock("agent_running")
                    main_chat_input.unlock("interview_active")
                status = self.app_state.get("_agent_status")
                if status:
                    status.text = "✅ 回覆完成"

    async def start_interview_from_agent(self, guidelines: str, fp: str, dt: str):
        cfg = self.load_config()
        ic_cfg = cfg.get("ic_subagent", {})

        if not ic_cfg.get("model_name"):
            self.refs["status"].text = "⚠️ 問診助理尚未設定 Model Name，無法啟動"
            self.write_session_log(fp, dt, "[IC_ERROR] IC Subagent model_name 未設定")
            self._finalize_failed_interview_start("IC Subagent model_name is not configured", fp, dt)
            return

        if self.interview_state.get_subagent() is None:
            self.interview_state.set_subagent(self.agent_factory.create_information_collection_agent(ic_cfg))

        ic_sub = self.interview_state.get_subagent()
        self.app_state["_ic_rollback_point"] = {
            "all_conversations_len": len(getattr(ic_sub, "all_conversations", [])),
            "dialogue_round": getattr(ic_sub, "dialogue_round", 0),
        }
        self.interview_state.start(ic_sub)
        main_chat_input = self.app_state.get("_main_chat_input")
        if main_chat_input:
            main_chat_input.unlock("agent_running")
            main_chat_input.lock("interview_active")

        self.refs["guidelines_label"].text = guidelines
        container = self.refs.get("chat_container")
        if container:
            container.clear()

        self.refs["status"].text = "⏳ 問診助理思考中..."

        hist = self.app_state.get("history")
        current = hist.get_current() if hist else None
        note_content = current["note"] if current else ""
        at_content = current["at"] if current else ""

        last_visit = self.get_last_visit_content(fp, dt)
        hist_summary = self.load_history_summary(fp, dt)

        agent = self.agent_run_state.get_agent()
        interaction_history = ""
        if agent and getattr(agent, "_suspended", None):
            suspended = getattr(agent, "_suspended", None) or {}
            interaction_history = suspended.get("conversation_history", "")
        elif agent and hasattr(agent, "turn_history"):
            for th in agent.turn_history:
                interaction_history += f"[Turn {th['turn']} 人類醫師的提問] {th['user_message']}\n"
                interaction_history += f"[Turn {th['turn']} AI主治醫師的回覆] {th['reply']}\n"

        def log_cb(msg):
            self.write_session_log(fp, dt, msg)

        forum_history_text = ""
        if agent and hasattr(agent, "_format_forum_history"):
            forum_history_text = agent._format_forum_history()

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ic_sub.start_collection(
                    guidelines=guidelines,
                    interaction_history=interaction_history,
                    note_content=note_content,
                    at_content=at_content,
                    last_visit_block=last_visit,
                    history_summary=hist_summary,
                    forum_history=forum_history_text,
                    log_callback=log_cb,
                    behavior_context={"folder_path": fp, "date_str": dt},
                ),
            )
            completed = self.handle_result(result, fp, dt)
            if completed:
                await self.resume_agent_after_interview(fp, dt)
        except Exception as e:
            self.interview_state.stop()
            if main_chat_input:
                main_chat_input.unlock("interview_active")
            self.refs["status"].text = f"❌ 問診助理啟動失敗：{e}"
            self.write_session_log(fp, dt, f"[IC_ERROR] 啟動失敗: {e}")
            self._finalize_failed_interview_start(str(e), fp, dt)
