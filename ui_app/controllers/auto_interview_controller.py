from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Any, Callable

from Information_Collection_Subagent import VOICE_ANSWER_MARK, voice_answer_content
from ui_app.controllers.agent_run_state_controller import get_agent_run_state
from ui_app.controllers.interview_state_controller import get_interview_state
from ui_app.rendering import build_conversation_text
from voice.textutil import format_clock

# How long the voice-answer window waits for the recognition to reach the moment the button was pressed before it lets the physician
# send what is there (the windows are a few seconds long; recognition and correction take a few more).
VOICE_WAIT_SECONDS = 30.0


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
        self._voice_answer_open = False          # one voice-answer window at a time (a double click opened two for the same round)
        # One step of the assistant at a time (a double click on 重試 ran two model calls and both changed the conversation). The step
        # is owned by the call that took it and only that call releases it: a later interview started while the Main Agent resumes
        # takes its own.
        self._assistant_step: object | None = None

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
                    btn_voice = ui.button("🎙 從逐字稿取回答", color="primary").props("outline").style("font-size: 14px;").tooltip(
                        "把這回合的問題口頭問完後按這裡：取這段時間的錄音逐字稿當作回答（這個就診要正在錄音或已有錄音）"
                    )
                    btn_retry = ui.button("🔁 重試", color="orange").style("font-size: 14px;").tooltip(
                        "問診助理上一步沒有成功（模型呼叫失敗或回覆無法使用）：再試一次；已送出的回覆都保留"
                    )
                    btn_stop = ui.button("⏹ 終止問診", color="red").props("outline").style("font-size: 14px;")
                    status = ui.label("").style("font-size: 13px; color: #666;")

        self.refs = {
            "guidelines_label": guidelines_label,
            "chat_container": chat_container,
            "chat_card": chat_card,
            "reply_input": reply_input,
            "btn_send": btn_send,
            "btn_voice": btn_voice,
            "btn_retry": btn_retry,
            "btn_stop": btn_stop,
            "status": status,
        }

        self._set_reply_enabled(False)
        self._show_retry(False)
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
            reply_input.value = ""
            await self.submit_answer(answer)

        async def on_ic_voice():
            await self.answer_from_transcript()

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
            self._set_reply_enabled(False)
            self._show_retry(False)
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
        btn_voice.on_click(on_ic_voice)
        btn_retry.on_click(self.retry_assistant)
        btn_stop.on_click(on_ic_stop)

    def _show_retry(self, visible: bool):
        button = self.refs.get("btn_retry")
        if button is not None:
            button.set_visibility(visible)

    def _show_failure(self, message: str, fp: str | None, dt: str | None):
        """A step of the assistant failed: the interview stays open, the replies already sent are kept, and the physician retries or stops."""
        self.refs["status"].text = f"❌ {message.rstrip('。. ')}。已送出的回覆都保留；可按「🔁 重試」再試一次，或終止問診"
        self._set_reply_enabled(False)
        self._show_retry(True)
        if self.refs.get("btn_stop"):
            self.refs["btn_stop"].enable()
        if fp and dt:
            self.write_session_log(fp, dt, f"[IC_ERROR] {message}")

    def _set_reply_enabled(self, enabled: bool):
        """The two ways to answer a round (typed, or asked aloud and taken from the transcript) are open and shut together."""
        for key in ("btn_send", "btn_voice"):
            button = self.refs.get(key)
            if button is not None:
                (button.enable if enabled else button.disable)()

    def _audio_clock(self, fp: str | None, dt: str | None) -> float | None:
        voice_manager = self.app_state.get("_voice_manager")
        return voice_manager.audio_clock(fp, dt) if voice_manager is not None and fp and dt else None

    def _transcript(self, fp: str | None, dt: str | None) -> str:
        voice_manager = self.app_state.get("_voice_manager")
        return voice_manager.snapshot_text(fp, dt) if voice_manager is not None and fp and dt else ""

    async def submit_answer(self, answer: str, *, voice: dict | None = None, answered_at: float | None = None):
        """Hand one round's reply to the interview assistant (typed, or the pointer of a round asked aloud) with the transcript as it stands
        now, and show what it does next. The next voice answer is counted from this moment of the recording."""
        app_state = self.app_state
        status = self.refs["status"]
        fp = app_state.get("selected_patient_folder")
        dt = app_state.get("selected_session_date")
        ic_sub = self.interview_state.get_subagent()
        if not ic_sub:
            status.text = "⚠️ 問診助理尚未啟動"
            return
        if self._assistant_step is not None:
            return
        step = self._assistant_step = object()
        try:
            await self._submit(ic_sub, answer, voice=voice, answered_at=answered_at, step=step)
        finally:
            self._release(step)

    def _release(self, step: object):
        if self._assistant_step is step:
            self._assistant_step = None

    async def _submit(self, ic_sub, answer: str, *, voice: dict | None, answered_at: float | None, step: object):
        app_state = self.app_state
        status = self.refs["status"]
        fp = app_state.get("selected_patient_folder")
        dt = app_state.get("selected_session_date")
        if fp and dt:
            ic_sub.behavior_context = {"folder_path": fp, "date_str": dt}

        run_id = self._next_ic_run_id()
        rollback_point = dict(app_state.get("_ic_rollback_point") or {})
        if answered_at is None:
            answered_at = self._audio_clock(fp, dt)
        if answered_at is not None:
            ic_sub.answer_from = answered_at
        voice_transcript = self._transcript(fp, dt)

        self.add_message("patient", answer, ic_sub.dialogue_round)
        self._set_reply_enabled(False)
        status.text = "⏳ 問診助理思考中..."

        hist = app_state.get("history")
        current = hist.get_current() if hist else None
        note_content = current["note"] if current else ""
        at_content = current["at"] if current else ""

        def log_cb(msg):
            if fp and dt:
                self.write_session_log(fp, dt, msg)

        await self._run_assistant(
            lambda: ic_sub.receive_answer(
                patient_input=answer,
                note_content=note_content,
                at_content=at_content,
                log_callback=log_cb,
                voice=voice,
                voice_transcript=voice_transcript,
            ),
            fp,
            dt,
            run_id=run_id,
            rollback_point=rollback_point,
            step=step,
        )

    async def retry_assistant(self):
        """Run the assistant's failed step again with the transcript of now; the reply it was answering is already recorded."""
        app_state = self.app_state
        fp = app_state.get("selected_patient_folder")
        dt = app_state.get("selected_session_date")
        ic_sub = self.interview_state.get_subagent()
        if not ic_sub or not self.interview_state.is_active():
            self.refs["status"].text = "⚠️ 問診助理尚未啟動"
            self._show_retry(False)
            return
        if self._assistant_step is not None:     # a second click before the first one's button went away: one model call, not two
            return
        step = self._assistant_step = object()
        try:
            await self._retry(ic_sub, fp, dt, step=step)
        finally:
            self._release(step)

    async def _retry(self, ic_sub, fp: str | None, dt: str | None, *, step: object):
        app_state = self.app_state
        if fp and dt:
            ic_sub.behavior_context = {"folder_path": fp, "date_str": dt}
        run_id = self._next_ic_run_id()
        rollback_point = dict(app_state.get("_ic_rollback_point") or {})
        voice_transcript = self._transcript(fp, dt)
        self._show_retry(False)
        self._set_reply_enabled(False)
        self.refs["status"].text = "⏳ 問診助理重試中..."

        hist = app_state.get("history")
        current = hist.get_current() if hist else None
        note_content = current["note"] if current else ""
        at_content = current["at"] if current else ""

        def log_cb(msg):
            if fp and dt:
                self.write_session_log(fp, dt, msg)

        await self._run_assistant(
            lambda: ic_sub.retry_turn(note_content=note_content, at_content=at_content, log_callback=log_cb,
                                      voice_transcript=voice_transcript),
            fp,
            dt,
            run_id=run_id,
            rollback_point=rollback_point,
            step=step,
        )

    async def _run_assistant(self, call: Callable[[], dict], fp: str | None, dt: str | None, *, run_id: int, rollback_point: dict,
                             step: object):
        """Run one step of the assistant off the web loop and show what it did. A result that arrives after the interview was stopped or
        moved on is discarded; an unexpected error is shown like a failed step (the reply is kept, the physician retries or stops)."""
        try:
            try:
                result = await asyncio.get_event_loop().run_in_executor(None, call)
            finally:
                self._release(step)              # the assistant's step is over; resuming the Main Agent is not part of it
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
            self._show_failure(f"問診助理錯誤：{e}", fp, dt)

    async def answer_from_transcript(self):
        """The physician asked this round aloud: the stretch of the recording from the previous reply to now is the answer. A small window
        waits for the recognition to reach this moment, shows the lines, takes an optional typed addition, and sends a pointer to them."""
        app_state = self.app_state
        status = self.refs["status"]
        fp = app_state.get("selected_patient_folder")
        dt = app_state.get("selected_session_date")
        ic_sub = self.interview_state.get_subagent()
        if not ic_sub or not self.interview_state.is_active():
            status.text = "⚠️ 問診助理尚未啟動"
            return
        end = self._audio_clock(fp, dt)
        if end is None:
            status.text = "⚠️ 語音功能目前不能用於這個就診（可能正在錄另一個就診）"
            return
        if end <= 0:
            status.text = "⚠️ 這個就診還沒有錄音：請先在「語音功能」開始錄音，口頭問完再按"
            return
        if self._voice_answer_open:
            return
        self._voice_answer_open = True
        try:
            start = min(ic_sub.answer_from or 0.0, end)          # set when the first question appeared, moved by every reply
            opened = (ic_sub, ic_sub.dialogue_round, app_state.get("_ic_run_id"))
            picked = await self._voice_answer_dialog(fp, dt, start, end, ic_sub.dialogue_round)
            if picked is None:
                return
            now = self.interview_state.get_subagent()
            if not self.interview_state.is_active() or (now, getattr(now, "dialogue_round", None), app_state.get("_ic_run_id")) != opened:
                status.text = f"⚠️ 問診已經不在 R{opened[1]}（已回覆、已終止或已換就診），這份語音回答沒有送出"
                return
            supplement, window = picked
            content = voice_answer_content(window.first, window.last, start, end, caught_up=window.caught_up, supplement=supplement)
            voice = {"first": window.first, "last": window.last, "start": start, "end": end}
            await self.submit_answer(content, voice=voice, answered_at=end)
        finally:
            self._voice_answer_open = False

    async def _voice_answer_dialog(self, fp: str, dt: str, start: float, end: float, round_num: int):
        """(the typed addition, the final window), or None when the physician goes back to asking."""
        ui = self.ui
        voice_manager = self.app_state.get("_voice_manager")
        state: dict[str, Any] = {"lines": None, "waited": 0.0}

        with ui.dialog().props("persistent") as dialog, ui.card().style("min-width: 560px; max-width: 820px;"):
            ui.label(f"🎙 R{round_num} 語音回答").style("font-weight: 700; font-size: 16px;")
            ui.label(f"本回合區間 {format_clock(start)}–{format_clock(end)}：從上一次回覆送出到按下按鈕。"
                     "問診紀錄只記錄這段逐字稿的位置，病歷直接引用逐字稿。").style("font-size: 13px; color: #666;")
            wait_label = ui.label("").style("font-size: 13px;")
            lines_box = ui.column().classes("w-full gap-1").style(
                "max-height: 320px; overflow-y: auto; border: 1px solid var(--border); border-radius: 8px; padding: 8px;"
            )
            supplement = ui.textarea(
                label="補充或更正（選填，照一般打字回覆記錄）",
                placeholder="例如：語音把「沒有夜尿」聽成「有夜尿」；或實習醫師量到的數值",
            ).classes("w-full")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("取消（繼續問）", on_click=lambda: dialog.submit(None)).props("flat")
                btn_ok = ui.button("送出", color="primary", on_click=lambda: dialog.submit((supplement.value or "").strip()))

            def refresh():
                window = voice_manager.answer_window(fp, dt, start, end) if voice_manager is not None else None
                if window is None:
                    wait_label.text = "⚠️ 語音功能目前不能用於這個就診；可以只送出下方的打字內容，或取消"
                    btn_ok.enable()
                    timer.deactivate()
                    return
                if window.lines != state["lines"]:
                    state["lines"] = list(window.lines)
                    lines_box.clear()
                    with lines_box:
                        if not window.lines:
                            ui.label("（這段時間還沒有辨識到對話）").style("color: #999; font-size: 13px;")
                        for _, line in window.lines:
                            ui.label(line).style("font-size: 13px; white-space: pre-wrap;")
                if window.caught_up:
                    wait_label.text = (f"✅ 辨識已跟上，共 {len(window.lines)} 段" if window.lines
                                       else "✅ 辨識已跟上：這段時間沒有辨識到對話")
                    btn_ok.enable()
                    timer.deactivate()
                    return
                state["waited"] += timer.interval
                wait_label.text = f"⏳ 辨識中…逐字稿目前到 {format_clock(window.reach)}，要等到 {format_clock(end)}"
                if state["waited"] >= VOICE_WAIT_SECONDS:
                    wait_label.text += "。等比較久了，可以先送出目前的段落（較晚辨識出的段落仍屬於本回合，之後的 Agent 看得到）"
                    btn_ok.enable()
                else:
                    btn_ok.disable()

            timer = ui.timer(0.5, refresh)
        refresh()
        dialog.open()
        try:
            picked = await dialog
        finally:
            timer.deactivate()
            dialog.delete()
        if picked is None:
            return None
        window = voice_manager.answer_window(fp, dt, start, end) if voice_manager is not None else None
        if window is None:
            from voice.pipeline import AnswerWindow
            window = AnswerWindow(start, end, [], False, 0.0)
        return picked, window

    def clear_ui(self):
        if self.refs.get("chat_container"):
            self.refs["chat_container"].clear()
        if self.refs.get("guidelines_label"):
            self.refs["guidelines_label"].text = "（尚未啟動問診助理）"
        if self.refs.get("status"):
            self.refs["status"].text = ""
        self._set_reply_enabled(False)
        self._show_retry(False)
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
        awaiting_assistant: bool = False,
    ):
        if self.refs.get("guidelines_label"):
            self.refs["guidelines_label"].text = guidelines or "（方針已載入）"
        if self.refs.get("chat_container"):
            self.refs["chat_container"].clear()
            for msg in conversations:
                self.add_message(msg["role"], msg["content"], msg.get("round", 0))
        if self.refs.get("status"):
            self.refs["status"].text = f"📂 已載入先前問診紀錄 (共 {total_rounds} 回合)。如需繼續問診請重新啟動。"
        self._set_reply_enabled(False)
        self._show_retry(False)
        if self.refs.get("btn_stop"):
            self.refs["btn_stop"].disable()
        if active and awaiting_assistant:
            # the assistant owes its next step: the first question, or the answer to the last reply (it failed, or the page went away)
            if self.refs.get("status"):
                self.refs["status"].text = "問診進行中，問診助理還沒完成下一步（上次失敗，或頁面在它思考時離開）：請按「🔁 重試」或手動停止"
            self._show_retry(True)
            if self.refs.get("btn_stop"):
                self.refs["btn_stop"].enable()
        elif active:
            if self.refs.get("status"):
                self.refs["status"].text = f"問診進行中，已恢復至 R{total_rounds}，請繼續回答或手動停止"
            self._set_reply_enabled(True)
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
        elif content.startswith(VOICE_ANSWER_MARK):
            bg = "#e3f2fd"
            label_prefix = f"🎙 語音回答 R{round_num}"
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

        if action == "error":
            self._show_failure(message, fp, dt)
            ic_sub = self.interview_state.get_subagent()
            if ic_sub and fp and dt:
                self.save_interview_state(fp, dt, ic_sub.export_state())
            return False

        self._show_retry(False)
        if action == "ask_patient" and not finished:
            self.add_message("subagent", message, round_num)
            self.refs["status"].text = f"📋 問診中 (R{round_num})"
            self._set_reply_enabled(True)
            self.refs["btn_stop"].enable()

            ic_sub = self.interview_state.get_subagent()
            if ic_sub is not None and ic_sub.answer_from is None:
                # the first question of this interview is on the screen: its voice answer is counted from here
                ic_sub.answer_from = self._audio_clock(fp, dt)
            if ic_sub and fp and dt:
                self.save_interview_state(fp, dt, ic_sub.export_state())
            return False

        if finished or action == "finish_collection":
            self.refs["status"].text = f"✅ 問診完成 (共 {round_num} 回合)"
            self._set_reply_enabled(False)
            self.refs["btn_stop"].disable()
            self.finalize_interview(fp, dt)
            return True

        self.refs["status"].text = f"⚠️ 未預期的動作: {action}"
        self._set_reply_enabled(False)
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
        # 問診期間錄音可能仍在進行：恢復時重新取一份最新的診間對話逐字稿（編號只會往後增加）。
        voice_manager = self.app_state.get("_voice_manager")
        voice_transcript = voice_manager.snapshot_text(fp, dt) if voice_manager is not None else None
        rec_clock = voice_manager.frozen_clock(fp, dt) if voice_manager is not None else None

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
                    voice_transcript=voice_transcript,
                    rec_clock=rec_clock,
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
        # The interview belongs to the Main Agent step that started it (it ends by resuming that same turn): its rounds are stamped with it.
        ic_sub.current_step = self.agent_run_state.get_agent().suspended_step_label
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

        self._show_retry(False)
        self._set_reply_enabled(False)
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
        voice_transcript = self._transcript(fp, dt)          # the assistant reads the room's conversation too: no need to ask it again

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
                    voice_transcript=voice_transcript,
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
