from __future__ import annotations

import asyncio
from typing import Any, Callable

from ui_app.controllers.agent_run_state_controller import get_agent_run_state
from ui_app.rendering import patient_info_html
from ui_app.services.llm_config_resolver import resolve_main_child_llm_config


class PatientSessionLifecycleController:
    """Own patient/session selection, creation, deletion, and loading flow."""

    def __init__(
        self,
        *,
        ui: Any,
        app_state: dict[str, Any],
        ui_state: dict[str, Any],
        app_context: Any,
        history: Any,
        refs: dict[str, Any],
        today: str,
        list_patients: Callable[[], list[dict]],
        list_sessions: Callable[[str], list[str]],
        get_session_summaries: Callable[[str, str], dict],
        load_patient: Callable[[str], dict | None],
        save_patient_info: Callable[[str, dict], str],
        load_session_content: Callable[[str, str], dict],
        save_session_summaries: Callable[[str, str, str, str], str],
        create_session: Callable[[str, str, str], str],
        delete_session: Callable[[str, str], str],
        write_session_log: Callable[[str, str, str], None],
        load_config: Callable[[], dict],
        record_snapshot_store: Any,
        generate_history_summary: Callable[[str, str], str],
        get_session_restorer: Callable[[], Any],
        update_display: Callable[[], None],
        update_buttons: Callable[[], None],
        render_chat: Callable[[], None],
        busy_guard: Any | None = None,
    ):
        self.ui = ui
        self.app_state = app_state
        self.ui_state = ui_state
        self.app_context = app_context
        self.history = history
        self.refs = refs
        self.today = today
        self.list_patients = list_patients
        self.list_sessions = list_sessions
        self.get_session_summaries = get_session_summaries
        self.load_patient = load_patient
        self.save_patient_info = save_patient_info
        self.load_session_content = load_session_content
        self.save_session_summaries = save_session_summaries
        self.create_session = create_session
        self.delete_session = delete_session
        self.write_session_log = write_session_log
        self.load_config = load_config
        self.record_snapshot_store = record_snapshot_store
        self.generate_history_summary = generate_history_summary
        self.get_session_restorer = get_session_restorer
        self.update_display = update_display
        self.update_buttons = update_buttons
        self.render_chat = render_chat
        self.agent_run_state = get_agent_run_state(app_state, ui_state)
        self.busy_guard = busy_guard
        self.last_restore_result: dict[str, Any] = {}

    def _sync_busy_navigation(self):
        if self.busy_guard:
            self.busy_guard.sync_navigation()

    def sync_global_settings_save_state(self):
        sync_fn = self.app_state.get("_sync_template_save_state")
        if sync_fn:
            sync_fn()
        sync_fn = self.app_state.get("_sync_model_settings_save_state")
        if sync_fn:
            sync_fn()
        sync_fn = self.app_state.get("_sync_professor_settings_save_state")
        if sync_fn:
            sync_fn()

    def refresh_patient_list(self):
        options = {}
        for patient in self.list_patients():
            bi = patient["basic_info"]
            label = f"{bi.get('id', '')} - {bi.get('name', '')} ({bi.get('birthday', '')})"
            options[patient["folder_path"]] = label
        self.refs["patient_select"].options = options
        self.refs["patient_select"].update()
        self._sync_busy_navigation()

    @staticmethod
    def _session_option_label(date_str: str, note_summary: str, limit: int = 30) -> str:
        summary = " ".join((note_summary or "").split()) or "空白"
        if len(summary) > limit:
            summary = f"{summary[:limit]}..."
        return f"{date_str}（{summary}）"

    def refresh_session_list(self):
        fp = self.app_state["selected_patient_folder"]
        session_select = self.refs["session_select"]
        template_select = self.refs["template_select"]
        if not fp:
            session_select.options = []
            session_select.update()
            self._sync_busy_navigation()
            return
        dates = self.list_sessions(fp)
        session_options = {}
        for dt in dates:
            summaries = self.get_session_summaries(fp, dt)
            session_options[dt] = self._session_option_label(dt, summaries.get("note_summary", ""))
        session_select.options = session_options if session_options else []
        template_select.options = ["（空白病歷）"] + dates
        session_select.update()
        template_select.update()
        self._sync_busy_navigation()

    def on_save_remark(self):
        if self._reject_if_busy():
            return
        fp = self.app_state["selected_patient_folder"]
        if not fp:
            self.refs["remark_status"].text = "⚠️ 請先選取患者"
            return
        info = self.load_patient(fp)
        if info:
            info["basic_info"]["remark"] = self.refs["remark_area"].value or ""
            self.save_patient_info(fp, info)
            self.app_state["selected_patient_info"] = info
            self.refs["remark_status"].text = "✅ 備註已儲存"

    def load_session_into_ui(self):
        fp = self.app_state["selected_patient_folder"]
        dt = self.app_state["selected_session_date"]
        if not fp or not dt:
            self.refs["note_display"].content = ""
            self.refs["at_display"].content = ""
            self.refs["note_editor"].value = ""
            self.refs["at_editor"].value = ""
            self.history.reset()
            return

        content = self.load_session_content(fp, dt)
        note_text = content.get("note", "")
        at_text = content.get("at", "")

        info = self.load_patient(fp)
        if info:
            self.refs["remark_area"].value = info.get("basic_info", {}).get("remark", "")

        restore_result = self.record_snapshot_store.restore_or_initialize(
            folder_path=fp,
            date_str=dt,
            history=self.history,
            note_text=note_text,
            at_text=at_text,
        )
        self.last_restore_result = restore_result

        self.update_display()
        self.update_buttons()
        session_restorer = self.get_session_restorer()
        session_restorer.restore_main_chat_state(fp, dt)
        session_restorer.restore_interview_state(fp, dt)
        status_text = f"已載入 {dt}"
        if restore_result.get("external_state_added"):
            status_text += "（偵測到外部檔案狀態，已新增版本）"
        self.refs["session_status"].text = status_text
        session_restorer.restore_forum_state(fp, dt)
        behavior_render = self.app_state.get("render_agent_behavior")
        if behavior_render:
            behavior_render()
        self._sync_busy_navigation()

    def refresh_linked_tabs(self, include_forum: bool = True):
        if self.app_state.get("_refresh_image_list"):
            self.app_state["_refresh_image_list"](clear_staged_files=True)
        if self.app_state.get("_refresh_medinfo_list"):
            self.app_state["_refresh_medinfo_list"]()
        if include_forum:
            render_fn = self.app_state.get("render_forum")
            if render_fn:
                render_fn()
        behavior_render = self.app_state.get("render_agent_behavior")
        if behavior_render:
            behavior_render()
        self._sync_busy_navigation()

    def reset_main_agent_input_state(self):
        self.agent_run_state.finish_run()
        main_chat_input = self.app_state.get("_main_chat_input")
        if main_chat_input:
            main_chat_input.reset()

    def _reject_if_busy(self) -> bool:
        """忙碌中禁止切換、刪除或寫入 session-sensitive 狀態。回傳 True 表示已拒絕。"""
        if self.busy_guard:
            return self.busy_guard.reject_if_busy(status_label=self.refs["session_status"])
        return False

    @staticmethod
    def _compact_summary(text: str, limit: int = 50) -> str:
        compact = " ".join((text or "").split())
        if len(compact) <= limit:
            return compact
        return compact[:limit]

    def _summarize_record_text(self, label: str, text: str) -> str:
        fallback = self._compact_summary(text)
        if not text.strip():
            return ""

        cfg = self.load_config()
        main_cfg = cfg.get("main_agent", {})
        llm_cfg = resolve_main_child_llm_config(
            main_cfg,
            "summary_exit",
            "summary_exit_model_name",
            default_max_tokens=128,
            default_temperature=0.2,
        )
        model_name = llm_cfg["model_name"]
        if not model_name:
            print(f"[Summary Exit] {label}: summary_exit_model_name/main_agent.model_name 未設定，使用前 50 字 fallback")
            print(f"[Summary Exit] {label} fallback: {fallback}")
            return fallback

        system_prompt = """你是病歷索引摘要員。
你的任務是將單一病歷欄位濃縮成一句 50 字以內的檔案索引短句，供檔案列表快速辨識。
摘要應以關鍵症狀、主要病程、診斷或治療主題為主，可用逗號串列呈現，例如「感冒第5天，頭痛、筋骨痠痛、喉嚨痛」。
不得新增原文沒有的資訊。"""
        user_prompt = f"""請將以下{label}濃縮成一句 50 字以內的病歷索引摘要。

要求：
1. 只輸出摘要本身，不要加標題、引號或條列。
2. 優先列出最能辨識本次就診的關鍵症狀、主要病程、診斷、治療或處置，可用逗號與頓號連接。
3. 不新增原文沒有的資訊。
4. 避免籠統描述，例如「病歷摘要」「追蹤治療」「中醫治療」；請直接寫出具體關鍵詞。

【{label}】
{text}
"""
        try:
            from openai import OpenAI

            print("\n" + "=" * 80)
            print(f"[Summary Exit] {label} 摘要模型呼叫")
            print("-" * 80)
            print("[Summary Exit] System Prompt:")
            print(system_prompt)
            print("-" * 80)
            print("[Summary Exit] User Prompt:")
            print(user_prompt)
            print("-" * 80)

            client = OpenAI(
                base_url=llm_cfg["api_url"],
                api_key=llm_cfg["api_key"],
            )
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=llm_cfg["max_tokens"],
                temperature=llm_cfg["temperature"],
            )
            summary = (resp.choices[0].message.content or "").strip()
            print("[Summary Exit] Raw Output:")
            print(summary)
            summary = summary.strip("\"'「」")
            final_summary = self._compact_summary(summary or fallback)
            print("[Summary Exit] Final Summary:")
            print(final_summary)
            print("=" * 80 + "\n")
            return final_summary
        except Exception as e:
            print(f"[Summary Exit] {label}: 摘要模型呼叫失敗，使用前 50 字 fallback: {e}")
            print(f"[Summary Exit] {label} fallback: {fallback}")
            return fallback

    def _summarize_current_session(self, fp: str, dt: str) -> tuple[str, str]:
        current = self.history.get_current()
        if current:
            note_text = current.get("note", "")
            at_text = current.get("at", "")
        else:
            content = self.load_session_content(fp, dt)
            note_text = content.get("note", "")
            at_text = content.get("at", "")

        note_summary = self._summarize_record_text("NOTE", note_text)
        at_summary = self._summarize_record_text("ASSESSMENT & TREATMENT", at_text)
        return note_summary, at_summary

    @staticmethod
    def _history_summary_failure_message(summary: str) -> str | None:
        text = (summary or "").strip()
        if text.startswith("（未設定模型"):
            return "未設定模型，無法產生歷史病歷摘要"
        if text.startswith("（歷史病歷摘要產生失敗："):
            detail = text.removeprefix("（歷史病歷摘要產生失敗：").removesuffix("）")
            return f"歷史病歷摘要產生失敗：{detail}"
        return None

    def _clear_current_session_ui(self):
        self.app_state["selected_session_date"] = None
        self.app_context.reset_agent_state()
        self.app_context.bump_session_generation()
        self.refs["session_select"].value = None

        self.ui_state["view_mode"] = "browse"
        if self.busy_guard:
            self.busy_guard.sync_navigation()
        if self.refs.get("browse_container"):
            self.refs["browse_container"].set_visibility(True)
        if self.refs.get("edit_container"):
            self.refs["edit_container"].set_visibility(False)
        self.refs["note_editor"].value = ""
        self.refs["at_editor"].value = ""
        if self.refs.get("record_status"):
            self.refs["record_status"].text = ""

        self.history.reset()
        self.update_display()
        self.update_buttons()

        self.ui_state["chat_messages"] = []
        self.reset_main_agent_input_state()
        self.render_chat()
        self.refs["agent_status"].text = ""

        render_fn = self.app_state.get("render_forum")
        if render_fn:
            render_fn()

        clear_interview_fn = self.app_state.get("_clear_interview_ui")
        if clear_interview_fn:
            clear_interview_fn()

        behavior_render = self.app_state.get("render_agent_behavior")
        if behavior_render:
            behavior_render()

    def on_confirm_patient(self):
        if self._reject_if_busy():
            return
        folder_path = self.refs["patient_select"].value
        if folder_path is None or folder_path not in (self.refs["patient_select"].options or {}):
            self.refs["session_status"].text = "⚠️ 請先從下拉選單選取一位患者"
            return

        self.app_state["selected_patient_folder"] = folder_path
        self.app_state["selected_session_date"] = None
        self.app_context.reset_agent_state()
        self.app_context.bump_session_generation()
        self.refs["session_select"].value = None

        self.ui_state["chat_messages"] = []
        self.reset_main_agent_input_state()
        self.render_chat()
        self.refs["agent_status"].text = ""

        info = self.load_patient(folder_path) if folder_path else None
        self.app_state["selected_patient_info"] = info

        if info:
            bi = info.get("basic_info", {})
            self.refs["patient_info_label"].content = patient_info_html(bi)
            self.refs["remark_area"].value = bi.get("remark", "")
            self.refs["session_status"].text = f"✅ 已載入患者：{bi.get('name', '')}"
        else:
            self.refs["patient_info_label"].content = ""
            self.refs["remark_area"].value = ""

        self.refresh_session_list()
        self.history.reset()
        self.update_display()
        self.update_buttons()
        self.refresh_linked_tabs()
        self.sync_global_settings_save_state()

    def clear_patient_ui(self):
        self.refs["patient_select"].value = None
        self.refs["session_select"].value = None
        self.refs["session_select"].options = []
        self.refs["patient_info_label"].content = ""
        self.refs["remark_area"].value = ""
        self.refs["remark_status"].text = ""
        self.refs["new_session_panel"].set_visibility(False)

        self.history.reset()
        self.update_display()
        self.update_buttons()

        self.refresh_linked_tabs(include_forum=False)

        self.ui_state["chat_messages"] = []
        self.reset_main_agent_input_state()
        self.render_chat()
        self.refs["agent_status"].text = ""

        render_fn = self.app_state.get("render_forum")
        if render_fn:
            render_fn()

        clear_interview_fn = self.app_state.get("_clear_interview_ui")
        if clear_interview_fn:
            clear_interview_fn()
        self.sync_global_settings_save_state()

    def update_patient_display(self):
        fp = self.app_state.get("selected_patient_folder")
        info = self.app_state.get("selected_patient_info")
        if fp and info:
            bi = info.get("basic_info", {})
            self.refs["patient_select"].value = fp
            self.refs["patient_select"].update()
            self.refs["patient_info_label"].content = patient_info_html(bi)
            self.refs["remark_area"].value = bi.get("remark", "")

    def on_exit_patient(self):
        if self._reject_if_busy():
            return
        self.app_context.reset_patient_selection()
        self.clear_patient_ui()
        self.refs["session_status"].text = "🚪 已退出患者"

    def on_refresh_patients(self):
        if self._reject_if_busy():
            return
        self.refresh_patient_list()
        self.refs["session_status"].text = "🔄 患者列表已重新掃描"

    def on_confirm_session(self):
        if self._reject_if_busy():
            return
        dt = self.refs["session_select"].value
        if not dt:
            self.refs["session_status"].text = "⚠️ 請先從下拉選單選取一個日期"
            return
        self.app_state["selected_session_date"] = dt
        self.app_context.reset_agent_state()
        self.app_context.bump_session_generation()
        self.reset_main_agent_input_state()
        self.load_session_into_ui()
        status_text = f"✅ 已載入就診日期：{dt}"
        if self.last_restore_result.get("external_state_added"):
            status_text += "（偵測到外部檔案狀態，已新增版本）"
        self.refs["session_status"].text = status_text

    async def on_summary_exit_session(self):
        fp = self.app_state["selected_patient_folder"]
        dt = self.app_state.get("selected_session_date")
        if not fp or not dt:
            self.refs["session_status"].text = "⚠️ 請先確認載入一個就診日期"
            return

        if self._reject_if_busy():
            return

        if self.busy_guard:
            self.busy_guard.begin_transition()
        self.refs["session_status"].text = f"⏳ 正在摘要 {dt} 並退出..."
        await asyncio.sleep(0)

        try:
            note_summary, at_summary = await asyncio.to_thread(self._summarize_current_session, fp, dt)
            msg = self.save_session_summaries(fp, dt, note_summary, at_summary)
            self.write_session_log(
                fp,
                dt,
                f"[SESSION_SUMMARY_EXIT] NOTE={note_summary or '（空白）'}; A&T={at_summary or '（空白）'}",
            )
            info = self.load_patient(fp)
            if info:
                self.app_state["selected_patient_info"] = info
            self._clear_current_session_ui()
            self.refresh_session_list()
            self._sync_busy_navigation()
            self.refs["session_status"].text = f"{msg}，已退出就診日期：{dt}"
        except Exception as e:
            self.refs["session_status"].text = f"❌ 摘要並退出失敗：{e}"
        finally:
            if self.busy_guard:
                self.busy_guard.end_transition()

    def on_new_session_click(self):
        if self._reject_if_busy():
            return
        self.refs["new_session_panel"].set_visibility(True)
        self.refs["new_session_date"].value = self.today

    def on_cancel_new_session(self):
        if self._reject_if_busy():
            return
        self.refs["new_session_panel"].set_visibility(False)

    async def on_confirm_new_session(self):
        if self._reject_if_busy():
            return
        fp = self.app_state["selected_patient_folder"]
        if not fp:
            self.refs["session_status"].text = "⚠️ 請先選取患者"
            return
        date_val = self.refs["new_session_date"].value or ""
        template_val = self.refs["template_select"].value or ""
        tmpl = "" if template_val == "（空白病歷）" else template_val
        btn_confirm_new = self.refs.get("btn_confirm_new")

        if self.busy_guard:
            self.busy_guard.begin_transition()
        created = False
        try:
            msg = self.create_session(fp, date_val, tmpl)
            self.refs["session_status"].text = msg
            if not msg.startswith("✅"):
                return
            created = True

            self.refs["new_session_panel"].set_visibility(False)
            self.refresh_session_list()
            self.refs["session_select"].value = date_val
            self.app_state["selected_session_date"] = date_val

            self.app_context.reset_agent_state()
            self.app_context.bump_session_generation()
            self.ui_state["chat_messages"] = []
            self.reset_main_agent_input_state()

            self.load_session_into_ui()
            self.write_session_log(fp, date_val, f"[SESSION] 建立新就診日期 (模板: {tmpl or '無'})")

            if btn_confirm_new:
                btn_confirm_new.disable()
            self._sync_busy_navigation()
            self.refs["session_status"].text = "⏳ 已建立，正在產生歷史病歷摘要，請稍候..."
            await asyncio.sleep(0)
            summary = await asyncio.to_thread(self.generate_history_summary, fp, date_val)
            summary_error = self._history_summary_failure_message(summary)
            if self.app_state.get("selected_patient_folder") == fp and self.app_state.get("selected_session_date") == date_val:
                if summary_error:
                    self.refs["session_status"].text = f"⚠️ 已建立，但{summary_error}"
                else:
                    self.refs["session_status"].text = "✅ 已建立，歷史病歷摘要已產生"
            if summary_error:
                self.write_session_log(fp, date_val, f"[SESSION] 歷史病歷摘要未產生: {summary_error}")
            else:
                self.write_session_log(fp, date_val, "[SESSION] 歷史病歷摘要已產生")
        except Exception as e:
            if not created:
                self.refs["session_status"].text = f"❌ 新增就診日期失敗：{e}"
            elif self.app_state.get("selected_patient_folder") == fp and self.app_state.get("selected_session_date") == date_val:
                self.refs["session_status"].text = f"⚠️ 已建立，但歷史病歷摘要產生失敗：{e}"
                self.write_session_log(fp, date_val, f"[SESSION] 歷史病歷摘要產生失敗: {e}")
        finally:
            if self.busy_guard:
                self.busy_guard.end_transition()
            if btn_confirm_new and not (self.busy_guard and self.busy_guard.is_busy()):
                btn_confirm_new.enable()

    def on_delete_session(self):
        if self._reject_if_busy():
            return
        fp = self.app_state["selected_patient_folder"]
        dt = self.refs["session_select"].value
        if not fp or not dt:
            self.refs["session_status"].text = "⚠️ 請先選取就診日期"
            return
        loaded_dt = self.app_state.get("selected_session_date")

        with self.ui.dialog() as dialog, self.ui.card().style("min-width: 320px;"):
            self.ui.label(f"確定要刪除 {dt} 的就診紀錄嗎？").style("font-size: 16px; font-weight: bold;")
            self.ui.label("此操作無法復原！").style("color: red; font-size: 14px;")
            with self.ui.row().classes("w-full justify-end gap-2 q-mt-md"):
                self.ui.button("取消", color="grey", on_click=dialog.close).props("flat")

                def do_delete():
                    if self._reject_if_busy():
                        dialog.close()
                        return
                    msg = self.delete_session(fp, dt)
                    self.refs["session_status"].text = msg
                    if msg.startswith("✅"):
                        if dt == loaded_dt:
                            self._clear_current_session_ui()
                            self.refresh_session_list()
                        else:
                            self.refresh_session_list()
                            self.refs["session_select"].value = loaded_dt
                            self.refs["session_select"].update()

                    dialog.close()

                self.ui.button("確認刪除", color="red", on_click=do_delete)
        dialog.open()
