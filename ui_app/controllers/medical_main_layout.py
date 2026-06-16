from __future__ import annotations

from typing import Any


def build_medical_main_layout(ui: Any, today: str) -> dict[str, Any]:
    refs: dict[str, Any] = {}

    with ui.row().classes("w-full items-start gap-3").style("flex-wrap: nowrap; min-height: calc(100vh - 160px);"):
        with ui.column().classes("card-panel").style("width: 280px; min-width: 260px; flex-shrink: 0; padding: 16px;"):
            ui.label("🏥 病歷導覽").classes("section-title")

            ui.label("選取患者").style("font-weight: 600; font-size: 14px; margin-top: 8px;")
            refs["patient_select"] = ui.select(
                label="搜尋 / 選取患者（可輸入 ID、姓名、生日）",
                options=[],
                with_input=True,
            ).classes("w-full")
            with ui.row().classes("w-full gap-2 q-mt-xs"):
                refs["btn_confirm_patient"] = ui.button("✅ 確認患者", color="primary").classes("flex-1").props("dense")
                refs["btn_refresh_patients"] = (
                    ui.button("🔄 重新掃描", color="blue-grey").classes("flex-1").props("dense outline")
                )
                refs["btn_exit_patient"] = ui.button("🚪 退出患者", color="red").classes("flex-1").props("dense outline")

            ui.separator().style("margin: 12px 0;")

            ui.label("就診日期").style("font-weight: 600; font-size: 14px;")
            refs["session_select"] = ui.select(label="選取日期", options=[]).classes("w-full")

            with ui.row().classes("w-full gap-2 q-mt-xs"):
                refs["btn_confirm_session"] = ui.button("✅ 確認日期", color="primary").classes("flex-1").props("dense")
                refs["btn_new_session"] = ui.button("➕ 新增", color="green").classes("flex-1").props("dense")
                refs["btn_del_session"] = ui.button("🗑️ 刪除", color="red").classes("flex-1").props("dense")
            refs["btn_summary_exit_session"] = (
                ui.button("🧾 摘要並退出", color="blue-grey")
                .props("dense outline")
                .classes("w-full q-mt-xs")
            )

            with ui.column().classes("w-full q-mt-sm") as new_session_panel:
                refs["new_session_date"] = ui.input(label="日期 (YYYY-MM-DD)", value=today).classes("w-full")
                refs["template_select"] = (
                    ui.select(label="選擇模板", options=["（空白病歷）"], value="（空白病歷）").classes("w-full")
                )
                with ui.row().classes("w-full gap-2"):
                    refs["btn_confirm_new"] = ui.button("✅ 確認新增", color="green").classes("flex-1").props("dense")
                    refs["btn_cancel_new"] = ui.button("取消", color="grey").classes("flex-1").props("dense")
            refs["new_session_panel"] = new_session_panel
            new_session_panel.set_visibility(False)

            refs["session_status"] = ui.label("").style("font-size: 12px; color: #888; margin-top: 4px;")

            ui.separator().style("margin: 12px 0;")
            refs["patient_info_label"] = ui.html("").style("font-size: 13px; color: #555;")

            ui.separator().style("margin: 12px 0;")
            ui.label("📌 備註").style("font-weight: 600; font-size: 14px;")
            refs["remark_area"] = ui.textarea(label="", placeholder="備註...").classes("w-full").style(
                "min-height: 60px;"
            )
            refs["remark_status"] = ui.label("").style("font-size: 11px; color: #888;")
            refs["btn_save_remark"] = ui.button("💾 更新備註", color="primary").props("dense").classes("w-full q-mt-xs")

        with ui.column().classes("card-panel flex-1").style("min-width: 480px; padding: 16px;"):
            ui.label("📝 病歷內容").classes("section-title")

            with ui.row().classes("w-full gap-2 q-mb-sm"):
                refs["btn_browse"] = ui.button("📖 一般瀏覽", color="green").props("dense").classes("mode-btn-active")
                refs["btn_diff"] = ui.button("🔍 差異瀏覽", color="blue").props("dense")
                refs["btn_edit_mode"] = ui.button("✏️ 修改模式", color="orange").props("dense")
                refs["btn_edit_done"] = ui.button("✅ 修改完成", color="green").props("dense")
                refs["btn_undo"] = ui.button("⬅️ 上一步", color="grey").props("dense")
                refs["btn_redo"] = ui.button("➡️ 下一步", color="grey").props("dense")

            refs["btn_edit_done"].disable()
            refs["version_label"] = ui.label("").style("font-size: 12px; color: #888; margin-bottom: 4px;")

            ui.separator().style("margin: 8px 0;")

            with ui.column().classes("w-full") as browse_container:
                ui.label("📋 NOTE").style("font-weight: 600; font-size: 15px;")
                refs["note_display"] = ui.html("").classes("w-full").style(
                    "min-height: 180px; padding: 12px; background: #fafafa; border: 1px solid #e8e8e8; border-radius: 8px; "
                    "white-space: pre-wrap; font-size: 14px; line-height: 1.7;"
                )

                ui.separator().style("margin: 8px 0;")

                ui.label("💊 ASSESSMENT & TREATMENT").style("font-weight: 600; font-size: 15px;")
                refs["at_display"] = ui.html("").classes("w-full").style(
                    "min-height: 120px; padding: 12px; background: #fafafa; border: 1px solid #e8e8e8; border-radius: 8px; "
                    "white-space: pre-wrap; font-size: 14px; line-height: 1.7;"
                )
            refs["browse_container"] = browse_container

            with ui.column().classes("w-full") as edit_container:
                ui.label("📋 NOTE (修改模式)").style("font-weight: 600; font-size: 15px;")
                refs["note_editor"] = ui.textarea(label="", placeholder="在此輸入 NOTE...").classes("w-full").style(
                    "min-height: 200px; font-family: monospace; font-size: 14px;"
                )

                ui.separator().style("margin: 8px 0;")

                ui.label("💊 ASSESSMENT & TREATMENT (修改模式)").style("font-weight: 600; font-size: 15px;")
                refs["at_editor"] = ui.textarea(label="", placeholder="在此輸入 A&T...").classes("w-full").style(
                    "min-height: 140px; font-family: monospace; font-size: 14px;"
                )
            refs["edit_container"] = edit_container
            edit_container.set_visibility(False)

            refs["record_status"] = ui.label("").style("font-size: 13px; margin-top: 4px;")

        with ui.column().classes("card-panel").style("width: 380px; min-width: 340px; flex-shrink: 0; padding: 16px;"):
            ui.label("🤖 AI 主治醫師").classes("section-title")

            refs["chat_container"] = ui.column().classes("w-full main-chat-scroll").style(
                "max-height: calc(100vh - 340px); overflow-y: auto; gap: 8px;"
            )

            ui.separator().style("margin: 8px 0;")

            with ui.row().classes("w-full gap-2 items-end"):
                refs["agent_input"] = ui.textarea(
                    label="訊息",
                    placeholder="輸入訊息給 AI 主治醫師...",
                ).classes("flex-1").style("min-height: 60px;")
                refs["btn_send"] = ui.button("📤", color="green").props("dense").style("height: 40px;")
                refs["btn_stop_agent"] = ui.button("⏹", color="orange").props("dense").style("height: 40px;")
                refs["btn_stop_agent"].disable()

            refs["agent_status"] = ui.label("").style("font-size: 12px; color: #888; margin-top: 4px;")

            refs["live_steps_container"] = ui.column().classes("w-full").style(
                "background: #e3f2fd; border-radius: 8px; padding: 10px; gap: 4px;"
            )
            refs["live_steps_container"].set_visibility(False)

    return refs
