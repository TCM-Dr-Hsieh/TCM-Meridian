from __future__ import annotations

from typing import Any, Callable

from ui_app.controllers.session_busy_guard import get_session_busy_guard


def build_record_template_tab(
    ui: Any,
    app_state: dict[str, Any],
    load_template: Callable[[], str],
    save_template: Callable[[str], None],
    reset_agent_state: Callable[[], None],
):
    """Render the record-template settings tab."""
    busy_guard = get_session_busy_guard(app_state)
    with ui.column().classes("w-full").style("max-width: 900px; margin: 0 auto;"):
        ui.label("📋 標準病歷模板設定").classes("section-title")
        ui.label(
            "設定病歷的標準格式模板。此模板會自動注入到 AI 主治醫師與病歷登載助理的 System Prompt 中，"
            "作為病歷書寫的參考依據。"
        ).style("color: #666; font-size: 14px; margin-bottom: 16px;")

        template_editor = ui.textarea(
            label="標準病歷模板內容",
            value=load_template(),
        ).classes("w-full").style(
            "min-height: 500px; font-family: 'Consolas', 'Monaco', monospace; "
            "font-size: 14px; line-height: 1.6;"
        )

        with ui.row().classes("w-full gap-3 items-center"):
            btn_save = ui.button("💾 儲存模板", color="green").style("font-size: 15px;")
            btn_reload = ui.button("🔄 重新載入", color="blue-grey").style("font-size: 14px;")
            template_status = ui.label("").style("font-size: 14px; line-height: 36px;")

        def patient_is_loaded() -> bool:
            return bool(app_state.get("selected_patient_folder"))

        def sync_save_state():
            if patient_is_loaded():
                btn_save.disable()
                template_status.text = "🔒 請先退出患者，才能儲存全域病歷模板"
                template_status.style("color: #999;")
            else:
                btn_save.enable()
                if template_status.text.startswith("🔒"):
                    template_status.text = ""

        app_state["_sync_template_save_state"] = sync_save_state
        sync_save_state()

        def on_save():
            if busy_guard.reject_if_busy(status_label=template_status):
                return
            if patient_is_loaded():
                sync_save_state()
                return
            save_template(template_editor.value or "")
            reset_agent_state()
            template_status.text = "✅ 模板已儲存（Agent 將在下次對話時重新載入）"
            template_status.style("color: var(--primary);")

        def on_reload():
            template_editor.value = load_template()
            template_status.text = "🔄 已重新載入"
            template_status.style("color: #666;")

        btn_save.on_click(on_save)
        btn_reload.on_click(on_reload)
