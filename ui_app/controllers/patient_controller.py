from __future__ import annotations

import os
from typing import Any, Callable

from ui_app.controllers.session_busy_guard import get_session_busy_guard


def build_patient_registration_tab(
    ui: Any,
    app_state: dict[str, Any],
    reset_patient_selection: Callable[[], None],
    *,
    list_patients: Callable[[], list[dict]],
    load_patient: Callable[[str], dict | None],
    create_patient: Callable[..., str],
    update_patient_basic_info: Callable[..., tuple[str, str]],
    delete_patient: Callable[[str], str],
):
    """Render patient registration, edit, delete, and search controls."""
    state = {
        "selected_folder_path": None,
        "mode": "idle",
    }
    busy_guard = get_session_busy_guard(app_state)

    with ui.row().classes("w-full items-start gap-6").style("flex-wrap: nowrap;"):
        with ui.column().classes("patient-list-card").style("width: 380px; min-width: 340px; flex-shrink: 0;"):
            ui.label("📋 患者列表").classes("section-title")

            search_input = ui.input(
                label="🔍 搜尋 (ID / 姓名)",
                placeholder="輸入關鍵字...",
            ).classes("w-full")

            patient_table = ui.table(
                columns=[
                    {"name": "id", "label": "ID", "field": "id", "align": "left", "sortable": True},
                    {"name": "name", "label": "姓名", "field": "name", "align": "left", "sortable": True},
                    {"name": "birthday", "label": "生日", "field": "birthday", "align": "left", "sortable": True},
                    {"name": "gender", "label": "性別", "field": "gender", "align": "center"},
                ],
                rows=[],
                selection="single",
                row_key="folder_path",
            ).classes("w-full").style("max-height: 480px;")
            patient_table.props("dense flat bordered")

            with ui.row().classes("w-full gap-2 q-mt-sm"):
                btn_new = ui.button("➕ 新增患者", color="green").classes("flex-1")
                btn_edit = ui.button("✏️ 編輯患者", color="blue").classes("flex-1")
                btn_delete = ui.button("🗑️ 刪除患者", color="red").classes("flex-1")

            btn_refresh = ui.button("🔄 重新掃描", color="grey").classes("w-full q-mt-xs")
            status_label = ui.label("").style("color: var(--text-secondary); font-size: 13px; margin-top: 4px;")

        with ui.column().classes("patient-form-card flex-1").style("min-width: 450px;"):
            form_title = ui.label("📝 患者基本資料").classes("section-title")

            with ui.column().classes("w-full gap-3"):
                with ui.row().classes("w-full gap-4"):
                    inp_id = ui.input(label="患者 ID *", placeholder="例如 A123456789").classes("flex-1")
                    inp_name = ui.input(label="姓名 *", placeholder="例如 張三").classes("flex-1")
                with ui.row().classes("w-full gap-4"):
                    inp_gender = ui.select(label="性別", options=["男", "女", "其他"], value="男").classes("flex-1")
                    inp_birthday = ui.input(label="生日 * (YYYY-MM-DD)", placeholder="例如 1990-01-01").classes("flex-1")
                with ui.row().classes("w-full gap-4"):
                    inp_phone = ui.input(label="電話", placeholder="例如 0912345678").classes("flex-1")
                    inp_address = ui.input(label="住址", placeholder="例如 台北市...").classes("flex-1")
                inp_remark = ui.textarea(
                    label="備註",
                    placeholder="特殊備註 (過敏史、特殊飲食習慣、身高體重等...)",
                ).classes("w-full").style("min-height: 80px;")

            with ui.row().classes("w-full gap-3 q-mt-md"):
                btn_save = ui.button("💾 儲存", color="green").classes("flex-1").style("font-size: 15px;")
                btn_cancel = ui.button("❌ 取消", color="grey").classes("flex-1").style("font-size: 15px;")

            form_status = ui.label("").style("font-size: 14px; margin-top: 8px;")

    def refresh_table(keyword: str = ""):
        patients = list_patients()
        kw = keyword.strip().lower()
        rows = []
        for p in patients:
            bi = p["basic_info"]
            if kw and kw not in bi.get("id", "").lower() and kw not in bi.get("name", "").lower():
                continue
            rows.append({
                "id": bi.get("id", ""),
                "name": bi.get("name", ""),
                "birthday": bi.get("birthday", ""),
                "gender": bi.get("gender", ""),
                "folder_path": p["folder_path"],
            })
        patient_table.rows = rows
        patient_table.selected = []
        status_label.text = f"共 {len(rows)} 位患者"

    def clear_form():
        inp_id.value = ""
        inp_name.value = ""
        inp_gender.value = "男"
        inp_birthday.value = ""
        inp_phone.value = ""
        inp_address.value = ""
        inp_remark.value = ""
        form_status.text = ""

    def fill_form(bi: dict):
        inp_id.value = bi.get("id", "")
        inp_name.value = bi.get("name", "")
        inp_gender.value = bi.get("gender", "男")
        inp_birthday.value = bi.get("birthday", "")
        inp_phone.value = bi.get("phone", "")
        inp_address.value = bi.get("address", "")
        inp_remark.value = bi.get("remark", "")

    def set_form_editable(editable: bool):
        for inp in [inp_id, inp_name, inp_gender, inp_birthday, inp_phone, inp_address, inp_remark]:
            inp.enable() if editable else inp.disable()

    def enter_mode(mode: str):
        state["mode"] = mode
        if mode == "idle":
            clear_form()
            set_form_editable(False)
            form_title.text = "📝 患者基本資料"
            btn_save.disable()
            btn_cancel.disable()
        elif mode == "create":
            clear_form()
            set_form_editable(True)
            form_title.text = "📝 新增患者"
            form_status.text = "請填寫患者資料後按「💾 儲存」"
            btn_save.enable()
            btn_cancel.enable()
        elif mode == "edit":
            set_form_editable(True)
            form_title.text = "✏️ 編輯患者資料"
            form_status.text = "修改後按「💾 儲存」以更新"
            btn_save.enable()
            btn_cancel.enable()

    def on_search_change(e):
        refresh_table(e.value if e.value else "")

    def on_refresh():
        refresh_table(search_input.value or "")
        form_status.text = "🔄 列表已重新載入"

    def on_select_patient():
        selected = patient_table.selected
        if not selected:
            state["selected_folder_path"] = None
            enter_mode("idle")
            return
        row = selected[0]
        folder_path = row.get("folder_path", "")
        state["selected_folder_path"] = folder_path
        info = load_patient(folder_path)
        if info:
            fill_form(info.get("basic_info", {}))
            set_form_editable(False)
            form_title.text = f"📋 {row.get('name', '')} 的基本資料"
            form_status.text = "已載入患者資料（點擊「✏️ 編輯患者」可修改）"
            btn_save.disable()
            btn_cancel.disable()
            state["mode"] = "idle"

    def on_new():
        if busy_guard.reject_if_busy(status_label=form_status):
            return
        state["selected_folder_path"] = None
        patient_table.selected = []
        enter_mode("create")

    def on_edit():
        if busy_guard.reject_if_busy(status_label=form_status):
            return
        if not state["selected_folder_path"]:
            form_status.text = "⚠️ 請先從左側列表選擇一位患者"
            return
        enter_mode("edit")

    def on_delete():
        if busy_guard.reject_if_busy(status_label=form_status):
            return
        selected = patient_table.selected
        if not selected:
            form_status.text = "⚠️ 請先從左側列表選擇要刪除的患者"
            return
        folder_path = selected[0].get("folder_path", "")
        patient_name = selected[0].get("name", "未知")
        with ui.dialog() as dlg, ui.card().style("min-width: 320px;"):
            ui.label(f"確定要刪除患者「{patient_name}」所有資料嗎？").style("font-size: 16px; font-weight: bold;")
            ui.label("此操作無法復原！").style("color: red; font-size: 14px;")
            with ui.row().classes("w-full justify-end gap-2 q-mt-md"):
                ui.button("取消", color="grey", on_click=dlg.close).props("flat")

                def do_del():
                    if busy_guard.reject_if_busy(status_label=form_status):
                        dlg.close()
                        return
                    msg = delete_patient(folder_path)
                    form_status.text = msg
                    state["selected_folder_path"] = None
                    enter_mode("idle")
                    refresh_table(search_input.value or "")
                    if app_state.get("selected_patient_folder") == folder_path:
                        reset_patient_selection()
                        clear_fn = app_state.get("_clear_patient_ui")
                        if clear_fn:
                            clear_fn()
                    refresh_fn = app_state.get("_refresh_patient_list")
                    if refresh_fn:
                        refresh_fn()
                    dlg.close()

                ui.button("確認刪除", color="red", on_click=do_del)
        dlg.open()

    def on_save():
        if busy_guard.reject_if_busy(status_label=form_status):
            return
        if state["mode"] == "create":
            result = create_patient(
                patient_id=inp_id.value or "",
                name=inp_name.value or "",
                gender=inp_gender.value or "男",
                birthday=inp_birthday.value or "",
                phone=inp_phone.value or "",
                address=inp_address.value or "",
                remark=inp_remark.value or "",
            )
            if result.startswith("❌"):
                form_status.text = result
                form_status.style("color: var(--danger);")
            else:
                form_status.text = f"✅ 患者已建立：{os.path.basename(result)}"
                form_status.style("color: var(--primary);")
                state["selected_folder_path"] = result
                refresh_table(search_input.value or "")
                enter_mode("idle")
                info = load_patient(result)
                if info:
                    fill_form(info.get("basic_info", {}))
                    set_form_editable(False)
                refresh_fn = app_state.get("_refresh_patient_list")
                if refresh_fn:
                    refresh_fn()
        elif state["mode"] == "edit":
            folder_path = state["selected_folder_path"]
            if not folder_path:
                form_status.text = "❌ 內部錯誤：未選中任何患者"
                return
            msg, new_path = update_patient_basic_info(
                folder_path,
                new_id=inp_id.value or "",
                new_name=inp_name.value or "",
                new_gender=inp_gender.value or "男",
                new_birthday=inp_birthday.value or "",
                new_phone=inp_phone.value or "",
                new_address=inp_address.value or "",
                new_remark=inp_remark.value or "",
            )
            state["selected_folder_path"] = new_path
            if app_state.get("selected_patient_folder") == folder_path:
                app_state["selected_patient_folder"] = new_path
                app_state["selected_patient_info"] = load_patient(new_path)
            if msg.startswith("❌"):
                form_status.text = msg
                form_status.style("color: var(--danger);")
            else:
                form_status.text = msg
                form_status.style("color: var(--primary);")
                refresh_table(search_input.value or "")
                enter_mode("idle")
                info = load_patient(new_path)
                if info:
                    fill_form(info.get("basic_info", {}))
                    set_form_editable(False)
                refresh_fn = app_state.get("_refresh_patient_list")
                if refresh_fn:
                    refresh_fn()
                update_fn = app_state.get("_update_patient_display")
                if update_fn:
                    update_fn()

    def on_cancel():
        if state["selected_folder_path"]:
            info = load_patient(state["selected_folder_path"])
            if info:
                fill_form(info.get("basic_info", {}))
        enter_mode("idle")
        form_status.text = "已取消"

    search_input.on("update:model-value", on_search_change)
    btn_refresh.on_click(on_refresh)
    patient_table.on("selection", lambda _: on_select_patient())
    btn_new.on_click(on_new)
    btn_edit.on_click(on_edit)
    btn_delete.on_click(on_delete)
    btn_save.on_click(on_save)
    btn_cancel.on_click(on_cancel)

    refresh_table()
    enter_mode("idle")
