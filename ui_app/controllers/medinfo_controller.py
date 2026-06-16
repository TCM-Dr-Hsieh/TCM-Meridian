from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any

from ui_app.controllers.session_busy_guard import get_session_busy_guard


def build_medinfo_tab(ui: Any, app_state: dict[str, Any], today: str):
    """Render medical text-file create, preview, edit, save, and delete controls."""
    mi_state: dict = {
        "file_list": [],
        "selected_idx": None,
        "editing": False,
    }
    busy_guard = get_session_busy_guard(app_state)

    ui.label("📋 醫療資訊檔案存放區").style("font-size: 20px; font-weight: bold; margin-bottom: 4px;")

    with ui.card().classes("w-full").style("padding: 16px; margin-bottom: 12px;"):
        ui.label("📄 新增檔案").style("font-size: 16px; font-weight: bold; margin-bottom: 8px;")

        with ui.row().classes("w-full items-end gap-3"):
            mi_date_input = ui.input(
                "日期 (YYYY-MM-DD)",
                value=today,
            ).classes("w-48").props("outlined dense")
            mi_suffix_input = ui.input(
                "檔名後綴 (XXX)",
                placeholder="例：血液檢查、CT報告、入院摘要",
            ).classes("w-64").props("outlined dense")

            def on_create():
                if busy_guard.reject_if_busy(status_label=mi_status):
                    return
                fp = app_state.get("selected_patient_folder")
                if not fp:
                    mi_status.text = "❌ 請先在「醫療系統主介面」選取患者"
                    mi_status.style("color: #e53935;")
                    return

                date_val = mi_date_input.value.strip()
                if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_val):
                    mi_status.text = "❌ 日期格式錯誤，請使用 YYYY-MM-DD"
                    mi_status.style("color: #e53935;")
                    return
                try:
                    datetime.strptime(date_val, "%Y-%m-%d")
                except ValueError:
                    mi_status.text = "❌ 無效日期，請檢查年月日是否合理"
                    mi_status.style("color: #e53935;")
                    return

                suffix_val = mi_suffix_input.value.strip()
                if not suffix_val:
                    mi_status.text = "❌ 請填寫檔名後綴 (XXX)"
                    mi_status.style("color: #e53935;")
                    return

                med_dir = os.path.join(fp, "Medical_information")
                os.makedirs(med_dir, exist_ok=True)

                target_name = f"{date_val}-{suffix_val}.txt"
                target_path = os.path.join(med_dir, target_name)

                if os.path.exists(target_path):
                    counter = 1
                    while True:
                        target_name = f"{date_val}-{suffix_val}_{counter}.txt"
                        target_path = os.path.join(med_dir, target_name)
                        if not os.path.exists(target_path):
                            break
                        counter += 1

                with open(target_path, "w", encoding="utf-8") as f:
                    f.write("")

                mi_status.text = f"✅ 已建立：{target_name}"
                mi_status.style("color: #2e7d32;")
                refresh_mi_list()

                for i, fi in enumerate(mi_state["file_list"]):
                    if fi["path"] == target_path:
                        select_file(i)
                        break

            ui.button("➕ 新增", on_click=on_create).props("color=primary")

        mi_status = ui.label("").style("margin-top: 6px; color: #666;")

    with ui.row().classes("w-full gap-3").style("flex-wrap: nowrap; align-items: flex-start;"):
        with ui.column().style("width: 320px; min-width: 280px; flex-shrink: 0;"):
            with ui.row().classes("items-center gap-2"):
                ui.label("📂 檔案列表").style("font-size: 16px; font-weight: bold;")
                ui.button("🔄", on_click=lambda: refresh_mi_list()).props("flat dense round size=sm").tooltip("重新掃描")

            mi_list_container = ui.column().classes("w-full gap-1").style(
                "max-height: 65vh; overflow-y: auto; padding: 4px;"
            )
            mi_count_label = ui.label("").style("color: #888; font-size: 13px; margin-top: 4px;")

        with ui.column().classes("flex-grow").style("min-width: 400px;"):
            mi_editor_title = ui.label("請在左側選取檔案").style(
                "font-size: 16px; font-weight: bold; margin-bottom: 8px; color: #555;"
            )

            mi_editor = ui.textarea("").classes("w-full").props("outlined").style(
                "min-height: 55vh; font-family: 'Consolas', 'Courier New', monospace; font-size: 14px;"
            )
            mi_editor.set_enabled(False)

            with ui.row().classes("gap-2 mt-2"):
                ui.button("✏️ 編輯", on_click=lambda: enter_edit())
                ui.button("💾 儲存", on_click=lambda: save_file()).props("color=primary")
                ui.button("❌ 取消", on_click=lambda: cancel_edit()).props("flat")
                ui.button("🗑️ 刪除檔案", on_click=lambda: delete_file()).props("flat color=negative")

            mi_edit_status = ui.label("").style("margin-top: 4px; color: #666; font-size: 13px;")

    def refresh_mi_list():
        fp = app_state.get("selected_patient_folder")
        mi_list_container.clear()
        mi_state["file_list"] = []
        mi_state["selected_idx"] = None

        if not fp:
            with mi_list_container:
                ui.label("尚未選取患者").style("color: #aaa;")
            mi_count_label.text = ""
            clear_editor()
            return

        med_dir = os.path.join(fp, "Medical_information")
        if not os.path.isdir(med_dir):
            with mi_list_container:
                ui.label("Medical_information 資料夾不存在").style("color: #aaa;")
            mi_count_label.text = ""
            clear_editor()
            return

        files = []
        for f in sorted(os.listdir(med_dir)):
            full = os.path.join(med_dir, f)
            if os.path.isfile(full):
                files.append({"name": f, "path": full})

        mi_state["file_list"] = files
        mi_count_label.text = f"共 {len(files)} 個檔案"

        if not files:
            with mi_list_container:
                ui.label("此患者尚無醫療資訊檔案").style("color: #aaa;")
            clear_editor()
            return

        with mi_list_container:
            for idx, fi in enumerate(files):
                build_mi_file_btn(idx, fi)

        clear_editor()

    def build_mi_file_btn(idx: int, fi: dict):
        ui.button(
            fi["name"],
            on_click=lambda idx_inner=idx: select_file(idx_inner),
        ).classes("w-full").props("flat align=left no-caps").style(
            "justify-content: flex-start; text-transform: none; font-size: 13px; "
            "padding: 6px 12px; border-radius: 6px;"
        )

    def select_file(idx: int):
        if mi_state["editing"]:
            mi_edit_status.text = "⚠️ 請先儲存或取消目前的編輯"
            mi_edit_status.style("color: #e65100;")
            return

        mi_state["selected_idx"] = idx
        fi = mi_state["file_list"][idx]

        try:
            with open(fi["path"], "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            content = f"（讀取失敗：{e}）"

        mi_editor_title.text = f"📄 {fi['name']}"
        mi_editor_title.style("color: #1565C0;")
        mi_editor.value = content
        mi_editor.set_enabled(False)
        mi_state["editing"] = False
        mi_edit_status.text = ""

    def clear_editor():
        mi_editor_title.text = "請在左側選取檔案"
        mi_editor_title.style("color: #555;")
        mi_editor.value = ""
        mi_editor.set_enabled(False)
        mi_state["editing"] = False
        mi_edit_status.text = ""

    def enter_edit():
        if busy_guard.reject_if_busy(status_label=mi_edit_status):
            return
        if mi_state["selected_idx"] is None:
            mi_edit_status.text = "⚠️ 請先選取檔案"
            mi_edit_status.style("color: #e65100;")
            return
        mi_editor.set_enabled(True)
        mi_state["editing"] = True
        mi_edit_status.text = "✏️ 編輯模式 — 修改後請按「💾 儲存」"
        mi_edit_status.style("color: #1565C0;")

    def save_file():
        if busy_guard.reject_if_busy(status_label=mi_edit_status):
            return
        idx = mi_state["selected_idx"]
        if idx is None:
            mi_edit_status.text = "⚠️ 請先選取檔案"
            mi_edit_status.style("color: #e65100;")
            return

        fi = mi_state["file_list"][idx]
        try:
            with open(fi["path"], "w", encoding="utf-8") as f:
                f.write(mi_editor.value or "")
            mi_edit_status.text = f"✅ 已儲存：{fi['name']}"
            mi_edit_status.style("color: #2e7d32;")
        except Exception as e:
            mi_edit_status.text = f"❌ 儲存失敗：{e}"
            mi_edit_status.style("color: #e53935;")

        mi_editor.set_enabled(False)
        mi_state["editing"] = False

    def cancel_edit():
        if mi_state["selected_idx"] is not None:
            select_file(mi_state["selected_idx"])
        else:
            clear_editor()
        mi_edit_status.text = "已取消編輯"
        mi_edit_status.style("color: #666;")

    def delete_file():
        if busy_guard.reject_if_busy(status_label=mi_edit_status):
            return
        idx = mi_state["selected_idx"]
        if idx is None:
            mi_edit_status.text = "⚠️ 請先選取檔案"
            mi_edit_status.style("color: #e65100;")
            return

        fi = mi_state["file_list"][idx]
        try:
            os.remove(fi["path"])
            mi_edit_status.text = f"🗑️ 已刪除：{fi['name']}"
            mi_edit_status.style("color: #e53935;")
        except Exception as e:
            mi_edit_status.text = f"❌ 刪除失敗：{e}"
            mi_edit_status.style("color: #e53935;")

        mi_state["selected_idx"] = None
        clear_editor()
        refresh_mi_list()

    refresh_mi_list()
    app_state["_refresh_medinfo_list"] = refresh_mi_list
