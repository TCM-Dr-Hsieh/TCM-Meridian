from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any

from ui_app.controllers.session_busy_guard import get_session_busy_guard


def build_image_tab(ui: Any, app_state: dict[str, Any], today: str):
    """Render image import, list, delete, and preview controls."""
    img_state: dict = {
        "file_list": [],
        "staged": [],
    }
    busy_guard = get_session_busy_guard(app_state)

    ui.label("🖼️ 影像檔查詢區").style("font-size: 20px; font-weight: bold; margin-bottom: 4px;")

    with ui.card().classes("w-full").style("padding: 16px; margin-bottom: 12px;"):
        ui.label("📥 匯入圖片").style("font-size: 16px; font-weight: bold; margin-bottom: 8px;")

        with ui.row().classes("w-full items-end gap-3"):
            date_input = ui.input(
                "日期 (YYYY-MM-DD)",
                value=today,
            ).classes("w-48").props("outlined dense")
            suffix_input = ui.input(
                "檔名後綴 (XXX)",
                placeholder="例：舌象、脈象、皮疹",
            ).classes("w-48").props("outlined dense")

        import_status = ui.label("").style("margin-top: 8px; color: #666;")

        upload_area = ui.upload(
            label="點此選擇圖片，或拖曳檔案至此",
            multiple=True,
            auto_upload=True,
        ).classes("w-full").style("margin-top: 8px;").props('accept="image/*"')

        staged_label = ui.label("").style("margin-top: 4px; color: #1565C0; font-size: 13px;")

        async def on_upload(e):
            fp = app_state.get("selected_patient_folder")
            if not fp:
                import_status.text = "❌ 請先在「醫療系統主介面」選取患者"
                import_status.style("color: #e53935;")
                upload_area.reset()
                return
            data = await e.file.read()
            img_state["staged"].append({
                "orig_name": e.file.name,
                "data_bytes": data,
                "patient_folder": fp,
            })
            staged_label.text = f"📎 已選取 {len(img_state['staged'])} 個檔案，請按「💾 儲存」寫入磁碟"

        upload_area.on_upload(on_upload)

        with ui.row().classes("gap-2 mt-2"):
            async def on_save():
                if busy_guard.reject_if_busy(status_label=import_status):
                    return
                fp = app_state.get("selected_patient_folder")
                if not fp:
                    import_status.text = "❌ 請先在「醫療系統主介面」選取患者"
                    import_status.style("color: #e53935;")
                    return

                if not img_state["staged"]:
                    import_status.text = "❌ 尚未選取任何圖片"
                    import_status.style("color: #e53935;")
                    return

                if any(item.get("patient_folder") != fp for item in img_state["staged"]):
                    clear_staged()
                    import_status.text = "❌ 患者已切換，已清除上一位患者的暫存圖片，請重新上傳"
                    import_status.style("color: #e53935;")
                    return

                date_val = date_input.value.strip()
                if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_val):
                    import_status.text = "❌ 日期格式錯誤，請使用 YYYY-MM-DD"
                    import_status.style("color: #e53935;")
                    return
                try:
                    datetime.strptime(date_val, "%Y-%m-%d")
                except ValueError:
                    import_status.text = "❌ 無效日期，請檢查年月日是否合理"
                    import_status.style("color: #e53935;")
                    return

                suffix_val = suffix_input.value.strip()
                if not suffix_val:
                    import_status.text = "❌ 請填寫檔名後綴 (XXX)"
                    import_status.style("color: #e53935;")
                    return

                pic_dir = os.path.join(fp, "Picture_Row")
                os.makedirs(pic_dir, exist_ok=True)

                saved_names = []
                for item in img_state["staged"]:
                    _, ext = os.path.splitext(item["orig_name"])
                    if not ext:
                        ext = ".png"

                    if len(img_state["staged"]) == 1:
                        target_name = f"{date_val}-{suffix_val}{ext}"
                    else:
                        target_name = f"{date_val}-{suffix_val}_{len(saved_names)+1}{ext}"

                    target_path = os.path.join(pic_dir, target_name)

                    counter = 1
                    while os.path.exists(target_path):
                        target_name = f"{date_val}-{suffix_val}_{len(saved_names)+counter}{ext}"
                        target_path = os.path.join(pic_dir, target_name)
                        counter += 1

                    with open(target_path, "wb") as f:
                        f.write(item["data_bytes"])
                    saved_names.append(target_name)

                clear_staged()

                import_status.text = f"✅ 已儲存 {len(saved_names)} 張：{', '.join(saved_names)}"
                import_status.style("color: #2e7d32;")
                refresh_file_list()

            ui.button("💾 儲存", on_click=on_save).props("color=primary")

            def on_clear_staged():
                clear_staged()
                import_status.text = "已清除暫存"
                import_status.style("color: #666;")

            ui.button("🗑️ 清除暫存", on_click=on_clear_staged).props("flat")

    with ui.row().classes("w-full gap-3").style("flex-wrap: nowrap; align-items: flex-start;"):
        with ui.column().style("width: 420px; min-width: 350px; flex-shrink: 0;"):
            with ui.row().classes("items-center gap-2"):
                ui.label("📂 已匯入的圖片").style("font-size: 16px; font-weight: bold;")
                ui.button("🔄 重新掃描", on_click=lambda: refresh_file_list()).props("flat dense")

            file_list_container = ui.column().classes("w-full gap-1").style("max-height: 60vh; overflow-y: auto; padding: 4px;")
            file_count_label = ui.label("").style("color: #888; font-size: 13px; margin-top: 4px;")

        with ui.column().classes("flex-grow"):
            ui.label("👁️ 圖片預覽").style("font-size: 16px; font-weight: bold; margin-bottom: 8px;")
            preview_container = ui.column().classes("w-full gap-3").style(
                "max-height: 70vh; overflow-y: auto; padding: 4px;"
            )
            with preview_container:
                ui.label("請在左側勾選圖片").style(
                    "color: #aaa; font-size: 15px; margin: 30px auto;"
                )

    def clear_staged():
        img_state["staged"].clear()
        staged_label.text = ""
        upload_area.reset()

    def refresh_file_list(*, clear_staged_files: bool = False):
        if clear_staged_files:
            clear_staged()
        fp = app_state.get("selected_patient_folder")
        file_list_container.clear()
        img_state["file_list"] = []

        if not fp:
            with file_list_container:
                ui.label("尚未選取患者").style("color: #aaa;")
            file_count_label.text = ""
            preview_container.clear()
            return

        pic_dir = os.path.join(fp, "Picture_Row")
        if not os.path.isdir(pic_dir):
            with file_list_container:
                ui.label("Picture_Row 資料夾不存在").style("color: #aaa;")
            file_count_label.text = ""
            preview_container.clear()
            return

        valid_exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
        files = []
        for f in sorted(os.listdir(pic_dir)):
            _, ext = os.path.splitext(f)
            if ext.lower() in valid_exts:
                files.append({"name": f, "path": os.path.join(pic_dir, f), "selected": False})

        img_state["file_list"] = files
        file_count_label.text = f"共 {len(files)} 張圖片"

        if not files:
            with file_list_container:
                ui.label("此患者尚無圖片").style("color: #aaa;")
            preview_container.clear()
            return

        with file_list_container:
            for idx, fi in enumerate(files):
                build_file_row(idx, fi)

    def build_file_row(idx: int, fi: dict):
        with ui.row().classes("items-center gap-2 w-full").style(
            "padding: 4px 8px; border-radius: 6px; background: #fafafa; "
            "border: 1px solid #eee;"
        ):
            cb = ui.checkbox(fi["name"], value=fi["selected"]).style("flex-grow: 1;")

            def on_toggle(e, idx_inner=idx):
                img_state["file_list"][idx_inner]["selected"] = e.value
                render_previews()

            cb.on_value_change(on_toggle)

            async def on_delete(idx_inner=idx):
                if busy_guard.reject_if_busy(status_label=import_status):
                    return
                fi_del = img_state["file_list"][idx_inner]
                try:
                    os.remove(fi_del["path"])
                except Exception:
                    pass
                refresh_file_list()
                render_previews()

            ui.button(icon="delete_outline", on_click=on_delete).props(
                "flat dense round color=negative size=sm"
            ).tooltip("刪除此圖片")

    def render_previews():
        preview_container.clear()
        selected = [fi for fi in img_state["file_list"] if fi["selected"]]

        if not selected:
            with preview_container:
                ui.label("請在左側勾選圖片").style(
                    "color: #aaa; font-size: 15px; margin: 30px auto;"
                )
            return

        with preview_container:
            for fi in selected:
                with ui.card().classes("w-full").style("padding: 8px;"):
                    ui.label(fi["name"]).style(
                        "font-weight: bold; font-size: 13px; color: #555; margin-bottom: 4px;"
                    )
                    ui.image(fi["path"]).style(
                        "max-width: 100%; max-height: 500px; object-fit: contain; "
                        "border: 1px solid #ddd; border-radius: 4px;"
                    )

    refresh_file_list()
    app_state["_refresh_image_list"] = refresh_file_list
