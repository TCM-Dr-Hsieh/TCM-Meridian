from __future__ import annotations

from typing import Any, Callable

from ui_app.rendering import generate_diff_html, simple_md_render, strip_citations, tag_human_edits


class MedicalRecordController:
    """Own browse/diff/edit/undo/redo behavior for the NOTE and A&T panel."""

    def __init__(
        self,
        *,
        app_state: dict[str, Any],
        ui_state: dict[str, Any],
        history: Any,
        refs: dict[str, Any],
        save_session_content: Callable[[str, str, str, str], None],
        write_session_log: Callable[[str, str, str], None],
        busy_guard: Any | None = None,
    ):
        self.app_state = app_state
        self.ui_state = ui_state
        self.history = history
        self.refs = refs
        self.save_session_content = save_session_content
        self.write_session_log = write_session_log
        self.busy_guard = busy_guard

    def update_display(self):
        current = self.history.get_current()
        note_display = self.refs["note_display"]
        at_display = self.refs["at_display"]
        if current is None:
            note_display.content = '<span style="color:#aaa;">請選取患者與就診日期</span>'
            at_display.content = ""
            return

        note_text = current["note"]
        at_text = current["at"]

        if self.ui_state["view_mode"] == "browse":
            note_browse = strip_citations(note_text) if note_text else ""
            at_browse = strip_citations(at_text) if at_text else ""
            note_display.content = (
                simple_md_render(note_browse) if note_browse else '<span style="color:#aaa;">（空白）</span>'
            )
            at_display.content = simple_md_render(at_browse) if at_browse else '<span style="color:#aaa;">（空白）</span>'

        elif self.ui_state["view_mode"] == "diff":
            prev = self.history.get_previous()
            if prev:
                note_display.content = generate_diff_html(prev["note"], note_text, "NOTE 差異")
                at_display.content = generate_diff_html(prev["at"], at_text, "A&T 差異")
            else:
                note_display.content = '<span style="color:#aaa;">（無前一版可比較）</span>'
                at_display.content = ""

        total = len(self.history.snapshots)
        idx = self.history.current_index + 1
        self.refs["version_label"].text = (
            f"版本 {idx}/{total}  |  來源: {current.get('source', '?')}  |  {current.get('timestamp', '')[:19]}"
        )

    def update_buttons(self):
        mode = self.ui_state["view_mode"]
        btn_browse = self.refs["btn_browse"]
        btn_diff = self.refs["btn_diff"]
        btn_edit_mode = self.refs["btn_edit_mode"]
        btn_edit_done = self.refs["btn_edit_done"]
        btn_undo = self.refs["btn_undo"]
        btn_redo = self.refs["btn_redo"]

        if mode == "edit":
            btn_browse.disable()
            btn_diff.disable()
            btn_edit_mode.disable()
            btn_undo.disable()
            btn_redo.disable()
            btn_edit_done.enable()
        else:
            btn_browse.enable()
            btn_diff.enable()
            btn_edit_mode.enable()
            btn_edit_done.disable()
            btn_undo.enable() if self.history.can_undo() else btn_undo.disable()
            btn_redo.enable() if self.history.can_redo() else btn_redo.disable()

    def save_current_to_disk(self):
        fp = self.app_state["selected_patient_folder"]
        dt = self.app_state["selected_session_date"]
        current = self.history.get_current()
        if fp and dt and current:
            self.save_session_content(fp, dt, current["note"], current["at"])
            self.write_session_log(fp, dt, f"[SAVE] 版本 {self.history.current_index + 1} 已存檔")

    def on_browse(self):
        self.ui_state["view_mode"] = "browse"
        if self.busy_guard:
            self.busy_guard.sync_navigation()
        self.refs["browse_container"].set_visibility(True)
        self.refs["edit_container"].set_visibility(False)
        self.update_display()
        self.update_buttons()
        self.refs["record_status"].text = "📖 一般瀏覽模式"

    def on_diff(self):
        self.ui_state["view_mode"] = "diff"
        if self.busy_guard:
            self.busy_guard.sync_navigation()
        self.refs["browse_container"].set_visibility(True)
        self.refs["edit_container"].set_visibility(False)
        self.update_display()
        self.update_buttons()
        self.refs["record_status"].text = "🔍 差異瀏覽模式"

    def on_edit_mode(self):
        if self.busy_guard and self.busy_guard.reject_if_busy(
            status_label=self.refs.get("record_status"),
            block_edit=False,
        ):
            return
        current = self.history.get_current()
        if current is None:
            self.refs["record_status"].text = "⚠️ 無內容可編輯"
            return

        self.ui_state["view_mode"] = "edit"
        if self.busy_guard:
            self.busy_guard.sync_navigation()
        self.refs["note_editor"].value = current["note"]
        self.refs["at_editor"].value = current["at"]
        self.refs["browse_container"].set_visibility(False)
        self.refs["edit_container"].set_visibility(True)
        self.update_buttons()
        self.refs["record_status"].text = "✏️ 修改模式 — 編輯完成後按「✅ 修改完成」"

    def on_edit_done(self):
        current = self.history.get_current()
        if current is None:
            return

        new_note = self.refs["note_editor"].value or ""
        new_at = self.refs["at_editor"].value or ""
        new_note = tag_human_edits(current["note"], new_note)
        new_at = tag_human_edits(current["at"], new_at)

        if new_note != current["note"] or new_at != current["at"]:
            self.history.push(new_note, new_at, source="人類修改")
            self.save_current_to_disk()
            fp = self.app_state["selected_patient_folder"]
            dt = self.app_state["selected_session_date"]
            if fp and dt:
                self.write_session_log(fp, dt, f"[HUMAN_EDIT] 人類修改病歷 (版本 {self.history.current_index + 1})")
            self.refs["record_status"].text = "✅ 修改已保存"
        else:
            self.refs["record_status"].text = "ℹ️ 未偵測到變更"

        self.ui_state["view_mode"] = "browse"
        if self.busy_guard:
            self.busy_guard.sync_navigation()
        self.refs["browse_container"].set_visibility(True)
        self.refs["edit_container"].set_visibility(False)
        self.update_display()
        self.update_buttons()

    def on_undo(self):
        if self.busy_guard and self.busy_guard.reject_if_busy(
            status_label=self.refs.get("record_status"),
            block_edit=False,
        ):
            return
        snap = self.history.undo()
        if snap:
            self.update_display()
            self.update_buttons()
            self.save_current_to_disk()
            self.refs["record_status"].text = f"⬅️ 回到版本 {self.history.current_index + 1}"

    def on_redo(self):
        if self.busy_guard and self.busy_guard.reject_if_busy(
            status_label=self.refs.get("record_status"),
            block_edit=False,
        ):
            return
        snap = self.history.redo()
        if snap:
            self.update_display()
            self.update_buttons()
            self.save_current_to_disk()
            self.refs["record_status"].text = f"➡️ 前進到版本 {self.history.current_index + 1}"
