from __future__ import annotations

from typing import Any, Callable

from record_edit_tags import MIN_SURVIVING_RUN, NoteEditError, editable_text, merge_human_edit
from ui_app.controllers.record_reference_panel import APP_STATE_KEY as REFERENCE_KEY
from ui_app.rendering import generate_diff_html, simple_md_render, strip_citations

_NOTHING_REMOVED = "儲存後不會移除任何來源標籤。（來源標籤與低信心標註已隱藏，可對照「參考版本」視窗。）"


def _clip(text: str, limit: int = 18) -> str:
    """The last `limit` characters of the words a tag covered (where the sentence ends is what the physician recognises)."""
    return text if len(text) <= limit else "…" + text[-limit:]


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
        record_snapshot_store: Any,
        write_session_log: Callable[[str, str, str], None],
        busy_guard: Any | None = None,
        load_config: Callable[[], dict] | None = None,
    ):
        self.app_state = app_state
        self.ui_state = ui_state
        self.history = history
        self.refs = refs
        self.save_session_content = save_session_content
        self.record_snapshot_store = record_snapshot_store
        self.write_session_log = write_session_log
        self.busy_guard = busy_guard
        self.load_config = load_config
        # The version the edit began on and the text the edit boxes were filled with: the boxes show no source tags nor
        # low-confidence marks, and the save puts them back by comparing against this (record_edit_tags.merge_human_edit).
        self._edit_base: dict[str, Any] | None = None
        self._preview_dirty = False
        for key in ("note_editor", "at_editor"):
            editor = refs.get(key)
            if editor is not None:
                editor.on_value_change(self.mark_preview_dirty)

    def _min_run(self) -> int:
        """`note_edit.min_surviving_run` of the settings (1..20), MIN_SURVIVING_RUN when absent or invalid."""
        try:
            value = ((self.load_config() if self.load_config else {}) or {}).get("note_edit", {}).get("min_surviving_run")
            value = int(value)
        except (TypeError, ValueError, AttributeError):
            return MIN_SURVIVING_RUN
        return value if 1 <= value <= 20 else MIN_SURVIVING_RUN

    def _main_turns_so_far(self) -> int:
        """How many Main Agent turns this session has had (0 when there is no agent yet: no turn has run)."""
        agent = self.app_state.get("agent_instance")
        try:
            return max(0, int(getattr(agent, "turn_count", 0) or 0))
        except (TypeError, ValueError):
            return 0

    def _recording_position(self) -> tuple[float | None, bool]:
        """(seconds recorded so far in this session, paused) -- (None, False) when the session has no voice recording."""
        manager = self.app_state.get("_voice_manager")
        if manager is None:
            return None, False
        return manager.recording_position(
            self.app_state.get("selected_patient_folder"), self.app_state.get("selected_session_date")
        )

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
        btn_edit_cancel = self.refs.get("btn_edit_cancel")
        btn_undo = self.refs["btn_undo"]
        btn_redo = self.refs["btn_redo"]

        if mode == "edit":
            btn_browse.disable()
            btn_diff.disable()
            btn_edit_mode.disable()
            btn_undo.disable()
            btn_redo.disable()
            btn_edit_done.enable()
            if btn_edit_cancel is not None:
                btn_edit_cancel.enable()
        else:
            btn_browse.enable()
            btn_diff.enable()
            btn_edit_mode.enable()
            btn_edit_done.disable()
            if btn_edit_cancel is not None:
                btn_edit_cancel.disable()
            btn_undo.enable() if self.history.can_undo() else btn_undo.disable()
            btn_redo.enable() if self.history.can_redo() else btn_redo.disable()
            self._edit_base = None
        reference = self.app_state.get(REFERENCE_KEY)
        if reference is not None:
            reference.sync(mode == "edit")

    def save_current_to_disk(self):
        fp = self.app_state["selected_patient_folder"]
        dt = self.app_state["selected_session_date"]
        current = self.history.get_current()
        if fp and dt and current:
            self.save_session_content(fp, dt, current["note"], current["at"])
            self.record_snapshot_store.save_history(fp, dt, self.history)
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
        self._edit_base = {"id": current.get("id"), "note": current["note"], "at": current["at"]}
        self.refs["note_editor"].value = editable_text(current["note"], marks=True)
        self.refs["at_editor"].value = editable_text(current["at"])
        self.refs["browse_container"].set_visibility(False)
        self.refs["edit_container"].set_visibility(True)
        reference = self.app_state.get(REFERENCE_KEY)
        if reference is not None:
            reference.begin(
                label=f"參考：版本 {self.history.current_index + 1}（修改前）",
                note=current["note"],
                at=current["at"],
            )
        self.refresh_preview(force=True)
        self.update_buttons()
        self._status("✏️ 修改模式 — 來源標籤與低信心標註已隱藏（可對照「參考版本」視窗），編輯完成後按「✅ 修改完成」；不想保留按「↩️ 放棄修改」")

    def _status(self, text: str, *, warn: bool = False):
        label = self.refs["record_status"]
        label.text = text
        label.style("color: #b3261e;" if warn else "color: inherit;")

    def mark_preview_dirty(self, _event=None):
        self._preview_dirty = True

    def _merge_edit(self, min_run: int):
        """(NOTE merge, A&T merge) of what the edit boxes hold now against the version the edit began on. NoteEditError: refused."""
        base = self._edit_base or {}
        note = merge_human_edit(base.get("note", ""), self.refs["note_editor"].value or "", min_run, marks=True)
        at = merge_human_edit(base.get("at", ""), self.refs["at_editor"].value or "", min_run)
        return note, at

    def refresh_preview(self, force: bool = False):
        """Under each edit box: which source tags (and low-confidence marks) the save would take away, because the words they
        covered are gone. Recomputed from a timer after the boxes changed, not on every keystroke."""
        if self.ui_state["view_mode"] != "edit" or self._edit_base is None:
            return
        if not (self._preview_dirty or force):
            return
        self._preview_dirty = False
        min_run = self._min_run()
        base = self._edit_base
        for field, editor_key, preview_key, marks in (
            ("note", "note_editor", "note_preview", True),
            ("at", "at_editor", "at_preview", False),
        ):
            preview = self.refs.get(preview_key)
            if preview is None:
                continue
            try:
                merge = merge_human_edit(base.get(field, ""), self.refs[editor_key].value or "", min_run, marks=marks)
            except NoteEditError as exc:
                preview.set_text(str(exc))
                preview.classes(replace="edit-preview warn")
                continue
            lines = []
            if merge.removed:
                shown = [f"{r.tag}（{_clip(r.text)}）" if r.text else r.tag for r in merge.removed[:5]]
                more = f" 等共 {len(merge.removed)} 個" if len(merge.removed) > 5 else ""
                lines.append("儲存後將移除來源標籤（它們涵蓋的字都已不在）：" + "、".join(shown) + more)
            if merge.dropped_marks:
                shown = [f"「{_clip(d.phrase)}」" for d in merge.dropped_marks[:5]]
                more = f" 等共 {len(merge.dropped_marks)} 個" if len(merge.dropped_marks) > 5 else ""
                lines.append("儲存後將移除低信心標註（該片段已被修改）：" + "、".join(shown) + more)
            preview.set_text("；".join(lines) if lines else _NOTHING_REMOVED)
            preview.classes(replace="edit-preview" + (" warn" if lines else ""))

    def _leave_edit_mode(self):
        self.ui_state["view_mode"] = "browse"
        self._edit_base = None
        if self.busy_guard:
            self.busy_guard.sync_navigation()
        self.refs["browse_container"].set_visibility(True)
        self.refs["edit_container"].set_visibility(False)
        self.update_display()
        self.update_buttons()

    def on_edit_cancel(self):
        if self.ui_state["view_mode"] != "edit":
            return
        self._leave_edit_mode()
        self._status("↩️ 已放棄修改，病歷維持原狀")

    def on_edit_done(self):
        current = self.history.get_current()
        if current is None:
            return
        base = self._edit_base
        if base is None:
            self._leave_edit_mode()
            return
        if (base.get("id"), base.get("note"), base.get("at")) != (current.get("id"), current["note"], current["at"]):
            # the record moved on while the boxes were open (another window, an undo): saving would overwrite the newer version
            self._status(
                "⚠️ 病歷在你修改的期間已被更新（另一個視窗、作業或復原），為避免蓋掉較新的版本，這次修改不能儲存。請按「↩️ 放棄修改」後重新進入修改模式。",
                warn=True,
            )
            return
        try:
            note_merge, at_merge = self._merge_edit(self._min_run())
        except NoteEditError as exc:               # a source tag typed or pasted into a box
            self._status(f"⚠️ {exc}", warn=True)
            self.refresh_preview(force=True)
            return
        new_note, new_at = note_merge.text, at_merge.text

        if new_note != current["note"] or new_at != current["at"]:
            before_index = self.history.current_index
            position = self._recording_position()
            rec_t, rec_paused = position[0], position[1]
            rec_hidden, rec_seen = getattr(position, "hidden", None), getattr(position, "last", None)
            removed = [{"field": "NOTE", "tag": r.tag, "text": r.text} for r in note_merge.removed] + [
                {"field": "A&T", "tag": r.tag, "text": r.text} for r in at_merge.removed
            ]
            dropped = [{"phrase": d.phrase, "reason": d.reason} for d in note_merge.dropped_marks]
            meta = {}
            if removed:
                meta["removed_tags"] = removed
            if dropped:
                meta["dropped_marks"] = dropped
            truncated = self.history.push(
                new_note, new_at, source="人類修改", rec_t=rec_t, rec_paused=rec_paused, rec_hidden=rec_hidden, rec_seen=rec_seen, meta=meta or None,
                after_turn=self._main_turns_so_far(),
            )
            if truncated:
                fp = self.app_state["selected_patient_folder"]
                dt = self.app_state["selected_session_date"]
                self.record_snapshot_store.append_audit_event(
                    fp,
                    dt,
                    event_type="truncate_redo",
                    source="人類修改",
                    current_index_before=before_index,
                    current_index_after=self.history.current_index,
                    snapshots=truncated,
                    note="Manual edit created a new linear version and replaced redo snapshots.",
                )
            self.save_current_to_disk()
            fp = self.app_state["selected_patient_folder"]
            dt = self.app_state["selected_session_date"]
            if fp and dt:
                detail = ""
                if removed:
                    detail += f"，移除來源標籤 {len(removed)} 個：" + "、".join(item["tag"] for item in removed)
                if dropped:
                    detail += f"，移除低信心標註 {len(dropped)} 個"
                self.write_session_log(fp, dt, f"[HUMAN_EDIT] 人類修改病歷 (版本 {self.history.current_index + 1}){detail}")
            extra = (f"；已移除 {len(removed)} 個來源標籤" if removed else "") + (
                f"、{len(dropped)} 個低信心標註" if dropped else ""
            )
            status = f"✅ 修改已保存{extra}"
        else:
            status = "ℹ️ 未偵測到變更"

        self._leave_edit_mode()
        self._status(status)

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
