"""The read-only 參考版本 window of the record edit.

The edit box shows the NOTE and the ASSESSMENT & TREATMENT without their source tags and low-confidence marks (they cannot be
damaged and do not get in the way); this window shows the version being edited as it is, with the tags in light colour and
the marks in bold, so the physician sees which sentence came from which source. A floating window like the voice window: no
backdrop (the record behind it stays usable), dragged and resized in the browser. It is built once, outside the tab panels
(a fixed-position element inside a tab panel can be displaced by the panel's transitions), and opened and closed by
`MedicalRecordController` as the edit begins and ends.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ui_app.rendering import reference_html

APP_STATE_KEY = "_record_reference"


class RecordReferencePanel:
    def __init__(self, *, ui: Any, app_state: dict[str, Any]):
        self.ui = ui
        self.app_state = app_state
        self.w = SimpleNamespace()
        self.opened = False
        self.dismissed = False          # the physician closed it during this edit: it does not open again until the next edit
        self.label = ""
        self.note = ""
        self.at = ""
        app_state[APP_STATE_KEY] = self

    def build(self) -> None:
        ui, w = self.ui, self.w
        with ui.element("div").classes("float-panel") as w.panel:
            # beside the voice window (560 px wide, 24 px from the right edge), not on top of it
            w.panel.props("data-default-w=480 data-default-h=560 data-default-right=596")
            with ui.element("div").classes("float-head").tooltip("拖曳標題列移動；連按兩下回到預設位置；右下角可調整大小"):
                w.title = ui.label("參考版本").classes("float-title")
                ui.element("div").style("flex:1 1 auto")
                ui.button(icon="close", on_click=self.dismiss).props("dense flat round") \
                    .tooltip("關閉（這次修改不再自動開啟）").mark("reference-close")
            with ui.element("div").classes("float-body"):
                ui.label("唯讀；淡色的是來源標籤，粗體是低信心標註。修改框裡看不到它們，儲存時由程式保留或移除。").classes("hint")
                ui.label("📋 NOTE").classes("ref-label")
                w.note_html = ui.html("", sanitize=False).classes("reference-text")
                ui.label("💊 ASSESSMENT & TREATMENT").classes("ref-label")
                w.at_html = ui.html("", sanitize=False).classes("reference-text")
        w.panel.set_visibility(False)

    # ------------------------------------------------------------------ edit lifecycle
    def begin(self, *, label: str, note: str, at: str) -> None:
        """An edit begins on this version of the record."""
        self.label, self.note, self.at = label, note, at
        self.dismissed = False
        self.sync(True)

    def sync(self, editing: bool) -> None:
        """Open the window while a record edit is going on, put it away when it is over (whichever way it ends). Closed by
        the physician: it stays closed for the rest of that edit."""
        if not editing:
            self.dismissed = False
            if self.opened:
                self._set_open(False)
        elif not self.opened and not self.dismissed:
            self._set_open(True)

    def dismiss(self) -> None:
        self.dismissed = True
        self._set_open(False)

    def _set_open(self, opened: bool) -> None:
        w = self.w
        self.opened = opened
        w.panel.set_visibility(opened)
        if opened:
            w.title.set_text(self.label or "參考版本")
            w.note_html.set_content(reference_html(self.note, marks=True))
            w.at_html.set_content(reference_html(self.at))
            self.ui.run_javascript(f"window.tcmFloat.show('c{w.panel.id}')")
