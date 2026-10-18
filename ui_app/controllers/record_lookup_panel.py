"""The 查閱病歷 window: the other visits of the patient that is open, read only, as the record's browse mode shows them.

A floating window like the voice window: no backdrop (the page behind it stays usable), dragged by its header, resized at its
bottom-right corner, put back by a double click on the header. Built once, outside the tab panels (a fixed-position element inside
a tab panel can be displaced by the panel's transitions), and opened by the 查閱病歷 button of the main page's left column.

It lists the visits of the selected patient, newest first, labelled like the visit selector (`2026-10-16（頭痛…）`); the visit that is
open is shown in light colour and cannot be opened here (it is on the page itself). A visit is shown as its saved final version,
NOTE and A&T, exactly as the browse mode shows a record: source tags hidden, low-confidence marks in bold with their reason. What it
shows is for the physician only; no agent receives it, and no lock (an AI turn, a recording) keeps it from opening. When the patient
or the visit that is open changes, the list follows; a record of another patient, or of the visit that is now open, is cleared.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

from ui_app.controllers.patient_session_lifecycle_controller import PatientSessionLifecycleController
from ui_app.rendering import simple_md_render, strip_citations

APP_STATE_KEY = "_record_lookup"
EMPTY = '<span style="color:#aaa;">（空白）</span>'


def browse_html(text: str) -> str:
    """A record as the browse mode of the main page shows it."""
    plain = strip_citations(text) if text else ""
    return simple_md_render(plain) if plain else EMPTY


class RecordLookupPanel:
    def __init__(
        self,
        *,
        ui: Any,
        app_state: dict[str, Any],
        list_sessions: Callable[[str], list[str]],
        get_session_summaries: Callable[[str, str], dict],
        load_session_content: Callable[[str, str], dict],
    ):
        self.ui = ui
        self.app_state = app_state
        self.list_sessions = list_sessions
        self.get_session_summaries = get_session_summaries
        self.load_session_content = load_session_content
        self.w = SimpleNamespace()
        self.button = None
        self.opened = False
        self.shown: tuple[str, str] | None = None             # (patient folder, date) on display
        self.listed_for: tuple[str | None, str | None] = (None, None)
        self.date_buttons: dict[str, Any] = {}
        app_state[APP_STATE_KEY] = self

    # ------------------------------------------------------------------ building
    def build(self) -> None:
        ui, w = self.ui, self.w
        with ui.element("div").classes("float-panel") as w.panel:
            # on the left half, beside the left column: the voice and reference windows open on the right
            w.panel.props("data-default-w=520 data-default-h=640 data-default-left=320")
            with ui.element("div").classes("float-head").tooltip("拖曳標題列移動；連按兩下回到預設位置；右下角可調整大小"):
                ui.label("查閱病歷").classes("float-title")
                ui.element("div").style("flex:1 1 auto")
                ui.button(icon="close", on_click=self.close).props("dense flat round").tooltip("關閉").mark("lookup-close")
            with ui.element("div").classes("float-body"):
                ui.label("唯讀；顯示各次就診存檔的最終版本，呈現與一般瀏覽相同（來源標籤隱藏、低信心標註照樣顯示）。"
                         "目前開啟的就診日期淡色、不能點。").classes("hint")
                w.dates = ui.column().classes("lookup-dates")
                w.heading = ui.label("").classes("lookup-heading")
                ui.label("📋 NOTE").classes("ref-label")
                w.note_html = ui.html("", sanitize=False).classes("lookup-text")
                ui.label("💊 ASSESSMENT & TREATMENT").classes("ref-label")
                w.at_html = ui.html("", sanitize=False).classes("lookup-text")
        w.panel.set_visibility(False)
        self.clear_record()
        ui.timer(1.0, self.follow_selection)

    def bind_button(self, button) -> None:
        """The 查閱病歷 button of the main page's left column."""
        self.button = button
        button.on_click(self.toggle)

    # ------------------------------------------------------------------ open / close
    def selection(self) -> tuple[str | None, str | None]:
        return self.app_state.get("selected_patient_folder"), self.app_state.get("selected_session_date")

    def toggle(self) -> None:
        if self.opened:
            self.close()
            return
        if not self.selection()[0]:
            self.ui.notify("請先在左欄選取並確認患者。", type="warning")
            return
        self.opened = True
        self.refresh()
        self.w.panel.set_visibility(True)
        self.ui.run_javascript(f"window.tcmFloat.show('c{self.w.panel.id}')")

    def close(self) -> None:
        self.opened = False
        self.w.panel.set_visibility(False)

    # ------------------------------------------------------------------ the list of visits
    def follow_selection(self) -> None:
        """Every second: the patient or the visit that is open has changed -> the list (and what is shown) follows."""
        if self.opened and self.selection() != self.listed_for:
            self.refresh()

    def refresh(self) -> None:
        folder, current = self.selection()
        self.listed_for = (folder, current)
        if self.shown is not None and (self.shown[0] != folder or self.shown[1] == current):
            self.clear_record()                          # another patient's record, or the visit that is open now
        self.w.dates.clear()
        self.date_buttons = {}
        with self.w.dates:
            if not folder:
                self.ui.label("請先在左欄選取並確認患者。").classes("hint")
                return
            dates = sorted(self.list_sessions(folder), reverse=True)
            if not dates:
                self.ui.label("這位患者還沒有任何就診紀錄。").classes("hint")
                return
            for date in dates:
                summary = (self.get_session_summaries(folder, date) or {}).get("note_summary", "")
                label = PatientSessionLifecycleController._session_option_label(date, summary)
                is_open = date == current
                button = self.ui.button(label + ("　目前開啟" if is_open else ""), on_click=lambda d=date: self.show_visit(d)) \
                    .props("flat dense no-caps align=left").classes("lookup-date")
                if is_open:
                    button.disable()
                    button.classes("lookup-open")
                elif self.shown == (folder, date):
                    button.classes("lookup-chosen")
                self.date_buttons[date] = button

    # ------------------------------------------------------------------ one visit
    def show_visit(self, date: str) -> None:
        folder, current = self.selection()
        if not folder or date == current:
            return
        content = self.load_session_content(folder, date) or {}
        self.shown = (folder, date)
        self.w.heading.set_text(f"🗓 {date} 的病歷")
        self.w.note_html.set_content(browse_html(content.get("note", "")))
        self.w.at_html.set_content(browse_html(content.get("at", "")))
        for other, button in self.date_buttons.items():
            if other != current:
                button.classes(remove="lookup-chosen")
                if other == date:
                    button.classes("lookup-chosen")

    def clear_record(self) -> None:
        self.shown = None
        self.w.heading.set_text("點選上方的日期查看那一天的病歷。")
        self.w.note_html.set_content("")
        self.w.at_html.set_content("")
