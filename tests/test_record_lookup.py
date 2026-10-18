"""The 查閱病歷 window: the other visits of the open patient, read only, as the browse mode shows them. No browser: a fake NiceGUI that
keeps the element tree, so the list of visits and what is shown can be looked at."""
from ui_app.controllers.record_lookup_panel import APP_STATE_KEY, EMPTY, RecordLookupPanel, browse_html

FOLDER_A, FOLDER_B = "patient_data/A1", "patient_data/B2"
RECORDS = {
    (FOLDER_A, "2026-10-01"): {"note": "甲- 現病史：頭痛三天[問診紀錄_患者R1]，**夜間加重（只有家屬轉述）**[問診紀錄_家屬R2]", "at": "A: 外感風熱[人類醫師_T2]"},
    (FOLDER_A, "2026-10-08"): {"note": "甲- 現病史：頭痛改善[語音#3]", "at": ""},
    (FOLDER_A, "2026-10-16"): {"note": "甲- 現病史：今天的病歷", "at": "今天的處置"},
    (FOLDER_B, "2026-09-30"): {"note": "乙- 過去病史：氣喘[上次病歷]", "at": ""},
}


class Element:
    def __init__(self, ui, kind, text="", on_click=None):
        self.ui, self.kind, self.text, self.content = ui, kind, text, ""
        self.children, self.props_, self.classes_ = [], "", set()
        self.enabled, self.visible, self.on_click_ = True, True, on_click
        self.id = len(ui.created)
        ui.created.append(self)

    def __enter__(self):
        self.ui.stack.append(self)
        return self

    def __exit__(self, *exc):
        self.ui.stack.pop()

    def props(self, value):
        self.props_ += " " + value
        return self

    def classes(self, add="", *, remove=None):
        self.classes_ |= set(add.split())
        self.classes_ -= set((remove or "").split())
        return self

    def style(self, value):
        return self

    def tooltip(self, text):
        return self

    def mark(self, name):
        return self

    def set_visibility(self, visible):
        self.visible = visible

    def disable(self):
        self.enabled = False

    def set_text(self, text):
        self.text = text

    def set_content(self, content):
        self.content = content

    def clear(self):
        self.children = []

    def click(self):
        assert self.enabled, f"{self.text} cannot be clicked"
        self.on_click_()


class FakeUI:
    def __init__(self):
        self.created, self.stack, self.notices, self.scripts, self.timers = [], [], [], [], []

    def _add(self, element):
        if self.stack:
            self.stack[-1].children.append(element)
        return element

    def element(self, tag):
        return self._add(Element(self, tag))

    def column(self):
        return self._add(Element(self, "column"))

    def label(self, text=""):
        return self._add(Element(self, "label", text))

    def html(self, content="", sanitize=True):
        element = self._add(Element(self, "html"))
        element.content = content
        return element

    def button(self, text="", *, icon=None, on_click=None):
        return self._add(Element(self, "button", text, on_click))

    def timer(self, interval, callback):
        self.timers.append(callback)

    def notify(self, message, type=None):
        self.notices.append((message, type))

    def run_javascript(self, code):
        self.scripts.append(code)


class Button:
    def __init__(self):
        self.handlers = []

    def on_click(self, handler):
        self.handlers.append(handler)

    def press(self):
        for handler in self.handlers:
            handler()


def make(folder=FOLDER_A, current="2026-10-16"):
    ui = FakeUI()
    state = {"selected_patient_folder": folder, "selected_session_date": current}
    loads = []

    def list_sessions(f):
        return sorted({date for (owner, date) in RECORDS if owner == f})            # oldest first, as the store gives them

    def load_session_content(f, date):
        loads.append((f, date))
        return RECORDS.get((f, date), {"note": "", "at": ""})

    panel = RecordLookupPanel(ui=ui, app_state=state, list_sessions=list_sessions,
                              get_session_summaries=lambda f, date: {"note_summary": RECORDS[(f, date)]["note"].split("：", 1)[1][:4]},
                              load_session_content=load_session_content)
    panel.build()
    button = Button()
    panel.bind_button(button)
    return panel, ui, state, button, loads


def dates(panel):
    return [(child.text, child.enabled) for child in panel.w.dates.children if child.kind == "button"]


def test_the_button_opens_the_window_with_the_patients_other_visits_newest_first_and_the_open_one_greyed(tmp_path=None):
    panel, ui, state, button, _ = make()
    assert state[APP_STATE_KEY] is panel and panel.w.panel.visible is False         # built hidden, outside the tabs
    assert "data-default-left=320" in panel.w.panel.props_                          # on the left half: the other windows open on the right
    button.press()
    assert panel.opened and panel.w.panel.visible and ui.scripts[-1] == f"window.tcmFloat.show('c{panel.w.panel.id}')"
    assert dates(panel) == [("2026-10-16（今天的病）　目前開啟", False), ("2026-10-08（頭痛改善）", True), ("2026-10-01（頭痛三天）", True)]
    assert "lookup-open" in panel.w.dates.children[0].classes_
    assert panel.w.heading.text == "點選上方的日期查看那一天的病歷。" and panel.w.note_html.content == ""
    button.press()                                                                 # the button closes it again
    assert not panel.opened and panel.w.panel.visible is False


def test_a_visit_is_shown_as_the_browse_mode_shows_it_and_the_open_one_cannot_be_shown(tmp_path=None):
    panel, ui, state, button, loads = make()
    button.press()
    panel.w.dates.children[2].click()                                              # 2026-10-01
    assert panel.w.heading.text == "🗓 2026-10-01 的病歷"
    assert panel.w.note_html.content == browse_html(RECORDS[(FOLDER_A, "2026-10-01")]["note"])
    assert "[問診紀錄" not in panel.w.note_html.content and "<b>夜間加重（只有家屬轉述）</b>" in panel.w.note_html.content
    assert panel.w.at_html.content == browse_html("A: 外感風熱[人類醫師_T2]") and "人類醫師" not in panel.w.at_html.content
    assert "lookup-chosen" in panel.w.dates.children[2].classes_
    panel.w.dates.children[1].click()                                              # another visit: the choice moves, an empty A&T says so
    assert "lookup-chosen" in panel.w.dates.children[1].classes_ and "lookup-chosen" not in panel.w.dates.children[2].classes_
    assert panel.w.at_html.content == EMPTY
    panel.show_visit("2026-10-16")                                                 # the open visit is on the page itself
    assert panel.shown == (FOLDER_A, "2026-10-08") and loads == [(FOLDER_A, "2026-10-01"), (FOLDER_A, "2026-10-08")]


def test_without_a_patient_the_button_asks_for_one_and_a_patient_without_a_visit_open_has_every_date(tmp_path=None):
    panel, ui, state, button, _ = make(folder=None, current=None)
    button.press()
    assert not panel.opened and ui.notices == [("請先在左欄選取並確認患者。", "warning")]
    panel, ui, state, button, _ = make(current=None)                               # a patient, no visit chosen yet
    button.press()
    assert all(enabled for _, enabled in dates(panel)) and len(dates(panel)) == 3


def test_the_window_follows_the_patient_and_the_visit_that_are_open(tmp_path=None):
    panel, ui, state, button, _ = make()
    button.press()
    panel.w.dates.children[1].click()                                              # showing 2026-10-08
    follow = ui.timers[0]
    state["selected_session_date"] = "2026-10-08"                                  # the physician opens that visit on the page
    follow()
    assert panel.shown is None and panel.w.note_html.content == ""                 # it is on the page now: cleared here
    assert dict(dates(panel))["2026-10-08（頭痛改善）　目前開啟"] is False and dict(dates(panel))["2026-10-16（今天的病）"] is True
    panel.w.dates.children[2].click()                                              # 2026-10-01
    state["selected_patient_folder"], state["selected_session_date"] = FOLDER_B, None
    follow()
    assert panel.shown is None and dates(panel) == [("2026-09-30（氣喘[上）", True)]     # another patient: nothing of the first one stays
    state["selected_patient_folder"] = None
    follow()
    assert dates(panel) == [] and panel.w.dates.children[0].text == "請先在左欄選取並確認患者。"


def test_a_closed_window_does_not_rebuild_its_list_until_it_is_opened(tmp_path=None):
    panel, ui, state, button, _ = make()
    follow = ui.timers[0]
    state["selected_session_date"] = "2026-10-08"
    follow()
    assert panel.w.dates.children == []                                            # closed: nothing is done every second
    button.press()
    assert dict(dates(panel))["2026-10-08（頭痛改善）　目前開啟"] is False           # opened: made for the selection of now
