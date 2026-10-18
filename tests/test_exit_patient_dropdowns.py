"""Leaving a patient empties the date dropdown in the browser, not only on the server: the browser had gone on listing the visits of
the patient that was left. A date with no patient open is not loaded either."""
from tests.test_reload_restore import FOLDER_A, make_controller


class BrowserSelect:
    """A dropdown as NiceGUI keeps it: assigning `options` changes the server's copy only; the browser gets the options on `update()`
    (which `set_options` calls)."""

    def __init__(self, options, value=None):
        self.options, self.value = options, value
        self.update()

    def update(self):
        self.in_browser = list(self.options)

    def set_options(self, options, *, value=...):
        self.options = options
        if value is not ...:
            self.value = value
        self.update()


def test_leaving_a_patient_empties_the_dropdowns_in_the_browser(tmp_path):
    controller, state, refs, _ = make_controller(tmp_path)
    refs["session_select"], refs["template_select"] = BrowserSelect([]), BrowserSelect(["（空白病歷）"], "（空白病歷）")
    refs["patient_select"].value = FOLDER_A
    controller.on_confirm_patient()
    assert refs["session_select"].in_browser == ["2026-10-16", "2026-10-17"]
    assert refs["template_select"].in_browser == ["（空白病歷）", "2026-10-16", "2026-10-17"]

    controller.on_exit_patient()
    assert refs["session_select"].in_browser == [] and refs["session_select"].value is None
    assert refs["template_select"].in_browser == ["（空白病歷）"] and refs["template_select"].value == "（空白病歷）"


def test_a_date_is_not_loaded_while_no_patient_is_open(tmp_path):
    controller, state, refs, _ = make_controller(tmp_path)
    refs["session_select"].value = "2026-10-16"                     # a date left in a page that has no patient
    controller.on_confirm_session()
    assert state["selected_session_date"] is None and refs["session_status"].text == "⚠️ 請先選取患者"
