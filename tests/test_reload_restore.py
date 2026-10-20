"""A page that is opened again (browser refresh, a connection lost for good) comes back to the patient and visit that were open; the
page asks before it is left while that would lose work; the connection is kept for a minute. No browser: the widgets are fakes."""
import json
import logging
from pathlib import Path
from types import SimpleNamespace

from ui_app.context import AppContext
from ui_app.controllers.patient_session_lifecycle_controller import PatientSessionLifecycleController
from ui_app.controllers.session_busy_guard import SessionBusyGuard
from ui_app.leave_warning import JS, leaving_would_lose_work, register_leave_warning
from ui_app.log_filters import DROPPED_PAGE_MESSAGE, DroppedPageFilter, install_dropped_page_filter, is_timer_of_a_deleted_page
from ui_app.services.last_selection import LastSelectionStore
from ui_app.services.snapshot_history import SnapshotHistory

ROOT = Path(__file__).resolve().parent.parent
FOLDER_A, FOLDER_B = "patient_data/A1_1980-01-01_王", "patient_data/B2_1990-02-02_李"


# ------------------------------------------------------------------------------------------------ the memory file
def test_the_store_keeps_one_selection_and_forgets_it_on_request(tmp_path):
    store = LastSelectionStore(tmp_path / "last_session.json")
    assert store.load() == (None, None)                                      # nothing stored yet
    store.save(FOLDER_A, "2026-10-16")
    assert store.load() == (FOLDER_A, "2026-10-16") and not list(tmp_path.glob("*.tmp"))
    store.save(FOLDER_A, None)                                               # a patient without a visit open
    assert store.load() == (FOLDER_A, None)
    store.save(None, "2026-10-16")                                           # no patient: nothing to come back to
    assert store.load() == (None, None) and not (tmp_path / "last_session.json").exists()
    store.save(None)                                                         # forgetting what is not there is fine


def test_a_damaged_or_odd_file_means_nothing_is_stored_and_a_failed_write_is_silent(tmp_path):
    path = tmp_path / "last_session.json"
    for text in ("not json", "[1, 2]", '{"folder": 5, "date": "x"}', '{"folder": "", "date": "x"}', '{"date": "x"}'):
        path.write_text(text, encoding="utf-8")
        assert LastSelectionStore(path).load() == (None, None), text
    path.write_text(json.dumps({"folder": FOLDER_A, "date": 7}), encoding="utf-8")
    assert LastSelectionStore(path).load() == (FOLDER_A, None)               # a date that is not text is dropped, the patient stays
    LastSelectionStore(tmp_path / "missing" / "last_session.json").save(FOLDER_A, "2026-10-16")      # the folder is not there: no error


# ------------------------------------------------------------------------------------------------ the controller
class Loose:
    """A widget: `value`, `options` and the like are plain attributes; every method is a harmless no-op."""

    def __init__(self, value=None, options=None):
        self.value, self.options, self.text, self.content = value, options if options is not None else [], "", ""

    def set_options(self, options, *, value=...):                    # as NiceGUI's select: the options, and the value if given
        self.options = options
        if value is not ...:
            self.value = value

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class Snapshots:
    def restore_or_initialize(self, **kwargs):
        return {}

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class SpyStore(LastSelectionStore):
    def __init__(self, path):
        super().__init__(path)
        self.saves = []

    def save(self, folder, date=None):
        self.saves.append((folder, date))
        super().save(folder, date)


def make_controller(tmp_path, *, voice=None, remembered=None, patients=(FOLDER_A, FOLDER_B), fail_load_patient=False, fail_load_session=False):
    context = AppContext()
    app_state = context.state
    ui_state = {"view_mode": "browse", "agent_running": False, "chat_messages": []}
    if voice is not None:
        app_state["_voice_manager"] = voice
    guard = SessionBusyGuard(app_state=app_state, ui_state=ui_state)
    store = SpyStore(tmp_path / "last_session.json")
    if remembered is not None:
        LastSelectionStore(store.path).save(*remembered)
    names = {FOLDER_A: "王", FOLDER_B: "李"}

    def load_patient(folder):
        if fail_load_patient:
            raise RuntimeError("the patient file is unreadable")
        return {"basic_info": {"id": "x", "name": names[folder], "birthday": "1980-01-01", "remark": ""}}

    def load_session_content(folder, date):
        if fail_load_session:
            raise OSError("the visit file is unreadable")
        return {"note": "n", "at": "a"}

    refs = {key: Loose() for key in (
        "template_select", "session_status", "new_session_panel", "new_session_date", "btn_confirm_new", "patient_info_label",
        "remark_area", "remark_status", "browse_container", "edit_container", "note_display", "at_display", "note_editor", "at_editor",
        "record_status", "agent_status")}
    refs["patient_select"], refs["session_select"] = Loose(options={}), Loose(options=[])
    controller = PatientSessionLifecycleController(
        ui=Loose(), app_state=app_state, ui_state=ui_state, app_context=context, history=SnapshotHistory(), refs=refs, today="2026-10-17",
        list_patients=lambda: [{"basic_info": {"id": "x", "name": names[f], "birthday": "1980-01-01"}, "folder_path": f} for f in patients],
        list_sessions=lambda folder: ["2026-10-16", "2026-10-17"], get_session_summaries=lambda folder, date: {"note_summary": "n"},
        load_patient=load_patient, save_patient_info=lambda *a: "", load_session_content=load_session_content,
        save_session_summaries=lambda *a: "", create_session=lambda *a: "", delete_session=lambda *a: "", write_session_log=lambda *a: None,
        load_config=lambda: {}, record_snapshot_store=Snapshots(), generate_history_summary=lambda *a: "",
        get_session_restorer=lambda: Loose(), update_display=lambda: None, update_buttons=lambda: None, render_chat=lambda: None,
        busy_guard=guard, last_selection=store)
    controller.refresh_patient_list()
    return controller, app_state, refs, store


def test_a_fresh_page_returns_to_the_patient_and_visit_that_were_open(tmp_path):
    controller, state, refs, store = make_controller(tmp_path, remembered=(FOLDER_B, "2026-10-16"))
    assert state["selected_patient_folder"] is None                         # what a reloaded page starts with
    assert controller.restore_last_selection() is True
    assert (state["selected_patient_folder"], state["selected_session_date"]) == (FOLDER_B, "2026-10-16")
    assert refs["patient_select"].value == FOLDER_B and refs["session_select"].value == "2026-10-16"
    assert "已回到上次開啟的患者與就診日期" in refs["session_status"].text
    assert store.saves == [(FOLDER_B, "2026-10-16")]                         # remembered once, not overwritten half way
    assert store.load() == (FOLDER_B, "2026-10-16")


def test_only_what_can_be_restored_is_restored(tmp_path):
    controller, state, refs, _ = make_controller(tmp_path, remembered=(FOLDER_A, None))
    assert controller.restore_last_selection() and state["selected_session_date"] is None
    assert "已回到上次開啟的患者" in refs["session_status"].text and "就診日期" not in refs["session_status"].text
    controller, state, _, _ = make_controller(tmp_path, remembered=(FOLDER_A, "2020-01-01"))                    # the visit is gone
    assert controller.restore_last_selection() and (state["selected_patient_folder"], state["selected_session_date"]) == (FOLDER_A, None)
    controller, state, _, _ = make_controller(tmp_path, remembered=("patient_data/gone", "2026-10-16"))          # the patient is gone
    assert controller.restore_last_selection() is False and state["selected_patient_folder"] is None
    controller, state, _, _ = make_controller(tmp_path)                                                          # nothing was stored
    assert controller.restore_last_selection() is False and state["selected_patient_folder"] is None


def test_a_page_that_already_has_a_patient_is_left_alone(tmp_path):
    controller, state, _, _ = make_controller(tmp_path, remembered=(FOLDER_B, "2026-10-16"))
    state["selected_patient_folder"] = FOLDER_A
    assert controller.restore_last_selection() is False and state["selected_patient_folder"] == FOLDER_A


def test_a_restore_that_fails_leaves_an_empty_page_that_says_so_and_keeps_the_memory(tmp_path):
    for failing in ({"fail_load_patient": True}, {"fail_load_session": True}):       # the patient file, then the visit file, cannot be read
        controller, state, refs, store = make_controller(tmp_path, remembered=(FOLDER_A, "2026-10-16"), **failing)
        assert controller.restore_last_selection() is False
        assert (state["selected_patient_folder"], state["selected_patient_info"], state["selected_session_date"]) == (None, None, None)
        assert refs["patient_select"].value is None and refs["session_select"].value is None       # not a patient with a blank record
        assert controller.history.current_index == -1 and controller._restoring is False
        assert "自動還原上次開啟的看診失敗" in refs["session_status"].text and "請手動選取" in refs["session_status"].text
        assert "已載入" not in refs["session_status"].text                                          # no "✅ 已載入患者" left over
        assert store.load() == (FOLDER_A, "2026-10-16")                                           # kept: the next load tries again
        assert store.saves == []                                                                   # clearing the page did not touch it
    controller, state, refs, store = make_controller(tmp_path, remembered=(FOLDER_A, "2026-10-16"))     # the same page, once it can be read
    assert controller.restore_last_selection() is True and state["selected_session_date"] == "2026-10-16"


def test_a_running_recording_brings_its_own_visit_back_through_the_navigation_lock(tmp_path):
    voice = SimpleNamespace(busy=True, session=SimpleNamespace(folder=FOLDER_A, date="2026-10-17"))
    controller, state, refs, store = make_controller(tmp_path, voice=voice, remembered=(FOLDER_B, "2026-10-16"))
    refs["patient_select"].value = FOLDER_A
    controller.on_confirm_patient()                                          # the recording locks the navigation of a page without a visit
    assert state["selected_patient_folder"] is None
    assert controller.restore_last_selection() is True                       # coming back to the recording's own visit is not leaving
    assert (state["selected_patient_folder"], state["selected_session_date"]) == (FOLDER_A, "2026-10-17")
    assert store.load() == (FOLDER_A, "2026-10-17")                          # the memory follows the recording
    refs["patient_select"].value = FOLDER_B
    controller.on_confirm_patient()                                          # the lock still holds against switching away
    assert state["selected_patient_folder"] == FOLDER_A


def test_the_memory_follows_the_visit_that_is_open(tmp_path):
    controller, state, refs, store = make_controller(tmp_path)
    refs["patient_select"].value = FOLDER_A
    controller.on_confirm_patient()
    assert store.load() == (FOLDER_A, None)
    refs["session_select"].value = "2026-10-17"
    controller.on_confirm_session()
    assert store.load() == (FOLDER_A, "2026-10-17")
    controller._clear_current_session_ui()                                  # the visit is closed, the patient stays
    assert store.load() == (FOLDER_A, None)
    refs["session_select"].value = "2026-10-16"
    controller.on_confirm_session()
    controller.on_exit_patient()                                            # nothing open: a reload starts empty
    assert store.load() == (None, None)


# ------------------------------------------------------------------------------------------------ the leave warning
def guard_for(**ui):
    state = {}
    return state, SessionBusyGuard(app_state=state, ui_state={"view_mode": "browse", "agent_running": False, **ui})


def test_the_page_asks_before_it_is_left_only_while_something_would_be_lost():
    state, _ = guard_for()
    assert leaving_would_lose_work(state) is False
    assert leaving_would_lose_work({}) is False                              # a page whose guard does not exist yet
    state, _ = guard_for(agent_running=True)
    assert leaving_would_lose_work(state) is True                            # an AI turn
    state, _ = guard_for(view_mode="edit")
    assert leaving_would_lose_work(state) is True                            # a record edit not finished
    state, _ = guard_for()
    state["interview_active"] = True
    assert leaving_would_lose_work(state) is True                            # an interview
    state, _ = guard_for()
    state["_session_transition_active"] = True
    assert leaving_would_lose_work(state) is True                            # a visit being switched or saved
    state, _ = guard_for()
    state["_voice_manager"] = SimpleNamespace(busy=True)
    assert leaving_would_lose_work(state) is False                           # a recording of the computer's microphone goes on without the page


def test_the_page_script_also_asks_while_this_tab_is_capturing_the_remote_microphone_and_the_flag_follows_the_state():
    assert "beforeunload" in JS and "window.tcmRemote" in JS and "tcmLeaveBusy" in JS
    sent, timers, heads = [], [], []
    ui = SimpleNamespace(add_head_html=heads.append, run_javascript=sent.append, timer=lambda seconds, callback: timers.append(callback))
    state, _ = guard_for()
    register_leave_warning(ui, state)
    assert heads and "beforeunload" in heads[0] and len(timers) == 1
    sync = timers[0]
    sync()
    sync()
    assert sent == ["window.tcmLeaveBusy = false;"]                          # told once, not every second
    state["_session_busy_guard"].ui_state["agent_running"] = True
    sync()
    sync()
    state["_session_busy_guard"].ui_state["agent_running"] = False
    sync()
    assert sent == ["window.tcmLeaveBusy = false;", "window.tcmLeaveBusy = true;", "window.tcmLeaveBusy = false;"]


# ------------------------------------------------------------------------------------------------ the connection
def test_the_connection_is_kept_for_a_minute_so_a_short_outage_does_not_reload_the_page():
    source = (ROOT / "TCM_Meridian_main.py").read_text(encoding="utf-8")
    assert "reconnect_timeout=RECONNECT_SECONDS" in source and 'os.environ.get("TCM_RECONNECT_SECONDS", "60")' in source
    assert "last_selection=last_selection_store" in source and "register_leave_warning(ui, app_state)" in source
    assert "restore_selection=nicegui_app.is_started" in source                     # not in the run NiceGUI makes before any browser connects
    controller_source = (ROOT / "ui_app" / "controllers" / "medical_main_controller.py").read_text(encoding="utf-8")
    assert 'app_state["_restore_last_selection"] = patient_session_controller.restore_last_selection' in controller_source
    assert "patient_session_controller.restore_last_selection()" not in controller_source   # not while the main tab is half built
    # run once every tab exists: an interview that was going on is shown in its own tab, which is built after the main one
    build = source[source.index("def build_ui():"):source.index("def _build_medical_main_tab():")]
    assert build.index("build_tab_shell(ui, app_state, _tab_specs())") < build.index('restore = app_state.pop("_restore_last_selection", None)')
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "last_session.json" in ignored                                    # the memory file is the physician's, not the repository's


def shaped_errors(tmp_path):
    """The "parent slot deleted" error raised along different paths. Each function is compiled under the folder, file and name it has in
    NiceGUI (or in the program), so the traceback looks like the real one; nothing of NiceGUI is imported. `read(entry)` reads a finished
    task the way NiceGUI does and returns the exception."""
    sources = {
        "nicegui/element.py": f"def parent_slot():\n    raise RuntimeError({DROPPED_PAGE_MESSAGE!r})\ndef use_child():\n    return parent_slot()\n",
        "nicegui/elements/timer.py": "def _get_context():\n    return parent_slot()\n",
        "nicegui/timer.py": "".join(f"def {name}():\n    with _get_context():\n        pass\n" for name in ("_run_in_loop", "_run_once", "_other_function")),
        "nicegui/background_tasks.py": "def _handle_exceptions(call):\n    try:\n        call()\n    except Exception as error:\n        return error\n",
        "program/page.py": "def callback():\n    return use_child()\n",
    }
    spaces = {}
    for relative, source in sources.items():
        spaces[relative] = {}
        exec(compile(source, str(tmp_path / relative), "exec"), spaces[relative])
    spaces["nicegui/elements/timer.py"]["parent_slot"] = spaces["nicegui/element.py"]["parent_slot"]
    spaces["nicegui/timer.py"]["_get_context"] = spaces["nicegui/elements/timer.py"]["_get_context"]
    spaces["program/page.py"]["use_child"] = spaces["nicegui/element.py"]["use_child"]
    entries = {"loop": spaces["nicegui/timer.py"]["_run_in_loop"], "once": spaces["nicegui/timer.py"]["_run_once"],
               "other": spaces["nicegui/timer.py"]["_other_function"], "program": spaces["program/page.py"]["callback"]}
    return lambda entry: spaces["nicegui/background_tasks.py"]["_handle_exceptions"](entries[entry])


def test_the_filter_drops_only_a_timer_that_woke_after_its_page_was_deleted(tmp_path):
    read = shaped_errors(tmp_path)
    for entry in ("loop", "once"):                                           # how a timer reports it, whichever kind of timer it is
        assert is_timer_of_a_deleted_page(read(entry)) is True, entry
    # the same words from a real UI problem keep their traceback: an element used after its parent was deleted, from the program's own code
    assert is_timer_of_a_deleted_page(read("program")) is False
    # a NiceGUI path it was not checked against is not guessed at
    assert is_timer_of_a_deleted_page(read("other")) is False
    # an error that is only like it, or no error at all
    assert is_timer_of_a_deleted_page(RuntimeError(DROPPED_PAGE_MESSAGE)) is False                        # no NiceGUI frames at all
    assert is_timer_of_a_deleted_page(ValueError(DROPPED_PAGE_MESSAGE)) is False
    assert is_timer_of_a_deleted_page(None) is False


def test_the_filter_works_on_the_log_record_and_is_installed_once(tmp_path):
    class Collect(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records = []

        def emit(self, record):
            self.records.append(record)

    read = shaped_errors(tmp_path)
    logger = logging.getLogger("tcm-test-dropped-page")
    logger.propagate, collected = False, Collect()
    logger.addHandler(collected)
    install_dropped_page_filter(logger)
    install_dropped_page_filter(logger)                                      # the script runs again for every connection: one filter only
    assert [type(item) for item in logger.filters] == [DroppedPageFilter]
    for entry in ("loop", "program"):
        error = read(entry)
        logger.error(str(error), exc_info=error)       # as NiceGUI logs it: the traceback of the exception that was caught
    logger.error("no exception at all")
    assert len(collected.records) == 2 and collected.records[1].getMessage() == "no exception at all"     # the timer of a deleted page is gone ...
    assert str(collected.records[0].exc_info[1]) == DROPPED_PAGE_MESSAGE                                 # ... the same words from the program's own code are kept
    assert "program" in str(collected.records[0].exc_info[2].tb_next.tb_frame.f_code.co_filename)
    assert "install_dropped_page_filter()" in (ROOT / "TCM_Meridian_main.py").read_text(encoding="utf-8")


def test_the_main_page_says_so_whenever_it_remakes_the_visit_list(tmp_path):
    controller, state, refs, _ = make_controller(tmp_path)
    calls = []
    state["_on_visits_changed"] = lambda: calls.append(state.get("selected_patient_folder"))
    refs["patient_select"].value = FOLDER_A
    controller.on_confirm_patient()                                          # another patient: the list is made again
    controller.refresh_session_list()                                        # what adding, deleting and summarising a visit call
    state["selected_patient_folder"] = None
    controller.refresh_session_list()                                        # no patient: told as well
    assert calls == [FOLDER_A, FOLDER_A, None]

