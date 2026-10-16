"""The record edit as the physician meets it: the boxes without tags, the save that puts them back, the reference window, the
version history (recording clock, what an edit removed) and what the agents read of it. No browser: the widgets are fakes."""
from types import SimpleNamespace

import pytest

from record_diff_context import build_record_diff_context
from record_edit_tags import HUMAN_EDIT_TAG
from ui_app.controllers.medical_record_controller import MedicalRecordController
from ui_app.controllers.record_reference_panel import APP_STATE_KEY, RecordReferencePanel
from ui_app.controllers.session_busy_guard import SessionBusyGuard
from ui_app.rendering import reference_html
from ui_app.services.snapshot_history import SnapshotHistory


class Widget:
    """Enough of a NiceGUI element for the controller."""

    def __init__(self, value=""):
        self.value = value
        self.text = ""
        self.content = ""
        self.visible = True
        self.enabled = True
        self.classes_ = ""
        self.style_ = ""
        self.handlers = []

    def on_value_change(self, handler):
        self.handlers.append(handler)

    def set_visibility(self, visible):
        self.visible = visible

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def set_text(self, text):
        self.text = text

    def classes(self, add="", *, replace=None):
        self.classes_ = replace if replace is not None else self.classes_ + " " + add
        return self

    def style(self, text="", *, remove=None):
        # NiceGUI's Style.parse wants "key: value" declarations: a bare property name raises ValueError
        for declaration in (text, remove):
            for word in (declaration or "").split(";"):
                if word.strip() and ":" not in word:
                    raise ValueError("not enough values to unpack (expected 2, got 1)")
        self.style_ = text
        return self


class FakeUI:
    def __init__(self):
        self.scripts = []

    def run_javascript(self, code):
        self.scripts.append(code)


class FakePanel:
    """RecordReferencePanel without the browser: the same lifecycle, widgets replaced."""

    def __init__(self):
        self.panel = RecordReferencePanel(ui=FakeUI(), app_state={})
        self.panel.w = SimpleNamespace(
            panel=SimpleNamespace(set_visibility=lambda v: None, id=7),
            title=Widget(), note_html=SimpleNamespace(set_content=lambda c: setattr(self, "note_html", c)),
            at_html=SimpleNamespace(set_content=lambda c: setattr(self, "at_html", c)),
        )
        self.panel.w.title.set_text = lambda t: setattr(self, "title", t)
        self.note_html = self.at_html = self.title = None


NOTE = ("甲- 現病史：**患者有高血壓（家屬R2未提及）**[問診紀錄_患者R1]。咳嗽三天伴隨痰多[問診紀錄_患者R1]\n"
        "乙- 個人史：不抽菸[問診紀錄_患者R4]")
AT = "一、西醫診斷：感冒[人類醫師_T2]"


def make(config=None, voice_position=(None, False)):
    history = SnapshotHistory()
    history.push(NOTE, AT, source="AI")
    refs = {k: Widget() for k in (
        "browse_container", "edit_container", "note_display", "at_display", "note_editor", "at_editor", "note_preview", "at_preview",
        "version_label", "record_status", "btn_browse", "btn_diff", "btn_edit_mode", "btn_edit_done", "btn_edit_cancel",
        "btn_undo", "btn_redo")}
    saved = []
    app_state = {"selected_patient_folder": "P", "selected_session_date": "2026-10-16"}
    fake = FakePanel()
    app_state[APP_STATE_KEY] = fake.panel
    ui_state = {"view_mode": "browse"}
    controller = MedicalRecordController(
        app_state=app_state, ui_state=ui_state, history=history, refs=refs,
        save_session_content=lambda *a: saved.append(a),
        record_snapshot_store=SimpleNamespace(save_history=lambda *a: None, append_audit_event=lambda *a, **k: None),
        write_session_log=lambda *a: saved.append(("log",) + a),
        busy_guard=None,
        load_config=lambda: config or {},
    )
    controller._recording_position = lambda: voice_position
    controller.update_display()
    controller.update_buttons()
    return SimpleNamespace(c=controller, history=history, refs=refs, ui_state=ui_state, saved=saved, panel=fake)


def test_edit_mode_fills_the_boxes_without_tags_or_marks_and_opens_the_reference_window():
    h = make()
    h.c.on_edit_mode()
    assert h.refs["note_editor"].value == "甲- 現病史：患者有高血壓。咳嗽三天伴隨痰多\n乙- 個人史：不抽菸"
    assert h.refs["at_editor"].value == "一、西醫診斷：感冒"
    assert h.ui_state["view_mode"] == "edit" and h.refs["btn_edit_done"].enabled and h.refs["btn_edit_cancel"].enabled
    assert h.panel.panel.opened and "參考：版本 1" in h.panel.title
    assert '<span class="src-tag">[問診紀錄_患者R1]</span>' in h.panel.note_html and "lc-note" in h.panel.note_html
    assert "家屬R2未提及" in h.panel.note_html                       # the reason is visible in the reference window
    assert "[人類醫師_T2]" in h.panel.at_html and h.refs["note_preview"].text.startswith("儲存後不會移除")


def test_saving_puts_the_tags_back_and_records_what_was_removed():
    h = make()
    h.c.on_edit_mode()
    h.refs["note_editor"].value = "甲- 現病史：患者有高血壓。咳嗽三天伴隨痰多\n完全改寫的內容"
    h.c.on_edit_done()
    snap = h.history.get_current()
    assert h.history.current_index == 1 and snap["source"] == "人類修改"
    lines = snap["note"].split("\n")
    assert lines[0] == NOTE.split("\n")[0]                           # the untouched line is exactly as it was, mark and tags included
    assert lines[1] == "完全改寫的內容" + HUMAN_EDIT_TAG
    assert snap["at"] == AT                                          # nothing changed in the A&T
    assert snap["meta"]["removed_tags"] == [{"field": "NOTE", "tag": "[問診紀錄_患者R4]", "text": "乙- 個人史：不抽菸"}]
    assert "dropped_marks" not in snap["meta"]
    assert h.ui_state["view_mode"] == "browse" and not h.panel.panel.opened and "已移除 1 個來源標籤" in h.refs["record_status"].text
    assert any(isinstance(item, tuple) and item[0] == "log" and "移除來源標籤 1 個" in item[3] for item in h.saved)


def test_editing_the_phrase_of_a_low_confidence_mark_drops_it_and_says_so():
    h = make()
    h.c.on_edit_mode()
    h.refs["note_editor"].value = h.refs["note_editor"].value.replace("患者有高血壓", "患者有糖尿病")
    h.c.refresh_preview(force=True)
    assert "低信心標註" in h.refs["note_preview"].text and "warn" in h.refs["note_preview"].classes_
    h.c.on_edit_done()
    snap = h.history.get_current()
    assert "**" not in snap["note"] and snap["meta"]["dropped_marks"] == [{"phrase": "患者有高血壓", "reason": "家屬R2未提及"}]
    assert "已移除 0 個來源標籤、1 個低信心標註" in h.refs["record_status"].text or "1 個低信心標註" in h.refs["record_status"].text


def test_a_typed_tag_is_refused_and_the_edit_stays_open():
    h = make()
    h.c.on_edit_mode()
    h.refs["note_editor"].value += "[語音#9]"
    h.c.on_edit_done()
    assert h.ui_state["view_mode"] == "edit" and h.history.current_index == 0
    assert "[語音#9]" in h.refs["record_status"].text and "b3261e" in h.refs["record_status"].style_
    h.c.refresh_preview(force=True)
    assert "[語音#9]" in h.refs["note_preview"].text
    h.c.on_edit_cancel()
    assert h.ui_state["view_mode"] == "browse" and h.history.current_index == 0
    assert not h.panel.panel.opened and "放棄" in h.refs["record_status"].text


def test_an_unchanged_save_adds_no_version():
    h = make()
    h.c.on_edit_mode()
    h.c.on_edit_done()
    assert h.history.current_index == 0 and "未偵測到變更" in h.refs["record_status"].text


def test_a_record_that_moved_on_during_the_edit_cannot_be_overwritten():
    h = make()
    h.c.on_edit_mode()
    h.history.push(NOTE + "\n丙- 新增", AT, source="AI")             # another window / an agent saved a newer version
    h.refs["note_editor"].value = "完全不同"
    h.c.on_edit_done()
    assert h.ui_state["view_mode"] == "edit" and h.history.get_current()["note"].endswith("丙- 新增")
    assert "已被更新" in h.refs["record_status"].text


def test_the_threshold_of_the_settings_decides_which_tags_stay():
    old = "甲- 現病史：大便偏乾，三天一次[問診紀錄_患者R1]"
    for run, expect_removed in ((3, False), (9, True)):
        h = make(config={"note_edit": {"min_surviving_run": run}})
        h.history.push(old, "", source="AI")
        h.c.on_edit_mode()
        h.refs["note_editor"].value = "甲- 現病史：大便偏乾，一日兩回"
        h.c.on_edit_done()
        assert ("removed_tags" in h.history.get_current().get("meta", {})) is expect_removed
    for bad in ({"note_edit": {"min_surviving_run": "x"}}, {"note_edit": {"min_surviving_run": 99}}, {}, {"note_edit": None}):
        assert make(config=bad).c._min_run() == 3


def test_the_reference_window_stays_closed_once_the_physician_closed_it_until_the_next_edit():
    h = make()
    h.c.on_edit_mode()
    h.panel.panel.dismiss()
    h.c.update_buttons()
    assert not h.panel.panel.opened
    h.c.on_edit_cancel()
    h.c.on_edit_mode()
    assert h.panel.panel.opened


def test_the_snapshot_keeps_the_recording_clock_and_the_edit_meta_across_a_save_and_a_restore():
    history = SnapshotHistory()
    history.push("a", "b", source="AI")
    history.push("c", "d", source="人類修改", rec_t=12.3456, rec_paused=True, meta={"removed_tags": [{"field": "NOTE", "tag": "[語音#1]", "text": "x"}]})
    copy = SnapshotHistory()
    copy.restore(history.to_dict())
    snap = copy.get_current()
    assert snap["rec_t"] == 12.35 and snap["rec_paused"] is True and snap["meta"]["removed_tags"][0]["tag"] == "[語音#1]"
    assert "rec_t" not in copy.snapshots[0] and "meta" not in copy.snapshots[0]


def test_the_diff_the_agents_read_orders_edits_against_the_transcript_and_says_what_an_edit_removed():
    history = SnapshotHistory()
    history.push("甲[語音#1]\n乙[語音#2]", "", source="AI", rec_t=60)
    history.push("甲[語音#1]\n丙" + HUMAN_EDIT_TAG, "", source="人類修改", rec_t=95.2, rec_paused=True,
                 meta={"removed_tags": [{"field": "NOTE", "tag": "[語音#2]", "text": "乙"}],
                       "dropped_marks": [{"phrase": "丙", "reason": "不明"}]})
    text = build_record_diff_context(history.snapshots, history.current_index)
    assert "@錄音 01:00" in text and "@錄音 01:35，錄音暫停中" in text
    assert "系統因此移除了來源標籤：NOTE：[語音#2]（原涵蓋：乙）" in text and "「丙」（不明）" in text
    # a session that was never recorded reads exactly as before
    plain = SnapshotHistory()
    plain.push("a", "", source="AI")
    plain.push("b", "", source="人類修改")
    out = build_record_diff_context(plain.snapshots, plain.current_index)
    assert "錄音" not in out and "系統因此移除" not in out
    # a version made before any recording in a session that has one says so
    mixed = SnapshotHistory()
    mixed.push("a", "", source="AI")
    mixed.push("b", "", source="AI", rec_t=3)
    assert "尚未錄音" in build_record_diff_context(mixed.snapshots, mixed.current_index)


def test_the_reference_html_escapes_text_shows_tags_light_and_marks_bold():
    html_text = reference_html("<b>x</b>[語音#3]**血壓高（無來源）**[歷史病歷]", marks=True)
    assert "&lt;b&gt;" in html_text and "<b>x" not in html_text
    assert '<span class="src-tag">[語音#3]</span>' in html_text and '<span class="src-tag">[歷史病歷]</span>' in html_text
    assert '<b class="lc-note">血壓高<span class="lc-why">（無來源）</span></b>' in html_text
    assert "**" in reference_html("**血壓高（無來源）**")                    # without marks (A&T) the bold is plain text
    assert "（空白）" in reference_html("  ")
    assert 'src-tag manual' in reference_html("乙" + HUMAN_EDIT_TAG)


def test_a_running_voice_recording_blocks_leaving_the_session_but_nothing_else():
    app_state = {"_voice_manager": SimpleNamespace(busy=True)}
    guard = SessionBusyGuard(app_state=app_state, ui_state={"view_mode": "browse"})
    assert guard.reason() is None                                          # agents, editing, ... go on
    assert guard.reason(include_voice=True) == "voice_recording"
    assert guard.is_busy(include_voice=True) and not guard.is_busy()
    assert "停止錄製" in guard.message("voice_recording")
    app_state["_voice_manager"].busy = False
    assert guard.reason(include_voice=True) is None
    guard.ui_state["view_mode"] = "edit"
    assert guard.reason() == "edit_mode" and "放棄修改" in guard.message("edit_mode")


def test_a_recording_that_began_after_the_last_version_still_marks_every_version_as_before_it():
    history = SnapshotHistory()
    history.push("a", "", source="AI")
    history.push("b", "", source="人類修改")                  # both made before the recording; nothing carries rec_t yet
    plain = build_record_diff_context(history.snapshots, history.current_index)
    assert "尚未錄音" not in plain and "@錄音" not in plain      # no transcript, no recording: no clock
    with_transcript = build_record_diff_context(history.snapshots, history.current_index, has_recording=True)
    assert with_transcript.count("尚未錄音") >= 2 and "@錄音" not in with_transcript


def test_a_manual_edit_is_stamped_with_the_main_turns_so_far_and_the_diff_says_so():
    h = make()
    h.c.app_state["agent_instance"] = SimpleNamespace(turn_count=3)
    h.c.on_edit_mode()
    h.refs["note_editor"].value = "改過的內容"
    h.c.on_edit_done()
    assert h.history.get_current()["after_turn"] == 3
    text = build_record_diff_context(h.history.snapshots, h.history.current_index)
    assert "第 3 主輪之後" in text
    # no agent yet in this session: no turn has run
    fresh = make()
    fresh.c.on_edit_mode()
    fresh.refs["note_editor"].value = "又一個內容"
    fresh.c.on_edit_done()
    assert fresh.history.get_current()["after_turn"] == 0
    assert "尚無任何主輪" in build_record_diff_context(fresh.history.snapshots, fresh.history.current_index)
    # the agents' own versions are not stamped (their source already says `Turn X-Y`)
    assert "主輪之後" not in build_record_diff_context(make().history.snapshots, 0)


def test_interview_rounds_carry_the_main_turn_that_started_them():
    from ui_app.rendering import format_interview_conversations
    text = format_interview_conversations([
        {"role": "subagent", "content": "哪裡不舒服？", "round": 1, "turn": 2},
        {"role": "patient", "content": "頭痛", "round": 1, "turn": 2},
        {"role": "subagent", "content": "多久了？", "round": 2, "turn": 3},
        {"role": "patient", "content": "三天", "round": 2, "turn": 3},
    ])
    assert "＜第1回合(R1，Turn 2)助理提問＞：哪裡不舒服？" in text and "＜第1回合(R1，Turn 2)收到回應＞：頭痛" in text
    assert "＜第2回合(R2，Turn 3)助理提問＞：多久了？" in text and "＜第2回合(R2，Turn 3)收到回應＞：三天" in text
