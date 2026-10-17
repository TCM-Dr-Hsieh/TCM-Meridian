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
    # a version made while a recording ran but nothing was transcribed yet is not "not recorded": it says there was no transcript
    started = SnapshotHistory()
    started.push("a", "", source="AI", rec_t=0)
    started.push("b", "", source="人類修改", rec_t=0, rec_paused=False)
    text = build_record_diff_context(started.snapshots, started.current_index)
    assert "@錄音 00:00，當時還沒有任何逐字稿" in text and "尚未錄音" not in text


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
        {"role": "subagent", "content": "哪裡不舒服？", "round": 1, "step": "Turn 2-1"},
        {"role": "patient", "content": "頭痛", "round": 1, "step": "Turn 2-1"},
        {"role": "subagent", "content": "多久了？", "round": 2, "step": "Turn 3-2"},
        {"role": "patient", "content": "三天", "round": 2, "step": "Turn 3-2"},
    ])
    assert "＜第1回合(R1，Turn 2-1)助理提問＞：哪裡不舒服？" in text and "＜第1回合(R1，Turn 2-1)收到回應＞：頭痛" in text
    assert "＜第2回合(R2，Turn 3-2)助理提問＞：多久了？" in text and "＜第2回合(R2，Turn 3-2)收到回應＞：三天" in text


def test_the_live_interview_output_carries_the_step_into_the_diff_order_note():
    """The real path: the assistant creates its round messages, `get_full_dialogue()` hands them to the Main Agent, the diff reads them."""
    from Information_Collection_Subagent import InformationCollectionSubagent
    from ui_app.rendering import format_interview_conversations
    ic = InformationCollectionSubagent({"api_url": "http://fake.invalid/v1", "api_key": "k", "model_name": "m"})
    log = lambda message: None
    ic.current_step = "Turn 1-1"
    ic._parse_and_execute('{"action": "ask_patient", "action_input": "哪裡不舒服？"}', log)       # R1, a question
    ic._execute_turn = lambda *args, **kwargs: {"action": "finish_collection", "message": "", "round": ic.dialogue_round, "finished": True}
    ic.receive_answer("頭痛三天")                                                                    # the answer to R1
    ic.current_step = "Turn 3-2"                                                                      # a later interview, a later step
    ic._parse_and_execute('{"action": "ask_patient", "action_input": "血糖？"}', log)                # R2
    ic.receive_answer("HbA1c 7.1%")
    dialogue = ic.get_full_dialogue()
    assert "＜第1回合(R1，Turn 1-1)收到回應＞：頭痛三天" in dialogue and "＜第2回合(R2，Turn 3-2)收到回應＞：HbA1c 7.1%" in dialogue
    assert format_interview_conversations(ic.all_conversations) == dialogue          # a restored session reads the same text
    history = SnapshotHistory()
    history.push("a", "", source="init")
    history.push("b", "", source="人類修改", after_turn=1)
    text = build_record_diff_context(history.snapshots, history.current_index, interview_dialogue=dialogue)
    assert "（版本 2 的先後：問診回合 R1 早於這個版本；問診回合 R2 晚於這個版本。）" in text


def test_the_interview_keeps_the_step_that_started_it_across_a_save_and_a_restore():
    from Information_Collection_Subagent import InformationCollectionSubagent
    config = {"api_url": "http://fake.invalid/v1", "api_key": "k", "model_name": "m"}
    started = InformationCollectionSubagent(config)
    assert started.current_step == ""                       # exists from the start, not only after reset()
    started.current_step = "Turn 3-2"
    started.all_conversations.append({"role": "patient", "content": "x", "round": 1, "step": "Turn 3-2"})
    reloaded = InformationCollectionSubagent(config)
    reloaded.restore_state(started.export_state())          # a new assistant, as after a restart
    assert reloaded.current_step == "Turn 3-2"
    assert reloaded.all_conversations[0]["step"] == "Turn 3-2"
    reloaded.reset()
    assert reloaded.current_step == ""


def test_the_main_agent_names_the_step_that_is_waiting_for_the_interview():
    from Main_Agent import MainAgent
    agent = MainAgent.__new__(MainAgent)
    agent.turn_count = 5
    agent._suspended = {"turn_num": 4, "sub_turn": 2}
    assert agent.suspended_step_label == "Turn 4-2"


def test_the_diff_says_which_interview_rounds_came_before_and_after_each_version():
    dialogue = ("＜第1回合(R1，Turn 1-1)助理提問＞：哪裡不舒服？\n＜第1回合(R1，Turn 1-1)收到回應＞：頭痛\n"
                "＜第2回合(R2，Turn 1-1)助理提問＞：多久？\n＜第2回合(R2，Turn 1-1)收到回應＞：三天\n"
                "＜第3回合(R3，Turn 3-2)助理提問＞：血糖？\n＜第3回合(R3，Turn 3-2)收到回應＞：7.1\n")
    history = SnapshotHistory()
    history.push("a", "", source="init")
    history.push("b", "", source="Turn 1-2 update_record")
    history.push("c", "", source="人類修改", after_turn=1)
    history.push("d", "", source="Turn 3-1 update_record")
    text = build_record_diff_context(history.snapshots, history.current_index, interview_dialogue=dialogue)
    # the agent's own version at Turn 1-2: both Turn 1-1 rounds are before it, the Turn 3-2 round after
    assert "（版本 2 的先後：問診回合 R1~R2 早於這個版本；問診回合 R3 晚於這個版本。）" in text
    # the manual edit after main turn 1: same split
    assert "（版本 3 的先後：問診回合 R1~R2 早於這個版本；問診回合 R3 晚於這個版本。）" in text
    # the doctor-instructed version at Turn 3-1 (agent-written): R3 started at Turn 3-2, so it is later
    assert "（版本 4 的先後：問診回合 R1~R2 早於這個版本；問診回合 R3 晚於這個版本。）" in text
    # a manual edit before any main turn: every round is after it
    first = SnapshotHistory()
    first.push("a", "", source="init")
    first.push("b", "", source="人類修改", after_turn=0)
    assert "（版本 2 的先後：問診回合 R1~R3 晚於這個版本。）" in build_record_diff_context(
        first.snapshots, first.current_index, interview_dialogue=dialogue)
    # without interview rounds there is nothing to say
    assert "的先後" not in build_record_diff_context(history.snapshots, history.current_index)


def make_processor(history, live_position):
    from ui_app.controllers.main_agent_result_processor import MainAgentResultProcessor
    nothing = lambda *args, **kwargs: None
    return MainAgentResultProcessor(
        app_state={"_voice_manager": SimpleNamespace(recording_position=lambda fp, dt: live_position)}, ui_state={}, history=history,
        agent_run_state=None, live_steps=None, main_chat_input=None, agent_status=None, save_current_to_disk=nothing, update_display=nothing,
        update_buttons=nothing, render_chat=nothing, write_session_log=nothing, save_chat_state=nothing, save_forum_state=nothing,
        save_conversation_file=nothing, record_snapshot_store=SimpleNamespace(append_audit_event=nothing))


def test_the_last_version_of_an_agent_turn_sits_where_the_turns_other_versions_do():
    history = SnapshotHistory()
    history.push("a", "", source="init")
    processor = make_processor(history, (99.0, False))                       # the live end of the transcript, far ahead
    result = {"note_changed": True, "at_changed": False, "turn_number": 3, "note": "d", "at": "",
              "record_snapshots": [{"note": "b", "at": "", "source": "Turn 3-1 update_record", "rec_t": 40.0, "rec_paused": True},
                                   {"note": "c", "at": "", "source": "Turn 3-2 update_record", "rec_t": 40.0, "rec_paused": True}]}
    processor._apply_record_changes(result, "fp", "dt")
    assert [snap.get("rec_t") for snap in history.snapshots] == [None, 40.0, 40.0, 40.0]
    assert history.snapshots[-1]["rec_paused"] is True
    # a turn that left no version of its own (nothing stamped): the live position
    history = SnapshotHistory()
    history.push("a", "", source="init")
    processor = make_processor(history, (99.0, False))
    processor._apply_record_changes({"note_changed": True, "at_changed": False, "turn_number": 4, "note": "e", "at": ""}, "fp", "dt")
    assert history.snapshots[-1]["rec_t"] == 99.0


TRANSCRIPT = ("【診間對話逐字稿 · 截至 01:00】\n語音#1 00:05–00:20 頭痛三天\n語音#2 00:20–00:35 家屬說最近一次檢查 HbA1c 是 7.1%\n"
              "語音#3 00:45–01:00 好，我們再看看")


def test_the_diff_roughly_says_which_transcript_lines_came_before_and_after_a_version():
    history = SnapshotHistory()
    history.push("甲", "", source="AI", rec_t=10)
    history.push("乙" + HUMAN_EDIT_TAG, "", source="人類修改", rec_t=40.7)
    out = build_record_diff_context(history.snapshots, 1, voice_transcript=TRANSCRIPT)
    assert "（版本 2 的逐字稿先後（以時間粗略估計）：逐字稿 #1~#2 約早於這個版本；逐字稿 #3 晚於這個版本。）" in out
    # the line that ends at the boundary itself is before it; whole seconds, as the lines show them: 00:35 is later than 34.9 s
    short = SnapshotHistory()
    short.push("甲", "", source="AI", rec_t=10)
    short.push("乙", "", source="人類修改", rec_t=34.9)
    assert "逐字稿 #1 約早於這個版本；逐字稿 #2~#3 晚於這個版本" in build_record_diff_context(short.snapshots, 1, voice_transcript=TRANSCRIPT)
    # a version made before the recording began: every line is after it; a session with no transcript lines gets no such note
    before = SnapshotHistory()
    before.push("甲", "", source="AI")
    before.push("乙", "", source="人類修改")
    assert "全部逐字稿都晚於這個版本" in build_record_diff_context(before.snapshots, 1, has_recording=True, voice_transcript=TRANSCRIPT)
    assert "逐字稿先後" not in build_record_diff_context(history.snapshots, 1)
    assert "逐字稿先後" not in build_record_diff_context(history.snapshots, 1, voice_transcript="【診間對話逐字稿 · 截至 00:00】\n（尚無逐字稿內容）")
    # hours in the clock
    long = SnapshotHistory()
    long.push("甲", "", source="AI", rec_t=10)
    long.push("乙", "", source="人類修改", rec_t=4000)
    assert "逐字稿 #9 約早於" in build_record_diff_context(long.snapshots, 1, voice_transcript="語音#9 1:02:03–1:05:00 好")


def test_the_main_agent_stamps_a_version_with_the_clock_it_was_given():
    from Main_Agent import MainAgent
    agent = MainAgent.__new__(MainAgent)
    agent.rec_clock = lambda: (9.0, True)
    assert agent._rec_fields() == {"rec_t": 9.0, "rec_paused": True}
    agent.rec_clock = lambda: (0.0, False)
    assert agent._rec_fields() == {"rec_t": 0.0}                         # a recording, no transcript yet
    agent.rec_clock = lambda: (None, False)
    assert agent._rec_fields() == {}                                     # no recording
    agent.rec_clock = None
    assert agent._rec_fields() == {}



