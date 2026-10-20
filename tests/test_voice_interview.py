"""Answering an interview round aloud: the physician asks the assistant's questions in the room and presses a button; the stretch of the
recording since the previous reply is the answer. The interview record keeps a pointer to the stretch, never a retelling of it (the facts
are cited from the transcript as [語音#N]); the assistant reads the latest transcript every round."""
import asyncio
from types import SimpleNamespace

import pytest

from Information_Collection_Subagent import (TYPED_SUPPLEMENT_MARK, VOICE_ANSWER_MARK, InformationCollectionSubagent,
                                             format_interview_dialogue, voice_answer_content)
from tests.test_voice_service import DATE, make_manager, record
from ui_app.controllers.auto_interview_controller import AutoInterviewController
from ui_app.controllers.interview_state_controller import get_interview_state
from voice.pipeline import AnswerWindow, TranscriptPipeline, gap_marker

CONFIG = {"api_url": "http://fake.invalid/v1", "api_key": "k", "model_name": "m"}


@pytest.fixture
def patient(tmp_path):
    folder = tmp_path / 'patient_data' / 'A123_1980-01-01_王'
    folder.mkdir(parents=True)
    return str(folder)


# ------------------------------------------------------------------------------------------------ the stretch of the transcript
def segment(index, start, end, new_start, text, kind="speech"):
    return {"index": index, "kind": kind, "start": start, "end": end, "new_start": new_start, "raw_asr": text, "raw": text,
            "added": text, "corrected": text}


def pipeline():
    made = TranscriptPipeline(asr=None, asr_settings=None, corrector=None)
    made.restore({"committed_end": 18.0, "segments": [
        segment(1, 0, 6, 0, "哪裡不舒服"), segment(2, 3, 9, 6, "頭痛三天"), segment(3, 6, 12, 9, "沒有發燒"),
        segment(4, 9, 15, 12, gap_marker(12, 15), kind="gap"), segment(5, 12, 18, 15, "晚上比較痛")]})
    return made


def test_a_stretch_holds_the_segments_that_cover_any_of_it_and_says_whether_the_recognition_has_reached_its_end():
    made = pipeline()
    window = made.answer_window(6.0, 12.0)
    assert [index for index, _ in window.lines] == [2, 3] and (window.first, window.last) == (2, 3)
    assert window.lines[0][1] == "語音#2 00:06–00:09 頭痛三天"                       # the line exactly as the agents read it
    wide = made.answer_window(5.0, 13.0)
    assert [index for index, _ in wide.lines] == [1, 2, 3, 4]                         # a straddling segment belongs to both sides
    assert "【音訊可能缺漏 00:12–00:15】" in wide.lines[-1][1]                         # a failed window is in the stretch, as a gap
    assert window.caught_up                                                            # a finished transcript has nothing more coming
    made.finished, made.committed_end = False, 10.0                                    # recording: the audio up to 10 s is recognized
    assert not made.answer_window(6.0, 12.0).caught_up and made.answer_window(6.0, 12.0).reach == 10.0
    made.committed_end = 12.0                                                          # silence and failed windows move it too
    assert made.answer_window(6.0, 12.0).caught_up
    empty = made.answer_window(30.0, 40.0)
    assert empty.lines == [] and empty.first is None and empty.last is None


async def test_the_manager_gives_the_recording_clock_and_the_stretch_of_the_session(patient, tmp_path):
    manager = make_manager(seconds_per_recording=7, text='我頭痛三天了')
    assert manager.audio_clock(patient, DATE) == 0.0                                   # nothing recorded yet
    nothing = manager.answer_window(patient, DATE, 0.0, 5.0)
    assert nothing.lines == [] and nothing.caught_up
    await record(manager)
    assert manager.audio_clock(patient, DATE) == pytest.approx(7.0, abs=0.2)           # the end of the file once it is stopped
    window = manager.answer_window(patient, DATE, 0.0, 7.0)
    assert window.caught_up and window.lines and window.lines[0][1].startswith("語音#1 ")
    assert manager.audio_clock(None, DATE) is None
    other = tmp_path / 'patient_data' / 'B456_1990-01-01_李'
    other.mkdir(parents=True)
    manager.session.state = 'recording'                                                # another visit is being recorded
    try:
        assert manager.audio_clock(str(other), DATE) is None and manager.answer_window(str(other), DATE, 0.0, 1.0) is None
    finally:
        manager.session.state = 'idle'


# ------------------------------------------------------------------------------------------------ the pointer and the assistant
def test_the_reply_of_a_round_asked_aloud_points_at_the_transcript_and_never_retells_it():
    text = voice_answer_content(45, 52, 192.0, 245.4)
    assert text.startswith(VOICE_ANSWER_MARK) and "【診間對話逐字稿】語音#45 至 #52（03:12–04:05）" in text
    assert "此處不轉述內容" in text and "[語音#N]" in text
    assert "語音#7（" in voice_answer_content(7, 7, 60, 70)
    late = voice_answer_content(7, 8, 60, 70, caught_up=False, supplement=" 沒有夜尿，語音聽錯 ")
    assert "還沒全部辨識完" in late and late.endswith(f"\n{TYPED_SUPPLEMENT_MARK}沒有夜尿，語音聽錯")
    assert "沒有辨識到對話" in voice_answer_content(None, None, 60, 70)


class FakeModel:
    def __init__(self):
        self.prompts = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.prompts.append(kwargs["messages"][-1]["content"])
        reply = '{"thinking": "t", "action": "ask_patient", "action_input": "晚上會比較痛嗎？", "next_step": "n"}'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


def test_the_assistant_reads_the_latest_transcript_every_round_and_keeps_where_a_voice_answer_is():
    assistant = InformationCollectionSubagent(CONFIG)
    assistant.client = model = FakeModel()
    assistant.current_step = "Turn 2-1"
    assistant.start_collection(guidelines="問頭痛", voice_transcript="【診間對話逐字稿 · 截至 00:10】\n語音#1 00:00–00:06 頭痛三天")
    assert "## 【診間對話逐字稿】\n【診間對話逐字稿 · 截至 00:10】" in model.prompts[-1] and assistant.answer_from is None
    pointer = voice_answer_content(2, 3, 10.0, 25.0)
    place = {"first": 2, "last": 3, "start": 10.0, "end": 25.0}
    assistant.receive_answer(pointer, voice=place, voice_transcript="【診間對話逐字稿 · 截至 00:25】\n語音#3 00:18–00:25 晚上比較痛")
    assert "截至 00:25" in model.prompts[-1] and "截至 00:10" not in model.prompts[-1]          # this round's transcript, not the first
    assert f"＜第1回合(R1)收到回應＞：{VOICE_ANSWER_MARK}" in model.prompts[-1]
    assert assistant.all_conversations[1]["voice"] == place
    assert f"＜第1回合(R1，Turn 2-1)收到回應＞：{pointer}" in format_interview_dialogue(assistant.all_conversations)
    assistant.receive_answer("打字的回覆")                                                       # no new transcript: the last one stays
    assert "截至 00:25" in model.prompts[-1] and "voice" not in assistant.all_conversations[3]


def test_where_the_next_voice_answer_starts_survives_a_save_and_a_restore():
    started = InformationCollectionSubagent(CONFIG)
    started.answer_from = 125.5
    reloaded = InformationCollectionSubagent(CONFIG)
    reloaded.restore_state(started.export_state())
    assert reloaded.answer_from == 125.5
    reloaded.reset()
    assert reloaded.answer_from is None


# ------------------------------------------------------------------------------------------------ the interview tab
class Thing:
    """A widget: chains, nests, and remembers whether it is enabled."""

    def __init__(self):
        self.text, self.value, self.enabled, self.visible = "", "", True, True

    def __getattr__(self, name):
        return lambda *args, **kwargs: self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def set_visibility(self, visible):
        self.visible = visible


class Voice:
    def __init__(self, clock):
        self.clock, self.snapshots = clock, 0

    def audio_clock(self, folder, date):
        return self.clock

    def snapshot_text(self, folder, date):
        self.snapshots += 1
        return f"【診間對話逐字稿 · 第 {self.snapshots} 份】"


class Assistant:
    def __init__(self, answer_from=None, conversations=()):
        self.answer_from, self.dialogue_round, self.all_conversations = answer_from, 3, list(conversations)
        self.received = []

    def receive_answer(self, **kwargs):
        self.received.append(kwargs)
        self.dialogue_round += 1                                                  # the next question is a new round
        return {"action": "ask_patient", "message": "下一題", "round": self.dialogue_round, "finished": False}

    def export_state(self):
        return {}

    def get_full_dialogue(self):
        return format_interview_dialogue(self.all_conversations)


def controller(voice, assistant):
    app_state = {"selected_patient_folder": "patient_data/A", "selected_session_date": DATE, "_voice_manager": voice}
    made = AutoInterviewController(
        Thing(), app_state, load_config=dict, write_session_log=lambda *a: None, save_chat_state=lambda *a: None,
        save_interview_state=lambda *a: None, save_forum_state=lambda *a: None, save_conversation_file=lambda *a: None,
        get_last_visit_content=lambda *a: "", load_history_summary=lambda *a: "", agent_factory=None)
    made.refs = {key: Thing() for key in ("status", "btn_send", "btn_voice", "btn_retry", "btn_stop", "chat_container", "reply_input")}
    state = get_interview_state(app_state)
    state.set_subagent(assistant)
    state.start(assistant)
    return made


async def test_the_voice_button_sends_a_pointer_to_the_stretch_since_the_previous_reply(monkeypatch):
    assistant = Assistant(answer_from=120.0)
    made = controller(Voice(clock=185.0), assistant)
    asked = {}

    async def dialog(fp, dt, start, end, round_num):
        asked.update(start=start, end=end, round=round_num)
        return "沒有夜尿", AnswerWindow(start, end, [(41, "語音#41 …"), (44, "語音#44 …")], True, 186.0)

    sent = {}

    async def submit(answer, *, voice=None, answered_at=None):
        sent.update(answer=answer, voice=voice, answered_at=answered_at)

    monkeypatch.setattr(made, "_voice_answer_dialog", dialog)
    monkeypatch.setattr(made, "submit_answer", submit)
    await made.answer_from_transcript()
    assert asked == {"start": 120.0, "end": 185.0, "round": 3}
    assert sent["answer"] == voice_answer_content(41, 44, 120.0, 185.0, supplement="沒有夜尿")
    assert sent["voice"] == {"first": 41, "last": 44, "start": 120.0, "end": 185.0} and sent["answered_at"] == 185.0


async def test_the_voice_button_needs_a_recording_of_this_visit(monkeypatch):
    for clock, says in ((0.0, "還沒有錄音"), (None, "不能用於這個就診")):
        made = controller(Voice(clock=clock), Assistant(answer_from=0.0))
        monkeypatch.setattr(made, "_voice_answer_dialog", lambda *a: pytest.fail("no window without a recording"))
        await made.answer_from_transcript()
        assert says in made.refs["status"].text


async def test_every_reply_hands_over_the_transcript_of_now_and_moves_the_start_of_the_next_answer():
    assistant = Assistant(answer_from=30.0)
    voice = Voice(clock=75.0)
    made = controller(voice, assistant)
    await made.submit_answer("頭痛三天")                                                # typed: counted from the moment it was sent
    assert assistant.answer_from == 75.0
    assert assistant.received[-1]["voice_transcript"] == "【診間對話逐字稿 · 第 1 份】" and assistant.received[-1]["voice"] is None
    place = {"first": 1, "last": 2, "start": 75.0, "end": 98.0}
    await made.submit_answer(voice_answer_content(1, 2, 75.0, 98.0), voice=place, answered_at=98.0)
    assert assistant.answer_from == 98.0 and assistant.received[-1]["voice"] == place
    assert assistant.received[-1]["voice_transcript"] == "【診間對話逐字稿 · 第 2 份】"
    assert made.refs["btn_send"].enabled and made.refs["btn_voice"].enabled           # the next question is on the screen


def test_the_first_question_on_the_screen_starts_the_first_voice_answer():
    assistant = Assistant(answer_from=None)
    made = controller(Voice(clock=42.0), assistant)
    made.handle_result({"action": "ask_patient", "message": "哪裡不舒服？", "round": 1}, "patient_data/A", DATE)
    assert assistant.answer_from == 42.0
    made.refs["btn_voice"].disable()
    made.handle_result({"action": "ask_patient", "message": "多久了？", "round": 2}, "patient_data/A", DATE)
    assert assistant.answer_from == 42.0 and made.refs["btn_voice"].enabled           # later rounds start where the last reply was sent
    made.handle_result({"action": "finish_collection", "message": "", "round": 2, "finished": True}, None, None)
    assert not made.refs["btn_send"].enabled and not made.refs["btn_voice"].enabled


async def test_one_voice_answer_window_at_a_time_and_nothing_is_sent_for_a_round_that_moved_on(monkeypatch):
    assistant = Assistant(answer_from=120.0)
    made = controller(Voice(clock=185.0), assistant)
    release, opened = asyncio.Event(), []

    async def dialog(fp, dt, start, end, round_num):
        opened.append(round_num)
        await release.wait()
        return "", AnswerWindow(start, end, [(41, "語音#41 …")], True, 186.0)

    monkeypatch.setattr(made, "_voice_answer_dialog", dialog)
    first = asyncio.ensure_future(made.answer_from_transcript())
    await asyncio.sleep(0)
    await asyncio.wait_for(made.answer_from_transcript(), 1)                         # a second click while the window is open: no window
    assert opened == [3]
    release.set()
    await first
    assert len(assistant.received) == 1 and assistant.dialogue_round == 4            # R3 answered once; R4 is on the screen
    release.clear()

    async def moved_on(fp, dt, start, end, round_num):                                 # the round moves on while the window is open
        assistant.dialogue_round += 1
        made._next_ic_run_id()
        return "", AnswerWindow(start, end, [(50, "語音#50 …")], True, 200.0)

    monkeypatch.setattr(made, "_voice_answer_dialog", moved_on)
    await made.answer_from_transcript()
    assert len(assistant.received) == 1 and "這份語音回答沒有送出" in made.refs["status"].text
    assert made._voice_answer_open is False                                           # the window can be opened again
