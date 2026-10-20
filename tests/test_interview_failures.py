"""A step of the interview assistant that fails (the model call fails, its output cannot be read, it names an action that does not exist)
is not a finished interview: it used to show "✅ 問診完成" and hand the Main Agent a half interview. Now the interview stays open, every
reply sent is kept, and the physician retries the step or stops the interview."""
from types import SimpleNamespace

from Information_Collection_Subagent import InformationCollectionSubagent
from tests.test_voice_interview import CONFIG, DATE, Assistant, Voice, controller


class Model:
    """Answers with the next item of `replies`; an exception in the list is raised instead."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


QUESTION = '{"thinking": "t", "action": "ask_patient", "action_input": "頭痛多久了？", "next_step": "n"}'


def test_a_failed_step_keeps_the_interview_open_and_a_retry_carries_on_from_the_same_reply():
    for failure, says in ((ConnectionError("timeout"), "模型呼叫失敗：timeout"),
                          ("完全不是 JSON", "輸出無法解析"),
                          ('{"thinking": "t", "action": "dance", "action_input": "x"}', "不認得的動作「dance」")):
        assistant = InformationCollectionSubagent(CONFIG)
        broken = [failure] * (assistant.MAX_JSON_RETRIES + 1) if failure == "完全不是 JSON" else [failure]
        assistant.client = Model(QUESTION, *broken, QUESTION)
        assistant.start_collection(guidelines="問頭痛", log_callback=lambda m: None)
        result = assistant.receive_answer("頭痛", log_callback=lambda m: None)
        assert result["action"] == "error" and result["finished"] is False and says in result["message"], says
        assert assistant.finished is False and assistant.all_conversations[-1]["content"] == "頭痛"     # the reply is kept
        again = assistant.retry_turn(log_callback=lambda m: None)
        assert again["action"] == "ask_patient" and again["round"] == 2
        assert [m["role"] for m in assistant.all_conversations] == ["subagent", "patient", "subagent"]    # nothing added twice


class Failing(Assistant):
    def __init__(self):
        super().__init__(answer_from=10.0)
        self.retried = []

    def receive_answer(self, **kwargs):
        self.received.append(kwargs)
        self.all_conversations.append({"role": "patient", "content": kwargs["patient_input"], "round": self.dialogue_round})
        return {"action": "error", "message": "問診助理模型呼叫失敗：timeout", "round": self.dialogue_round, "finished": False}

    def retry_turn(self, **kwargs):
        self.retried.append(kwargs)
        self.dialogue_round += 1
        return {"action": "ask_patient", "message": "頭痛多久了？", "round": self.dialogue_round, "finished": False}


async def test_the_tab_shows_a_failure_not_a_finished_interview_and_the_retry_button_carries_on():
    assistant = Failing()
    made = controller(Voice(clock=40.0), assistant)
    await made.submit_answer("頭痛")
    refs = made.refs
    assert refs["status"].text.startswith("❌ 問診助理模型呼叫失敗：timeout") and "問診完成" not in refs["status"].text
    assert refs["btn_retry"].visible and refs["btn_stop"].enabled
    assert not refs["btn_send"].enabled and not refs["btn_voice"].enabled         # the reply is in: the next step is the assistant's
    assert made.interview_state.is_active()                                        # not handed to the Main Agent as finished
    await made.retry_assistant()
    assert assistant.retried[-1]["voice_transcript"] == "【診間對話逐字稿 · 第 2 份】"     # the transcript of now
    assert refs["status"].text == "📋 問診中 (R4)" and not refs["btn_retry"].visible and refs["btn_send"].enabled


async def test_an_unexpected_error_is_shown_like_a_failed_step_instead_of_reopening_the_reply():
    assistant = Assistant(answer_from=10.0)

    def broken(**kwargs):
        raise RuntimeError("boom")

    assistant.receive_answer = broken
    made = controller(Voice(clock=40.0), assistant)
    await made.submit_answer("頭痛")
    assert "問診助理錯誤：boom" in made.refs["status"].text and made.refs["btn_retry"].visible
    assert not made.refs["btn_send"].enabled                                        # sending the reply again would record it twice


def test_a_page_restored_while_the_assistant_owed_its_next_step_offers_the_retry():
    made = controller(Voice(clock=40.0), Assistant(answer_from=10.0))
    asked = [{"role": "subagent", "content": "哪裡不舒服？", "round": 1}]
    made.load_history_ui("方針", asked + [{"role": "patient", "content": "頭痛", "round": 1}], 1, active=True, awaiting_assistant=True)
    assert made.refs["btn_retry"].visible and not made.refs["btn_send"].enabled and "還沒完成下一步" in made.refs["status"].text
    made.load_history_ui("方針", asked, 1, active=True)                              # waiting for the physician: reply as usual
    assert not made.refs["btn_retry"].visible and made.refs["btn_send"].enabled
    made.load_history_ui("方針", asked, 1, active=False, awaiting_assistant=True)    # a finished interview has nothing to retry
    assert not made.refs["btn_retry"].visible


class Shown:
    def __init__(self):
        self.calls = []

    def __call__(self, guidelines, conversations, total, *, active=False, awaiting_assistant=False):
        self.calls.append({"conversations": conversations, "active": active, "awaiting": awaiting_assistant})


def restore(state):
    from ui_app.controllers.session_state_restorer import SessionStateRestorer
    app_state, ui_state = {}, {"chat_messages": []}
    shown = app_state["_load_interview_ui"] = Shown()
    restorer = SessionStateRestorer(
        app_state=app_state, ui_state=ui_state, load_config=lambda: {"ic_subagent": CONFIG}, load_chat_state=lambda *a: None,
        load_interview_state=lambda *a: state, load_forum_state=lambda *a: [], save_chat_state=lambda *a: None,
        agent_factory=SimpleNamespace(create_information_collection_agent_if_configured=InformationCollectionSubagent),
        render_chat=lambda: None)
    waiting = SimpleNamespace(_suspended={"step": "Turn 2-1"}, export_state=dict)   # the Main Agent waits for this interview
    restorer.agent_run_state.set_agent(waiting)
    restorer.restore_interview_state("patient_data/A", DATE)
    return restorer, shown, waiting


def test_an_interview_whose_first_question_failed_is_restored_waiting_for_the_retry():
    started = InformationCollectionSubagent(CONFIG)
    started.client = Model(ConnectionError("timeout"))
    assert started.start_collection(guidelines="問頭痛", log_callback=lambda m: None)["action"] == "error"
    state = started.export_state()
    assert state["active"] and state["all_conversations"] == []                     # nothing to show, but the interview is on
    restorer, shown, waiting = restore(state)
    assert restorer.interview_state.is_active() and waiting._suspended                # neither the interview nor the Main Agent dropped
    assert shown.calls == [{"conversations": [], "active": True, "awaiting": True}]
    restorer.interview_state.get_subagent().client = Model(QUESTION)
    assert restorer.interview_state.get_subagent().retry_turn(log_callback=lambda m: None)["action"] == "ask_patient"


def test_whether_the_assistant_owes_a_step_is_read_from_this_interview_not_the_earlier_ones_of_the_visit():
    earlier = [{"role": "subagent", "content": "q", "round": 1, "step": "Turn 1-1"},
               {"role": "patient", "content": "a", "round": 1, "step": "Turn 1-1"}]
    base = {"active": True, "finished": False, "dialogue_round": 1, "guidelines": "g", "all_conversations": earlier}
    _, shown, _ = restore({**base, "conversation": []})                               # a second interview, its first question failed
    assert shown.calls[-1]["awaiting"] is True
    asked = earlier + [{"role": "subagent", "content": "q2", "round": 2, "step": "Turn 2-1"}]
    _, shown, _ = restore({**base, "all_conversations": asked, "conversation": asked[2:]})   # its question is on the screen
    assert shown.calls[-1]["awaiting"] is False
    _, shown, waiting = restore({**base, "active": False, "finished": True, "conversation": []})  # a finished one is just shown
    assert shown.calls[-1] == {"conversations": earlier, "active": False, "awaiting": False} and waiting._suspended is None


async def test_a_double_click_on_retry_runs_one_step_and_a_later_step_keeps_its_own_guard():
    import asyncio
    import threading
    assistant = Failing()
    gate, calls = threading.Event(), []

    def slow_retry(**kwargs):
        calls.append(kwargs)
        gate.wait(5)
        assistant.dialogue_round += 1
        return {"action": "ask_patient", "message": "頭痛多久了？", "round": assistant.dialogue_round, "finished": False}

    assistant.retry_turn = slow_retry
    made = controller(Voice(clock=40.0), assistant)
    first = asyncio.ensure_future(made.retry_assistant())
    await asyncio.sleep(0.05)
    await asyncio.wait_for(made.retry_assistant(), 1)                                # the second click returns at once
    gate.set()
    await first
    assert len(calls) == 1 and assistant.dialogue_round == 4 and made._assistant_step is None
    mine, later = object(), object()
    made._assistant_step = later
    made._release(mine)                                                               # an earlier call finishing late
    assert made._assistant_step is later
