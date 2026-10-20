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


def test_a_page_restored_while_the_assistant_owed_an_answer_offers_the_retry():
    made = controller(Voice(clock=40.0), Assistant(answer_from=10.0))
    asked = [{"role": "subagent", "content": "哪裡不舒服？", "round": 1}]
    made.load_history_ui("方針", asked + [{"role": "patient", "content": "頭痛", "round": 1}], 1, active=True)
    assert made.refs["btn_retry"].visible and not made.refs["btn_send"].enabled and "還沒回應" in made.refs["status"].text
    made.load_history_ui("方針", asked, 1, active=True)                              # waiting for the physician: reply as usual
    assert not made.refs["btn_retry"].visible and made.refs["btn_send"].enabled
    made.load_history_ui("方針", asked + [{"role": "patient", "content": "頭痛", "round": 1}], 1, active=False)
    assert not made.refs["btn_retry"].visible                                       # a finished interview has nothing to retry
