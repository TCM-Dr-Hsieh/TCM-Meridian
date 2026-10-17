"""The agents' models copy the tab-indented record lines into JSON string values as raw tab characters; the replies must still parse.
The model clients are fakes; the replies are built by hand because `json.dumps` would escape the tabs."""
import json

import pytest

import model_json
from Hallucination_Subagent import HallucinationSubagent
from Information_Collection_Subagent import InformationCollectionSubagent
from Main_Agent import MainAgent
from Note_Review_Subagent import NoteReviewSubagent
from Record_Subagent import RecordSubagent
from tests.test_review_fixes import CONFIG, NOTE, fake_client, lc

TAB = "\t"


def test_the_helper_accepts_raw_tabs_and_newlines_in_strings_but_not_broken_json():
    text = '{"a": "x' + TAB + TAB + 'y\nz", "b": [1]}'
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)                                       # what the agents did before
    assert model_json.loads(text) == {"a": "x\t\ty\nz", "b": [1]}
    for broken in ('{"a": ', '{"a": "x" "b"}', "not json"):
        with pytest.raises(json.JSONDecodeError):
            model_json.loads(broken)


def test_the_record_writer_reads_an_operation_whose_content_holds_raw_tabs():
    reply = ('{"thinking": "t' + TAB + 'x", "target_field": "note", "operations": '
             '[{"op": "replace", "line": 2, "content": "' + TAB + TAB + '血壓： 128/82 mmHg[人類醫師_T1]"}]}')

    class Reviewer:
        def review(self, **kwargs):
            return {"agree": "yes", "comment": "", "thinking": ""}

    writer = RecordSubagent(CONFIG, {**CONFIG, "detection_strength": 1, "max_review_rounds": 1})
    writer.hallucination_reviewer = Reviewer()
    writer.client = fake_client(reply)
    result = writer.execute(note_content="己- 基本理學檢查：\n" + TAB + TAB + "血壓： / mmHg", at_content="", target_field="note", guidelines="g")
    assert result["success"] is True and len(writer.client.calls) == 1             # parsed at the first try, no retry wasted
    assert result["note"].split("\n")[1] == TAB + TAB + "血壓： 128/82 mmHg[人類醫師_T1]"


def test_the_record_writers_rewrite_after_a_refusal_reads_raw_tabs_too():
    first = '{"thinking": "t", "target_field": "note", "operations": [{"op": "replace", "line": 1, "content": "甲"}]}'
    rewrite = ('{"thinking": "t' + TAB + '", "target_field": "note", "notes_for_main": "提醒' + TAB + '一下", '
               '"operations": [{"op": "replace", "line": 1, "content": "' + TAB + '乙"}]}')
    answers = iter([first, rewrite])

    class Reviewer:
        calls = 0

        def review(self, **kwargs):
            Reviewer.calls += 1
            return {"agree": "no", "comment": "改", "thinking": ""} if Reviewer.calls == 1 else {"agree": "yes", "comment": "", "thinking": ""}

    writer = RecordSubagent(CONFIG, {**CONFIG, "detection_strength": 1, "max_review_rounds": 3})
    writer.hallucination_reviewer = Reviewer()
    writer.client = fake_client(lambda kwargs: next(answers))
    result = writer.execute(note_content="x", at_content="", target_field="note", guidelines="g")
    assert result["success"] is True and result["note"] == TAB + "乙" and result["notes_for_main"] == "提醒" + TAB + "一下"


def test_the_reviewer_reads_a_verdict_with_raw_tabs_in_its_reasoning():
    reply = '{"thinking": "1.' + TAB + '核對\n2.' + TAB + '通過", "agree": "yes", "comment": ""}'
    reviewer = HallucinationSubagent({**CONFIG, "detection_strength": 1})
    reviewer.client = fake_client(reply)
    result = reviewer.review(field="note", content="甲")
    assert result["agree"] == "yes" and len(reviewer.client.calls) == 1


def test_the_low_confidence_scan_reads_a_reason_with_raw_tabs():
    reply = ('{"intercept": "detected", "low_confidence_phrases": '
             '[{"original_phrase": "否認胸痛", "reason": "只有家屬' + TAB + '轉述"}]}')
    agent = lc(strength=1, rounds=1)
    agent.client = fake_client(reply)
    result = agent.execute(note_content=NOTE)
    assert result["scan_errors"] == 0 and result["total_annotated"] == 1 and "**否認胸痛（只有家屬" + TAB + "轉述）**" in result["annotated_note"]


def test_the_note_review_reads_raw_tabs():
    reply = '{"thinking": "t' + TAB + '", "action_input": "", "update_reminder": "請確認' + TAB + '舌象"}'
    agent = NoteReviewSubagent(CONFIG)
    agent.client = fake_client(reply)
    result = agent.execute(note_content=NOTE)
    assert result["success"] is True and result["update_reminder"] == "請確認" + TAB + "舌象" and len(agent.client.calls) == 1


def test_the_interview_assistant_reads_raw_tabs():
    reply = '{"thinking": "t' + TAB + '", "action": "ask_patient", "action_input": "哪裡' + TAB + '不舒服？", "next_step": ""}'
    agent = InformationCollectionSubagent({**CONFIG, "max_collection_rounds": 3})
    result = agent._parse_and_execute(reply, lambda message: None)
    assert result["action"] == "ask_patient" and result["finished"] is False and result["message"] == "哪裡" + TAB + "不舒服？"


def test_the_main_agent_reads_its_own_reply_and_its_action_inputs_with_raw_tabs():
    agent = MainAgent(CONFIG, CONFIG, None, CONFIG, CONFIG)
    reply = ('{"thinking": "t' + TAB + '", "action": "update_record", '
             '"action_input": "{\\"target_field\\": \\"note\\", \\"guidelines\\": \\"更新' + TAB + '血壓\\"}", "next_step": ""}')
    agent.client = fake_client(reply)
    parsed = agent._call_llm([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], "Turn 1-1", lambda message: None, "2026-10-17", 1)
    assert parsed["action"] == "update_record" and len(agent.client.calls) == 1
    # the action input is itself a JSON string: it carries raw tabs when the model copies a template line into it
    inner = '{"target_field": "note", "guidelines": "更新' + TAB + '血壓"}'
    seen = {}
    script = iter([{"thinking": "t", "action": "update_record", "action_input": inner, "next_step": ""},
                   {"thinking": "t", "action": "reply", "action_input": "好", "next_step": ""}])
    agent._call_llm = lambda *args, **kwargs: next(script)
    agent.record_subagent.execute = lambda **kwargs: seen.update(kwargs) or {
        "success": True, "note": "乙", "at": "", "operations": [], "review_result": "agree", "review_rounds": 1}
    agent.process_message("更新", "甲", "", session_date="2026-10-17")
    assert seen["target_field"] == "note" and seen["guidelines"] == "更新" + TAB + "血壓"        # not the whole JSON text taken as the guideline


def test_the_turn_history_keeps_the_updated_field_when_the_action_input_holds_raw_tabs():
    agent = MainAgent.__new__(MainAgent)
    inner = '{"target_field": "note",' + TAB + '"guidelines": "更新' + TAB + '血壓"}'
    steps = [{"step_label": "Turn 1-1", "action": "update_record", "action_input": inner, "result": "成功執行 1 個操作", "thinking": "t", "next_step": ""}]
    summary = agent._build_sub_steps_summary(steps)
    assert summary[0]["action_summary"] == "update_record: 更新 note, 成功執行 1 個操作"          # not "更新 ?"
    broken = [{"step_label": "Turn 1-2", "action": "update_record", "action_input": "沒有 JSON 的方針", "result": "r", "thinking": "", "next_step": ""}]
    assert agent._build_sub_steps_summary(broken)[0]["action_summary"] == "update_record: 更新 ?, r"       # a plain-text guideline still reads as unknown
