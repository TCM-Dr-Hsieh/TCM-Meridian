"""Findings of the prompt-contract review: a failed low-confidence scan is not a pass, a doubtful phrase can be named by occurrence, the
reviewer is told the scope of the update, the writer can warn the Main Agent. The agents' model clients are fakes."""
import json
from types import SimpleNamespace

import pytest

from Hallucination_Subagent import HallucinationSubagent
from Low_Confidence_Subagent import LowConfidenceSubagent, _replace_occurrence
from Record_Subagent import RecordSubagent

CONFIG = {"api_url": "http://fake.invalid/v1", "api_key": "k", "model_name": "m", "max_tokens": 1000, "temperature": 0.0}


def fake_client(reply):
    """A client whose chat.completions.create answers `reply` (a str, or a callable returning a str / raising)."""
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        text = reply(kwargs) if callable(reply) else reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), calls=calls)


def lc(strength=2, rounds=4):
    agent = LowConfidenceSubagent({**CONFIG, "detection_strength": strength, "max_scan_rounds": rounds})
    return agent


def scripted(agent, results):
    queue = list(results)
    agent._scan_round = lambda **kwargs: queue.pop(0)


NOTE = "甲- 現病史：否認胸痛。\n乙- 其他：否認胸痛。"
ERROR = {"intercept": "error", "phrases": [], "error": "boom"}
PASS = {"intercept": "pass", "phrases": []}


def test_two_failed_scans_in_a_row_are_incomplete_not_a_pass():
    agent = lc()
    scripted(agent, [ERROR, ERROR, PASS, PASS])
    result = agent.execute(note_content=NOTE)
    assert result["success"] is True and result["incomplete"] is True and result["scan_errors"] == 2
    assert result["total_annotated"] == 0 and result["annotated_note"] == NOTE and result["total_rounds"] == 2   # it stopped, it did not go on to "pass"


def test_a_failed_round_followed_by_enough_passes_still_completes():
    agent = lc()
    scripted(agent, [ERROR, PASS, PASS])
    result = agent.execute(note_content=NOTE)
    assert result["incomplete"] is False and result["scan_errors"] == 1 and result["total_rounds"] == 3


def test_what_was_marked_before_the_failure_is_kept_and_the_scan_is_reported_incomplete():
    agent = lc()
    found = {"intercept": "detected", "phrases": [{"original_phrase": "否認胸痛", "reason": "原始問答沒有提到胸痛"}]}
    scripted(agent, [found, ERROR, ERROR])
    result = agent.execute(note_content=NOTE)
    assert result["incomplete"] is True and result["total_annotated"] == 1
    assert result["annotated_note"].startswith("甲- 現病史：**否認胸痛（原始問答沒有提到胸痛）**")


def test_the_real_scan_round_turns_a_dead_model_into_an_error_not_a_pass():
    agent = lc()

    def dead(kwargs):
        raise ConnectionError("down")
    agent.client = fake_client(dead)
    result = agent.execute(note_content=NOTE)
    assert result["incomplete"] is True and result["scan_errors"] == 2 and result["total_annotated"] == 0
    agent.client = fake_client("not json at all")                                         # an answer nobody can read is the same
    result = agent.execute(note_content=NOTE)
    assert result["incomplete"] is True and result["scan_errors"] == 2


def test_a_real_pass_is_still_a_pass():
    agent = lc()
    agent.client = fake_client(json.dumps({"intercept": "pass", "low_confidence_phrases": []}))
    result = agent.execute(note_content=NOTE)
    assert result["incomplete"] is False and result["scan_errors"] == 0 and result["total_rounds"] == 2


def test_the_occurrence_names_which_place_of_a_repeated_phrase_is_doubtful():
    text = "否認胸痛。否認胸痛[問診紀錄_患者R1]。否認胸痛"
    marked, status = _replace_occurrence(text, "否認胸痛", "**否認胸痛（x）**", 3)
    assert status == "replaced" and marked.endswith("。**否認胸痛（x）**") and marked.count("**") == 2
    assert _replace_occurrence(text, "否認胸痛", "**X**", 4) == (text, "not_found")
    already = "**否認胸痛（x）**。否認胸痛"
    assert _replace_occurrence(already, "否認胸痛", "**Y**", 1)[1] == "already_tagged"      # a place in bold counts, and is not marked twice
    assert _replace_occurrence(already, "否認胸痛", "**Y**", 2)[1] == "replaced"
    assert _replace_occurrence("無異常[2026-01-02-胸部X光]", "異常[2026-01-02", "**Z**", 1)[1] == "citation_tag"


def test_the_scan_marks_the_second_place_when_the_model_says_occurrence_2_and_ignores_a_bad_value():
    agent = lc(strength=1, rounds=2)
    scripted(agent, [{"intercept": "detected", "phrases": [{"original_phrase": "否認胸痛", "reason": "r", "occurrence": 2}]}, PASS])
    result = agent.execute(note_content=NOTE)
    assert result["annotated_note"] == "甲- 現病史：否認胸痛。\n乙- 其他：**否認胸痛（r）**。"
    for bad in (0, "2", True, None, -1):
        agent = lc(strength=1, rounds=2)
        scripted(agent, [{"intercept": "detected", "phrases": [{"original_phrase": "否認胸痛", "reason": "r", "occurrence": bad}]}, PASS])
        assert agent.execute(note_content=NOTE)["annotated_note"].startswith("甲- 現病史：**否認胸痛（r）**")     # the old rule: the first place


def test_the_reviewer_is_told_the_scope_of_the_update():
    reviewer = HallucinationSubagent({**CONFIG, "detection_strength": 1})
    reviewer.client = fake_client(json.dumps({"thinking": "", "agree": "yes", "comment": ""}))
    reviewer.review(field="note", content="血壓 120/80", update_guidelines="局部更新：只更新血壓，其餘內容不動")
    user = reviewer.client.calls[-1]["messages"][1]["content"]
    assert "## 【本次登載方針（病歷登載助理收到的指示）】" in user and "局部更新：只更新血壓，其餘內容不動" in user
    reviewer.review(field="note", content="血壓 120/80")
    assert "本次登載方針" not in reviewer.client.calls[-1]["messages"][1]["content"]


def test_the_record_writer_hands_its_guidelines_to_the_reviewer_and_its_warning_to_the_main_agent():
    seen = {}

    class Reviewer:
        def review(self, **kwargs):
            seen.update(kwargs)
            return {"agree": "yes", "comment": "", "thinking": ""}

    writer = RecordSubagent(CONFIG, {**CONFIG, "detection_strength": 1, "max_review_rounds": 2})
    writer.hallucination_reviewer = Reviewer()
    reply = {"thinking": "t", "target_field": "note", "notes_for_main": "對話中患者說對青黴素過敏，這次是局部更新，沒有寫入",
             "operations": [{"op": "replace", "line": 1, "content": "血壓 120/80[人類醫師_T1]"}]}
    writer.client = fake_client(json.dumps(reply, ensure_ascii=False))
    guidelines = "局部更新：只更新血壓，其餘內容不動"
    result = writer.execute(note_content="血壓：", at_content="", target_field="note", guidelines=guidelines)
    assert result["success"] is True and seen["update_guidelines"] == guidelines
    assert result["notes_for_main"] == "對話中患者說對青黴素過敏，這次是局部更新，沒有寫入"
    # nothing to change: the warning still reaches the Main Agent
    reply2 = {"thinking": "t", "target_field": "note", "notes_for_main": "只有提醒", "operations": []}
    writer.client = fake_client(json.dumps(reply2, ensure_ascii=False))
    assert writer.execute(note_content="血壓：", at_content="", target_field="note", guidelines=guidelines)["notes_for_main"] == "只有提醒"
    # no warning: the field is empty
    reply3 = {"thinking": "t", "target_field": "note", "operations": []}
    writer.client = fake_client(json.dumps(reply3))
    assert writer.execute(note_content="血壓：", at_content="", target_field="note", guidelines=guidelines)["notes_for_main"] == ""


def test_the_reviewer_sees_what_this_update_changed_and_the_writer_supplies_the_note_before_it():
    reviewer = HallucinationSubagent({**CONFIG, "detection_strength": 1})
    reviewer.client = fake_client(json.dumps({"thinking": "", "agree": "yes", "comment": ""}))
    reviewer.review(field="note", content="甲- 現病史：頭痛三天\n癸- Lab：HbA1c 7.3%[2026-10-16-Lab_data]",
                    previous_content="甲- 現病史：頭痛三天\n癸- Lab：")
    user = reviewer.client.calls[-1]["messages"][1]["content"]
    assert "## 【本次更新的改動（`-` 修改前、`+` 即將登載）】" in user
    assert "-癸- Lab：\n+癸- Lab：HbA1c 7.3%[2026-10-16-Lab_data]" in user and "+甲- 現病史：頭痛三天" not in user
    reviewer.review(field="note", content="same", previous_content="same")
    assert "（沒有改動）" in reviewer.client.calls[-1]["messages"][1]["content"]
    reviewer.review(field="note", content="x")                                             # an old caller: no section at all
    assert "本次更新的改動" not in reviewer.client.calls[-1]["messages"][1]["content"]

    seen = {}

    class Spy:
        def review(self, **kwargs):
            seen.update(kwargs)
            return {"agree": "yes", "comment": "", "thinking": ""}

    writer = RecordSubagent(CONFIG, {**CONFIG, "detection_strength": 1, "max_review_rounds": 2})
    writer.hallucination_reviewer = Spy()
    writer.client = fake_client(json.dumps({"thinking": "t", "target_field": "note",
                                            "operations": [{"op": "replace", "line": 1, "content": "血壓 120/80[人類醫師_T1]"}]}))
    writer.execute(note_content="血壓：", at_content="", target_field="note", guidelines="g")
    assert seen["previous_content"] == "血壓：" and seen["content"] == "血壓 120/80[人類醫師_T1]"
