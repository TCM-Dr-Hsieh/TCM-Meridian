"""Findings of the prompt-contract review: a failed low-confidence scan is not a pass, a doubtful phrase can be named by occurrence, the
reviewer is told the scope of the update, the writer can warn the Main Agent. The agents' model clients are fakes."""
import json
from types import SimpleNamespace

import pytest

from Hallucination_Subagent import HallucinationSubagent
from Low_Confidence_Subagent import LowConfidenceSubagent, _apply_replacements, _plan_replacements
from Main_Agent import lc_incomplete_text, record_warning_suffix
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


def plan(note, *items):
    phrases = [dict(item) for item in items]
    statuses, claims = _plan_replacements(note, phrases)
    return [statuses[id(p)] for p in phrases], _apply_replacements(note, claims)


def test_the_occurrence_names_which_place_of_a_repeated_phrase_is_doubtful():
    text = "否認胸痛。否認胸痛[問診紀錄_患者R1]。否認胸痛"
    statuses, marked = plan(text, {"original_phrase": "否認胸痛", "reason": "x", "occurrence": 3})
    assert statuses == ["replaced"] and marked.endswith("。**否認胸痛（x）**") and marked.count("**") == 2
    assert plan(text, {"original_phrase": "否認胸痛", "reason": "x", "occurrence": 4}) == (["not_found"], text)
    already = "**否認胸痛（x）**。否認胸痛"
    assert plan(already, {"original_phrase": "否認胸痛", "reason": "y", "occurrence": 1})[0] == ["already_tagged"]    # a place in bold counts
    assert plan(already, {"original_phrase": "否認胸痛", "reason": "y", "occurrence": 2})[0] == ["replaced"]
    assert plan("無異常[2026-01-02-胸部X光]", {"original_phrase": "異常[2026-01-02", "reason": "z"})[0] == ["citation_tag"]


def test_several_phrases_of_one_round_are_all_placed_on_the_note_the_model_saw():
    # the reason of the first phrase contains the second phrase's words: with one-by-one editing the count of the second would shift
    note = "胸痛三天。胸痛加重。"
    statuses, marked = plan(note,
                            {"original_phrase": "胸痛", "reason": "胸痛的來源不明", "occurrence": 1},
                            {"original_phrase": "胸痛", "reason": "只有家屬轉述", "occurrence": 2})
    assert statuses == ["replaced", "replaced"]
    assert marked == "**胸痛（胸痛的來源不明）**三天。**胸痛（只有家屬轉述）**加重。"
    # without occurrence, two entries for the same words take two different places; a repeated request for one place is not marked twice
    statuses, marked = plan(note, {"original_phrase": "胸痛", "reason": "a"}, {"original_phrase": "胸痛", "reason": "b"})
    assert statuses == ["replaced", "replaced"] and marked.count("**") == 4 and "（a）" in marked and "（b）" in marked
    statuses, marked = plan(note, {"original_phrase": "胸痛", "reason": "a", "occurrence": 1}, {"original_phrase": "胸痛", "reason": "dup", "occurrence": 1})
    assert statuses == ["replaced", "already_tagged"] and "dup" not in marked


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
    assert "沒有列出的內容，這次沒動" in user                               # the diff shows the changed lines only: far-away facts are untouched
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


@pytest.mark.parametrize("reply", [
    "{}",
    '{"intercept": "detected", "low_confidence_phrases": [{}]}',
    '{"intercept": "detected", "low_confidence_phrases": [{"original_phrase": "", "reason": "x"}, "text"]}',
    '{"intercept": "pass"}',
    '{"low_confidence_phrases": []}',
    '{"intercept": "maybe", "low_confidence_phrases": []}',
    '{"intercept": "pass", "low_confidence_phrases": "none"}',
    '[1, 2, 3]',
])
def test_a_reply_with_the_wrong_structure_is_a_failed_scan_not_a_pass(reply):
    agent = lc()
    agent.client = fake_client(reply)
    result = agent.execute(note_content=NOTE)
    assert result["incomplete"] is True and result["scan_errors"] == 2 and result["total_annotated"] == 0


def test_a_well_formed_reply_is_accepted_even_when_it_contradicts_itself():
    agent = lc(strength=1, rounds=2)
    agent.client = fake_client(json.dumps({"intercept": "detected", "low_confidence_phrases": []}))      # detected, nothing named: a pass
    result = agent.execute(note_content=NOTE)
    assert result["incomplete"] is False and result["scan_errors"] == 0
    agent = lc(strength=1, rounds=1)
    both = {"intercept": "pass", "low_confidence_phrases": [{"original_phrase": "否認胸痛", "reason": "r"}, {"bad": 1}]}
    agent.client = fake_client(json.dumps(both))                                                          # a valid phrase among invalid ones is kept
    assert agent.execute(note_content=NOTE)["total_annotated"] == 1


def test_the_scan_prompt_carries_the_scope_of_this_turns_update_only_when_there_is_one():
    agent = lc(strength=1, rounds=1)
    agent.client = fake_client(json.dumps({"intercept": "pass", "low_confidence_phrases": []}))
    agent.execute(note_content=NOTE, update_guidelines="局部更新：只更新血壓，其餘內容不動")
    user = agent.client.calls[-1]["messages"][1]["content"]
    assert "## 【本輪病歷更新的登載方針】" in user and "局部更新：只更新血壓，其餘內容不動" in user
    agent.execute(note_content=NOTE)
    assert "本輪病歷更新的登載方針" not in agent.client.calls[-1]["messages"][1]["content"]


def refusing_writer(reviewer):
    writer = RecordSubagent(CONFIG, {**CONFIG, "detection_strength": 1, "max_review_rounds": 1})
    writer.hallucination_reviewer = reviewer
    reply = {"thinking": "t", "target_field": "note", "notes_for_main": "患者說對青黴素過敏，這次局部更新沒寫入",
             "operations": [{"op": "replace", "line": 1, "content": "血壓 120/80[人類醫師_T1]"}]}
    writer.client = fake_client(json.dumps(reply, ensure_ascii=False))
    return writer


def test_the_writers_warning_survives_a_refused_update_and_an_update_without_a_reviewer():
    class Refuser:
        def review(self, **kwargs):
            return {"agree": "no", "comment": "不通過", "thinking": ""}

    refused = refusing_writer(Refuser()).execute(note_content="血壓：", at_content="", target_field="note", guidelines="g")
    assert refused["success"] is False and refused["notes_for_main"] == "患者說對青黴素過敏，這次局部更新沒寫入"
    unreviewed = refusing_writer(None)
    unreviewed.hallucination_reviewer = None
    done = unreviewed.execute(note_content="血壓：", at_content="", target_field="note", guidelines="g")
    assert done["success"] is True and done["notes_for_main"] == "患者說對青黴素過敏，這次局部更新沒寫入"


def test_the_main_agent_puts_the_warning_in_the_step_result_whether_or_not_the_record_was_written():
    sub = {"notes_for_main": "患者說對青黴素過敏"}
    assert record_warning_suffix(sub, written=True) == "；病歷登載助理提醒：患者說對青黴素過敏"
    refused = record_warning_suffix(sub, written=False)
    assert "病歷雖未寫入，仍須轉告醫師" in refused and "患者說對青黴素過敏" in refused
    assert record_warning_suffix({"notes_for_main": " "}, written=True) == "" and record_warning_suffix({}, written=False) == ""


def test_phrases_that_overlap_in_one_round_become_one_mark_with_both_reasons():
    note = "否認胸痛且無呼吸困難，持續三天"
    statuses, marked = plan(note, {"original_phrase": "胸痛", "reason": "a"}, {"original_phrase": "胸痛且無呼吸困難", "reason": "b"})
    assert statuses == ["replaced", "replaced"] and marked == "否認**胸痛且無呼吸困難（a；b）**，持續三天"    # 無呼吸困難 is not left bare
    # the longer one first: the shorter lies wholly inside it, nothing is lost by not marking it again
    statuses, marked = plan(note, {"original_phrase": "胸痛且無呼吸困難", "reason": "b"}, {"original_phrase": "胸痛", "reason": "a"})
    assert statuses == ["replaced", "already_tagged"] and marked == "否認**胸痛且無呼吸困難（b）**，持續三天"
    # a partial overlap from either side joins the two spans
    statuses, marked = plan("無胸痛且無呼吸困難", {"original_phrase": "無胸痛", "reason": "a"}, {"original_phrase": "痛且無呼吸困難", "reason": "b"})
    assert statuses == ["replaced", "replaced"] and marked == "**無胸痛且無呼吸困難（a；b）**"
    # the same reason twice is said once; phrases that do not touch stay two marks
    assert plan(note, {"original_phrase": "胸痛", "reason": "a"}, {"original_phrase": "胸痛且無呼吸困難", "reason": "a"})[1] == "否認**胸痛且無呼吸困難（a）**，持續三天"
    assert plan(note, {"original_phrase": "否認", "reason": "a"}, {"original_phrase": "持續三天", "reason": "b"})[1].count("**") == 4


def test_the_scan_counts_a_merged_mark_once():
    answers = iter([
        {"intercept": "detected", "low_confidence_phrases": [{"original_phrase": "胸痛", "reason": "a"}, {"original_phrase": "胸痛且無呼吸困難", "reason": "b"}]},
        {"intercept": "pass", "low_confidence_phrases": []},
    ])
    agent = lc(strength=1, rounds=3)
    agent.client = fake_client(lambda kwargs: json.dumps(next(answers), ensure_ascii=False))
    result = agent.execute(note_content="否認胸痛且無呼吸困難")
    assert result["annotated_note"] == "否認**胸痛且無呼吸困難（a；b）**" and result["total_annotated"] == 1 and result["incomplete"] is False


MIXED = {"intercept": "detected", "low_confidence_phrases": [
    {"original_phrase": "否認胸痛", "reason": "只有家屬轉述"},
    {"reason": "過敏資訊遺漏"},
]}


def test_an_item_without_a_phrase_is_asked_for_again_and_the_repaired_answer_is_used():
    repaired = {"intercept": "detected", "low_confidence_phrases": [
        {"original_phrase": "否認胸痛", "reason": "只有家屬轉述"},
        {"original_phrase": "乙- 其他", "reason": "過敏資訊遺漏，沒有來源"},
    ]}
    answers = iter([MIXED, repaired, {"intercept": "pass", "low_confidence_phrases": []}])
    agent = lc(strength=1, rounds=3)
    agent.client = fake_client(lambda kwargs: json.dumps(next(answers), ensure_ascii=False))
    result = agent.execute(note_content=NOTE)
    assert result["skipped_items"] == [] and result["incomplete"] is False and result["total_annotated"] == 2
    asked = agent.client.calls[1]["messages"]
    assert asked[-2]["role"] == "assistant" and "過敏資訊遺漏" in asked[-1]["content"] and "original_phrase" in asked[-1]["content"]


def test_an_item_that_stays_unusable_is_reported_not_dropped_and_the_valid_marks_stay():
    agent = lc(strength=1, rounds=1)
    agent.client = fake_client(json.dumps(MIXED, ensure_ascii=False))
    result = agent.execute(note_content=NOTE)
    assert len(agent.client.calls) == 3                                            # asked twice more, then gave up on the item
    assert result["total_annotated"] == 1 and "**否認胸痛（只有家屬轉述）**" in result["annotated_note"]
    assert result["incomplete"] is True and result["scan_errors"] == 0
    assert [entry["round"] for entry in result["skipped_items"]] == [1] and "過敏資訊遺漏" in result["skipped_items"][0]["item"]


def test_the_step_result_says_what_the_unfinished_scan_lacked():
    both = {"scan_errors": 1, "total_rounds": 3, "total_annotated": 2, "skipped_items": [{"round": 2, "item": '{"reason": "過敏資訊遺漏"}'}]}
    text = lc_incomplete_text(both)
    assert text.startswith("低信心標註未完成：") and "輸出解析失敗 1 次" in text and "過敏資訊遺漏" in text and "已跑 3 輪、標註 2 個片段" in text
    assert "請在回覆中明確告訴醫師低信心標註沒有完成" in text
    assert "輸出解析失敗" not in lc_incomplete_text({"scan_errors": 0, "total_rounds": 1, "total_annotated": 1, "skipped_items": [{"item": "x"}]})


def test_the_scope_of_the_update_survives_a_save_and_restore_during_the_interview():
    from Main_Agent import MainAgent

    def build():
        return MainAgent(CONFIG, CONFIG, None, CONFIG, CONFIG)

    first = build()
    script = iter([
        {"thinking": "t", "action": "update_record", "action_input": {"target_field": "note", "guidelines": "局部更新：只更新血壓"}, "next_step": ""},
        {"thinking": "t", "action": "information_collection_subagent", "action_input": "問頭痛", "next_step": ""},
    ])
    first._call_llm = lambda *args, **kwargs: next(script)
    first.record_subagent.execute = lambda **kwargs: {"success": True, "note": "血壓 128/82[人類醫師_T1]", "at": "", "operations": [],
                                                      "review_result": "agree", "review_rounds": 1}
    started = first.process_message("更新血壓", "", "", session_date="2026-10-17")
    assert started["waiting_for_interview"] is True

    second = build()                                                   # a new assistant, as after a restart
    second.restore_state(json.loads(json.dumps(first.export_state())))
    seen = {}

    def spy(**kwargs):
        seen.update(kwargs)
        return {"success": True, "annotated_note": kwargs["note_content"], "total_rounds": 1, "total_annotated": 0, "all_phrases": [],
                "scan_errors": 0, "skipped_items": [], "incomplete": False, "error": None}

    second.lc_subagent.execute = spy
    after = iter([
        {"thinking": "t", "action": "low_confidence_check", "action_input": "", "next_step": ""},
        {"thinking": "t", "action": "reply", "action_input": "好", "next_step": ""},
    ])
    second._call_llm = lambda *args, **kwargs: next(after)
    done = second.continue_after_interview("", "問診完成", "血壓 128/82[人類醫師_T1]", "")
    assert done["reply"] == "好" and seen["update_guidelines"] == "局部更新：只更新血壓"


def test_a_failed_call_while_asking_for_the_repair_keeps_what_the_first_answer_held():
    def reply(kwargs):
        if len(kwargs["messages"]) > 2:
            raise ConnectionError("down")                               # the call that asks for the repair
        return json.dumps(MIXED, ensure_ascii=False)

    agent = lc(strength=1, rounds=1)
    agent.client = fake_client(reply)
    result = agent.execute(note_content=NOTE)
    assert result["total_annotated"] == 1 and "**否認胸痛（只有家屬轉述）**" in result["annotated_note"]
    assert result["incomplete"] is True and result["scan_errors"] == 1
    assert len(result["skipped_items"]) == 1 and "過敏資訊遺漏" in result["skipped_items"][0]["item"]
    # nothing found before the failure: a plain failed scan, as before
    agent = lc(strength=1, rounds=1)
    agent.client = fake_client(lambda kwargs: (_ for _ in ()).throw(ConnectionError("down")))
    failed = agent.execute(note_content=NOTE)
    assert failed["total_annotated"] == 0 and failed["skipped_items"] == [] and failed["incomplete"] is True and failed["scan_errors"] == 1


def test_the_files_read_this_turn_survive_a_save_and_restore_during_the_interview(tmp_path):
    from Main_Agent import MainAgent

    (tmp_path / "Medical_information").mkdir()
    (tmp_path / "Medical_information" / "2026-10-16-Lab_data.md").write_text("HbA1c 7.3%", encoding="utf-8")

    def build():
        return MainAgent(CONFIG, CONFIG, None, CONFIG, CONFIG)

    first = build()
    script = iter([
        {"thinking": "t", "action": "read_patient_file", "action_input": "2026-10-16-Lab_data.md", "next_step": ""},
        {"thinking": "t", "action": "information_collection_subagent", "action_input": "問頭痛", "next_step": ""},
    ])
    first._call_llm = lambda *args, **kwargs: next(script)
    started = first.process_message("看檢驗", "", "", patient_folder=str(tmp_path), session_date="2026-10-17")
    assert started["waiting_for_interview"] is True and len(first._loaded_files) == 1

    second = build()                                                   # a new assistant, as after a restart
    second.restore_state(json.loads(json.dumps(first.export_state())))
    seen = {}

    def spy(**kwargs):
        seen.update(kwargs)
        return {"success": True, "annotated_note": kwargs["note_content"], "total_rounds": 1, "total_annotated": 0, "all_phrases": [],
                "scan_errors": 0, "skipped_items": [], "incomplete": False, "error": None}

    second.lc_subagent.execute = spy
    after = iter([
        {"thinking": "t", "action": "low_confidence_check", "action_input": "", "next_step": ""},
        {"thinking": "t", "action": "reply", "action_input": "好", "next_step": ""},
    ])
    second._call_llm = lambda *args, **kwargs: next(after)
    second.continue_after_interview("", "問診完成", "檢驗：HbA1c 7.3%[2026-10-16-Lab_data]", "")
    assert "HbA1c 7.3%" in seen["loaded_files_block"] and "2026-10-16-Lab_data.md" in seen["loaded_files_block"]
    # the next main turn starts clean, as before
    again = iter([{"thinking": "t", "action": "reply", "action_input": "嗨", "next_step": ""}])
    second._call_llm = lambda *args, **kwargs: next(again)
    second.process_message("下一輪", "", "", patient_folder=str(tmp_path), session_date="2026-10-17")
    assert second._loaded_files == []
