"""Findings of the prompt-contract review: a failed low-confidence scan is not a pass, a doubtful phrase can be named by occurrence, the
reviewer is told the scope of the update, the writer can warn the Main Agent. The agents' model clients are fakes."""
import json
import os
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


def test_the_record_writer_hands_its_warning_to_the_main_agent():
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
    assert result["success"] is True and "update_guidelines" not in seen          # the scope is read from the chat, not from this text
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
    assert seen["image_files"] is second._loaded_files                                          # the main agent hands the files (and so the images) to the scan
    # the next main turn starts clean, as before
    again = iter([{"thinking": "t", "action": "reply", "action_input": "嗨", "next_step": ""}])
    second._call_llm = lambda *args, **kwargs: next(again)
    second.process_message("下一輪", "", "", patient_folder=str(tmp_path), session_date="2026-10-17")
    assert second._loaded_files == []


def test_the_main_agent_hands_the_writer_a_diff_that_says_which_transcript_lines_came_after_each_version():
    from Main_Agent import MainAgent
    agent = MainAgent(CONFIG, CONFIG, None, CONFIG, CONFIG)
    script = iter([{"thinking": "t", "action": "update_record", "action_input": {"target_field": "note", "guidelines": "g"}, "next_step": ""},
                   {"thinking": "t", "action": "reply", "action_input": "好", "next_step": ""}])
    agent._call_llm = lambda *args, **kwargs: next(script)
    seen = {}

    def writer(**kwargs):
        seen.update(kwargs)
        return {"success": True, "note": "乙", "at": "", "operations": [], "review_result": "agree", "review_rounds": 1}

    agent.record_subagent.execute = writer
    transcript = "【診間對話逐字稿 · 截至 01:00】\n語音#1 00:05–00:20 頭痛三天\n語音#2 00:20–00:35 家屬說 HbA1c 7.1%\n語音#3 00:45–01:00 好"
    snapshots = [{"note": "甲", "at": "", "source": "Turn 1-1 update_record", "rec_t": 10.0},
                 {"note": "甲2", "at": "", "source": "人類修改", "rec_t": 40.0, "after_turn": 1}]
    agent.process_message("更新", "甲2", "", record_history_snapshots=snapshots, record_history_current_index=1,
                          voice_transcript=transcript, session_date="2026-10-17")
    assert "（版本 2 的逐字稿先後（以時間粗略估計）：逐字稿 #1~#2 約早於這個版本；逐字稿 #3 晚於這個版本。）" in seen["record_diff_context"]


def test_the_scope_is_read_from_the_physicians_words_in_the_chat_not_from_a_guideline_text():
    import inspect
    assert "update_guidelines" not in inspect.signature(HallucinationSubagent.review).parameters
    assert "update_guidelines" not in inspect.signature(LowConfidenceSubagent.execute).parameters
    said = "【人類醫師】\n只幫我更新血壓，其他不要動"
    reviewer = HallucinationSubagent({**CONFIG, "detection_strength": 1})
    reviewer.client = fake_client(json.dumps({"thinking": "", "agree": "yes", "comment": ""}))
    reviewer.review(field="note", content="血壓 120/80", conversation_history=said, previous_content="血壓：")
    user = reviewer.client.calls[-1]["messages"][1]["content"]
    assert said in user and "登載方針" not in user and "## 【本次更新的改動" in user
    agent = lc(strength=1, rounds=1)
    agent.client = fake_client(json.dumps({"intercept": "pass", "low_confidence_phrases": []}))
    agent.execute(note_content=NOTE, conversation_history=said)
    assert said in agent.client.calls[-1]["messages"][1]["content"] and "登載方針" not in agent.client.calls[-1]["messages"][1]["content"]
    from Main_Agent import MainAgent
    assert not hasattr(MainAgent.__new__(MainAgent), "_last_note_update_guidelines")


# ------------------------------------------------------------------------------------------------ the low-confidence scan sees the images read this turn
PNG_1X1 = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c6360f8cfc0f00000040001ee8d6eb70000000049454e44ae426082")


def image_entry(tmp_path, name="bp_photo.png"):
    path = tmp_path / name
    path.write_bytes(PNG_1X1)
    return {"name": name, "type": "image", "path": str(path)}


def test_the_scan_gets_the_images_read_this_turn_in_every_round(tmp_path):
    agent = lc(strength=2, rounds=3)
    agent.client = fake_client(json.dumps({"intercept": "pass", "low_confidence_phrases": []}))
    agent.execute(note_content=NOTE, loaded_files_block="### 🖼️ bp_photo.png\n（圖片已載入，見多模態訊息）", image_files=[image_entry(tmp_path)])
    assert len(agent.client.calls) == 2                                                     # two passes: both carry the picture
    for call in agent.client.calls:
        user = call["messages"][1]["content"]
        assert isinstance(user, list) and user[0]["type"] == "text" and "bp_photo.png" in user[0]["text"]
        assert [part["type"] for part in user[1:]] == ["image_url"] and user[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_without_a_readable_image_the_message_stays_plain_text(tmp_path):
    for files in (None, [], [{"name": "lab.md", "type": "text", "content": "HbA1c 7.3%"}],
                  [{"name": "gone.png", "type": "image", "path": str(tmp_path / "gone.png")}]):         # nothing, a text file, an image that is no longer there
        agent = lc(strength=1, rounds=1)
        agent.client = fake_client(json.dumps({"intercept": "pass", "low_confidence_phrases": []}))
        agent.execute(note_content=NOTE, image_files=files)
        assert isinstance(agent.client.calls[0]["messages"][1]["content"], str), files


def test_the_picture_goes_along_when_the_scan_asks_the_model_to_repair_its_answer(tmp_path):
    mixed = {"intercept": "detected", "low_confidence_phrases": [{"original_phrase": "否認胸痛", "reason": "r"}, {"reason": "沒有片段"}]}
    agent = lc(strength=1, rounds=1)
    agent.client = fake_client(json.dumps(mixed, ensure_ascii=False))
    agent.execute(note_content=NOTE, image_files=[image_entry(tmp_path)])
    assert len(agent.client.calls) == 3                                                     # asked, asked to repair, asked again
    for call in agent.client.calls:
        assert isinstance(call["messages"][1]["content"], list) and call["messages"][1]["content"][1]["type"] == "image_url"


def test_a_picture_is_said_to_be_attached_only_when_it_can_be(tmp_path):
    from Main_Agent import MainAgent
    agent = MainAgent(CONFIG, CONFIG, None, CONFIG, CONFIG)
    here = image_entry(tmp_path, "here.png")
    gone = {"name": "gone.png", "type": "image", "path": str(tmp_path / "gone.png")}
    nameless = {"name": "nameless.png", "type": "image"}                          # a restored entry without a path
    agent._loaded_files = [here, gone, nameless, {"name": "lab.md", "type": "text", "content": "HbA1c 7.3%"}]
    block = agent._format_loaded_files_block()
    assert "### 🖼️ here.png\n（圖片已載入，見多模態訊息）" in block
    for name in ("gone.png", "nameless.png"):
        assert f"### 🖼️ {name}\n（圖片檔目前讀取不到，沒有附上圖片本體，內容無法核對）" in block
    assert block.count("見多模態訊息") == 1 and "### 📄 lab.md\nHbA1c 7.3%" in block
    # the main agent's own prompt says the same thing, from the same place
    prompt = agent._build_context_prompt("看圖", "", "", None, "2026-10-17", [])
    assert "### 🖼️ gone.png\n（圖片檔目前讀取不到" in prompt and prompt.count("見多模態訊息") == 1


def test_the_scan_is_not_told_a_missing_picture_is_attached(tmp_path):
    from Main_Agent import MainAgent
    agent = MainAgent(CONFIG, CONFIG, None, CONFIG, CONFIG)
    agent._loaded_files = [{"name": "bp_photo.png", "type": "image", "path": str(tmp_path / "bp_photo.png")}]        # read once, gone since
    scan = lc(strength=1, rounds=1)
    scan.client = fake_client(json.dumps({"intercept": "pass", "low_confidence_phrases": []}))
    scan.execute(note_content=NOTE, loaded_files_block=agent._format_loaded_files_block(), image_files=agent._loaded_files)
    user = scan.client.calls[0]["messages"][1]["content"]
    assert isinstance(user, str)                                                         # no picture went out ...
    assert "讀取不到，沒有附上圖片本體" in user and "見多模態訊息" not in user               # ... and the text does not say one did


def scan_message_for(tmp_path, entry):
    """What the scan receives when the main agent hands it this one picture: the stash text built by the main agent, and the files."""
    from Main_Agent import MainAgent
    agent = MainAgent(CONFIG, CONFIG, None, CONFIG, CONFIG)
    agent._loaded_files = [entry]
    scan = lc(strength=1, rounds=1)
    scan.client = fake_client(json.dumps({"intercept": "pass", "low_confidence_phrases": []}))
    scan.execute(note_content=NOTE, loaded_files_block=agent._format_loaded_files_block(), image_files=agent._loaded_files)
    return scan.client.calls[0]["messages"][1]["content"]


def test_a_picture_that_is_there_but_cannot_be_opened_is_not_said_to_be_attached(tmp_path, monkeypatch):
    import builtins
    import multimodal_utils
    entry = image_entry(tmp_path, "locked.png")
    real_open = builtins.open

    def refusing_open(file, *args, **kwargs):
        if str(file) == entry["path"]:
            raise PermissionError(13, "The process cannot access the file because it is being used by another process", entry["path"])
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(multimodal_utils, "open", refusing_open, raising=False)          # the module's own `open`: the check and the encoding
    assert multimodal_utils.image_is_readable(entry["path"]) is False
    user = scan_message_for(tmp_path, entry)
    assert isinstance(user, str) and "讀取不到，沒有附上圖片本體" in user and "見多模態訊息" not in user
    monkeypatch.undo()
    user = scan_message_for(tmp_path, entry)                                             # the same file, once it opens
    assert isinstance(user, list) and "見多模態訊息" in user[0]["text"] and user[1]["type"] == "image_url"


@pytest.mark.skipif(__import__("sys").platform != "win32", reason="an exclusive open is a Windows lock")
def test_a_picture_another_program_holds_open_exclusively_is_not_said_to_be_attached(tmp_path):
    import ctypes
    from ctypes import wintypes
    import multimodal_utils
    entry = image_entry(tmp_path, "held.png")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    generic_read, no_sharing, open_existing, normal = 0x80000000, 0, 3, 0x80
    handle = kernel32.CreateFileW(entry["path"], generic_read, no_sharing, None, open_existing, normal, None)
    assert handle not in (None, wintypes.HANDLE(-1).value), ctypes.get_last_error()
    try:
        assert os.access(entry["path"], os.R_OK) is True                                 # what the old check looked at: it says yes
        assert multimodal_utils.image_is_readable(entry["path"]) is False
        user = scan_message_for(tmp_path, entry)
        assert isinstance(user, str) and "讀取不到，沒有附上圖片本體" in user and "見多模態訊息" not in user
    finally:
        kernel32.CloseHandle(handle)
    assert multimodal_utils.image_is_readable(entry["path"]) is True                    # released: it can be attached again
