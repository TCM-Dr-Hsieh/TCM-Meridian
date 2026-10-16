"""The agent prompts as text: they must survive the way the agents fill them in, and carry what the voice transcript, the
(d) status and the doctor-edit principles added. Read from the files, as the agents do."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = {
    "main": "prompt_main_agent.txt",
    "record": "prompt_record_update.txt",
    "hallucination": "prompt_hallucination_check.txt",
    "lc": "prompt_low_confidence_check.txt",
    "nr": "prompt_note_review.txt",
}


def read(name: str) -> str:
    return (ROOT / PROMPTS.get(name, name)).read_text(encoding="utf-8")


def test_the_low_confidence_prompt_goes_through_str_format_and_keeps_its_literal_braces():
    text = read("lc")
    filled = text.format(last_visit_block="LAST", history_summary="HIST")
    assert "LAST" in filled and "HIST" in filled and "{last_visit_block}" not in filled
    assert "[問診紀錄_患者R{N}]" in filled and "[Forum_D{N}]" in filled         # {{N}} came out as {N}
    assert "{{" not in filled


@pytest.mark.parametrize("name,known", [
    ("record", {"record_template", "last_visit_block", "history_summary"}),
    ("hallucination", {"last_visit_block", "history_summary"}),
    ("nr", {"record_template"}),
    ("main", {"physician_preferences", "professor_list", "record_template", "last_visit_block", "history_summary"}),
])
def test_prompts_filled_by_replace_hold_no_placeholder_nobody_fills(name, known):
    # the agents fill these with str.replace on exact "{name}" tokens: a stray {thing} would reach the model as is
    found = set(re.findall(r"(?<![{])\{([a-z_]+)\}(?![}])", read(name)))
    assert found <= known, found - known


def test_every_agent_that_reads_the_transcript_is_told_about_it_and_the_interview_and_professors_are_not():
    for name in ("main", "record", "hallucination", "lc", "nr"):
        assert "診間對話逐字稿" in read(name), name
    for name in ("record", "hallucination", "lc"):
        text = read(name)
        assert "[語音#N]" in text and "[語音#N-M]" in text and "[語音#N, #M]" in text, name
        assert "11 種" in text or name == "record", name
    assert "逐字稿" not in read("prompt_information_collection_subagent.txt")
    for professor_prompt in ROOT.glob("professor*/**/*.txt"):
        assert "診間對話逐字稿" not in professor_prompt.read_text(encoding="utf-8", errors="ignore"), professor_prompt


def test_the_exam_findings_status_d_is_in_the_three_copies_and_briefly_in_the_main_prompt():
    for name in ("record", "hallucination", "lc"):
        text = read(name)
        assert "(d) 檢查所見未口述" in text, name
        assert "未知（未口述，待醫師輸入）" in text and "未知（尚未輸入，待醫師輸入）" in text, name
        assert "不要寫「尚未檢查」「尚未評估」" in text or "尚未檢查" in text, name
    main = read("main")
    assert "(d) 檢查所見未口述" in main and "未知（未口述，待醫師輸入）" in main and "未知(尚未檢查)" not in main


def test_the_doctor_edit_principles_replace_the_old_keep_the_old_source_beside_it_rule():
    for name in ("record", "hallucination", "lc"):
        text = read(name)
        assert "醫師修正與資訊衝突處理原則" in text, name
        assert "被該次修正取代或明確否定的舊資訊" in text and "@錄音" in text, name
        assert "HbA1c 7.8%[人類醫師_手動修改]；另有紀錄 HbA1c 7.1%" not in text, name
    assert "醫師修正與資訊衝突處理原則" in read("main")


def test_low_confidence_has_the_weak_reminder_for_a_doctor_value_that_differs_from_objective_data_and_the_reviewers_do_not():
    lc = read("lc")
    assert "醫師修正值與 X 不同，請確認" in lc and "客觀來源" in lc
    assert "醫師修正值與 X 不同" not in read("hallucination") and "醫師修正值與 X 不同" not in read("record")


def test_note_review_hands_the_tongue_and_pulse_to_the_doctor_only_when_there_is_a_transcript():
    nr = read("nr")
    assert "沒有【診間對話逐字稿】時" in nr and "有【診間對話逐字稿】時" in nr
    assert "不得列入補問清單" in nr and "請醫師口述或手動輸入" in nr
    assert "請實習醫師執行" in nr                                     # the old rule is still there for sessions without a transcript


def test_the_speaker_marks_and_the_gap_and_still_correcting_rules_are_explained_where_the_transcript_is_used():
    for name in ("record", "hallucination", "lc"):
        text = read(name)
        assert "說話者標記由程式自動判斷" in text and "醫師:" in text and "患者或家屬:" in text and "不明:" in text, name
        assert "整批相反" in text and "仍在校稿中" in text, name
    assert "【音訊可能缺漏】" in read("record") and "【音訊可能缺漏】" in read("hallucination")


def test_the_three_reviewing_prompts_order_edits_against_interviews_and_files_by_main_turn():
    for name in ("record", "hallucination", "lc"):
        text = read(name)
        assert "主輪編號" in text and "第 N 主輪之後" in text and "尚無任何主輪" in text, name
        assert "必然晚於版本歷史中所有人工修改" in text and "回合標頭沒有 `Turn` 標示" in text, name
        assert "只依問診紀錄與檔案取得的先後判斷" not in text, name


def test_the_reviewing_prompts_make_the_model_compare_the_source_turn_with_the_edit_turn():
    for name in ("record", "hallucination", "lc"):
        text = read(name)
        assert "逐一檢查" in text and "T > N 才是修改之後" in text and "寫在你的 thinking 裡" in text, name
