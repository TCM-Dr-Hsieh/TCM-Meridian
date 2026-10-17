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
        assert "手上的逐字稿最後一段的結束時間" in text and "不是錄音機錄到的長度" in text and "`@錄音 00:00`" in text, name
        assert "那是醫師修改當下已錄下的音訊長度" not in text, name
        assert "這只是粗略的先後界線" in text and "不證明醫師看過或裁決過每一句話" in text, name       # the boundary promises no more than it knows
        assert "已能看到" not in text and "在畫面上" not in text and "當時還沒出現在畫面上" not in text, name
        assert "（版本 N 的逐字稿先後（以時間粗略估計）：…）" in text and "直接採用" in text, name
        assert text.index("**先看程式算好的先後**") < text.index("沒有這一行時才自己比"), name
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
        assert "步驟編號" in text and "第 N 主輪之後" in text and "尚無任何主輪" in text, name
        assert "必然晚於版本歷史中所有人工修改" in text, name
        assert "只依問診紀錄與檔案取得的先後判斷" not in text, name


def test_the_reviewing_prompts_make_the_model_compare_the_source_step_with_the_edit_position():
    for name in ("record", "hallucination", "lc"):
        text = read(name)
        assert "版本 N 的先後" in text, name
        assert "逐一檢查" in text and "X > N 者晚於這次修改" in text and "先比 X，X 相同再比 Y" in text and "寫在你的 thinking 裡" in text, name


def test_the_review_contract_fixes_are_in_the_prompts():
    three = ("record", "hallucination", "lc")
    for name in three:
        text = read(name)
        assert "不加來源標籤，也不得為了湊標籤而虛構標籤" in text, name                    # process-type unknowns (a)(d)
        assert "程度或頻率的描述（稍微、輕微、偶有、有時）不算不確定" in text, name      # degree words are not uncertainty
        assert "「稍微口乾」、「偶有胸悶」、「疑為" not in text, name
    record = read("record")
    assert "### 更新範圍：局部更新與全面整併" in record and "notes_for_main" in record
    assert "範圍以人類醫師本人的話為準，不以主治醫師的方針為準" in record and "不採信" in record      # the writer stays the guard against a narrowing guideline
    assert "### 既有的低信心警示" in record and "不需要、也不可替它加來源標籤" in record
    hallucination = read("hallucination")
    assert "### 重要：醫師限定的更新範圍決定遺漏檢查的範圍" in hallucination and "【本次更新的改動】" in hallucination
    assert "範圍以人類醫師本人的話為準，不以主治醫師的方針為準" in hallucination and "【本次登載方針】" not in hallucination
    assert "判斷看**事實內容**，不是看整行有沒有出現在 `+` 行" in hallucination and "忠實揭示的不確定性" in hallucination
    assert "逐個比對數值與結論" in hallucination and "數值或結論不同就是改寫" in hallucination
    assert "而且沒有出現在【本次更新的改動】" not in hallucination and "引用與數值都沒變的既有事實，先放行" in hallucination and "新增或改寫的，不放行" in hallucination
    assert "離改動處較遠的內容不會出現，代表這次沒動" in hallucination and "先看它有沒有出現在【本次更新的改動】的 `+` 行" in hallucination
    assert "### 重要：醫師限定的更新範圍決定遺漏判定的範圍" in read("lc") and "本輪病歷更新的登載方針" not in read("lc")
    lc = read("lc")
    assert "你不會收到討論區內容" in lc and "`occurrence`（選填，整數）" in lc and "最近的欄位標題" in lc
    assert "不得提醒補回" in read("nr")
    main = read("main")
    assert "局部更新：只更新血壓，其餘內容不動" in main and "低信心標註未完成" in main
    assert "不要自行縮小範圍" in main and "方針自己寫成的局部更新不會被採信" in main
    assert "稍微口乾" not in main
    assert "方針範圍與症狀追問的分工" in read("prompt_information_collection_subagent.txt")
