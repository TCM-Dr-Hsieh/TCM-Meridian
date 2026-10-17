"""The edit box of the record shows no source tags and no low-confidence marks; on save the program puts them back and takes away
the tags whose words are gone (`merge_human_edit`). Pure text logic."""
import pytest

from record_edit_tags import (
    HUMAN_EDIT_TAG as MANUAL_TAG,
    NoteEditError,
    editable_text,
    merge_human_edit,
    strip_source_tags,
    tag_human_edits,
    tokenize,
    typed_tags,
)

EXAMPLE = "胃口佳[語音#1]。輕微脹氣[語音#2]。大便1天2次，偏軟。[語音#3]"


def tags_of(merge):
    return [r.tag for r in merge.removed]


def test_the_edit_box_text_has_no_tags_and_an_unchanged_save_changes_nothing():
    clean = editable_text(EXAMPLE)
    assert clean == "胃口佳。輕微脹氣。大便1天2次，偏軟。"
    merge = merge_human_edit(EXAMPLE, clean)
    assert merge.text == EXAMPLE and merge.removed == ()


def test_every_tag_family_of_the_record_is_hidden_and_other_brackets_are_not():
    note = ("甲- 現病史：頭痛三天[問診紀錄_患者R1][問診紀錄_家屬R2]，口乾[問診紀錄_實習醫師R3][問診紀錄_病歷系統R4]。"
            "血壓130/80[人類醫師_T2][人類醫師_手動修改][上次病歷][歷史病歷][Forum_D4][2026-06-02-Lab_data]"
            "[語音#3-5][語音#7, #9][語音#11]")
    assert editable_text(note) == "甲- 現病史：頭痛三天，口乾。血壓130/80"
    # a bracket that is not a source tag stays: the physician can see and edit it
    assert editable_text("診斷：[疑似]感冒[語音#1]") == "診斷：[疑似]感冒"
    assert typed_tags("甲[語音#1]乙[語音#1][歷史病歷]") == ["[語音#1]", "[歷史病歷]"]
    assert strip_source_tags("甲 [語音#1]乙") == "甲乙"


def test_the_example_of_the_physician_the_tag_of_the_deleted_words_goes_the_others_stay():
    edited = editable_text(EXAMPLE).replace("。大便1天2次，偏軟", "，胃食道逆流，大便偏硬")
    assert edited == "胃口佳。輕微脹氣，胃食道逆流，大便偏硬。"
    merge = merge_human_edit(EXAMPLE, edited)
    # "大便" was typed again, but two letters are chance, not the old words surviving
    assert merge.text == "胃口佳[語音#1]。輕微脹氣[語音#2]，胃食道逆流，大便偏硬。" + MANUAL_TAG
    assert tags_of(merge) == ["[語音#3]"] and merge.removed[0].text == "。大便1天2次，偏軟。"


def test_unchanged_lines_keep_their_tags_exactly_and_only_the_changed_line_is_marked():
    old = "甲- 現病史：頭痛三天 [問診紀錄_患者R1]\n高血壓[歷史病歷]\n丙- 個人史：不抽菸[問診紀錄_患者R4]"
    edited = editable_text(old).replace("高血壓", "糖尿病")
    merge = merge_human_edit(old, edited)
    lines = merge.text.split("\n")
    assert lines[0] == "甲- 現病史：頭痛三天 [問診紀錄_患者R1]" and lines[2] == "丙- 個人史：不抽菸[問診紀錄_患者R4]"   # even the blank before the tag
    assert lines[1] == "糖尿病" + MANUAL_TAG
    assert tags_of(merge) == ["[歷史病歷]"]


def test_no_field_name_is_assumed_a_word_left_standing_keeps_the_tag_even_if_it_was_a_label():
    old = "辛.9- 大便：一日三次[問診紀錄_患者R8]"
    left = merge_human_edit(old, "辛.9- 大便：偏乾")
    assert left.removed == () and "[問診紀錄_患者R8]" in left.text
    assert tags_of(merge_human_edit("大便一日三次[問診紀錄_患者R8]", "偏乾")) == ["[問診紀錄_患者R8]"]
    kept = merge_human_edit("患者說：頭痛三天[語音#2]", "患者說：頭痛四天")
    assert kept.removed == () and kept.text == "患者說：頭痛四天[語音#2]" + MANUAL_TAG


def test_words_that_survive_keep_the_tag_even_when_a_part_is_rewritten():
    old = "大便1天2次，偏軟[語音#3]"
    merge = merge_human_edit(old, "大便1天3次，偏軟")
    assert merge.text == "大便1天3次，偏軟[語音#3]" + MANUAL_TAG and merge.removed == ()


def test_a_short_span_that_was_only_extended_keeps_its_tag():
    merge = merge_human_edit("口乾[語音#5]", "口乾多飲")
    assert merge.text == "口乾[語音#5]多飲" + MANUAL_TAG and merge.removed == ()      # the new words come after the tag


def test_text_typed_right_after_a_sentence_is_not_covered_by_that_sentences_tag():
    merge = merge_human_edit("胃口佳[語音#1]。", "胃口佳，嘔吐。")
    assert merge.text == "胃口佳[語音#1]，嘔吐。" + MANUAL_TAG and merge.removed == ()


def test_tags_written_one_after_another_cover_the_same_words_and_go_together():
    old = "大便偏乾[語音#1][問診紀錄_患者R2]。睡眠差[語音#3]"
    untouched = merge_human_edit(old, "大便偏乾。睡眠好")
    assert untouched.text.startswith("大便偏乾[語音#1][問診紀錄_患者R2]。") and tags_of(untouched) == ["[語音#3]"]
    gone = merge_human_edit(old, "食慾差。睡眠差")
    assert tags_of(gone) == ["[語音#1]", "[問診紀錄_患者R2]"] and "[語音#3]" in gone.text


def test_tags_with_no_words_between_them_to_begin_with_are_not_removed():
    merge = merge_human_edit("胃口佳[語音#1]。[語音#2]", "胃口差。")
    assert tags_of(merge) == ["[語音#1]"] and "[語音#2]" in merge.text


def test_a_deleted_line_takes_its_tags_along_and_they_are_reported():
    old = "甲- 現病史：頭痛三天[語音#1]\n乙- 個人史：不抽菸[問診紀錄_患者R4]"
    merge = merge_human_edit(old, "甲- 現病史：頭痛三天")
    assert merge.text == "甲- 現病史：頭痛三天[語音#1]"
    assert [(r.tag, r.text) for r in merge.removed] == [("[問診紀錄_患者R4]", "乙- 個人史：不抽菸")]


def test_a_new_line_gets_the_manual_tag_and_blank_lines_stay_blank():
    merge = merge_human_edit("甲- 現病史：頭痛三天[語音#1]", "甲- 現病史：頭痛三天\n\n乙- 醫師加的")
    assert merge.text == "甲- 現病史：頭痛三天[語音#1]\n\n乙- 醫師加的" + MANUAL_TAG


def test_the_manual_tag_is_not_listed_as_removed_nor_stacked():
    merge = merge_human_edit("大便偏乾" + MANUAL_TAG, "大便偏硬，三天一次")
    assert merge.text == "大便偏硬，三天一次" + MANUAL_TAG and merge.removed == ()
    again = merge_human_edit(merge.text, editable_text(merge.text).replace("三天", "四天"))
    assert again.text.count(MANUAL_TAG) == 1


def test_a_tag_typed_or_pasted_into_the_edit_box_is_refused_not_stored_as_a_real_one():
    for typed in ("[語音#9]", "[歷史病歷]", MANUAL_TAG, "[語音#3-5]", "[語音#2, #4]", "[問診紀錄_患者R9]", "[Forum_D2]", "[2026-06-02-Lab_data]"):
        for text in ("胃口佳" + typed, typed + "胃口佳", "胃口佳 " + typed + "。"):
            with pytest.raises(NoteEditError) as error:
                merge_human_edit("胃口佳[語音#1]", text)
            assert typed in str(error.value)
    with pytest.raises(NoteEditError):
        merge_human_edit("", "新的一行[語音#1]")                         # also into an empty record
    assert merge_human_edit("胃口佳[語音#1]", "胃口佳[Lab]").text == "胃口佳[語音#1][Lab]" + MANUAL_TAG   # other brackets are plain text


def test_every_manual_tag_stays_even_when_the_words_before_them_are_gone():
    merge = merge_human_edit("舊內容" + MANUAL_TAG + "[語音#8]" + MANUAL_TAG, "新內容")
    assert merge.text == "新內容" + MANUAL_TAG + MANUAL_TAG          # nothing merged, nothing de-duplicated
    assert tags_of(merge) == ["[語音#8]"]
    both = merge_human_edit("甲" + MANUAL_TAG + "乙" + MANUAL_TAG, "甲丙")
    assert both.text.count(MANUAL_TAG) >= 2 and both.removed == ()


def test_windows_line_breaks_from_the_browser_do_not_change_the_lines():
    old = "甲- 現病史：頭痛三天[語音#1]\n乙- 個人史：不抽菸[問診紀錄_患者R4]"
    assert merge_human_edit(old, editable_text(old).replace("\n", "\r\n")).text == old


def test_the_whole_record_rewritten_loses_every_tag_and_empty_to_text_has_none_to_lose():
    old = "甲- 現病史：頭痛三天[語音#1]\n乙- 個人史：不抽菸[問診紀錄_患者R4]"
    merge = merge_human_edit(old, "完全不同的內容")
    assert tags_of(merge) == ["[語音#1]", "[問診紀錄_患者R4]"] and merge.text == "完全不同的內容" + MANUAL_TAG
    assert merge_human_edit("", "第一行").text == "第一行" + MANUAL_TAG


def test_tokenize_splits_text_and_tags():
    assert tokenize("甲[語音#1]乙[歷史病歷]" + MANUAL_TAG) == [
        ("甲", False), ("[語音#1]", True), ("乙", False), ("[歷史病歷]", True), (MANUAL_TAG, True)]
    assert tokenize("[Lab] 血壓") == [("[Lab] 血壓", False)]


def test_the_threshold_decides_how_many_left_over_letters_keep_a_tag():
    old = "大便偏乾，三天一次[語音#1]"
    edited = "大便偏乾，一日兩回"                                         # "大便偏乾" (4 letters) stays, the rest is rewritten
    assert merge_human_edit(old, edited).removed == ()                    # default 3: four left over keep the tag
    assert merge_human_edit(old, edited, 4).removed == ()
    assert tags_of(merge_human_edit(old, edited, 5)) == ["[語音#1]"]        # stricter: four are not enough
    again = editable_text(EXAMPLE).replace("。大便1天2次，偏軟", "，胃食道逆流，大便偏硬")
    assert tags_of(merge_human_edit(EXAMPLE, again)) == ["[語音#3]"]
    assert merge_human_edit(EXAMPLE, again, 1).removed == ()


def test_the_old_tag_function_still_appends_to_added_or_changed_lines_only():
    assert tag_human_edits("甲\n乙", "甲\n丙\n\n丁") == "甲\n丙" + MANUAL_TAG + "\n\n丁" + MANUAL_TAG
    assert tag_human_edits("甲", "甲") == "甲"


# ---------------------------------------------------------------------------------------------------------------------
# Low-confidence marks (**phrase（reason）**), the NOTE only
# ---------------------------------------------------------------------------------------------------------------------
MARKED = "甲- 現病史：**患者有高血壓（家屬R2未提及）**[問診紀錄_患者R1]。咳嗽三天[問診紀錄_患者R1]"


def test_the_edit_box_hides_the_marks_of_the_note_but_not_of_the_other_record():
    assert editable_text(MARKED, marks=True) == "甲- 現病史：患者有高血壓。咳嗽三天"
    assert editable_text(MARKED, marks=False) == "甲- 現病史：**患者有高血壓（家屬R2未提及）**。咳嗽三天"   # A&T: plain text


def test_an_unchanged_save_changes_nothing_with_marks():
    merge = merge_human_edit(MARKED, editable_text(MARKED, marks=True), marks=True)
    assert merge.text == MARKED and merge.removed == () and merge.dropped_marks == ()


def test_a_mark_comes_back_when_its_phrase_is_untouched_and_the_edit_is_elsewhere():
    old = "甲- 現病史：**患者有高血壓（家屬R2未提及）**[問診紀錄_患者R1]。咳嗽三天伴隨痰多[問診紀錄_患者R1]"
    merge = merge_human_edit(old, "甲- 現病史：患者有高血壓。咳嗽四天伴隨痰多", marks=True)
    assert merge.text == ("甲- 現病史：**患者有高血壓（家屬R2未提及）**[問診紀錄_患者R1]。咳嗽四天伴隨痰多[問診紀錄_患者R1]" + MANUAL_TAG)
    assert merge.dropped_marks == () and merge.removed == ()


def test_a_mark_is_dropped_when_its_phrase_is_edited_even_a_little():
    merge = merge_human_edit(MARKED, "甲- 現病史：患者有糖尿病。咳嗽三天", marks=True)
    assert "**" not in merge.text and "家屬R2未提及" not in merge.text
    assert [(d.phrase, d.reason) for d in merge.dropped_marks] == [("患者有高血壓", "家屬R2未提及")]
    assert "[問診紀錄_患者R4]" not in merge.text
    # words typed right after the phrase are not part of it: the mark stays on the phrase as it was
    extended = merge_human_edit(MARKED, "甲- 現病史：患者有高血壓病史。咳嗽三天", marks=True)
    assert extended.text.startswith("甲- 現病史：**患者有高血壓（家屬R2未提及）**[問診紀錄_患者R1]病史。") and extended.dropped_marks == ()


def test_a_mark_goes_with_a_deleted_line_and_is_reported():
    old = "甲- 現病史：**患者有高血壓（家屬R2未提及）**[問診紀錄_患者R1]\n乙- 個人史：不抽菸[問診紀錄_患者R4]"
    merge = merge_human_edit(old, "乙- 個人史：不抽菸", marks=True)
    assert merge.text == "乙- 個人史：不抽菸[問診紀錄_患者R4]"
    assert [d.phrase for d in merge.dropped_marks] == ["患者有高血壓"]


def test_marks_and_tags_keep_their_order_after_an_edit_before_them():
    old = "前文。**血壓偏高（來源不明）**[問診紀錄_患者R1]尾巴[語音#2]"
    merge = merge_human_edit(old, "新增前文。血壓偏高尾巴", marks=True)
    assert "**血壓偏高（來源不明）**[問診紀錄_患者R1]尾巴[語音#2]" in merge.text
    assert merge.dropped_marks == ()


def test_a_mark_with_parentheses_in_its_phrase_is_still_one_mark():
    old = "**血壓（收縮壓）130（數值無來源）**[問診紀錄_患者R1]"
    assert editable_text(old, marks=True) == "血壓（收縮壓）130"
    assert merge_human_edit(old, "血壓（收縮壓）130", marks=True).text == old


def test_bold_that_is_not_a_low_confidence_mark_is_ordinary_text():
    note = "**重點** 與 **粗體（）** 與 **no reason**"
    assert editable_text(note, marks=True) == "**重點** 與 粗體 與 **no reason**"        # an empty reason is still a mark
    assert merge_human_edit(note, editable_text(note, marks=True), marks=True).text == note


def test_a_bold_span_with_a_source_tag_inside_is_left_alone():
    note = "**患者有高血壓[語音#3]（原因）**"
    assert editable_text(note, marks=True) == "**患者有高血壓（原因）**"
    assert merge_human_edit(note, "**患者有高血壓（原因）**", marks=True).text == note


def test_a_reason_with_parentheses_of_its_own_is_still_one_mark():
    # the LC agent writes the model's reason as it is: it may hold full-width parentheses
    old = "甲- 現病史：**夜間加重（僅家屬轉述（患者未確認））**[問診紀錄_患者R1]。咳嗽三天"
    assert editable_text(old, marks=True) == "甲- 現病史：夜間加重。咳嗽三天"
    assert merge_human_edit(old, "甲- 現病史：夜間加重。咳嗽三天", marks=True).text == old                  # an unchanged save
    edited = merge_human_edit(old, "甲- 現病史：夜間不會加重。咳嗽三天", marks=True)
    assert [(d.phrase, d.reason) for d in edited.dropped_marks] == [("夜間加重", "僅家屬轉述（患者未確認）")]
    # parentheses in the phrase and in the reason together; the reason is the last pair that closes at the end
    assert editable_text("**血壓（坐姿）130（來源（R2）不明）**", marks=True) == "血壓（坐姿）130"
    # two marks in a row, the second with a nested reason
    assert editable_text("**口乾（未詢問）****胸悶（家屬說（未確認））**", marks=True) == "口乾胸悶"


def test_parentheses_that_do_not_pair_are_not_a_mark():
    for text in ("**夜間加重（原因）多）**", "**（只有原因）**", "**夜間加重（原因**"):
        assert editable_text(text, marks=True) == text, text
    # a bold span that is not a mark does not keep the next one from being found
    assert editable_text("**重點** 與 **口乾（未詢問（R1））**", marks=True) == "**重點** 與 口乾"

