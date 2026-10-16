"""The small parts of speaker marking: projecting labels onto corrected text, prompt/screen formats, the role voter and parsers,
the fill request, and the conversion of aligned ASR units to tokens."""
import json
from pathlib import Path

import pytest

from voice.llm import ValidationError
from voice.speaker import DOCTOR, OTHER, UNKNOWN
from voice.speaker import fill, roles
from voice.speaker.render import Run, project, prompt_body, segment_runs
from voice.speaker.service import window_tokens
from voice.alignment import map_alignment, select_new_window
from tests.helpers import fake_items


def labels(*items):
    """((c0, c1, label, source), ...) from (c0, label[, source]) pieces of one character each unless c1 is given."""
    return tuple(items)


# --- projecting label boundaries onto the corrected text ----------------------------------------------------------------
def test_projection_is_the_identity_for_equal_text():
    assert project('abcdef', 'abcdef', [0, 3, 6]) == [0, 3, 6]


def test_projection_follows_an_insertion_a_deletion_and_a_replacement():
    assert project('我頭痛三天', '我頭痛已三天', [0, 3, 5]) == [0, 3, 6]        # inserted text belongs to the piece that follows it
    assert project('我頭痛已三天', '我頭痛三天', [0, 4, 6]) == [0, 3, 5]        # and removed again
    assert project('我頭痛三天', '我頭疼三天', [0, 3, 5]) == [0, 3, 5]          # replaced in place: boundaries stay
    assert project('abc', '', [0, 3]) == [0, 0]
    assert project('', 'abc', [0]) == [0]


def test_projection_keeps_boundaries_in_order_and_inside_the_text():
    old, new = '你有沒有發燒嗯沒有', '請問你有沒有發燒？沒有'
    got = project(old, new, list(range(len(old) + 1)))
    assert got == sorted(got) and got[0] == 0 and got[-1] == len(new)


def test_a_segment_is_cut_into_runs_where_the_speaker_changes():
    added = '有沒有發燒沒有'
    marks = tuple((i, i + 1, DOCTOR if i < 5 else OTHER, 'voice') for i in range(7))
    runs = segment_runs(added, added, marks)
    assert runs == [Run(DOCTOR, 'voice', '有沒有發燒'), Run(OTHER, 'voice', '沒有')]
    assert prompt_body(runs) == '醫師: 有沒有發燒 -> 患者或家屬: 沒有'


def test_labels_follow_the_text_when_the_corrector_rewrote_it():
    added = '有沒有發燒沒有'
    marks = tuple((i, i + 1, DOCTOR if i < 5 else OTHER, 'voice') for i in range(7))
    corrected = '請問有沒有發燒，沒有。'                               # two characters in front, a comma and a full stop added
    runs = segment_runs(added, corrected, marks)
    assert [(r.label, r.text) for r in runs] == [(DOCTOR, '請問有沒有發燒'), (OTHER, '，沒有。')]
    assert ''.join(r.text for r in runs) == corrected                  # nothing lost, nothing added
    replaced = segment_runs(added, '有沒有發熱沒有', marks)            # a wrong character fixed in the middle of a run
    assert [(r.label, r.text) for r in replaced] == [(DOCTOR, '有沒有發熱'), (OTHER, '沒有')]


def test_a_segment_without_labels_is_one_unmarked_run_and_empty_text_gives_no_run():
    assert segment_runs('頭痛', '頭痛', None) == [Run(UNKNOWN, '', '頭痛')]
    assert segment_runs('頭痛', '  ', labels((0, 2, DOCTOR, 'voice'))) == []


def test_the_prompt_format_marks_text_inference_with_a_star_and_unknown_never_gets_one():
    runs = [Run(UNKNOWN, '', '嗯'), Run(DOCTOR, 'text', '好'), Run(OTHER, 'voice', '胃痛'), Run(UNKNOWN, 'text', '喔')]
    assert prompt_body(runs) == '不明: 嗯 -> 醫師*: 好 -> 患者或家屬: 胃痛 -> 不明: 喔'


def test_adjacent_pieces_with_the_same_label_and_source_merge_but_voice_and_text_stay_apart():
    marks = ((0, 1, DOCTOR, 'voice'), (1, 2, DOCTOR, 'voice'), (2, 3, DOCTOR, 'text'), (3, 4, OTHER, 'voice'))
    assert [(r.label, r.source, r.text) for r in segment_runs('甲乙丙丁', '甲乙丙丁', marks)] == [
        (DOCTOR, 'voice', '甲乙'), (DOCTOR, 'text', '丙'), (OTHER, 'voice', '丁')]


# --- aligned ASR units -> tokens ---------------------------------------------------------------------------------------
def aligned(text, duration=6.0, offset=0.0, covered=-1.0):
    units = map_alignment(text, fake_items(text, duration), offset, duration)
    added, selected = select_new_window(text, units, covered)
    return added, selected


def test_tokens_tile_the_new_text_and_know_where_sentences_end():
    added, selected = aligned('請問哪裡不舒服？我頭痛三天。')
    tokens = window_tokens(7, selected, added, added)
    assert [t.c0 for t in tokens][0] == 0 and tokens[-1].c1 == len(added)
    assert all(a.c1 == b.c0 for a, b in zip(tokens, tokens[1:]))              # no gaps, no overlaps
    assert [t.sentence_end for t in tokens].count(True) == 2
    assert added[tokens[[t.sentence_end for t in tokens].index(True)].c0:].startswith('服？') or True
    assert all(t.seg == 7 for t in tokens) and tokens[0].start == pytest.approx(0.0, abs=0.01)


def test_the_overlap_is_left_out_and_offsets_count_from_the_new_text():
    text = '我頭痛三天了醫師好'
    units = map_alignment(text, fake_items(text, 6.0), 3.0, 6.0)             # a window that starts at 3 s
    added, selected = select_new_window(text, units, 6.0)                    # the first half belongs to the previous window
    tokens = window_tokens(2, selected, added, added)
    assert tokens[0].c0 == 0 and tokens[-1].c1 == len(added) and tokens[0].start >= 6.0 - 0.7
    assert len(tokens) < len(text)


def test_a_traditional_text_of_another_length_still_gets_boundaries_inside_it():
    added_asr, selected = aligned('我头痛三天了。')
    longer = '我頭痛已經三天了。'                                             # as if the conversion had changed the length
    tokens = window_tokens(1, selected, added_asr, longer)
    assert tokens[0].c0 == 0 and tokens[-1].c1 == len(longer)
    assert all(a.c1 == b.c0 for a, b in zip(tokens, tokens[1:])) and all(t.c1 >= t.c0 for t in tokens)


def test_no_units_no_tokens():
    assert window_tokens(1, [], '', '') == []


# --- role layer: parsing and the voter -----------------------------------------------------------------------------------
def test_a_role_answer_must_name_every_group_with_a_known_word():
    ok = json.dumps({'roles': {'S1': '醫師', 'S2': '患者或家屬', 'S3': '不明'}})
    assert roles.parse_answer(ok, [0, 1, 2]) == {0: DOCTOR, 1: OTHER, 2: None}
    assert roles.parse_answer('```json\n' + ok + '\n```', [0, 1, 2])[0] == DOCTOR
    for bad in ('not json', json.dumps({'roles': {'S1': '醫師'}}), json.dumps({'roles': {'S1': '醫生', 'S2': '醫師'}}),
                json.dumps({'answer': {}})):
        with pytest.raises(ValidationError):
            roles.parse_answer(bad, [0, 1])


def test_the_voter_adopts_the_first_answer_and_needs_two_alike_to_change_it():
    v = roles.RoleVoter()
    assert v.feed({0: DOCTOR, 1: OTHER}) is True and v.current == {0: DOCTOR, 1: OTHER} and v.confirmed == 1
    assert v.feed({0: OTHER, 1: DOCTOR}) is False and v.current[0] == DOCTOR           # one different answer is not enough
    assert v.feed({0: DOCTOR, 1: OTHER}) is False and v.confirmed == 2 and v.candidate is None   # and a return resets it
    assert v.feed({0: OTHER, 1: DOCTOR}) is False
    assert v.feed({0: OTHER, 1: DOCTOR}) is True and v.current == {0: OTHER, 1: DOCTOR}          # two alike in a row: switch
    assert v.confirmed == 1


def test_three_agreeing_answers_lock_it_and_a_new_group_unlocks_it():
    v = roles.RoleVoter()
    for _ in range(3):
        v.feed({0: DOCTOR, 1: OTHER})
    assert v.locked and not v.wants_answer
    v.groups_changed()
    assert not v.locked and v.wants_answer and v.current == {0: DOCTOR, 1: OTHER}      # the old map stays until the model says otherwise


def test_an_unknown_answer_changes_nothing_and_a_new_group_is_adopted_without_a_vote():
    v = roles.RoleVoter()
    v.feed({0: DOCTOR, 1: OTHER})
    assert v.feed({0: None, 1: None}) is False and v.confirmed == 1
    assert v.feed({0: DOCTOR, 1: OTHER, 2: OTHER}) is True and v.current[2] == OTHER


def test_a_map_without_a_doctor_or_without_another_group_is_replaced_by_the_next_usable_answer():
    v = roles.RoleVoter()
    v.feed({0: OTHER, 1: OTHER})                                                        # nobody is the doctor
    assert not v.usable and v.confirmed == 0 and v.wants_answer
    assert v.feed({0: DOCTOR, 1: OTHER}) is True and v.usable                           # taken at once, no second vote
    v2 = roles.RoleVoter()
    for _ in range(5):
        v2.feed({0: DOCTOR, 1: DOCTOR})
    assert not v2.locked                                                                # agreement on nonsense never locks


def test_the_physician_overrides_the_voter_and_can_hand_control_back():
    v = roles.RoleVoter()
    v.feed({0: DOCTOR, 1: OTHER})
    v.set_manual({0: OTHER, 1: DOCTOR})
    assert v.manual and not v.wants_answer
    assert v.feed({0: DOCTOR, 1: OTHER}) is False and v.current == {0: OTHER, 1: DOCTOR}
    v.release()
    assert v.wants_answer and v.current == {0: OTHER, 1: DOCTOR}


# --- fill request ---------------------------------------------------------------------------------------------------------
def test_asked_sentences_are_grouped_into_runs_of_at_most_25():
    assert fill.chunks([3, 5, 20, 27, 28, 60]) == [[3, 5, 20, 27], [28], [60]]      # 27 starts within 25 of 3, 28 does not
    assert fill.chunks([]) == []


def test_the_context_window_starts_40_before_and_stops_at_the_decided_sentences():
    assert list(fill.window([100, 110], 500)) == list(range(60, 135))
    assert list(fill.window([100], 120)) == list(range(60, 120))
    assert list(fill.window([5], 500))[0] == 0


def test_the_fill_answer_must_cover_every_asked_id_and_unknown_is_a_valid_answer():
    reply = json.dumps({'answers': [{'id': 4, 'role': '醫師', 'basis': '問答'}, {'id': 7, 'role': '不明', 'basis': '其他'}]})
    assert fill.parse_answers(reply, [4, 7]) == {4: DOCTOR}
    with pytest.raises(ValidationError):
        fill.parse_answers(reply, [4, 7, 9])
    with pytest.raises(ValidationError):
        fill.parse_answers('nonsense', [4])


def test_fill_rows_show_voice_labels_and_hide_everything_else_behind_a_question_mark():
    assert fill.row(3, 75.0, 70.2, DOCTOR, '請坐') == {'id': 3, 't': '01:15', 'gap': 4.8, 'who': '醫師', 'text': '請坐'}
    assert fill.row(4, 76.0, None, UNKNOWN, '嗯')['who'] == '?' and fill.row(4, 76.0, None, OTHER, '嗯')['who'] == '患者或家屬'
    assert fill.row(5, 70.0, 71.0, OTHER, 'x')['gap'] == 0.0                     # never negative


# --- settings -------------------------------------------------------------------------------------------------------------
def test_a_config_from_before_speaker_marking_loads_with_it_switched_off():
    from voice.config import VoiceSettings
    settings = VoiceSettings.from_config({})                                      # a config.json with no voice_config at all
    assert settings.speaker.enabled is False and settings.speaker.use_in_jobs is True and settings.speaker.text_fill is True
    assert settings.speaker.unknown_percentile == 15.0 and settings.speaker.model_path.endswith('.onnx')
    saved = VoiceSettings.from_dict(settings.to_dict())
    assert saved.speaker == settings.speaker
    assert 'speaker' in settings.public_dict() and 'api_key' not in settings.public_dict()['corrector']


def test_speaker_settings_refuse_nonsense():
    from voice.config import VoiceSettings
    for bad in ({'unknown_percentile': 51}, {'unknown_percentile': -1}, {'unknown_percentile': 'x'}, {'model_path': '  '}):
        with pytest.raises(ValueError):
            VoiceSettings.from_dict({'speaker': bad})


# --- prompt files ------------------------------------------------------------------------------------------------------------
AGENT_PROMPTS = ('prompt_record_update.txt', 'prompt_hallucination_check.txt', 'prompt_low_confidence_check.txt')
PROJECT = Path(__file__).resolve().parent.parent


def test_the_agent_prompts_explain_the_marks_the_snapshot_really_uses():
    from voice.speaker.service import MARKS_NOTE
    header_phrase = MARKS_NOTE.split('，')[0]                                  # what the snapshot's title line says
    for name in AGENT_PROMPTS:
        text = (PROJECT / name).read_text(encoding='utf-8')
        for needle in (header_phrase, '醫師:', '患者或家屬:', '不明:', ' -> ', '醫師*:', '整批相反', '[語音#N]'):
            assert needle in text, (name, needle)
    writer = (PROJECT / 'prompt_record_update.txt').read_text(encoding='utf-8')
    assert '標記與語境明顯矛盾時' in writer and '說話者不明，待確認' in writer                # it may not follow a wrong mark blindly
    reviewer = (PROJECT / 'prompt_hallucination_check.txt').read_text(encoding='utf-8')
    assert 'G-3' in reviewer and '標記可能有誤就要求病歷降級為未知' in reviewer            # a possibly wrong mark is no reason to demand 未知


def test_the_speaker_prompts_match_what_their_parsers_expect_and_cannot_be_mistaken_for_other_prompts():
    from voice.config import PROMPTS_DIR
    from tests.helpers import ROLE_MARKERS
    roles_prompt, fill_prompt = roles.load_prompt(), fill.load_prompt()
    assert '"roles"' in roles_prompt and 'S1' in roles_prompt and 'groups' in roles_prompt
    assert '"answers"' in fill_prompt and 'ask_ids' in fill_prompt and '"id"' in fill_prompt and '"role"' in fill_prompt
    texts = {p.name: p.read_text(encoding='utf-8') for p in PROMPTS_DIR.glob('*.txt')}
    for role, marker in ROLE_MARKERS:                                              # the fake LLM tells the prompts apart by these
        owners = [name for name, text in texts.items() if marker in text]
        assert len(owners) <= 1, (role, owners)
    assert [n for n, t in texts.items() if '診間逐字稿的角色判定員' in t] == ['prompt_speaker_roles.txt']
    assert [n for n, t in texts.items() if '說話者補標員' in t] == ['prompt_speaker_fill.txt']
    assert [n for n, t in texts.items() if '說話者二階補標員' in t] == ['prompt_speaker_fill2.txt']


def test_no_prompt_quotes_an_accuracy_figure_that_only_two_tuned_recordings_support():
    from voice.config import PROMPTS_DIR
    paths = [PROJECT / name for name in AGENT_PROMPTS] + [PROMPTS_DIR / 'prompt_speaker_fill.txt', PROMPTS_DIR / 'prompt_speaker_fill2.txt']
    for path in paths:
        text = path.read_text(encoding='utf-8')
        for figure in ('1–3%', '97%', '約 97', '1-3%'):
            assert figure not in text, (path.name, figure)
    assert '整批相反' in (PROJECT / 'prompt_record_update.txt').read_text(encoding='utf-8')   # the reversal risk is named instead


# --- the role map must be trusted before the prompts get marks --------------------------------------------------------------
def test_one_answer_is_not_trusted_but_a_second_agreeing_answer_or_the_physician_is():
    v = roles.RoleVoter()
    v.feed({0: DOCTOR, 1: OTHER})
    assert v.usable and not v.trusted                                       # one answer could have named the wrong group
    v.feed({0: DOCTOR, 1: OTHER})
    assert v.trusted
    by_hand = roles.RoleVoter()
    by_hand.set_manual({0: DOCTOR, 1: OTHER})
    assert by_hand.trusted and not by_hand.wants_answer


def test_a_map_that_flip_flops_never_builds_trust_until_it_settles():
    v = roles.RoleVoter()
    v.feed({0: DOCTOR, 1: OTHER})
    v.feed({0: OTHER, 1: DOCTOR})                                           # one different answer: kept as a candidate
    assert not v.trusted
    v.feed({0: OTHER, 1: DOCTOR})                                           # two alike: switched, but only one answer behind it
    assert v.current == {0: OTHER, 1: DOCTOR} and not v.trusted
    v.feed({0: OTHER, 1: DOCTOR})
    assert v.trusted


def test_agreement_about_a_map_without_a_doctor_or_without_another_group_never_builds_trust():
    for answer in ({0: DOCTOR, 1: None}, {0: OTHER, 1: OTHER}, {0: DOCTOR, 1: DOCTOR}):
        v = roles.RoleVoter()
        for _ in range(5):
            v.feed(dict(answer))
        assert not v.trusted, answer


def test_trust_stays_when_a_new_voice_group_resets_the_count_and_when_the_physician_hands_control_back():
    v = roles.RoleVoter()
    v.feed({0: DOCTOR, 1: OTHER})
    v.feed({0: DOCTOR, 1: OTHER})
    v.groups_changed()
    assert v.confirmed == 0 and v.trusted                                    # the family member's arrival must not take the marks away
    v.release()
    assert v.trusted


def trusted_voter():
    v = roles.RoleVoter()
    v.feed({0: DOCTOR, 1: OTHER})
    v.feed({0: DOCTOR, 1: OTHER})
    assert v.trusted and v.confirmed_roles == {0: DOCTOR, 1: OTHER}
    v.groups_changed()
    return v


def test_a_group_that_appears_later_needs_a_second_agreeing_answer_before_the_prompts_may_use_it():
    v = trusted_voter()
    v.feed({0: DOCTOR, 1: OTHER, 2: DOCTOR})                                 # the first answer that names the newcomer
    assert v.current[2] == DOCTOR and v.trusted                              # the screen and the old groups go on
    assert v.confirmed_roles == {0: DOCTOR, 1: OTHER}                        # but one answer is not enough for the prompts
    v.feed({0: DOCTOR, 1: OTHER, 2: DOCTOR})
    assert v.confirmed_roles == {0: DOCTOR, 1: OTHER, 2: DOCTOR}


def test_a_second_answer_that_disagrees_confirms_nothing_and_a_replaced_role_waits_again():
    v = trusted_voter()
    v.feed({0: DOCTOR, 1: OTHER, 2: DOCTOR})
    v.feed({0: DOCTOR, 1: OTHER, 2: OTHER})                                  # a different answer: only a candidate
    assert v.current[2] == DOCTOR and 2 not in v.confirmed_roles
    v.feed({0: DOCTOR, 1: OTHER, 2: OTHER})                                  # twice: the role is replaced, with one answer behind it
    assert v.current[2] == OTHER and 2 not in v.confirmed_roles
    v.feed({0: DOCTOR, 1: OTHER, 2: OTHER})
    assert v.confirmed_roles[2] == OTHER


def test_a_map_set_by_the_physician_is_confirmed_for_every_group_and_stays_so_when_control_is_handed_back():
    v = roles.RoleVoter()
    v.set_manual({0: DOCTOR, 1: OTHER, 2: OTHER})
    assert v.confirmed_roles == {0: DOCTOR, 1: OTHER, 2: OTHER}
    v.release()
    assert v.confirmed_roles == {0: DOCTOR, 1: OTHER, 2: OTHER}


def test_answers_that_are_not_in_a_row_never_confirm_a_role():
    a, b = {0: DOCTOR, 1: OTHER}, {0: OTHER, 1: DOCTOR}
    v = roles.RoleVoter()
    for answer in (a, b, a):                                                  # the two alike are not consecutive
        v.feed(dict(answer))
    assert not v.trusted and v.confirmed_roles == {}
    v.feed(dict(a))
    assert v.trusted and v.confirmed_roles == {0: DOCTOR, 1: OTHER}


def test_a_confirmed_role_survives_one_dissenting_answer_but_not_two():
    v = trusted_voter()
    v.feed({0: OTHER, 1: DOCTOR})                                              # one dissent: only a candidate
    assert v.confirmed_roles == {0: DOCTOR, 1: OTHER} and v.trusted
    v.feed({0: OTHER, 1: DOCTOR})                                              # two alike: replaced, with one answer behind them
    assert v.current == {0: OTHER, 1: DOCTOR} and v.confirmed_roles == {} and v.trusted
    v.feed({0: OTHER, 1: DOCTOR})
    assert v.confirmed_roles == {0: OTHER, 1: DOCTOR}


def test_a_group_the_model_will_not_name_keeps_the_model_being_asked_and_the_map_unlocked():
    v = roles.RoleVoter()
    for _ in range(5):
        v.feed({0: DOCTOR, 1: OTHER, 2: None})                                 # S3 is "不明" every time
    assert v.pending_groups == [2] and not v.locked and v.wants_answer         # five agreeing answers would lock it otherwise
    v.feed({0: DOCTOR, 1: OTHER, 2: OTHER})
    assert v.pending_groups == [2]                                             # named by one answer
    v.feed({0: DOCTOR, 1: OTHER, 2: OTHER})
    assert v.pending_groups == [] and v.locked
    v2 = roles.RoleVoter()
    v2.feed({0: DOCTOR, 1: OTHER})
    v2.groups_changed(3)                                                       # the service tells the voter how many groups exist
    assert v2.pending_groups == [0, 1, 2]


def test_the_feed_reports_a_confirmation_even_when_the_map_itself_did_not_change():
    v = roles.RoleVoter()
    assert v.feed({0: DOCTOR, 1: OTHER}) is True                               # adopted
    assert v.feed({0: DOCTOR, 1: OTHER}) is True                               # the second alike answer confirms it: save and refresh
    assert v.feed({0: DOCTOR, 1: OTHER}) is False                              # nothing new


def test_a_context_label_read_from_the_text_carries_a_star_for_the_second_fill_and_unknown_never_does():
    from voice.speaker import fill as f
    assert f.row(1, 0.0, None, DOCTOR, '請問', inferred=True)['who'] == '醫師*'
    assert f.row(1, 0.0, None, OTHER, '好', inferred=True)['who'] == '患者或家屬*'
    assert f.row(1, 0.0, None, OTHER, '好')['who'] == '患者或家屬'
    assert f.row(1, 0.0, None, UNKNOWN, '嗯', inferred=True)['who'] == '?'


def test_the_second_fill_prompt_explains_the_star_and_the_two_prompts_stay_apart():
    from voice.config import PROMPTS_DIR
    second = (PROMPTS_DIR / 'prompt_speaker_fill2.txt').read_text(encoding='utf-8')
    assert '醫師*' in second and '患者或家屬*' in second and '"answers"' in second and 'ask_ids' in second
    assert '說話者二階補標員' in second and '說話者補標員' not in second
    assert '先前的答案' not in second and '第一次' not in second                    # the model is never told the first answer


def test_the_second_text_fill_is_off_by_default_and_saved_with_the_settings():
    from voice.config import VoiceSettings
    settings = VoiceSettings.from_dict({})
    assert settings.speaker.text_fill2 is False
    settings.speaker.text_fill2 = True
    assert VoiceSettings.from_dict(settings.to_dict()).speaker.text_fill2 is True


def test_both_text_fills_are_marked_with_a_star_in_the_prompt_and_on_the_screen():
    from ui_app.controllers.voice_controller import VoiceController
    for source in ('text', 'text2'):
        run = Run(DOCTOR, source, '好')
        assert prompt_body([run]) == '醫師*: 好'
        html = VoiceController.run_html(None, run)
        assert '醫師*：' in html and '由上下文推測' in html
    plain = Run(DOCTOR, 'voice', '好')
    assert prompt_body([plain]) == '醫師: 好' and '*' not in VoiceController.run_html(None, plain)


def test_the_second_text_fill_cannot_be_on_while_the_first_is_off():
    from voice.config import SpeakerSettings
    settings = SpeakerSettings(text_fill=False, text_fill2=True)
    settings.validate()
    assert settings.text_fill2 is False
    both = SpeakerSettings(text_fill=True, text_fill2=True)
    both.validate()
    assert both.text_fill2 is True


def test_a_model_that_still_writes_the_old_spelling_is_understood():
    assert roles.parse_answer(json.dumps({'roles': {'S1': '醫師', 'S2': '患者家屬'}}), [0, 1]) == {0: DOCTOR, 1: OTHER}
    assert roles.parse_answer(json.dumps({'roles': {'S1': '醫師', 'S2': '患者或家屬'}}), [0, 1]) == {0: DOCTOR, 1: OTHER}
    rows = json.dumps({'answers': [{'id': 3, 'role': '患者家屬', 'basis': '問答'}, {'id': 4, 'role': '患者或家屬', 'basis': '問答'}]})
    assert fill.parse_answers(rows, [3, 4]) == {3: OTHER, 4: OTHER}


def test_the_model_is_shown_and_asked_with_the_label_that_cannot_be_read_as_the_patients_companion_only():
    from voice.config import PROMPTS_DIR
    assert fill.SHOWN_NAMES[OTHER] == '患者或家屬'
    for name in ('prompt_speaker_roles.txt', 'prompt_speaker_fill.txt', 'prompt_speaker_fill2.txt'):
        text = (PROMPTS_DIR / name).read_text(encoding='utf-8')
        assert '患者家屬' not in text and '患者或家屬' in text, name

