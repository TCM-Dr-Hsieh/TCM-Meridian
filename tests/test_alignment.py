"""Mapping the aligner's units onto the ASR text: what is accepted, what stops the window (and so makes a gap)."""
import pytest

from voice.alignment import AlignmentError, map_alignment, select_new_window, select_uncommitted


def item(text, start, end):
    return {'text': text, 'start': start, 'end': end}


def test_mixed_language_offsets_and_punctuation_are_preserved():
    text = '用 API，做 C++。'
    units = map_alignment(text, [item('用', 0, .2), item('API', .2, .5),
                                item('做', .6, .8), item('C', .8, 1)], 4, 2)
    added, pending, selected, _ = select_uncommitted(text, units, -1)
    assert added == text and not pending
    assert selected[0].start == 4 and selected[-1].end == 5
    assert select_uncommitted(text, units, 4.5)[0] == '做 C++。'


@pytest.mark.parametrize('items', [[], [item('假', 0, 1)], [item('你', 0, 1)],
                                   [item('你', 0, 1), item('好', .5, .8)],
                                   [item('你好', -1, 1)], [item('你好', 0, 10)], [item('你好', 0, float('nan'))],
                                   [item('你好', 1.5, 1.0)]])                      # (the last: ends before it starts)
def test_bad_alignment_cannot_silently_drop_words(items):
    with pytest.raises(AlignmentError):
        map_alignment('你好', items, 0, 2)


def test_out_of_range_alignment_reports_the_offending_item():
    with pytest.raises(AlignmentError, match=r'第 2 項.*音訊長度 2 秒'):
        map_alignment('你好', [item('你', 0, .2), item('好', .3, 3)], 0, 2)


def test_repeated_words_at_different_times_are_retained():
    units = map_alignment('好好', [item('好', 0, .4), item('好', .6, 1)], 3, 2)
    added, _, selected, _ = select_uncommitted('好好', units, 3.4, previous_unit=units[0])
    assert added == '好' and selected[0].start == 3.6


def test_small_alignment_drift_matches_same_word_not_next_repetition():
    old = map_alignment('好', [item('好', 0, .5)], 3, 2)[0]
    new = map_alignment('好好', [item('好', 0, .56), item('好', .6, 1)], 3, 2)
    assert select_uncommitted('好好', new, 3.5, previous_unit=old)[0] == '好'


def test_long_window_defers_right_context():
    text = '甲乙丙丁'
    units = map_alignment(text, [item(c, i, i + .8) for i, c in enumerate(text)], 8, 4)
    added, pending, selected, _ = select_uncommitted(text, units, 8.8, 10)
    assert added == '乙' and pending == '丙丁'
    assert selected[-1].end == 9.8


# ---- a word cut off by the end of the window (the aligner stretches it past the audio)

def test_a_word_cut_off_by_the_end_of_the_window_is_kept_with_its_times_held_at_the_end():
    # the shape of the windows that stopped real recordings: the last word's end 0.52 s (and its start 0.12 s) past the audio
    words = [item('Isolation', 8.0, 8.4), item('and', 8.4, 8.88), item('call', 8.88, 9.12), item('system', 9.12, 9.52)]
    units = map_alignment('Isolation and call system', words, 51, 9)
    assert [u.text for u in units] == ['Isolation', 'and', 'call', 'system']
    assert units[-1].start == 60.0 and units[-1].end == 60.0                           # not past the end of the window (51 + 9)
    assert units[-2].end == 60.0 and units[-3].end == pytest.approx(59.88)             # the others are as the aligner gave them
    cut = map_alignment('在 Kubernetes', [item('在', 8.0, 8.2), item('Kubernetes', 8.8, 9.28)], 108, 9)
    assert cut[-1].start == pytest.approx(116.8) and cut[-1].end == 117.0
    added, selected = select_new_window('在 Kubernetes', cut, 114.0)
    assert added == '在 Kubernetes' and len(selected) == 2                           # no word is lost


def test_no_unit_ends_before_it_starts_when_a_word_starts_past_the_end_of_the_audio():
    """A start 0.04 s past the end of the audio used to be kept as it was while the end was held at the audio's end: a unit
    that ended before it began (seen in a recorded visit: 330.04 → 330.0)."""
    units = map_alignment('甲乙', [item('甲', 1.0, 1.9), item('乙', 2.04, 2.3)], 328, 2)
    assert (units[-1].start, units[-1].end) == (330.0, 330.0)
    assert all(u.start <= u.end for u in units)


@pytest.mark.parametrize('items', [
    [item('你', 0, .2), item('好', .3, 2.9)],                                          # the end 0.9 s past the audio
    [item('你', 0, .2), item('好', 2.9, 3.0)],                                         # a word that starts 0.9 s past it
], ids=['end too far out', 'start too far out'])
def test_times_further_out_than_the_edge_allowance_still_stop_the_window(items):
    with pytest.raises(AlignmentError, match='音訊長度 2 秒'):
        map_alignment('你好', items, 0, 2)


def test_only_the_last_word_may_run_further_past_the_end_than_the_usual_allowance():
    for first, second in [(item('甲', 1.8, 2.3), item('乙', 2.3, 2.4)),             # a word in the middle 0.3 s past the audio: the times are broken
                          (item('甲', 1.0, 2.25), item('乙', 2.25, 2.3))]:
        with pytest.raises(AlignmentError, match=r'第 1 項.*音訊長度 2 秒'):
            map_alignment('甲乙', [first, second], 0, 2)
    units = map_alignment('甲乙', [item('甲', 1.8, 2.15), item('乙', 2.15, 2.6)], 0, 2)      # 0.15 s for the first, 0.6 s for the cut last word
    assert [(u.start, u.end) for u in units] == [(1.8, 2.0), (2.0, 2.0)]


def test_a_last_word_that_lies_wholly_outside_the_audio_is_not_a_cut_off_word():
    with pytest.raises(AlignmentError, match=r'第 2 項.*音訊長度 2 秒'):              # 2.7-2.8 s of 2 s: it never started inside the audio
        map_alignment('甲乙', [item('甲', 1.0, 1.9), item('乙', 2.7, 2.8)], 0, 2)
    units = map_alignment('甲乙', [item('甲', 1.0, 1.9), item('乙', 2.15, 2.6)], 0, 2)      # it starts 0.15 s past the end: still a cut word
    assert units[-1].start == units[-1].end == 2.0
    with pytest.raises(AlignmentError):                                              # 0.25 s past the end, not 0.2
        map_alignment('甲乙', [item('甲', 1.0, 1.9), item('乙', 2.25, 2.4)], 0, 2)


def test_the_edge_allowance_does_not_excuse_times_that_go_backwards():
    with pytest.raises(AlignmentError, match='前一項'):
        map_alignment('你好', [item('你', 1.9, 2.4), item('好', 1.8, 2.3)], 0, 2)


def test_the_edge_allowance_is_for_the_last_word_only_a_single_word_window_is_its_own_last_word():
    units = map_alignment('甲', [item('甲', 1.5, 2.7)], 0, 2)                         # the end 0.7 s past the audio
    assert (units[0].start, units[0].end) == (1.5, 2.0)
    with pytest.raises(AlignmentError):
        map_alignment('甲', [item('甲', 1.5, 2.9)], 0, 2)                             # 0.9 s
