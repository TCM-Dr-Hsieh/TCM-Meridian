"""Units and sentences: the cuts, the merging of short fragments, and the incremental stream against the one-shot build."""
import random

import numpy as np
import pytest

from voice.speaker.mathutil import kmeans, tied_two_component, unit_rows
from voice.speaker.units import MAX_MERGED, MIN_UNIT, Token, UnitStream, build_units


def tok(start, end, *, seg=1, c0=0, c1=1, stop=False):
    return Token(seg, c0, c1, start, end, stop)


def spans(tokens, units):
    return [(tokens[u[0]].start, tokens[u[-1]].end) for u in units]


def test_a_unit_ends_after_about_eight_tenths_of_a_second():
    tokens = [tok(i * 0.25, i * 0.25 + 0.25) for i in range(12)]          # continuous speech, 3 s
    assert build_units(tokens) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]


def test_a_pause_or_sentence_mark_ends_the_unit():
    tokens = [tok(0.0, 0.5), tok(0.5, 0.75), tok(1.25, 1.75), tok(1.75, 2.0, stop=True), tok(2.0, 2.5)]
    # a 0.5 s pause separates [0, 1] from [2, 3]; the sentence mark closes [2, 3]; [4] is a new 0.5 s unit
    assert build_units(tokens) == [[0, 1], [2, 3], [4]]


def test_a_short_fragment_joins_the_previous_unit_in_the_same_sentence():
    tokens = [tok(0.0, 0.9), tok(0.9, 1.2)]
    assert build_units(tokens) == [[0, 1]]                                 # 0.3 s alone is too short; 1.2 s together is fine


def test_a_short_fragment_that_would_make_the_unit_too_long_is_dropped():
    tokens = [tok(0.0, 1.2), tok(1.2, 1.5)]
    assert build_units(tokens) == [[0]]


def test_a_short_fragment_joins_the_next_unit_when_it_cannot_join_the_previous():
    tokens = [tok(0.0, 0.9, stop=True), tok(0.9, 1.1), tok(1.1, 1.6)]
    assert build_units(tokens) == [[0], [1, 2]]


def test_a_fragment_across_a_sentence_end_is_dropped_not_merged():
    tokens = [tok(0.0, 0.9, stop=True), tok(1.0, 1.2)]
    assert build_units(tokens) == [[0]]                                    # [1] is 0.2 s, in another sentence, nothing to join


def test_a_lone_fragment_with_a_gap_is_dropped():
    tokens = [tok(0.0, 0.9), tok(2.0, 2.2), tok(5.0, 5.9)]
    assert build_units(tokens) == [[0], [2]]


def test_merged_units_never_exceed_the_cap():
    rng = random.Random(3)
    for _ in range(200):
        tokens, t = [], 0.0
        for _ in range(rng.randint(1, 30)):
            gap = rng.choice([0.0, 0.0, 0.1, 0.25, 0.6])
            length = rng.choice([0.1, 0.2, 0.3, 0.5, 0.9])
            tokens.append(tok(t + gap, t + gap + length, stop=rng.random() < 0.15))
            t += gap + length
        for unit in build_units(tokens):
            start, end = tokens[unit[0]].start, tokens[unit[-1]].end
            assert end - start >= MIN_UNIT - 1e-9
            assert unit == list(range(unit[0], unit[-1] + 1))              # contiguous
            assert end - start <= MAX_MERGED + 0.9 + 1e-9                  # a 0.9 s unit absorbing a fragment stays near the cap


def random_tokens(rng):
    tokens, t = [], 0.0
    for _ in range(rng.randint(0, 60)):
        gap = rng.choice([0.0, 0.0, 0.0, 0.1, 0.25, 0.35, 0.6, 2.0])
        length = rng.choice([0.05, 0.1, 0.2, 0.3, 0.5, 0.9])
        tokens.append(tok(t + gap, t + gap + length, stop=rng.random() < 0.12))
        t += gap + length
    return tokens


def test_the_stream_gives_the_same_units_as_the_one_shot_build():
    rng = random.Random(11)
    for _ in range(400):
        tokens = random_tokens(rng)
        stream, got = UnitStream(), []
        i = 0
        while i < len(tokens):
            step = rng.randint(1, 7)
            got += stream.add(tokens[i:i + step])
            i += step
        got += stream.finish()
        assert got == build_units(tokens)


def test_the_stream_never_changes_a_unit_it_has_already_emitted():
    rng = random.Random(5)
    for _ in range(200):
        tokens = random_tokens(rng)
        stream, emitted = UnitStream(), []
        for i in range(0, len(tokens), 3):
            emitted += stream.add(tokens[i:i + 3])
            assert emitted == build_units(tokens[:i + 3])[:len(emitted)]


def test_kmeans_finds_two_separated_clouds_and_a_warm_start_keeps_the_numbering():
    rng = np.random.default_rng(0)
    a = unit_rows(rng.normal(size=(40, 16)) * 0.2 + np.eye(16)[0])
    b = unit_rows(rng.normal(size=(30, 16)) * 0.2 + np.eye(16)[1])
    points, weights = np.vstack([a, b]), np.ones(70)
    centres, labels, _ = kmeans(points, weights, 2)
    assert len(set(labels[:40])) == 1 and len(set(labels[40:])) == 1 and labels[0] != labels[-1]
    swapped, again, _ = kmeans(points, weights, 2, init=centres[::-1])
    assert (again == 1 - labels).all()                                     # the first start stays group 0
    assert np.allclose(swapped[0], centres[1], atol=1e-6)


def test_kmeans_needs_at_least_k_points():
    with pytest.raises(ValueError):
        kmeans(np.ones((1, 4)), np.ones(1), 2)


def test_tied_mixture_recovers_two_means_and_a_shared_variance():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(-1.0, 0.5, 300), rng.normal(1.5, 0.5, 200)])
    low, high, var = tied_two_component(x)
    assert low == pytest.approx(-1.0, abs=0.15) and high == pytest.approx(1.5, abs=0.15)
    assert var == pytest.approx(0.25, abs=0.06)


def test_tied_mixture_survives_degenerate_input():
    low, high, var = tied_two_component([0.5] * 10)
    assert low == pytest.approx(0.5) and high == pytest.approx(0.5) and var > 0
