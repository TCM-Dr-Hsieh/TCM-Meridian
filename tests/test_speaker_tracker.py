"""The speaker tracker on synthetic consultations: groups, roles, the per-sentence vote, growth, and what must NOT happen."""
import numpy as np
import pytest

from voice.speaker import DOCTOR, OTHER, UNKNOWN
from voice.speaker.mathutil import unit_rows
from voice.speaker.tracker import Params, Sentence, SpeakerTracker, Unit
from tests.speaker_fakes import FakeEmbedder, bases, build, feed, make_turns, window_of, windows


def session(pattern, *, similarity=0.0, noise=0.15, seed=1, lo=1.6, hi=5.5, speakers=None):
    turns = make_turns(pattern, seed=seed, lo=lo, hi=hi)
    audio, tokens, truth = build(turns)
    voices = bases(speakers or (max(pattern) + 1), similarity)
    tracker = SpeakerTracker(FakeEmbedder(voices, noise=noise, seed=seed))
    return tracker, audio, tokens, truth, voices


def doctor_group(tracker, voices):
    return int(np.argmax(tracker._centres @ voices[0]))


def named(tracker, voices):
    """Tell the tracker which group is the doctor (what the role layer does), from the truth the test knows."""
    g = doctor_group(tracker, voices)
    return tracker.set_roles({k: (DOCTOR if k == g else OTHER) for k in range(tracker.group_count)})


def shown(tracker, tokens):
    """The label the tracker shows for every token, in token order."""
    got = {}
    for seg in {t.seg for t in tokens}:
        labels = tracker.labels_for(seg)
        for c0, c1, label, source in labels or ():
            got[(seg, c0)] = (label, source)
    return [got.get((t.seg, t.c0), (UNKNOWN, '')) for t in tokens]


def scores(tracker, tokens, truth, *, after=60.0):
    labels = shown(tracker, tokens)
    rows = [(got[0], 'doctor' if s == 0 else 'other') for got, s, t in zip(labels, truth, tokens) if t.start >= after]
    labelled = [(a, b) for a, b in rows if a != UNKNOWN]
    wrong = sum(a != b for a, b in labelled)
    return len(labelled) / len(rows), wrong / max(1, len(labelled))


def alternating(n, third=None):
    return [0 if i % 2 == 0 else 1 for i in range(n)]


def test_nothing_is_labelled_until_the_doctor_group_is_named():
    tracker, audio, tokens, truth, voices = session(alternating(40))
    feed(tracker, audio, tokens)
    tracker.finish()
    assert tracker.group_count == 2
    assert all(label == UNKNOWN for label, _ in shown(tracker, tokens))
    assert tracker.stats()['labelled'] == 0


def test_two_speakers_are_labelled_with_few_mistakes_once_the_doctor_is_named():
    tracker, audio, tokens, truth, voices = session(alternating(60))
    feed(tracker, audio, tokens)
    assert tracker.group_count == 2
    named(tracker, voices)
    tracker.finish()
    coverage, wrong = scores(tracker, tokens, truth)
    assert coverage > 0.8
    assert wrong < 0.02


def test_the_first_minute_is_labelled_afterwards_when_the_groups_exist():
    tracker, audio, tokens, truth, voices = session(alternating(60))
    feed(tracker, audio, tokens, last=25)                       # about 80 s of audio: the first refit (60 s) has happened
    assert tracker.group_count == 2
    first_minute = [t for t in tokens if t.start < 50.0]
    assert all(label == UNKNOWN for label, _ in shown(tracker, first_minute))
    named(tracker, voices)
    labels = shown(tracker, first_minute)
    assert sum(label != UNKNOWN for label, _ in labels) > 0.6 * len(first_minute)
    wrong = sum(label != ('doctor' if s == 0 else 'other') for (label, _), s, t in zip(labels, truth, tokens) if label != UNKNOWN and t.start < 50.0)
    assert wrong <= 0.03 * len(first_minute)


def test_swapping_the_roles_relabels_everything_already_decided():
    tracker, audio, tokens, truth, voices = session(alternating(40))
    feed(tracker, audio, tokens)
    named(tracker, voices)
    before = shown(tracker, tokens)
    g = doctor_group(tracker, voices)
    tracker.set_roles({k: (OTHER if k == g else DOCTOR) for k in range(2)})
    after = shown(tracker, tokens)
    flipped = {DOCTOR: OTHER, OTHER: DOCTOR, UNKNOWN: UNKNOWN}
    assert [flipped[a[0]] for a in before] == [b[0] for b in after]


def test_one_voice_alone_never_makes_groups():
    tracker, audio, tokens, truth, voices = session([0] * 30, speakers=2)
    feed(tracker, audio, tokens)
    assert tracker.group_count == 0
    assert tracker.stats()['labelled'] == 0


def test_a_stray_second_voice_that_is_too_small_does_not_make_a_group():
    pattern = [0] * 30 + [1]
    tracker, audio, tokens, truth, voices = session(pattern)
    feed(tracker, audio, tokens)
    assert tracker.group_count == 0


def test_a_second_voice_that_arrives_later_makes_the_groups_then():
    pattern = [0] * 14 + [1, 0] * 20
    tracker, audio, tokens, truth, voices = session(pattern, lo=2.0, hi=4.0)
    first_other = next(t.start for t, s in zip(tokens, truth) if s == 1)
    feed(tracker, audio, tokens, last=int(first_other // 3) - 1)
    assert tracker.group_count == 0
    feed(tracker, audio, tokens, first=int(first_other // 3))
    assert tracker.group_count == 2


def test_two_speakers_never_make_a_third_group():
    for seed in range(1, 6):
        tracker, audio, tokens, truth, voices = session(alternating(80), seed=seed)
        feed(tracker, audio, tokens)
        tracker.finish()
        assert tracker.group_count == 2, seed
        assert not [e for e in tracker.events if e['event'] == 'speaker_group_added']


def test_a_third_voice_makes_a_third_group_and_is_not_taken_for_the_doctor():
    pattern = [0, 1] * 25 + [0, 1, 2] * 14
    tracker, audio, tokens, truth, voices = session(pattern, lo=1.8, hi=4.0)
    feed(tracker, audio, tokens)
    named(tracker, voices)
    tracker.finish()
    assert tracker.group_count == 3
    assert [e for e in tracker.events if e['event'] == 'speaker_group_added']
    third_start = next(t.start for t, s in zip(tokens, truth) if s == 2)
    late = [(t, s) for t, s in zip(tokens, truth) if t.start >= third_start + 90]
    labels = shown(tracker, [t for t, _ in late])
    as_doctor = sum(1 for (label, _), (_, s) in zip(labels, late) if s == 2 and label == DOCTOR)
    assert as_doctor <= 0.05 * sum(1 for _, s in late if s == 2)
    coverage, wrong = scores(tracker, tokens, [0 if s == 0 else 1 for s in truth], after=third_start + 90)
    assert wrong < 0.05


def newcomer_session(pattern, *, similarity, seed):
    """Feed window by window and name the doctor as soon as two groups exist (what the role layer does). Returns how many seconds
    after its first words the third voice got its own group, the share of its words shown as the doctor until then, the tracker."""
    tracker, audio, tokens, truth, voices = session(pattern, lo=1.8, hi=4.0, similarity=similarity, seed=seed)
    for k in range(windows(audio)):
        feed(tracker, audio, tokens, first=k, last=k)
        if tracker.group_count == 2 and not tracker.roles:
            named(tracker, voices)
    tracker.finish()
    start = next(t.start for t, s in zip(tokens, truth) if s == 2)
    added = [e['at'] for e in tracker.events if e['event'] == 'speaker_group_added']
    before = [label for (label, _), t, s in zip(shown(tracker, tokens), tokens, truth) if s == 2 and (not added or t.start < added[0])]
    return (added[0] - start if added else None), before.count(DOCTOR) / max(1, len(before)), tracker


NEWCOMERS = {'quiet (one turn in three)': [0, 1] * 25 + [0, 1, 2] * 14,
             'talkative (one turn in two)': [0, 1] * 20 + [2, 0, 2, 1] * 16,
             'late and quiet (after about 8 minutes)': [0, 1] * 80 + [0, 1, 2] * 10}


@pytest.mark.parametrize('similarity', [0.0, 1.0], ids=['distinct voices', 'voices at cosine 0.5'])
@pytest.mark.parametrize('kind', list(NEWCOMERS))
def test_a_newcomer_gets_its_own_group_within_a_couple_of_minutes_and_is_hardly_taken_for_the_doctor(kind, similarity):
    for seed in (1, 2, 3):
        delay, as_doctor, tracker = newcomer_session(NEWCOMERS[kind], similarity=similarity, seed=seed)
        assert delay is not None and delay <= 130, (seed, delay)
        assert as_doctor <= 0.2, (seed, as_doctor)
        assert tracker.group_count == 3


def test_two_speakers_with_similar_voices_do_not_make_a_third_group_either():
    for similarity in (1.0, 1.5):                                        # cosine 0.5 and 0.69 between the two voices
        for seed in range(1, 5):
            tracker, audio, tokens, truth, voices = session([0, 1, 0] * 55, similarity=similarity, noise=0.35, seed=seed)
            feed(tracker, audio, tokens)
            tracker.finish()
            assert tracker.group_count == 2, (similarity, seed)


def grow(groups, newcomer_ends, *, seconds=0.8, contaminated=0.0, horizon=600.0):
    """The minute's refit on hand-made voiceprints: `groups` known voices with 100 units each spread over the visit (24 s each in the
    last three minutes) and a newcomer with the given unit end times. `contaminated` pulls the first centre towards the newcomer,
    as the units absorbed into it would have done."""
    eye = np.eye(16)
    rng = np.random.default_rng(0)
    ends = [np.linspace(5.0, horizon, 100) for _ in range(groups)] + [np.asarray(newcomer_ends, dtype=float)]
    vectors = [eye[g] + rng.normal(0, 0.03, (len(e), 16)) for g, e in enumerate(ends)]
    seconds_each = [np.full(len(e), 0.8) for e in ends[:-1]] + [np.full(len(ends[-1]), seconds)]
    tracker = SpeakerTracker(FakeEmbedder(bases(1)))
    centres = eye[:groups].copy()
    centres[0] = centres[0] + contaminated * eye[groups]
    tracker._centres = unit_rows(centres)
    tracker._update_groups(unit_rows(np.vstack(vectors)), np.concatenate(seconds_each), np.concatenate(ends), horizon)
    return tracker


def test_a_further_group_needs_enough_units_a_share_of_the_recent_speech_and_a_free_slot():
    spoke_lately = np.linspace(560, 600, 12)
    tracker = grow(2, spoke_lately)                                     # 12 units, 9.6 s of the 58 s spoken in the last three minutes
    assert tracker.group_count == 3
    assert [e['event'] for e in tracker.events] == ['speaker_group_added'] and tracker.groups_version == 1
    assert grow(2, np.linspace(560, 600, 7), seconds=1.3).group_count == 2        # 7 units: too few, however long they are
    assert grow(2, np.linspace(560, 600, 10), seconds=0.4).group_count == 2       # 4 s of 52 s: under a tenth of the recent speech
    assert grow(3, np.linspace(540, 600, 20)).group_count == 4                    # 16 s of 88 s: a fourth group is allowed ...
    assert grow(4, np.linspace(540, 600, 20)).group_count == 4                    # ... a fifth is not (16 s of 112 s would pass the share)


def test_a_newcomer_counts_by_its_share_of_the_last_three_minutes_not_of_the_whole_visit():
    # 9.6 s are 5.7% of the 170 s of speech in the whole visit but 17% of the last three minutes; it spoke two minutes ago
    assert grow(2, np.linspace(440, 480, 12)).group_count == 3
    assert grow(2, np.linspace(150, 190, 12)).group_count == 2                    # long before the window: not a recent voice


def test_units_nobody_sounds_like_do_not_drag_a_centre_towards_them():
    # seven units of a voice that is far from both groups (too few to be a group): the centres must stay where the two voices are
    tracker = grow(2, np.linspace(560, 600, 7), seconds=1.3)
    assert tracker.group_count == 2
    eye = np.eye(16)
    assert float(tracker._centres[0] @ eye[0]) > 0.999 and float(tracker._centres[1] @ eye[1]) > 0.999


def decided_session(**kwargs):
    tracker, audio, tokens, truth, voices = session(alternating(80), **kwargs)
    feed(tracker, audio, tokens)
    named(tracker, voices)
    tracker.finish()
    return tracker


def test_an_unknown_sentence_is_scored_again_and_a_labelled_one_is_never_touched():
    tracker = decided_session()
    labelled = [i for i, s in enumerate(tracker.sentences) if s.gid is not None and s.source == 'voice' and abs(s.node) > 3]
    target, bystander = labelled[3], labelled[7]
    right = tracker.sentences[target].gid
    tracker.sentences[target].gid, tracker.sentences[target].source = None, ''                 # as if it had been left unknown
    tracker.sentences[target].lean = None
    wrong = 1 - tracker.sentences[bystander].gid
    tracker.sentences[bystander].gid = wrong                                                    # a (wrong) label stays as it is
    tracker._history = [0.5] * 30 + [100.0] * 70                           # today's threshold is the 15th percentile of this: 0.5
    labelled_before = sum(1 for s in tracker.sentences if s.gid is not None)
    changed = tracker._rescore()
    promoted = sum(1 for s in tracker.sentences if s.gid is not None) - labelled_before
    assert tracker.sentences[target].gid == right and tracker.sentences[target].source == 'voice'
    assert tracker.sentences[bystander].gid == wrong
    assert promoted >= 1 and tracker.rescored == promoted and tracker.stats()['rescored'] == promoted
    assert tracker._sentence_segments(tracker.sentences[target]) <= changed
    event = next(e for e in tracker.events if e['event'] == 'speaker_rescored')
    assert event['promoted'] == promoted


def test_a_sentence_that_stays_unknown_is_left_exactly_as_it_was_and_is_not_asked_again():
    tracker = decided_session()
    tracker._history = [1e9] * 20                                      # nothing can clear the threshold now
    strong = [i for i, s in enumerate(tracker.sentences) if s.gid is not None and s.source == 'voice' and abs(s.node) > 3]
    target = strong[2]
    s = tracker.sentences[target]
    s.gid, s.source, s.asked, s.lean, s.node = None, '', True, None, 0.0     # unknown, already given to the text fill once
    assert tracker._rescore() == set()
    assert s.gid is None and s.asked is True and s.lean is None and s.node == 0.0
    assert tracker.rescored == 0 and not [e for e in tracker.events if e['event'] == 'speaker_rescored']
    assert target not in tracker.fill_candidates(final=True)


def test_nothing_is_scored_again_while_there_is_no_doctor_group_to_compare_with():
    tracker = decided_session()
    strong = [i for i, s in enumerate(tracker.sentences) if s.gid is not None and s.source == 'voice' and abs(s.node) > 3]
    tracker.sentences[strong[1]].gid = None
    tracker._roles = {}
    assert tracker._rescore() == set() and tracker.rescored == 0


def test_the_unknown_count_includes_the_sentences_of_a_group_nobody_has_named():
    tracker = decided_session(noise=0.35, similarity=0.5, seed=2)
    before = tracker.stats()['unknown']
    doctor = next(g for g, r in tracker.roles.items() if r == DOCTOR)
    tracker.set_roles({doctor: DOCTOR})
    shown = sum(1 for s in tracker.sentences[:tracker._cursor] if tracker.role_of(s.gid) == UNKNOWN)
    stats = tracker.stats()
    assert stats['unknown'] == shown and shown > before
    assert stats['labelled'] + stats['unknown'] == stats['decided']
    assert tracker._unknown_share(0.0, 1e9) == round(shown / stats['decided'], 3)


def test_a_newcomers_first_sentences_get_its_group_once_it_exists():
    pattern = NEWCOMERS['quiet (one turn in three)']
    tracker, audio, tokens, truth, voices = session(pattern, lo=1.8, hi=4.0, seed=2)
    for k in range(windows(audio)):
        feed(tracker, audio, tokens, first=k, last=k)
        if tracker.group_count == 2 and not tracker.roles:
            named(tracker, voices)
        if tracker.group_count == 3:
            break                                                       # the visit is still going on: no finish() has run
    assert not tracker.finished
    added_at = next(e['at'] for e in tracker.events if e['event'] == 'speaker_group_added')
    groups = tracker.all_groups()
    early = [(t.seg, t.c0) for t, s in zip(tokens, truth) if s == 2 and t.start < added_at]
    carried = 0
    for seg, c0 in early:
        indexes = tracker.segment_token_indexes(seg)
        pos = next(p for p, i in enumerate(indexes) if tracker.tokens[i].c0 == c0)
        carried += groups[seg][pos] == 2
    assert len(early) > 50 and carried > 0.8 * len(early)             # they were unknown, nobody sounded like them yet
    assert tracker.rescored >= 3
    feed(tracker, audio, tokens, first=k + 1)
    tracker.finish()
    assert [e for e in tracker.events if e['event'] == 'speaker_rescored']


def test_a_new_group_leaves_the_old_centres_free_of_the_newcomer():
    tracker = grow(2, np.linspace(560, 600, 12), contaminated=0.5)               # the first centre has drifted towards the newcomer
    assert tracker.group_count == 3
    assert float(tracker._centres[0] @ np.eye(16)[0]) > 0.98


def test_a_decided_sentence_is_never_changed_by_later_audio_only_by_roles_or_fill():
    tracker, audio, tokens, truth, voices = session(alternating(80))
    feed(tracker, audio, tokens, last=40)
    named(tracker, voices)
    early = {seg: tracker.labels_for(seg) for seg in range(1, 25) if tracker.labels_for(seg)}
    decided_early = tracker._cursor
    feed(tracker, audio, tokens, first=41)
    for seg, labels in early.items():
        now = tracker.labels_for(seg)
        for old, new in zip(labels, now):
            if old[2] != UNKNOWN or tracker.sentences[tracker._sentence_of[tracker._seg_tokens[seg][0]]].decided is False:
                assert old == new
    assert tracker._cursor >= decided_early


def test_similar_voices_are_still_separated_with_unknown_instead_of_mistakes():
    tracker, audio, tokens, truth, voices = session(alternating(100), similarity=0.75, noise=0.12)
    feed(tracker, audio, tokens)
    named(tracker, voices)
    tracker.finish()
    coverage, wrong = scores(tracker, tokens, truth)
    assert wrong < 0.06
    assert coverage > 0.5


def test_a_weak_sentence_is_left_unknown_and_the_percentile_controls_how_many():
    unknown = {}
    for percentile in (0.0, 15.0, 40.0):
        tracker, audio, tokens, truth, voices = session(alternating(100), similarity=0.3, noise=0.35, seed=2, lo=2.0, hi=4.0)
        tracker.params = Params(unknown_percentile=percentile)
        for k in range(windows(audio)):                            # as live: the doctor is named as soon as the groups exist,
            feed(tracker, audio, tokens, first=k, last=k)          # so most sentences are judged against the PAST ones
            if tracker.group_count == 2 and not tracker.roles:
                named(tracker, voices)
        tracker.finish()
        unknown[percentile] = tracker.stats()['unknown'] / tracker.stats()['sentences']
    assert unknown[0.0] < unknown[15.0] < unknown[40.0]
    assert unknown[0.0] < 0.05 and unknown[40.0] > 0.25            # p = 0 decides nearly everything, p = 40 leaves a good part open


# ---- the gates of the first groups, the far-unit rule, the horizon and the waiting rule, one at a time ----------------
def voice(axis, extra=0.0, noise=0.02, seed=0):
    v = np.zeros(16)
    v[axis] = 1.0
    v[15] = extra
    return unit_rows(v + np.random.default_rng(seed).normal(0, noise, 16))


def inject(tracker, parts):
    """parts: [(count, seconds each, axis, extra)] laid out one after the other from t = 0, one sentence per unit."""
    t, n = 0.0, 0
    for count, seconds, axis, extra in parts:
        for _ in range(count):
            tracker.units.append(Unit(n, n, t, t + seconds, n, voice(axis, extra, seed=n)))
            t += seconds + 0.01
            n += 1
    return tracker


def fresh():
    return SpeakerTracker(lambda samples: None)


def test_a_second_voice_needs_enough_units_to_make_a_group():
    few = inject(fresh(), [(40, 1.0, 0, 0), (7, 3.0, 1, 0)])
    few._refit(300.0)
    assert few.group_count == 0                                    # 7 units of 3 s: plenty of speech, but too few units
    enough = inject(fresh(), [(40, 1.0, 0, 0), (8, 3.0, 1, 0)])
    enough._refit(300.0)
    assert enough.group_count == 2


def test_a_second_voice_needs_enough_speech_to_make_a_group():
    little = inject(fresh(), [(50, 1.0, 0, 0), (12, 0.5, 1, 0)])
    little._refit(300.0)
    assert little.group_count == 0                                 # 12 units but only ~10% of the speech
    enough = inject(fresh(), [(50, 1.0, 0, 0), (12, 1.0, 1, 0)])
    enough._refit(300.0)
    assert enough.group_count == 2


def test_two_halves_of_one_voice_are_not_two_voices_but_two_different_voices_are():
    alike = inject(fresh(), [(30, 1.0, 0, 3.0), (30, 1.0, 1, 3.0)])      # centroid cosine about 0.9
    alike._refit(300.0)
    assert alike.group_count == 0
    apart = inject(fresh(), [(30, 1.0, 0, 1.0), (30, 1.0, 1, 1.0)])      # about 0.5
    apart._refit(300.0)
    assert apart.group_count == 2


def test_a_refit_never_looks_at_units_that_end_after_its_minute():
    t = inject(fresh(), [(30, 1.0, 0, 0), (30, 1.0, 1, 0)])
    before = sum(1 for u in t.units if u.t1 < 40.0)
    t._refit(40.0)
    assert t.minutes[-1]['units'] == before and before < len(t.units)


def test_a_unit_that_sounds_like_nobody_does_not_vote():
    t = inject(fresh(), [(30, 1.0, 0, 0), (30, 1.0, 1, 0)])
    t._refit(300.0)
    doctor = int(np.argmax(t._centres @ voice(0)))
    t.set_roles({g: (DOCTOR if g == doctor else OTHER) for g in range(2)})
    other = 1 - doctor
    stranger = np.zeros(16)
    stranger[other], stranger[10] = 0.2, 0.98                      # leans a little to the patient, but is close to neither
    t.units = [Unit(i, i, i, i + 1.0, 0, unit_rows(v)) for i, v in enumerate([voice(0), stranger, stranger])]
    sentence = Sentence(0, 2, 0.0, 3.0, closed=True, units=[0, 1, 2])
    t._calibration = (-0.8, 0.8, 0.1)
    node, lean = t._score(sentence, [doctor], [other])
    assert lean == doctor and node == pytest.approx(t.params.clip)      # only the clear doctor unit was heard


def test_nothing_is_decided_until_there_are_eight_sentences_to_judge_weakness_against():
    tracker, audio, tokens, truth, voices = session([0, 1] * 6, lo=9.0, hi=11.0)
    feed(tracker, audio, tokens, last=25)                          # about 78 s: the groups exist, only a handful of sentences are closed
    assert tracker.group_count == 2
    named(tracker, voices)
    assert 0 < len([s for s in tracker.sentences if s.closed]) < 8
    assert tracker.stats()['decided'] == 0
    feed(tracker, audio, tokens, first=26)
    assert tracker.stats()['decided'] >= 8


def test_a_sentence_is_decided_only_after_all_its_units_are_final():
    tracker, audio, tokens, truth, voices = session(alternating(40))
    for k in range(0, 100):
        feed(tracker, audio, tokens, first=k, last=k)
        if tracker.group_count == 2 and not tracker.roles:
            named(tracker, voices)
        decided = tracker.sentences[:tracker._cursor]
        assert all(s.last < tracker.stream.settled for s in decided), k


def test_voiceprints_that_cannot_be_measured_leave_the_sentence_unknown():
    tracker, audio, tokens, truth, voices = session(alternating(40))
    tracker._embed = lambda samples: None
    feed(tracker, audio, tokens)
    assert tracker.group_count == 0
    tracker.finish()
    assert all(label == UNKNOWN for label, _ in shown(tracker, tokens))


def test_text_fill_is_accepted_only_when_it_agrees_with_the_weak_voice():
    tracker, audio, tokens, truth, voices = session(alternating(80), similarity=0.7, noise=0.2, seed=4)
    feed(tracker, audio, tokens)
    named(tracker, voices)
    tracker.finish()
    pending = tracker.fill_candidates(final=True)
    assert pending, 'this session should leave some sentences unknown'
    index = next(i for i in pending if tracker.sentences[i].lean is not None)
    lean = tracker.lean_role(index)
    other = OTHER if lean == DOCTOR else DOCTOR
    assert tracker.apply_fill(index, other) is False
    assert tracker.sentences[index].gid is None
    assert tracker.apply_fill(index, lean) is True
    sentence = tracker.sentences[index]
    assert sentence.source == 'text' and tracker.role_of(sentence.gid) == lean
    seg = tracker.tokens[sentence.first].seg
    assert ('text' in {s for _, _, _, s in tracker.labels_for(seg)})


def test_fill_candidates_wait_for_context_after_them_until_the_end():
    tracker, audio, tokens, truth, voices = session(alternating(80), similarity=0.7, noise=0.2, seed=4)
    feed(tracker, audio, tokens)
    named(tracker, voices)
    live = tracker.fill_candidates()
    tracker.finish()
    final = tracker.fill_candidates(final=True)
    assert set(live) <= set(final) and len(final) >= len(live)
    assert all(tracker._cursor - 1 - i >= 10 for i in live)
    tracker.mark_asked(final)
    assert tracker.fill_candidates(final=True) == []


def test_group_information_for_the_dialog():
    tracker, audio, tokens, truth, voices = session(alternating(60))
    feed(tracker, audio, tokens)
    named(tracker, voices)
    infos = tracker.group_info()
    assert [i.name for i in infos] == ['S1', 'S2']
    assert sum(i.share for i in infos) == pytest.approx(1.0)
    assert {i.role for i in infos} == {DOCTOR, OTHER}
    assert all(i.units > 5 for i in infos)
    doctor = next(i for i in infos if i.role == DOCTOR)
    assert doctor.examples and all(tracker.sentences[e].gid == doctor.gid for e in doctor.examples)


def test_the_audio_buffer_handles_overlap_and_gaps():
    from voice.speaker.tracker import AudioBuffer
    buffer = AudioBuffer(keep_seconds=10)
    buffer.add(0.0, np.full(6 * 16000, 1.0, dtype=np.float32))
    buffer.add(3.0, np.full(6 * 16000, 2.0, dtype=np.float32))
    assert buffer.slice(1.0, 2.0).mean() == pytest.approx(1.0)
    assert buffer.slice(4.0, 5.0).mean() == pytest.approx(2.0)
    assert buffer.slice(8.0, 10.0) is None                      # not recorded yet
    buffer.add(20.0, np.full(2 * 16000, 3.0, dtype=np.float32))  # a gap: older audio is dropped, not glued on
    assert buffer.slice(4.0, 5.0) is None
    assert buffer.slice(20.5, 21.0).mean() == pytest.approx(3.0)
    buffer.add(22.0, np.full(12 * 16000, 4.0, dtype=np.float32))
    assert len(buffer.samples) <= 10 * 16000


def test_window_helper_matches_the_pipeline_arithmetic():
    assert window_of(0.0) == 0 and window_of(5.9) == 0 and window_of(6.0) == 1 and window_of(8.9) == 1 and window_of(9.0) == 2


# ---- a group nobody has named is not the patient ---------------------------------------------------------------------------
def test_a_group_nobody_named_is_shown_as_unknown_and_never_as_the_patient():
    tracker, audio, tokens, truth, voices = session(alternating(100), similarity=0.3, noise=0.35, seed=2, lo=2.0, hi=4.0)
    feed(tracker, audio, tokens)
    g = doctor_group(tracker, voices)
    tracker.set_roles({g: DOCTOR})                                           # the model named the doctor and said "不明" for the other
    tracker.finish()
    assert tracker.role_of(1 - g) == UNKNOWN and tracker.role_of(g) == DOCTOR
    labels = [label for label, _ in shown(tracker, tokens)]
    assert OTHER not in labels and labels.count(DOCTOR) > 0.3 * len(labels)
    assert {i.role for i in tracker.group_info()} == {DOCTOR, None}
    leaning = next(i for i, s in enumerate(tracker.sentences) if s.lean == 1 - g and s.gid is None)
    assert tracker.apply_fill(leaning, OTHER) is False and tracker.apply_fill(leaning, DOCTOR) is False   # no role to agree with


def test_a_voice_group_that_appears_later_stays_unknown_until_it_is_named():
    pattern = [0, 1] * 20 + [2, 0, 2, 1] * 16                                # the newcomer speaks a lot, so its group forms soon
    tracker, audio, tokens, truth, voices = session(pattern, lo=1.8, hi=4.0)
    third_start = next(t.start for t, s in zip(tokens, truth) if s == 2)
    feed(tracker, audio, tokens, last=int(third_start // 3) - 1)
    assert tracker.group_count == 2
    named(tracker, voices)                                                   # roles for the two groups known so far
    feed(tracker, audio, tokens, first=int(third_start // 3))
    tracker.finish()
    assert tracker.group_count == 3
    added_at = next(e['at'] for e in tracker.events if e['event'] == 'speaker_group_added')
    # Until the third group exists the newcomer is taken for the nearest group (see SPEC 4.3, limits); once it exists, nobody
    # has named it, so its sentences must be unknown, not "the patient".
    third = [(label, t) for (label, _), t, s in zip(shown(tracker, tokens), tokens, truth) if s == 2 and t.start >= added_at + 30]
    assert len(third) > 50 and all(label != OTHER for label, _ in third)
    assert sum(label == UNKNOWN for label, _ in third) > 0.8 * len(third)
    other = next(g for g, r in tracker.roles.items() if r == OTHER)
    doctor = doctor_group(tracker, voices)
    tracker.set_roles({doctor: DOCTOR, other: OTHER, 3 - doctor - other: OTHER})   # now the physician (or the model) names it
    after = [label for (label, _), s, t in zip(shown(tracker, tokens), truth, tokens) if s == 2 and t.start >= added_at + 30]
    assert sum(label == OTHER for label in after) > 0.7 * len(after)


def test_the_published_voice_group_of_every_label_is_kept_for_the_record():
    tracker, audio, tokens, truth, voices = session(alternating(60), similarity=0.3, noise=0.35, seed=2, lo=2.0, hi=4.0)
    feed(tracker, audio, tokens)
    named(tracker, voices)
    tracker.finish()
    labels, groups = tracker.all_labels(), tracker.all_groups()
    assert set(labels) == set(groups) and all(len(labels[s]) == len(groups[s]) for s in labels)
    seen_unknown = seen_labelled = False
    for seg, items in labels.items():
        for (c0, c1, label, source), gid in zip(items, groups[seg]):
            if label == UNKNOWN:
                seen_unknown = seen_unknown or gid is None
            else:
                seen_labelled = True
                assert gid is not None and tracker.role_of(gid) == label    # the role is exactly what that group stands for now
    assert seen_unknown and seen_labelled


def promoted_after_naming_only_the_doctor(configure):
    """A finished session in which one clear sentence of the doctor is made unknown again and the other group is not shown with its
    role (`configure` decides how); the rescoring promotes it and writes its log line."""
    tracker = decided_session()
    doctor = next(g for g, r in tracker.roles.items() if r == DOCTOR)
    index = next(i for i, s in enumerate(tracker.sentences) if s.gid == doctor and s.source == 'voice' and abs(s.node) > 3)
    tracker.sentences[index].gid, tracker.sentences[index].source = None, ''
    tracker._history = [0.5] * 30 + [100.0] * 70
    configure(tracker, doctor)
    tracker._rescore()
    event = next(e for e in tracker.events if e['event'] == 'speaker_rescored')
    named_unknown = sum(1 for s in tracker.sentences[:tracker._cursor] if tracker.role_of(s.gid) == UNKNOWN)
    shown_unknown = sum(1 for s in tracker.sentences[:tracker._cursor] if tracker.shown_role(s.gid) == UNKNOWN)
    return event, named_unknown, shown_unknown, tracker, index


def test_the_rescoring_log_counts_the_unknown_sentences_the_way_the_screen_does():
    def other_group_unnamed(tracker, doctor):
        tracker._roles = {doctor: DOCTOR}

    event, named_unknown, shown_unknown, tracker, index = promoted_after_naming_only_the_doctor(other_group_unnamed)
    assert tracker.sentences[index].gid is not None and event['promoted'] >= 1
    assert event['unknown'] == shown_unknown > sum(1 for s in tracker.sentences[:tracker._cursor] if s.gid is None)


def test_only_the_confirmed_groups_are_shown_with_their_role_and_counted_as_labelled():
    tracker = decided_session()
    doctor = next(g for g, r in tracker.roles.items() if r == DOCTOR)
    patient = next(g for g, r in tracker.roles.items() if r == OTHER)
    seen = lambda: {label for labels in tracker.all_labels().values() for _, _, label, _ in labels}       # noqa: E731
    before = tracker.stats()
    assert OTHER in seen()
    changed = tracker.set_confirmed({doctor})
    assert changed and OTHER not in seen() and DOCTOR in seen()                      # shown as unknown, although the roles are named
    assert any(g == patient for gids in tracker.all_groups().values() for g in gids)   # the voice group behind each label is kept
    after = tracker.stats()
    assert after['unknown'] > before['unknown'] and after['labelled'] + after['unknown'] == after['decided']
    assert tracker._unknown_share(0.0, 1e9) == round(after['unknown'] / after['decided'], 3)       # the per-minute record agrees
    assert tracker.set_confirmed({doctor}) == set()                                  # nothing new
    tracker.set_confirmed(None)                                                      # before the first confirmation: the tentative roles show
    assert OTHER in seen() and tracker.stats() == before
    tracker.set_confirmed({doctor, patient})
    assert OTHER in seen()


def test_the_rescoring_log_counts_an_unconfirmed_group_as_unknown_too():
    def other_group_unconfirmed(tracker, doctor):
        tracker.set_confirmed({doctor})

    event, named_unknown, shown_unknown, tracker, index = promoted_after_naming_only_the_doctor(other_group_unconfirmed)
    assert event['unknown'] == shown_unknown > named_unknown


def fill_session():
    """A noisy finished session with unknown sentences, the way the first text fill meets them."""
    tracker = decided_session(noise=0.35, similarity=0.5, seed=2)
    withlean = [i for i, s in enumerate(tracker.sentences) if s.gid is None and s.lean is not None]
    assert len(withlean) >= 5
    return tracker, withlean


def test_a_first_answer_that_is_not_accepted_is_kept_for_the_second_fill_and_one_that_is_accepted_is_not_needed():
    tracker, ids = fill_session()
    first, second = ids[0], ids[1]
    lean_role = tracker.lean_role(first)
    other_role = OTHER if lean_role == DOCTOR else DOCTOR
    assert tracker.apply_fill(first, other_role) is False and tracker.sentences[first].said == other_role     # refused, remembered
    assert tracker.apply_fill(second, tracker.lean_role(second)) is True and tracker.sentences[second].source == 'text'
    tracker.sentences[ids[2]].lean = None                                                                   # no voiceprint to compare with
    assert tracker.apply_fill(ids[2], DOCTOR) is False and tracker.sentences[ids[2]].said == DOCTOR


def test_a_sentence_reaches_the_second_fill_one_refit_after_the_first_answer_came_back_and_only_with_a_first_answer():
    tracker, ids = fill_session()
    a, b = ids[0], ids[1]
    tracker.mark_asked([a, b])                                                  # both went to the first fill; only `a` is answered
    tracker.minutes.append({})                                                  # a refit happens while that request is out
    other_role = OTHER if tracker.lean_role(a) == DOCTOR else DOCTOR
    assert tracker.apply_fill(a, other_role) is False                           # the answer comes back: refused, remembered
    assert tracker.sentences[a].said_refit == len(tracker.minutes)
    assert tracker.fill2_candidates() == []                                      # the refit during the request does not count
    tracker.minutes.append({})                                                  # a minute passes after the answer
    assert tracker.fill2_candidates() == [a]                                     # `b` was never answered
    tracker.mark_asked2([a])
    assert tracker.fill2_candidates() == []                                      # asked once
    tracker.mark_asked2([a], False)
    assert tracker.fill2_candidates() == [a]


def test_the_second_fill_waits_for_twelve_sentences():
    tracker, _ = fill_session()
    assert not tracker.fill2_due(list(range(11))) and tracker.fill2_due(list(range(12)))


def test_the_second_answer_decides_whenever_it_names_a_role_if_the_first_fill_saw_the_sentence_and_it_is_still_unknown():
    tracker, ids = fill_session()
    i = ids[0]
    s = tracker.sentences[i]
    assert tracker.apply_fill2(i, DOCTOR) is False                                # the first fill never answered it
    s.said = DOCTOR
    assert tracker.apply_fill2(i, 'unknown') is False and s.gid is None            # not a role
    assert tracker.apply_fill2(i, OTHER) is True                                  # a different answer from the first one still decides
    assert s.source == 'text2' and tracker.role_of(s.gid) == OTHER and tracker.stats()['text_filled2'] == 1
    assert tracker.apply_fill2(i, DOCTOR) is False                                # no longer unknown
    labels = [label for labels in tracker.all_labels().values() for _, _, label, source in labels if source == 'text2']
    assert labels and set(labels) == {OTHER}                                     # published, with the source kept
    j = ids[1]
    against = OTHER if tracker.lean_role(j) == DOCTOR else DOCTOR                   # against the weak voice lean as well
    tracker.sentences[j].said = tracker.lean_role(j)
    assert tracker.apply_fill2(j, against) is True and tracker.role_of(tracker.sentences[j].gid) == against


def test_without_a_group_of_that_role_the_second_answer_cannot_be_used():
    tracker, ids = fill_session()
    tracker.sentences[ids[0]].said = OTHER
    doctor = next(g for g, r in tracker.roles.items() if r == DOCTOR)
    tracker._roles = {doctor: DOCTOR}                                            # nobody is named patient
    assert tracker.apply_fill2(ids[0], OTHER) is False and tracker.sentences[ids[0]].gid is None


def test_the_second_fill_puts_a_sentence_into_the_nearest_group_of_the_role_and_falls_back_to_the_nearest_earlier_sentence():
    tracker, audio, tokens, truth, voices = session(NEWCOMERS['talkative (one turn in two)'], lo=1.8, hi=4.0, seed=2)
    feed(tracker, audio, tokens)
    tracker.finish()
    assert tracker.group_count == 3
    doctor = int(np.argmax(tracker._centres @ voices[0]))
    first = int(np.argmax(tracker._centres @ voices[1]))
    newcomer = int(np.argmax(tracker._centres @ voices[2]))
    tracker.set_roles({doctor: DOCTOR, first: OTHER, newcomer: OTHER})

    def with_voiceprint(index):
        return tracker.sentences[index].decided and any(tracker.units[n].vec is not None for n in tracker.sentences[index].units)

    for speaker, group in ((1, first), (2, newcomer)):
        k = next(k for k, s in enumerate(truth) if s == speaker and k > 200 and with_voiceprint(tracker._sentence_of[k]))
        index = tracker._sentence_of[k]
        s = tracker.sentences[index]
        s.gid, s.source, s.said = None, '', OTHER
        assert tracker.apply_fill2(index, OTHER) is True and s.gid == group           # the voice decides between the two patient groups
    patient_groups = sorted((first, newcomer))

    def nearest_earlier(index):
        return next(tracker.sentences[j].gid for j in range(index - 1, -1, -1) if tracker.sentences[j].gid in patient_groups)

    # a sentence without any voiceprint whose nearest earlier patient sentence is NOT the first patient group in the list
    k = next(k for k, s in enumerate(truth) if s == 2 and k > 400 and nearest_earlier(tracker._sentence_of[k]) != patient_groups[0])
    index = tracker._sentence_of[k]
    s = tracker.sentences[index]
    s.gid, s.source, s.said = None, '', OTHER
    for n in s.units:
        tracker.units[n].vec = None                                               # no voiceprint at all
    expected = nearest_earlier(index)
    assert tracker.apply_fill2(index, OTHER) is True and s.gid == expected != patient_groups[0]
    # nothing earlier of that role and no voiceprint, with two groups of that role: there is no basis to choose, so it stays unknown
    s.gid, s.source, s.said = None, '', OTHER
    for j in range(index):
        if tracker.sentences[j].gid in patient_groups:
            tracker.sentences[j].gid = None
    assert tracker.apply_fill2(index, OTHER) is False and s.gid is None
