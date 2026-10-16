"""Who is speaking: voice groups built during the visit, scored per sentence. Pure computation, no I/O, no LLM.

Flow (SPEC 4.3): aligned characters -> word-sized units -> one voiceprint per unit -> voice groups (k-means, built once the first
minute is in, refitted every minute on PAST units only, grown only on strong evidence) -> per-sentence vote -> a sentence is
labelled with its voice group, or left "unknown" when the evidence is weaker than the p-th percentile of the past sentences.
Which group is the doctor is not decided here: `set_roles` is called by the role layer (an LLM reads the dialogue by group) or by
the physician. Until it is known nothing is labelled. Labels are stored as voice groups, so changing a group's role relabels
everything already decided. Centroids (the voiceprints) live in memory only and are never written anywhere.

Thread model: mutators take a lock and may run in a worker thread; readers (`labels_for`, `group_info`, ...) never wait, they
read immutable values that are replaced as a whole.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import BACKGROUND, BACKGROUND_GID, DOCTOR, OTHER, UNKNOWN
from .mathutil import kmeans, tied_two_component, unit_rows
from .units import DEFAULT_PARAMS as DEFAULT_UNIT_PARAMS, Token, UnitParams, UnitStream

RATE = 16000
MIN_HISTORY = 8              # past sentences needed before the percentile is taken from the past
FILL_AFTER = 10              # a text fill waits until this many later sentences are decided (they are its context)
FILL_BATCH = 6               # ... and until this many sentences are waiting, or the oldest has waited FILL_MAX_AGE seconds
FILL_MAX_AGE = 120.0
FILL2_BATCH = 12             # the second text fill waits until this many sentences are waiting for it (the final pass takes what is left)
FILL2_REFITS = 1             # ... and each of them went to the first text fill at least this many refits (minutes) ago


@dataclass(frozen=True)
class Params:
    unknown_percentile: float = 15.0     # a sentence weaker than this percentile of the past sentences is "unknown"
    refit_seconds: float = 60.0          # groups are refitted once per minute of audio
    first_fit_units: int = 8             # units needed before the first groups are built
    min_group_share: float = 0.15        # the smaller of the first two groups must hold this share of the speech ...
    min_group_units: int = 8             # ... and at least this many units, otherwise there is only one voice so far
    first_split_max_cos: float = 0.75    # ... and be at most this similar (halves of ONE voice measured 0.74-0.89, real pairs 0.15-0.71)
    new_group_max_cos: float = 0.60      # a further group must be at most this similar to every existing group
    new_group_share: float = 0.10        # ... and hold this share of the speech of the last `new_group_window` seconds
    new_group_window: float = 180.0      # (recent, not cumulative: somebody who arrives late must not wait for a share of the whole visit)
    max_groups: int = 4
    far_cos: float = 0.25                # a unit this unlike every group is not used for the vote
    clip: float = 6.0                    # log-likelihood ratios are clipped to +-clip
    background_rel_db: float = -8.0      # background voices: a sentence this much quieter than the doctor's usual level ...
    background_max_cos: float = 0.425    # ... AND fitting no voice group better than this (mean best cosine of its units); -inf / 0 switch it off
    quiet_gate_db: float = 10.0          # a unit this much quieter than the `quiet_gate_percentile` level of all units is left out of the group fit
    quiet_gate_percentile: float = 75.0  # (inf switches it off); such units are still assigned to the nearest group afterwards
    min_history: int = MIN_HISTORY       # past sentences needed before the percentile is taken from the past
    fill_after: int = FILL_AFTER         # a text fill waits until this many later sentences are decided (they are its context)
    fill_batch: int = FILL_BATCH         # ... and until this many sentences are waiting, or the oldest has waited `fill_max_age` seconds
    fill_max_age: float = FILL_MAX_AGE
    fill2_batch: int = FILL2_BATCH       # the second text fill waits until this many sentences are waiting for it (the final pass takes what is left)
    fill2_refits: int = FILL2_REFITS     # ... and each of them went to the first text fill at least this many refits (minutes) ago
    units: UnitParams = DEFAULT_UNIT_PARAMS      # how the text is cut into units and sentences


class AudioBuffer:
    """Recent audio by absolute time. ASR windows overlap, so a new window overwrites the overlap with the same samples."""

    def __init__(self, keep_seconds: float = 40.0):
        self.keep = int(keep_seconds * RATE)
        self.start = 0.0
        self.samples = np.empty(0, dtype=np.float32)

    def add(self, start: float, samples: np.ndarray):
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        end = self.start + len(self.samples) / RATE
        if not len(self.samples) or start > end + 0.05 or start < self.start:
            self.start, self.samples = start, samples.copy()
        else:
            cut = int(round((start - self.start) * RATE))
            self.samples = np.concatenate((self.samples[:cut], samples))
        if len(self.samples) > self.keep:
            drop = len(self.samples) - self.keep
            self.start += drop / RATE
            self.samples = self.samples[drop:]

    def slice(self, t0: float, t1: float) -> np.ndarray | None:
        a, b = int(round((t0 - self.start) * RATE)), int(round((t1 - self.start) * RATE))
        return self.samples[a:b] if 0 <= a < b <= len(self.samples) else None


@dataclass
class Unit:
    first: int                   # token indexes first..last
    last: int
    t0: float
    t1: float
    sentence: int
    vec: np.ndarray | None       # unit-length voiceprint, None when it could not be measured
    level: float = float('nan')  # mean-square level of its audio in dBFS, nan when the audio was not at hand

    @property
    def seconds(self) -> float:
        return self.t1 - self.t0


@dataclass
class Sentence:
    first: int
    last: int
    t0: float
    t1: float
    closed: bool = False
    units: list[int] = field(default_factory=list)
    decided: bool = False
    gid: int | None = None       # the voice group it is labelled with; None = unknown
    lean: int | None = None      # the group the (possibly weak) evidence points to; None = no usable voiceprint
    node: float = float('nan')   # mean log-likelihood ratio of its units (positive = doctor side)
    source: str = ''             # 'voice' | 'text' (first text fill) | 'text2' (second text fill)
    asked: bool = False          # already shown to the text-fill step
    said: str = ''               # the role the first text fill answered when that answer was not accepted (DOCTOR | OTHER), else ''
    said_refit: int = 0          # how many refits there had been when that answer came back
    asked2: bool = False         # already shown to the second text fill
    background: bool = False     # judged to be somebody else's voice (another room): shown grey, kept out of the record and the votes


@dataclass(frozen=True)
class GroupInfo:
    gid: int
    name: str                    # 'S1', 'S2', ...
    role: str | None             # DOCTOR | OTHER | None = not decided yet
    units: int
    seconds: float
    share: float                 # of all measured speech so far
    examples: tuple              # sentence indexes (recent ones clearly labelled with this group)


def group_name(gid: int) -> str:
    return f'S{gid + 1}'


class SpeakerTracker:
    def __init__(self, embed: Callable[[np.ndarray], np.ndarray | None], params: Params | None = None):
        self._embed = embed
        self.params = params or Params()
        self._lock = threading.RLock()
        self.audio = AudioBuffer()
        self.stream = UnitStream(self.params.units)
        self.units: list[Unit] = []
        self.sentences: list[Sentence] = []
        self._sentence_of: list[int] = []            # token index -> sentence index
        self._unit_of: dict[int, int] = {}           # token index -> unit index
        self._seg_tokens: dict[int, list[int]] = {}
        self._centres: np.ndarray | None = None     # (groups, 192), rows unit length; memory only
        self._roles: dict[int, str | None] = {}
        self._confirmed: frozenset[int] | None = None  # None = every named group is shown; else only these groups are (the others show unknown)
        self._calibration: tuple[float, float, float] | None = None
        self._ref_level: float | None = None         # the doctor's usual level (dBFS): the yardstick for "much quieter" (background voices)
        self._next_horizon = self.params.refit_seconds
        self._model_horizon = 0.0
        self._cursor = 0                             # sentences before it are decided
        self._history: list[float] = []              # |node| of the sentences decided so far
        self._view: dict[int, tuple] = {}
        self._gview: dict[int, tuple] = {}           # the voice group behind each published label (None = undecided)
        self.events: list[dict] = []
        self.minutes: list[dict] = []
        self.groups_version = 0                      # moves whenever a group is built
        self.roles_version = 0
        self.rescored = 0                            # unknown sentences a later model has labelled by their voice
        self.finished = False

    # ------------------------------------------------------------------ read side (never waits)
    @property
    def tokens(self) -> list[Token]:
        return self.stream.tokens

    @property
    def group_count(self) -> int:
        centres = self._centres
        return 0 if centres is None else len(centres)

    @property
    def roles(self) -> dict[int, str | None]:
        return dict(self._roles)

    def labels_for(self, seg: int) -> tuple | None:
        """((c0, c1, label, source), ...) for the segment's tokens in order, or None when the segment has no timed text."""
        return self._view.get(seg)

    @property
    def doctor_known(self) -> bool:
        return DOCTOR in self._roles.values()

    def role_of(self, gid: int | None) -> str:
        """DOCTOR / OTHER / UNKNOWN for a voice group. Only a role that was named counts: a group nobody has named yet (the model
        said "不明", or it appeared after the last answer) stays UNKNOWN instead of being taken for the patient."""
        role = self._roles.get(gid) if gid is not None else None
        return role if role in (DOCTOR, OTHER) else UNKNOWN

    def shown_role(self, gid: int | None) -> str:
        """The role the labels, the counts and the logs show: `role_of`, but unknown for a group whose role is not confirmed yet
        (see `set_confirmed`)."""
        if self._confirmed is not None and gid not in self._confirmed:
            return UNKNOWN
        return self.role_of(gid)

    def group_info(self) -> list[GroupInfo]:
        centres = self._centres
        if centres is None:
            return []
        measured = [u for u in list(self.units) if u.vec is not None and not self.sentences[u.sentence].background]    # background voices are counted apart
        seconds = np.zeros(len(centres))
        counts = np.zeros(len(centres), dtype=int)
        if measured:
            nearest = (np.stack([u.vec for u in measured]) @ centres.T).argmax(1)
            for u, g in zip(measured, nearest):
                seconds[g] += u.seconds
                counts[g] += 1
        total = float(seconds.sum()) or 1.0
        infos = []
        for g in range(len(centres)):
            examples = [i for i in range(self._cursor - 1, -1, -1)
                        if self.sentences[i].gid == g and self.sentences[i].source == 'voice' and not self.sentences[i].background
                        and self.sentences[i].last - self.sentences[i].first >= 2][:2]
            role = self.role_of(g)
            infos.append(GroupInfo(g, group_name(g), None if role == UNKNOWN else role, int(counts[g]), float(seconds[g]),
                                   float(seconds[g]) / total, tuple(examples)))
        return infos

    def voice_groups_for(self, seg: int) -> list[tuple[int, int, int | None]]:
        """(c0, c1, voice group) for each timed piece of a segment: the group its voiceprint is nearest to, None when it has none,
        BACKGROUND_GID for a piece of a sentence judged to be background voices."""
        centres = self._centres
        result = []
        for i in list(self._seg_tokens.get(seg, [])):
            token, unit = self.tokens[i], self._unit_of.get(i)
            vec = self.units[unit].vec if unit is not None else None
            gid = int((centres @ vec).argmax()) if centres is not None and vec is not None else None
            if self._background(i):
                gid = BACKGROUND_GID
            result.append((token.c0, token.c1, gid))
        return result

    def all_labels(self) -> dict[int, tuple]:
        return dict(self._view)

    def all_groups(self) -> dict[int, tuple]:
        """Per segment, the voice group of each published label (the same order as `all_labels`; None where undecided)."""
        return dict(self._gview)

    def segment_token_indexes(self, seg: int) -> list[int]:
        return list(self._seg_tokens.get(seg, []))

    def sentence_range(self, index: int) -> tuple[int, int]:
        sentence = self.sentences[index]
        return sentence.first, sentence.last

    def stats(self) -> dict:
        decided = [s for s in self.sentences[:self._cursor]]
        background = [s for s in decided if s.background]
        labelled = [s for s in decided if not s.background and self.shown_role(s.gid) != UNKNOWN]       # what the screen shows: an unnamed or unconfirmed group is unknown
        return {'sentences': len(self.sentences), 'decided': len(decided), 'labelled': len(labelled),
                'unknown': len(decided) - len(labelled) - len(background), 'background': len(background), 'groups': self.group_count,
                'text_filled': sum(1 for s in labelled if s.source == 'text'),
                'text_filled2': sum(1 for s in labelled if s.source == 'text2'),
                'rescored': self.rescored,
                'units': len(self.units), 'measured_units': sum(1 for u in self.units if u.vec is not None)}

    # ------------------------------------------------------------------ feeding
    def add_audio(self, start: float, samples: np.ndarray):
        with self._lock:
            self.audio.add(start, samples)

    def add_tokens(self, tokens: list[Token]) -> set[int]:
        """New aligned text, in time order. Returns the segment numbers whose labels may have changed."""
        with self._lock:
            base = len(self.stream.tokens)
            self.stream.extend(tokens)
            touched = set()
            for k, token in enumerate(tokens):
                self._add_token(base + k)
                touched.add(token.seg)
            self._take_units(self.stream.emit())
            return self._advance(touched)

    def finish(self) -> set[int]:
        """No more text: close the open sentence and decide whatever is left."""
        with self._lock:
            if self.finished:
                return set()
            self.finished = True
            if self.sentences:
                self.sentences[-1].closed = True
            self._take_units(self.stream.finish())
            return self._advance(set())

    def reopen(self):
        """The recording goes on after a stop (`finish` decided everything): the tracker takes new text again. Sentences and units
        decided so far stay as they are; the first new word starts a new sentence."""
        with self._lock:
            self.finished = False

    def _add_token(self, i: int):
        token = self.stream.tokens[i]
        if self.sentences and not self.sentences[-1].closed and token.start - self.stream.tokens[i - 1].end >= self.params.units.sent_pause:
            self.sentences[-1].closed = True
        if not self.sentences or self.sentences[-1].closed:
            self.sentences.append(Sentence(i, i, token.start, token.end))
        sentence = self.sentences[-1]
        sentence.last, sentence.t1 = i, token.end
        self._sentence_of.append(len(self.sentences) - 1)
        self._seg_tokens.setdefault(token.seg, []).append(i)
        if token.sentence_end:
            sentence.closed = True

    def _take_units(self, finished: list[list[int]]):
        tokens = self.stream.tokens
        for group in finished:
            first, last = group[0], group[-1]
            t0, t1 = tokens[first].start, tokens[last].end
            vec, level = None, float('nan')
            audio = self.audio.slice(t0, t1)
            if audio is not None:
                level = float(10.0 * np.log10(max(float(np.mean(np.square(audio, dtype=np.float64))), 1e-12)))
                raw = self._embed(audio)
                if raw is not None and np.all(np.isfinite(raw)) and np.linalg.norm(raw) > 0:
                    vec = unit_rows(raw)
            sentence = self._sentence_of[first]
            number = len(self.units)
            self.units.append(Unit(first, last, t0, t1, sentence, vec, level))
            self.sentences[sentence].units.append(number)
            for i in range(first, last + 1):
                self._unit_of[i] = number

    def _advance(self, touched: set[int]) -> set[int]:
        newest = self.units[-1].t1 if self.units else 0.0
        changed = set(touched)
        refitted = False
        while newest >= self._next_horizon:
            self._refit(self._next_horizon)
            self._next_horizon += self.params.refit_seconds
            refitted = True
        if refitted or self.finished:
            changed |= self._rescore()
            changed |= self._remark_background()
        before = self._cursor
        self._decide()
        for s in self.sentences[before:self._cursor]:
            changed |= self._sentence_segments(s)
        self._publish(changed)
        return changed

    def _sentence_segments(self, s: Sentence) -> set[int]:
        return {self.tokens[i].seg for i in range(s.first, s.last + 1)}

    # ------------------------------------------------------------------ voice groups
    def _doctor_groups(self) -> list[int]:
        return [g for g in range(self.group_count) if self._roles.get(g) == DOCTOR]

    def _refit(self, horizon: float):
        p = self.params
        self._model_horizon = horizon
        past = [u for u in self.units if u.vec is not None and u.t1 < horizon]
        fit = self._without_quiet(past)
        if len(fit) >= p.first_fit_units:
            self._fit(fit, horizon)
        self._update_reference()
        self._recalibrate()
        self.minutes.append({'minute': round(horizon / 60), 'units': len(past), 'groups': self.group_count,
                             'sentences_decided': self._cursor, 'left_out_of_fit': len(past) - len(fit),
                             'unknown_share': self._unknown_share(horizon - p.refit_seconds, horizon)})

    def _fit(self, units: list[Unit], horizon: float):
        X = np.stack([u.vec for u in units])
        w = np.array([u.seconds for u in units])
        if self._centres is None:
            self._build_first_groups(X, w, horizon)
        else:
            self._update_groups(X, w, np.array([u.t1 for u in units]), horizon)

    def _without_quiet(self, units: list[Unit]) -> list[Unit]:
        """The units the voice groups are fitted on. The ones much quieter than the rest (somebody in another room, or far from the
        microphone) are left out: the most DISTINCT voice would otherwise take a whole group, and two voices in the room that sound alike
        would be merged into one (2026-10-11 recordings, SPEC 4.3). They still get the nearest group afterwards."""
        p = self.params
        levels = np.array([u.level for u in units])
        known = np.isfinite(levels)
        if not known.any():
            return units
        floor = float(np.percentile(levels[known], p.quiet_gate_percentile)) - p.quiet_gate_db
        return [u for u, level, ok in zip(units, levels, known) if not ok or level >= floor]

    def _unknown_share(self, t0: float, t1: float) -> float | None:
        window = [s for s in self.sentences[:self._cursor] if t0 <= s.t0 < t1]
        return round(sum(self._is_unknown(s) for s in window) / len(window), 3) if window else None

    def _is_unknown(self, s: Sentence) -> bool:
        return not s.background and self.shown_role(s.gid) == UNKNOWN

    def _build_first_groups(self, X: np.ndarray, w: np.ndarray, horizon: float):
        p = self.params
        centres, labels, _ = kmeans(X, w, 2, n_init=10)
        seconds = [float(w[labels == j].sum()) for j in (0, 1)]
        counts = [int((labels == j).sum()) for j in (0, 1)]
        small = int(np.argmin(seconds))
        halves = unit_rows(centres)
        if (counts[small] < p.min_group_units or seconds[small] / sum(seconds) < p.min_group_share
                or float(halves[0] @ halves[1]) >= p.first_split_max_cos):
            return                                               # still one voice (or a stray one): try again next minute
        self._centres = halves
        self.groups_version += 1
        self.events.append({'event': 'speaker_groups_built', 'groups': 2, 'units': len(X), 'at': round(horizon, 1)})

    def _update_groups(self, X: np.ndarray, w: np.ndarray, ends: np.ndarray, horizon: float):
        """The minute's refit. A further group is looked for FIRST, against the groups as they stand: if the groups were updated
        first, a newcomer's units (taken for the nearest group meanwhile) would drag the centres towards it, and the newcomer
        would look closer to them than it is, so it would never get its own group."""
        p = self.params
        k = len(self._centres)
        if k < p.max_groups and len(X) > k:
            candidates, labels, _ = kmeans(X, w, k + 1, n_init=10)
            candidates = unit_rows(candidates)
            closeness = (candidates @ self._centres.T).max(1)
            j = int(np.argmin(closeness))
            member = labels == j
            recent = ends >= horizon - p.new_group_window
            share = float(w[member & recent].sum()) / max(float(w[recent].sum()), 1e-9)
            if closeness[j] < p.new_group_max_cos and int(member.sum()) >= p.min_group_units and share >= p.new_group_share:
                centres, _, _ = kmeans(X, w, k + 1, init=np.vstack([self._centres, candidates[j]]))
                self._centres = unit_rows(centres)
                self.groups_version += 1
                self.events.append({'event': 'speaker_group_added', 'groups': k + 1, 'units': int(member.sum()),
                                    'at': round(horizon, 1)})
                return
        near = (X @ self._centres.T).max(1) >= p.far_cos           # somebody nobody sounds like must not drag a centre towards it
        if int(near.sum()) > k:
            centres, _, _ = kmeans(X[near], w[near], k, init=self._centres)
            self._centres = unit_rows(centres)

    # ------------------------------------------------------------------ background voices
    def _update_reference(self):
        """The doctor's usual level: the median level of the past units nearest to a doctor group. None until a doctor group is named."""
        self._ref_level = None
        doctor = self._doctor_groups()
        if self._centres is None or not doctor:
            return
        past = [u for u in self.units if u.vec is not None and np.isfinite(u.level) and u.t1 < self._model_horizon]
        if len(past) < self.params.first_fit_units:
            return
        nearest = (np.stack([u.vec for u in past]) @ self._centres.T).argmax(1)
        mine = [u.level for u, g in zip(past, nearest) if g in doctor]
        if len(mine) >= self.params.first_fit_units:
            self._ref_level = float(np.median(mine))

    def _is_background(self, s: Sentence) -> bool:
        """Another room's voices: clearly quieter than the doctor AND unlike every voice group (a conservative two-cue rule; on the 15
        labelled recordings it caught about 73% of the background words and took none of the doctor's or the patient's; a weak-voiced
        patient sounds like background from cosine 0.45 up, so the threshold stays below that, SPEC 4.3). Needs a named doctor group (the yardstick for "quieter") and a voiceprint."""
        if self._ref_level is None or self._centres is None or self.params.background_max_cos <= 0:     # (<= 0: switched off; a cosine can be negative)
            return False
        units = [self.units[n] for n in s.units if self.units[n].vec is not None and np.isfinite(self.units[n].level)]
        if not units:
            return False
        seconds = np.array([u.seconds for u in units])
        power = float(np.average(10.0 ** (np.array([u.level for u in units]) / 10.0), weights=seconds))
        quieter = 10.0 * np.log10(max(power, 1e-12)) - self._ref_level
        best = float(np.mean([(self._centres @ u.vec).max() for u in units]))
        return quieter < self.params.background_rel_db and best < self.params.background_max_cos

    def _remark_background(self) -> set[int]:
        """The groups, the doctor's level or the roles have just changed: every decided sentence is judged again (either way)."""
        changed = set()
        for s in self.sentences[:self._cursor]:
            flag = self._is_background(s)
            if flag != s.background:
                s.background = flag
                changed |= self._sentence_segments(s)
        return changed

    def _background(self, token: int) -> bool:
        s = self.sentences[self._sentence_of[token]]
        return s.decided and s.background

    def _recalibrate(self):
        """Map the doctor-minus-others similarity to a log-likelihood ratio with a two-component mixture on the PAST units."""
        self._calibration = None
        doctor = self._doctor_groups()
        if self._centres is None or not doctor or len(doctor) >= len(self._centres):
            return
        other = [g for g in range(len(self._centres)) if g not in doctor]
        past = [u.vec for u in self.units if u.vec is not None and u.t1 < self._model_horizon]
        if len(past) < self.params.first_fit_units:
            return
        sims = np.stack(past) @ self._centres.T
        low, high, variance = tied_two_component(sims[:, doctor].max(1) - sims[:, other].max(1))
        if high - low > 1e-6:
            self._calibration = (low, high, variance)

    # ------------------------------------------------------------------ roles
    def set_roles(self, roles: dict[int, str | None]) -> set[int]:
        """Name the doctor group(s). Everything already decided is relabelled; waiting sentences are decided now."""
        with self._lock:
            known = {g: r for g, r in roles.items() if 0 <= g < self.group_count and r in (DOCTOR, OTHER)}
            if known == self._roles:
                return set()
            self._roles = known
            self.roles_version += 1
            self._update_reference()
            self._recalibrate()
            self._remark_background()
            self._decide()
            changed = set(self._seg_tokens)
            self._publish(changed)
            return changed

    def set_confirmed(self, groups) -> set[int]:
        """Once the role map is trusted, only the groups whose role is confirmed are shown with it (screen, files, counts, prompts);
        every other group, including one that appears later, is shown as unknown until it is confirmed. None = show all named groups
        (before the first confirmation the screen shows the tentative roles). Returns the segments whose labels changed."""
        with self._lock:
            groups = None if groups is None else frozenset(groups)
            if groups == self._confirmed:
                return set()
            self._confirmed = groups
            changed = set(self._seg_tokens)
            self._publish(changed)
            return changed

    # ------------------------------------------------------------------ sentences
    def _score(self, sentence: Sentence, doctor: list[int], other: list[int]):
        low, high, variance = self._calibration
        slope, middle = (high - low) / variance, (low + high) / 2.0
        clip = self.params.clip
        ratios, sims = [], []
        for n in sentence.units:
            vec = self.units[n].vec
            if vec is None:
                continue
            c = self._centres @ vec
            if c.max() < self.params.far_cos:
                continue
            ratios.append(float(np.clip(slope * (c[doctor].max() - c[other].max() - middle), -clip, clip)))
            sims.append(c)
        if not ratios:
            return None
        node = float(np.mean(ratios))
        mean_sim = np.mean(sims, axis=0)
        pool = doctor if node > 0 else other
        return node, pool[int(np.argmax(mean_sim[pool]))]

    def _rescore(self) -> set[int]:
        """The model has just changed (a refit, or the end of the visit): sentences still unknown are scored again with it, against
        today's threshold. One that now clears the threshold is labelled by its voice; one that does not stays unknown, untouched, and
        is not given to the text fill again. Sentences that are already labelled, by voice or by text, are never touched."""
        doctor = self._doctor_groups()
        other = [g for g in range(self.group_count) if g not in doctor]
        if self._calibration is None or not doctor or not other or len(self._history) < self.params.min_history:
            return set()
        threshold = float(np.percentile(self._history, self.params.unknown_percentile))
        changed, promoted = set(), 0
        for s in self.sentences[:self._cursor]:
            if s.gid is not None or s.background:
                continue
            got = self._score(s, doctor, other)
            if got is None:
                continue
            node, lean = got
            if abs(node) >= threshold and node != 0.0:
                s.node, s.lean, s.gid, s.source = node, lean, lean, 'voice'
                changed |= self._sentence_segments(s)
                promoted += 1
        if promoted:
            self.rescored += promoted
            self.events.append({'event': 'speaker_rescored', 'promoted': promoted,
                                'unknown': sum(1 for s in self.sentences[:self._cursor] if self._is_unknown(s))})
        return changed

    def _decide(self):
        if self._calibration is None or self._centres is None:
            return
        doctor = self._doctor_groups()
        other = [g for g in range(len(self._centres)) if g not in doctor]
        settled = self.stream.settled
        ready = []
        for i in range(self._cursor, len(self.sentences)):
            s = self.sentences[i]
            if not ((s.closed or self.finished) and s.last < settled):
                break
            ready.append(i)
        if not ready:
            return
        scores = {i: self._score(self.sentences[i], doctor, other) for i in ready}
        magnitudes = [abs(v[0]) for v in scores.values() if v is not None]
        if not self.finished and len(self._history) + len(magnitudes) < self.params.min_history:
            return                                               # too little to judge "weak" against: wait for more sentences
        percentile = self.params.unknown_percentile
        seeded = len(self._history) < self.params.min_history
        fixed = float(np.percentile(magnitudes, percentile)) if seeded and magnitudes else 0.0
        if seeded:
            self._history.extend(magnitudes)                     # the first batch is judged against itself
        for i in ready:
            s, got = self.sentences[i], scores[i]
            s.decided, s.source = True, 'voice'
            s.background = self._is_background(s)
            if got is not None:
                node, lean = got
                threshold = fixed if seeded else (float(np.percentile(self._history, percentile)) if self._history else 0.0)
                s.node, s.lean = node, lean
                if abs(node) >= threshold and node != 0.0:
                    s.gid = lean
                if not seeded:
                    self._history.append(abs(node))
        self._cursor = ready[-1] + 1

    # ------------------------------------------------------------------ text fill
    def fill_candidates(self, final: bool = False) -> list[int]:
        """Unknown sentences not yet shown to the text fill, with enough decided sentences after them to read as context."""
        with self._lock:
            return [i for i in range(self._cursor)
                    if self.sentences[i].gid is None and not self.sentences[i].asked and not self.sentences[i].background
                    and (final or self._cursor - 1 - i >= self.params.fill_after)]

    def fill_due(self, wanted: list[int]) -> bool:
        """Enough waiting sentences, or the oldest has waited long enough, to be worth one LLM call."""
        if not wanted:
            return False
        newest = self.sentences[self._cursor - 1].t1
        return len(wanted) >= self.params.fill_batch or newest - self.sentences[wanted[0]].t1 >= self.params.fill_max_age

    def mark_asked(self, indexes: list[int], asked: bool = True):
        with self._lock:
            for i in indexes:
                self.sentences[i].asked = asked

    def apply_fill(self, index: int, role: str) -> bool:
        """Accept the text's answer only when it agrees with the side the sentence's voice (weakly) leans to. An answer that is not
        accepted is kept (`said`) for the second text fill."""
        with self._lock:
            s = self.sentences[index]
            if index >= self._cursor or s.gid is not None or role not in (DOCTOR, OTHER):
                return False
            s.said, s.said_refit = role, len(self.minutes)
            if s.lean is None or self.role_of(s.lean) != role:
                return False
            s.gid, s.source = s.lean, 'text'
            self._publish(self._sentence_segments(s))
            return True

    # ------------------------------------------------------------------ second text fill
    def fill2_candidates(self) -> list[int]:
        """Unknown sentences the first text fill answered without being accepted, not yet shown to the second fill, whose first
        answer came back at least `fill2_refits` refits ago (the model and the neighbours' labels have moved on since; a refit that
        happened while the first request was out does not count)."""
        with self._lock:
            refits = len(self.minutes)
            return [i for i in range(self._cursor)
                    if self.sentences[i].gid is None and self.sentences[i].said and not self.sentences[i].asked2
                    and not self.sentences[i].background and refits >= self.sentences[i].said_refit + self.params.fill2_refits]

    def fill2_due(self, wanted: list[int]) -> bool:
        return len(wanted) >= self.params.fill2_batch

    def mark_asked2(self, indexes: list[int], asked: bool = True):
        with self._lock:
            for i in indexes:
                self.sentences[i].asked2 = asked

    def apply_fill2(self, index: int, role: str) -> bool:
        """The second answer decides whenever it names a role, whatever the first answer or the weak voice lean said: this pass sees the
        neighbours' text-filled labels, the first one did not, and it was the more accurate of the two on the recordings measured
        (SPEC 4.3). Only a sentence the first fill was given (`said`) is eligible. The sentence goes to the voice group of that role that
        its voiceprint is nearest to (or, without a voiceprint, to the nearest earlier sentence of that role): a label always belongs to
        a voice group, so a later correction of the role map still moves it. When no group can be chosen on evidence, the sentence
        stays unknown."""
        with self._lock:
            s = self.sentences[index]
            if index >= self._cursor or s.gid is not None or not s.said or role not in (DOCTOR, OTHER):
                return False
            gid = self._group_for_role(index, role)
            if gid is None:
                return False
            s.gid, s.source = gid, 'text2'
            self._publish(self._sentence_segments(s))
            return True

    def _group_for_role(self, index: int, role: str) -> int | None:
        groups = [g for g in range(self.group_count) if self.role_of(g) == role]
        if len(groups) <= 1:
            return groups[0] if groups else None
        sims = [self._centres @ self.units[n].vec for n in self.sentences[index].units if self.units[n].vec is not None]
        if sims:
            return groups[int(np.argmax(np.mean(sims, axis=0)[groups]))]
        for j in range(index - 1, -1, -1):
            if self.sentences[j].gid in groups:
                return self.sentences[j].gid
        return None                                    # several groups of that role, no voiceprint, no earlier sentence: no basis to choose

    def lean_role(self, index: int) -> str:
        s = self.sentences[index]
        return self.role_of(s.lean) if s.lean is not None else UNKNOWN

    # ------------------------------------------------------------------ publishing
    def _label(self, i: int) -> tuple[str, str]:
        s = self.sentences[self._sentence_of[i]]
        if s.decided and s.background:
            return BACKGROUND, 'voice'
        if not s.decided or s.gid is None:
            return UNKNOWN, ''
        role = self.shown_role(s.gid)
        return role, s.source if role != UNKNOWN else ''

    def _group(self, i: int) -> int | None:
        s = self.sentences[self._sentence_of[i]]
        return s.gid if s.decided else None

    def _publish(self, segments: set[int]):
        for seg in segments:
            indexes = self._seg_tokens.get(seg)
            if indexes:
                self._gview[seg] = tuple(self._group(i) for i in indexes)
                self._view[seg] = tuple((self.tokens[i].c0, self.tokens[i].c1, *self._label(i)) for i in indexes)
