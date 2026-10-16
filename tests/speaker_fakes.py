"""A synthetic consultation for the speaker tests: audio whose samples encode who is speaking, a fake voiceprint model that
reads that back, and a feeder that hands the tracker 6-second windows with a 3-second stride, like the real pipeline."""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from voice.speaker.units import Token

RATE = 16000
DIM = 16


class FakeEmbedder:
    """Voiceprint = the speaker's base vector plus noise. The audio carries the speaker as a constant sample value (0.1 * (n + 1))."""

    def __init__(self, bases: dict[int, np.ndarray], noise: float = 0.15, seed: int = 0):
        self.bases = bases
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.calls = 0

    def __call__(self, samples: np.ndarray):
        self.calls += 1
        level = float(np.mean(samples))
        if abs(level) < 0.05:
            return None
        speaker = int(round(level / 0.1)) - 1
        return self.bases[speaker] + self.rng.normal(0, self.noise, DIM)


def bases(count: int, similarity: float = 0.0) -> dict[int, np.ndarray]:
    """Unit base vectors; with `similarity` > 0 every speaker also shares a common component (voices that sound alike)."""
    out = {}
    for n in range(count):
        v = np.zeros(DIM)
        v[n] = 1.0
        v[DIM - 1] = similarity
        out[n] = v / np.linalg.norm(v)
    return out


@dataclass
class Turn:
    speaker: int
    start: float
    end: float


def make_turns(pattern: list[int], *, seed: int = 1, first: float = 0.5, lo: float = 1.6, hi: float = 5.5, pause: float = 0.7):
    rng = random.Random(seed)
    turns, t = [], first
    for speaker in pattern:
        length = round(rng.uniform(lo, hi) * 4) / 4
        turns.append(Turn(speaker, t, t + length))
        t += length + pause
    return turns


def window_of(t: float) -> int:
    """Which window first covers time t: window 0 is [0, 6), window k >= 1 adds [3k + 3, 3k + 6) (6 s windows, 3 s overlap)."""
    return 0 if t < 6.0 else 1 + int((t - 6.0) // 3.0)


def build(turns: list[Turn], *, step: float = 0.25):
    """(audio, tokens, truth): a token every `step` seconds inside each turn, the last one carrying the sentence mark.
    A token's `seg` is its window number + 1, like the pipeline's segment numbers."""
    total = turns[-1].end + 2.0
    audio = np.zeros(int(total * RATE), dtype=np.float32)
    tokens, truth, counters = [], [], {}
    for turn in turns:
        audio[int(turn.start * RATE):int(turn.end * RATE)] = 0.1 * (turn.speaker + 1)
        t = turn.start
        while t + step <= turn.end + 1e-9:
            seg = window_of(t) + 1
            c0 = counters.get(seg, 0)
            counters[seg] = c0 + 1
            last = t + 2 * step > turn.end + 1e-9
            tokens.append(Token(seg, c0, c0 + 1, t, t + step, last))
            truth.append(turn.speaker)
            t += step
    return audio, tokens, truth


def windows(audio) -> int:
    return int(np.ceil(len(audio) / RATE / 3.0))


def feed(tracker, audio, tokens, first: int = 0, last: int | None = None) -> set[int]:
    """Hand the tracker windows `first`..`last` (inclusive): the audio, then the tokens that window adds. Returns the segments touched."""
    changed: set[int] = set()
    for k in range(first, (windows(audio) if last is None else last + 1)):
        tracker.add_audio(3.0 * k, audio[int(3.0 * k * RATE):int((3.0 * k + 6.0) * RATE)])
        changed |= tracker.add_tokens([t for t in tokens if window_of(t.start) == k])
    return changed


# ---- whole-visit fakes ------------------------------------------------------------------------------------------------
DOCTOR_LINE, PATIENT_LINE = '請問有沒有發燒。', '我頭痛好幾天了。'


def conversation(turns: int = 40, seconds: float = 3.0):
    """Doctor and patient alternate every `seconds` (even turns are the doctor), one 7-character sentence per turn.

    Returns (audio, replies, truth). `replies[k]` is what a fake ASR says for the k-th 6-second window (it covers turns k and
    k + 1; the last reply is the 3-second tail), so the words and the voices agree. `truth[k]` is 'doctor' or 'other'."""
    levels = [0.1 if i % 2 == 0 else 0.2 for i in range(turns)]
    audio = np.concatenate([np.full(int(seconds * RATE), level, dtype=np.float32) for level in levels])
    lines = [DOCTOR_LINE if i % 2 == 0 else PATIENT_LINE for i in range(turns)]
    replies = {k: lines[k] + lines[k + 1] for k in range(turns - 1)}
    replies[turns - 1] = lines[turns - 1]
    return audio, replies, ['doctor' if i % 2 == 0 else 'other' for i in range(turns)]


def scripted_source(settings, audio: np.ndarray, recording_path=None, block: float = 0.1):
    """A real AudioSource that plays `audio` (float32) as fast as it can."""
    from voice.audio import AudioSource

    def make_iterator(stop):
        step = int(block * RATE)
        for i in range(0, len(audio), step):
            if stop.is_set():
                return
            yield audio[i:i + step]

    return AudioSource(make_iterator, window_seconds=settings.asr.window_seconds,
                       overlap_seconds=settings.asr.overlap_seconds, recording_path=recording_path)
