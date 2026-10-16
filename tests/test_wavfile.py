"""The recording file: created, continued (appended to), cut for replay while it is still being written."""
import wave

import numpy as np
import pytest

from voice.replay import clip_wav
from voice.wavfile import HEADER_BYTES, RATE, WavError, WavRecorder, sample_count


def tone(seconds: float, level: float = 0.2) -> np.ndarray:
    total = int(seconds * RATE)
    return (np.sin(2 * np.pi * 220 * np.arange(total) / RATE) * level).astype(np.float32)


def test_a_new_recording_is_a_valid_wav(tmp_path):
    path = tmp_path / 'a.wav'
    with WavRecorder(path) as recorder:
        recorder.write(tone(2))
    with wave.open(str(path), 'rb') as saved:
        assert (saved.getnchannels(), saved.getsampwidth(), saved.getframerate()) == (1, 2, RATE)
        assert saved.getnframes() == 2 * RATE
    assert sample_count(path) == 2 * RATE


def test_append_continues_the_same_file(tmp_path):
    path = tmp_path / 'a.wav'
    with WavRecorder(path) as first:
        first.write(tone(2))
    with WavRecorder(path, append=True) as second:
        assert second.samples == 2 * RATE                      # it knows where the recording stands
        second.write(tone(3))
    with wave.open(str(path), 'rb') as saved:
        assert saved.getnframes() == 5 * RATE                  # one file, longer; the header says so too
    assert sample_count(path) == 5 * RATE


def test_append_to_a_missing_or_empty_file_starts_a_new_recording(tmp_path):
    path = tmp_path / 'a.wav'
    with WavRecorder(path, append=True) as recorder:
        recorder.write(tone(1))
    assert sample_count(path) == RATE
    empty = tmp_path / 'b.wav'
    empty.write_bytes(b'')
    with WavRecorder(empty, append=True) as recorder:
        recorder.write(tone(1))
    assert sample_count(empty) == RATE


def test_a_half_sample_left_by_a_crash_is_dropped_on_append(tmp_path):
    path = tmp_path / 'a.wav'
    with WavRecorder(path) as recorder:
        recorder.write(tone(1))
    with open(path, 'ab') as handle:
        handle.write(b'\x01')                                  # a write cut in the middle of a sample
    with WavRecorder(path, append=True) as recorder:
        assert recorder.samples == RATE
        recorder.write(tone(1))
    assert sample_count(path) == 2 * RATE
    with wave.open(str(path), 'rb') as saved:
        assert saved.getnframes() == 2 * RATE


def test_a_recording_with_the_wrong_rate_is_not_continued(tmp_path):
    path = tmp_path / 'a.wav'
    with wave.open(str(path), 'wb') as other:
        other.setnchannels(1)
        other.setsampwidth(2)
        other.setframerate(44100)
        other.writeframes(b'\x00\x00' * 100)
    with pytest.raises(WavError):
        WavRecorder(path, append=True)


def test_the_header_is_patched_while_recording_so_a_crash_leaves_a_playable_file(tmp_path):
    path = tmp_path / 'a.wav'
    recorder = WavRecorder(path)
    recorder.write(tone(1))
    recorder.patch_header()
    with wave.open(str(path), 'rb') as peek:                   # read while the recorder still holds the file open
        assert peek.getnframes() == RATE
    recorder.write(tone(1))
    recorder.close()
    assert sample_count(path) == 2 * RATE


def test_replay_can_cut_a_clip_across_the_join_of_two_recordings(tmp_path):
    path = tmp_path / 'a.wav'
    with WavRecorder(path) as first:
        first.write(np.zeros(RATE, dtype=np.float32))
    with WavRecorder(path, append=True) as second:
        second.write(np.full(RATE, 0.5, dtype=np.float32))
    clip = clip_wav(path, 0.5, seconds=1.0)                    # half silence from the first, half tone from the second
    out = tmp_path / 'clip.wav'
    out.write_bytes(clip)
    with wave.open(str(out), 'rb') as saved:
        samples = np.frombuffer(saved.readframes(saved.getnframes()), dtype='<i2')
    assert len(samples) == RATE
    assert np.all(samples[:RATE // 2 - 10] == 0) and np.all(samples[RATE // 2 + 10:] > 10000)


def test_sample_count_of_a_missing_or_broken_file_is_zero(tmp_path):
    assert sample_count(tmp_path / 'none.wav') == 0
    broken = tmp_path / 'broken.wav'
    broken.write_bytes(b'not a wav at all' * 4)
    assert sample_count(broken) == 0
    assert HEADER_BYTES == 44
