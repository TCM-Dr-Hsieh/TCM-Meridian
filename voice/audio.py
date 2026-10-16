"""Local microphone capture, fixed overlapping windows, and continuous WAV recording.

Differences from voice_to_text: microphone only (plus a dev-only file replay), an unbounded
chunk queue (recording never stops because processing is slow), and WAV headers that are
patched about once a second so a crash leaves a playable file.

Differences from TCM-Meridian-mini: a session has ONE recording that is stopped and started again, so a source can
continue an existing WAV (`append=True`) and its clock starts where the file ends (`start_samples`).
"""
from __future__ import annotations

import asyncio
import os
import queue
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .wavfile import RATE, WavRecorder

BLOCK = 1600
HARDWARE_LOCK = threading.Lock()
S_OK, S_FALSE, COINIT_MULTITHREADED = 0, 1, 0         # HRESULTs and the apartment mode SoundCard itself uses
RPC_E_CHANGED_MODE = -2147417850                      # 0x80010106 as the signed HRESULT ctypes returns


@dataclass
class Chunk:
    index: int
    start: float
    samples: np.ndarray
    final: bool = True
    reason: str = 'stop'          # 'window' = full window, more speech may follow; 'stop' = tail

    @property
    def duration(self) -> float:
        return len(self.samples) / RATE


class WindowChunker:
    """Emit [0, L], [L-Z, 2L-Z], ... and one unfinished tail on flush."""

    def __init__(self, window_seconds: float, overlap_seconds: float, start_sample: int = 0):
        self.length = round(window_seconds * RATE)
        self.overlap = round(overlap_seconds * RATE)
        self.stride = self.length - self.overlap
        self.buffer = np.empty(0, dtype=np.float32)
        # A continued recording starts its windows where the file ends: the clock is the position in the whole recording.
        self.start_sample = start_sample
        self.received_samples = start_sample
        self.last_emitted_end = start_sample
        self.index = 0

    def feed(self, samples):
        incoming = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not len(incoming):
            return
        self.received_samples += len(incoming)
        self.buffer = np.concatenate((self.buffer, incoming))
        while len(self.buffer) >= self.length:
            end = self.start_sample + self.length
            yield Chunk(self.index, self.start_sample / RATE, self.buffer[:self.length].copy(), True, 'window')
            self.index += 1
            self.last_emitted_end = end
            self.buffer = self.buffer[self.stride:].copy()
            self.start_sample += self.stride

    def flush(self):
        if self.received_samples <= self.last_emitted_end or not len(self.buffer):
            return None
        chunk = Chunk(self.index, self.start_sample / RATE, self.buffer.copy(), True, 'stop')
        self.last_emitted_end = self.received_samples
        self.buffer = np.empty(0, dtype=np.float32)
        return chunk


def _ole32():
    """The Windows COM library, or None where there is no COM."""
    if os.name != 'nt':
        return None
    import ctypes
    library = ctypes.windll.ole32
    library.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    library.CoInitializeEx.restype = ctypes.c_long
    return library


@contextmanager
def com_initialised():
    """Initialise COM on the calling thread for the duration of the block (Windows; a no-op elsewhere).

    SoundCard initialises COM only on the thread that first imports it. Every visit records on a new thread, and the thread
    that imported SoundCard during the first visit ends with it: from then on no thread holds COM, and the next visit's
    thread fails with 0x800401F0 (CO_E_NOTINITIALIZED). So each thread that calls SoundCard initialises COM itself, and
    balances it. Two rules: call this AFTER `import soundcard` (that import initialises COM by itself and fails if the
    thread already did), and release what SoundCard handed out (COM objects) before the block ends.

    Only a successful call is balanced (S_OK, and S_FALSE: already initialised here, which still needs its own
    CoUninitialize). RPC_E_CHANGED_MODE means the thread is already in another apartment: COM is usable, nothing to undo.
    Any other HRESULT (E_INVALIDARG, E_OUTOFMEMORY, E_UNEXPECTED) means the thread has no COM: fail here, with the code,
    instead of letting SoundCard fail later with something that points nowhere.
    """
    library = _ole32()
    result = library.CoInitializeEx(None, COINIT_MULTITHREADED) if library is not None else None
    if library is not None and result not in (S_OK, S_FALSE, RPC_E_CHANGED_MODE):
        raise RuntimeError(f'無法初始化 Windows COM（CoInitializeEx 回傳 0x{result & 0xFFFFFFFF:08x}）。')
    try:
        yield
    finally:
        if result in (S_OK, S_FALSE):
            library.CoUninitialize()


def list_microphones() -> dict[str, str]:
    import soundcard as sc                      # first: see com_initialised
    with com_initialised():
        microphones = sc.all_microphones(include_loopback=False)
        try:
            return {d.id: d.name for d in microphones}
        finally:
            microphones = None                  # COM objects: free them while COM is still initialised on this thread


def mic_iterator(device_id: str, stop: threading.Event):
    import soundcard as sc                      # first: see com_initialised
    if not HARDWARE_LOCK.acquire(blocking=False):
        raise RuntimeError('已有其他程式或視窗正在使用錄音裝置。')
    mic = recorder = None
    try:
        with com_initialised():
            try:
                mic = sc.get_microphone(id=device_id, include_loopback=False) if device_id else sc.default_microphone()
                # SoundCard's WASAPI backend has a known single-channel capture issue:
                # request all physical channels, then downmix explicitly.
                with mic.recorder(samplerate=RATE, channels=None, blocksize=BLOCK) as recorder:
                    while not stop.is_set():
                        samples = recorder.record(numframes=BLOCK)
                        yield samples.mean(axis=1).astype(np.float32)
            finally:
                mic = recorder = None           # COM objects: free them while COM is still initialised on this thread
    finally:
        HARDWARE_LOCK.release()


def file_iterator(path: str, stop: threading.Event, speed: float = 1.0):
    """Development aid: replay an audio file as if it were the microphone (paced in real time)."""
    import av
    with av.open(path) as container:
        if not container.streams.audio:
            raise ValueError('檔案沒有可讀取的音軌。')
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=RATE)
        pending = np.empty(0, dtype=np.float32)

        def frames():
            for frame in container.decode(audio=0):
                for converted in resampler.resample(frame):
                    yield converted.to_ndarray().reshape(-1)
            for converted in resampler.resample(None):
                yield converted.to_ndarray().reshape(-1)

        for samples in frames():
            pending = np.concatenate((pending, samples))
            while len(pending) >= BLOCK:
                if stop.is_set():
                    return
                yield pending[:BLOCK].copy()
                pending = pending[BLOCK:]
                time.sleep(BLOCK / RATE / max(speed, 0.1))
        if len(pending):
            yield pending


class AudioSource:
    """Producer thread: iterator -> WAV file + window chunks. The queue is unbounded."""

    def __init__(self, make_iterator, *, window_seconds: float, overlap_seconds: float,
                 recording_path: Path | None, start_samples: int = 0, append: bool = False):
        self._make_iterator = make_iterator
        self.recording_path = recording_path
        self.append = append                       # continue the recording that is already in the file
        self.start_samples = start_samples         # samples already in it: this source's clock starts there
        self.chunker = WindowChunker(window_seconds, overlap_seconds, start_samples)
        self.queue: queue.Queue[Chunk] = queue.Queue()
        self.stop = threading.Event()
        self.cancelled = threading.Event()
        self.done = threading.Event()
        self.error = ''
        self.level = 0.0
        self.recorded_samples = start_samples
        self.thread = threading.Thread(target=self._produce, daemon=True, name='audio-source')

    @property
    def audio_time(self) -> float:
        return self.recorded_samples / RATE

    def start(self):
        self.thread.start()

    def _produce(self):
        last_patch = time.monotonic()
        try:
            context = WavRecorder(self.recording_path, append=self.append) if self.recording_path else nullcontext()
            with context as recording:
                iterator = self._make_iterator(self.stop)
                try:
                    for samples in iterator:
                        if self.cancelled.is_set():
                            break
                        if recording is not None:
                            recording.write(samples)
                            if time.monotonic() - last_patch > 1:
                                recording.patch_header()
                                last_patch = time.monotonic()
                        self.recorded_samples += len(samples)
                        self.level = min(1.0, float(np.sqrt(np.mean(samples ** 2))) * 5)
                        for chunk in self.chunker.feed(samples):
                            self.queue.put(chunk)
                        if self.stop.is_set():
                            break
                finally:
                    close = getattr(iterator, 'close', None)
                    if close:
                        close()
        except Exception as exc:
            self.error = f'音訊來源錯誤：{exc}'
            if '0x80070005' in str(exc).lower():
                self.error += ('\nWindows 拒絕存取錄音裝置。請確認「設定 → 隱私權與安全性 → 麥克風」'
                               '已允許桌面應用程式存取麥克風。')
        finally:
            if not self.cancelled.is_set():
                tail = self.chunker.flush()
                if tail is not None:
                    self.queue.put(tail)
            self.level = 0.0
            self.done.set()

    async def next_chunk(self) -> Chunk | None:
        while True:
            try:
                return self.queue.get_nowait()
            except queue.Empty:
                if self.done.is_set():
                    try:
                        return self.queue.get_nowait()   # recheck after the completion barrier
                    except queue.Empty:
                        return None
                await asyncio.sleep(0.05)

    def request_stop(self):
        self.stop.set()

    async def close(self):
        self.cancelled.set()
        self.stop.set()
        if self.thread.is_alive():
            await asyncio.to_thread(self.thread.join, 3)


def make_source(asr_settings, recording_path: Path | None, *, start_samples: int = 0, append: bool = False) -> AudioSource:
    """Create the recording's audio source: the microphone, or a replayed file when TCM_FAKE_AUDIO is set."""
    fake = os.environ.get('TCM_FAKE_AUDIO', '').strip()
    if fake:
        speed = float(os.environ.get('TCM_FAKE_AUDIO_SPEED', '1') or 1)

        def make_iterator(stop):
            return file_iterator(fake, stop, speed)
    else:
        device_id = asr_settings.microphone_id

        def make_iterator(stop):
            return mic_iterator(device_id, stop)
    return AudioSource(make_iterator, window_seconds=asr_settings.window_seconds,
                       overlap_seconds=asr_settings.overlap_seconds, recording_path=recording_path,
                       start_samples=start_samples, append=append)
