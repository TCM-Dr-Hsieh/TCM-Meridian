"""COM on the audio threads (Windows).

Every visit records on a NEW thread, and SoundCard initialises COM only on the thread that first imports it. When that
thread (the first visit's) ends, no thread holds COM any more and the next visit's thread fails with 0x800401F0
(CO_E_NOTINITIALIZED): "the second visit has no microphone". Each thread that calls SoundCard now initialises COM itself,
in this order: import soundcard, CoInitializeEx, use it, free what it returned, CoUninitialize.

The tests marked Windows run the real library in a fresh process (COM state is per process); the others use a fake
`soundcard` and a fake `ole32` and run anywhere.
"""
import importlib.abc
import importlib.util
import subprocess
import sys
import textwrap
import threading
import types
from pathlib import Path

import numpy as np
import pytest

from voice import audio

ROOT = Path(__file__).resolve().parent.parent
WINDOWS = pytest.mark.skipif(sys.platform != 'win32', reason='COM exists only on Windows')
RPC_E_CHANGED_MODE = -2147417850                    # 0x80010106 as the signed HRESULT ctypes returns


# ---------------------------------------------------------------------------------------- the real library (Windows)
@WINDOWS
def test_successive_threads_can_all_use_soundcard_in_a_fresh_process():
    """The reported failure, without touching any audio: the first thread imports SoundCard and ends, a later one used to
    get 0x800401F0 from the very same call."""
    script = textwrap.dedent('''
        import sys, threading
        sys.path.insert(0, sys.argv[1])
        from voice.audio import list_microphones
        outcome = []

        def run():
            try:
                list_microphones()
                outcome.append('ok')
            except Exception as exc:
                outcome.append(repr(exc))

        for _ in range(3):
            thread = threading.Thread(target=run)
            thread.start()
            thread.join()
        print('|'.join(outcome))
    ''')
    result = subprocess.run([sys.executable, '-c', script, str(ROOT)], capture_output=True, text=True, encoding='utf-8',
                            errors='replace', timeout=60)
    outcome = result.stdout.strip().split('|')
    if outcome[0] != 'ok':                                  # no SoundCard or no audio service here: nothing to compare
        pytest.skip(f'the first listing already fails on this machine: {outcome[0] or result.stderr.strip()[-200:]}')
    assert outcome == ['ok', 'ok', 'ok'], outcome


@WINDOWS
def test_com_initialised_is_balanced_on_the_thread_that_uses_it():
    seen = {}

    def run():
        library = audio._ole32()
        with audio.com_initialised():
            seen['inside'] = library.CoInitializeEx(None, 0)        # S_FALSE: this thread is already initialised ...
            library.CoUninitialize()                                 # ... so give that extra call back
        seen['after'] = library.CoInitializeEx(None, 0)             # S_OK: the block left nothing behind
        library.CoUninitialize()

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    assert seen == {'inside': audio.S_FALSE, 'after': audio.S_OK}


# ------------------------------------------------------------------------------------ a fake SoundCard and a fake ole32
class FakeCom:
    """ole32 for one thread: a count like the real one (S_OK the first time, S_FALSE after), and a log shared with SoundCard."""

    def __init__(self, events, forced=None):
        self.events, self.depth, self.forced = events, 0, forced

    def CoInitializeEx(self, reserved, mode):                      # noqa: N802  (the Windows name)
        assert reserved is None and mode == audio.COINIT_MULTITHREADED
        self.events.append('CoInitializeEx')
        if self.forced is not None:
            return self.forced
        self.depth += 1
        return audio.S_OK if self.depth == 1 else audio.S_FALSE

    def CoUninitialize(self):                                      # noqa: N802
        self.events.append('CoUninitialize')
        self.depth -= 1

    def initialise_like_the_soundcard_import(self):
        """What the real library does the moment it is imported: initialise COM on the importing thread, and FAIL if the
        thread already did (it treats S_FALSE as an error: 'Error 0x100000001')."""
        self.depth += 1
        if self.depth != 1:
            raise RuntimeError('Error 0x100000001')


@pytest.fixture
def world(monkeypatch):
    """A fake `soundcard` that is imported for real (so the order of the import and CoInitializeEx is observable) and a fake
    `audio._ole32`. Yields (events, com, settings): `events` logs every step in order; `settings` can make the device fail."""
    events: list[str] = []
    com = FakeCom(events)
    settings = {'fail_after': None}

    class Recorder:
        def __init__(self):
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            events.append('recorder closed')

        def record(self, numframes):
            self.calls += 1
            if settings['fail_after'] is not None and self.calls > settings['fail_after']:
                raise RuntimeError('device unplugged')
            return np.zeros((numframes, 2), dtype=np.float32)

        def __del__(self):
            events.append('recorder freed')

    class Microphone:
        id, name = 'mic-1', 'Fake microphone'

        def recorder(self, samplerate, channels, blocksize):
            return Recorder()

        def __del__(self):
            events.append('microphone freed')

    class Loader(importlib.abc.Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            events.append('import soundcard')
            com.initialise_like_the_soundcard_import()

            def default_microphone():
                events.append('default_microphone')
                return Microphone()

            def get_microphone(id, include_loopback):                  # noqa: A002
                events.append('get_microphone')
                return Microphone()

            def all_microphones(include_loopback):
                events.append('all_microphones')
                return [Microphone(), Microphone()]

            module.default_microphone, module.get_microphone, module.all_microphones = (
                default_microphone, get_microphone, all_microphones)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            return importlib.util.spec_from_loader('soundcard', Loader()) if name == 'soundcard' else None

    monkeypatch.setitem(sys.modules, 'soundcard', types.ModuleType('placeholder'))      # restored (or removed) at the end
    del sys.modules['soundcard']                                                          # so the next import is a first one
    monkeypatch.setattr(sys, 'meta_path', [Finder(), *sys.meta_path])
    monkeypatch.setattr(audio, '_ole32', lambda: com)
    return events, com, settings


def only(events, name):
    return [i for i, event in enumerate(events) if event == name]


def test_the_microphone_thread_imports_then_initialises_com_and_releases_everything_before_uninitialising(world):
    events, com, _ = world
    generator = audio.mic_iterator('', threading.Event())
    next(generator), next(generator)
    generator.close()
    assert events[:3] == ['import soundcard', 'CoInitializeEx', 'default_microphone']     # import first, or it fails
    assert only(events, 'CoInitializeEx') == [1] and len(only(events, 'CoUninitialize')) == 1
    uninitialise = events.index('CoUninitialize')
    assert uninitialise == len(events) - 1                                                    # nothing happens after it
    for name in ('recorder closed', 'recorder freed', 'microphone freed'):
        assert events.index(name) < uninitialise, (name, events)                              # COM objects die while COM lives
    assert com.depth == 1                                                                     # only the import's own init is left
    assert audio.HARDWARE_LOCK.acquire(blocking=False)                                        # and the device lock is free again
    audio.HARDWARE_LOCK.release()


def test_a_device_that_fails_mid_recording_is_still_released_before_com_is_uninitialised(world):
    events, _, settings = world
    settings['fail_after'] = 1
    with pytest.raises(RuntimeError, match='device unplugged') as caught:
        for _ in audio.mic_iterator('', threading.Event()):
            pass
    # `caught` holds the traceback, which keeps the failing `record()` frame -- and so the recorder instance -- alive, as it
    # would in a real failure. That is fine: SoundCard releases the recorder's COM objects in __exit__ ('recorder closed'),
    # not when it is garbage collected. What the generator itself holds (the microphone) must still be freed inside the block.
    assert caught.value is not None
    uninitialise = events.index('CoUninitialize')
    for name in ('recorder closed', 'microphone freed'):
        assert events.index(name) < uninitialise, (name, events)
    assert 'recorder freed' not in events                          # (pinned by the traceback, as explained above)
    assert audio.HARDWARE_LOCK.acquire(blocking=False)
    audio.HARDWARE_LOCK.release()


def test_listing_the_microphones_initialises_com_after_the_import_and_frees_the_devices_inside(world):
    events, com, _ = world
    assert audio.list_microphones() == {'mic-1': 'Fake microphone'}
    assert events[:3] == ['import soundcard', 'CoInitializeEx', 'all_microphones']
    uninitialise = events.index('CoUninitialize')
    assert events.count('microphone freed') == 2 and uninitialise == len(events) - 1
    assert all(i < uninitialise for i in only(events, 'microphone freed'))
    assert com.depth == 1


def test_a_second_thread_gets_its_own_initialisation(world):
    """The same module is already imported (as in a second visit): the thread initialises COM itself, and nothing else."""
    events, com, _ = world
    audio.list_microphones()                                       # the first import
    com.depth = 0                                                  # that thread ended: its COM went with it
    del events[:]
    audio.list_microphones()
    assert events[0] == 'CoInitializeEx' and 'import soundcard' not in events
    assert only(events, 'CoUninitialize') == [len(events) - 1] and com.depth == 0


@pytest.mark.parametrize('result, uninitialised', [
    (audio.S_OK, True), (audio.S_FALSE, True),                     # a success: SoundCard is used, and the call is balanced
    (RPC_E_CHANGED_MODE, False)])                                  # the thread is in another apartment: usable, nothing to undo
def test_only_a_successful_initialisation_is_balanced(world, result, uninitialised):
    events, com, _ = world
    com.forced = result
    generator = audio.mic_iterator('', threading.Event())
    next(generator)
    generator.close()
    assert 'default_microphone' in events and ('CoUninitialize' in events) is uninitialised


@pytest.mark.parametrize('result, code', [
    (-2147024809, '0x80070057'),                                   # E_INVALIDARG
    (-2147024882, '0x8007000e'),                                   # E_OUTOFMEMORY
    (-2147418113, '0x8000ffff')])                                  # E_UNEXPECTED
def test_a_failed_initialisation_stops_there_with_its_code_and_touches_nothing(world, result, code):
    """The thread has no COM: say so now, with the HRESULT, instead of letting SoundCard fail later with something that
    points nowhere (and there is nothing to balance)."""
    events, com, _ = world
    com.forced = result
    with pytest.raises(RuntimeError, match=code) as caught:
        next(audio.mic_iterator('', threading.Event()))
    assert '無法初始化 Windows COM' in str(caught.value)
    assert 'default_microphone' not in events and 'CoUninitialize' not in events
    assert audio.HARDWARE_LOCK.acquire(blocking=False)             # the device lock is not left held either
    audio.HARDWARE_LOCK.release()
    with pytest.raises(RuntimeError, match=code):                  # listing the microphones stops the same way
        audio.list_microphones()
    assert 'all_microphones' not in events and 'CoUninitialize' not in events


def test_a_failed_initialisation_is_the_audio_source_error_the_physician_sees(world):
    """Through the real producer thread: the status bar shows `音訊來源錯誤：…`, with the code, not a later SoundCard error."""
    events, com, _ = world
    com.forced = -2147024882                                       # E_OUTOFMEMORY
    source = audio.AudioSource(lambda stop: audio.mic_iterator('', stop), window_seconds=6, overlap_seconds=3,
                               recording_path=None)
    source.start()
    source.thread.join(5)
    assert source.done.is_set() and not source.thread.is_alive()
    assert source.error.startswith('音訊來源錯誤：無法初始化 Windows COM') and '0x8007000e' in source.error
    assert 'default_microphone' not in events


def test_without_com_nothing_is_called(world, monkeypatch):
    """Not Windows, or no ole32: the audio thread works as before."""
    events, _, _ = world
    monkeypatch.setattr(audio, '_ole32', lambda: None)
    generator = audio.mic_iterator('', threading.Event())
    next(generator)
    generator.close()
    assert 'CoInitializeEx' not in events and 'CoUninitialize' not in events and 'default_microphone' in events


def test_there_is_no_com_off_windows(monkeypatch):
    """`ctypes.windll` does not exist elsewhere: the helper must not reach for it."""
    monkeypatch.setattr(audio.os, 'name', 'posix')
    assert audio._ole32() is None
    with audio.com_initialised():                                  # a plain no-op, and nothing to undo
        pass


def test_initialising_com_before_the_first_import_would_break_the_import(world, monkeypatch):
    """Why the order matters (the real library raises 'Error 0x100000001' when COM is already initialised on the thread
    that imports it): the fake reproduces that, so a refactor that initialises first fails here."""
    events, com, _ = world
    com.depth = 1                                                  # COM already initialised on this thread
    with pytest.raises(RuntimeError, match='0x100000001'):
        audio.mic_iterator('', threading.Event()).__next__()
    assert 'default_microphone' not in events
