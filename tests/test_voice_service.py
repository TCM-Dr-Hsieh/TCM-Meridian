"""VoiceManager: one recording and one transcript per session; stop, continue, restore; the transcript for the agents."""
import asyncio
import json
import wave
from types import SimpleNamespace

import pytest

from tests.helpers import FakeASR, FakeLLM, make_settings, silent_source
from tests.test_remote_audio import blocks, open_socket, wait_for
from voice.config import VoiceSettings
from voice.llm import LLMClient
from voice.remote import REMOTE_SOURCES, RemoteAudioSource
from voice.replay import replay_response
from voice.service import NO_MODELS_MESSAGE, VoiceError, VoiceManager, voice_paths
from voice.wavfile import RATE, sample_count

DATE = '2026-10-16'


class Request:
    def __init__(self, **headers):
        self.headers = {key.replace('_', '-').lower(): value for key, value in headers.items()}


@pytest.fixture(autouse=True)
def clean_registry():
    yield
    REMOTE_SOURCES.clear()


@pytest.fixture
def patient(tmp_path):
    folder = tmp_path / 'patient_data' / 'A123_1980-01-01_王'
    folder.mkdir(parents=True)
    return str(folder)


def make_manager(seconds_per_recording=7, text='我頭痛三天了', settings=None, llm=None):
    settings = settings or make_settings()
    config = {'voice_config': settings.to_dict()}
    seconds = [seconds_per_recording]
    fake = llm or FakeLLM()
    manager = VoiceManager(
        load_config=lambda: config, asr=FakeASR(default=text), llm_client=LLMClient(fake.transport()), llm_backoff=0,
        source_factory=lambda path, **kw: silent_source(settings, seconds[0], recording_path=path, **kw),
        behavior_log=lambda *args, **kwargs: None)
    manager.models_problem = lambda settings=None: ''              # the models are not needed with a fake ASR
    manager.next_seconds = lambda value: seconds.__setitem__(0, value)
    manager.fake_llm = fake
    return manager


async def record(manager, seconds=None):
    if seconds is not None:
        manager.next_seconds(seconds)
    await manager.start()
    await asyncio.wait_for(manager.session._tasks[0], 30)         # the fake microphone ends by itself: the transcript is done
    await manager.stop()


# ---------------------------------------------------------------------------------------------------------------------
async def test_a_recording_makes_the_wav_the_transcript_and_the_events(patient):
    manager = make_manager()
    manager.attach(patient, DATE)
    await record(manager)
    paths = voice_paths(patient, DATE)
    assert sample_count(paths.audio) == 7 * RATE and manager.session.state == 'idle'
    assert [s.index for s in manager.session.pipeline.segments] == [1, 2]
    assert '我頭痛三天了' in paths.transcript_txt.read_text(encoding='utf-8')
    saved = json.loads(paths.transcript_json.read_text(encoding='utf-8'))
    assert len(saved['segments']) == 2 and saved['committed_end'] == 7.0
    events = [json.loads(line) for line in paths.events.read_text(encoding='utf-8').splitlines()]
    started = next(e for e in events if e['type'] == 'recording_started')
    assert started['audio_source'] == 'local' and started['continued'] is False
    assert any(e['type'] == 'recording_stopped' for e in events)
    assert paths.llm_log.exists()                                  # every correction call is kept with its full input and output


async def test_recording_again_continues_the_same_file_and_the_numbering(patient):
    manager = make_manager()
    manager.attach(patient, DATE)
    await record(manager, 7)
    first_lines = manager.session.pipeline.transcript_txt()
    await record(manager, 4)
    paths = voice_paths(patient, DATE)
    assert sample_count(paths.audio) == 11 * RATE                  # one file, longer
    with wave.open(str(paths.audio), 'rb') as saved:
        assert saved.getnframes() == 11 * RATE
    segments = manager.session.pipeline.segments
    assert [s.index for s in segments] == list(range(1, len(segments) + 1)) and len(segments) >= 3
    assert segments[2].start >= 7.0 - 1e-6                         # the clock goes on where the audio ended
    assert manager.session.pipeline.transcript_txt().startswith(first_lines)      # earlier lines are unchanged
    events = [json.loads(line) for line in paths.events.read_text(encoding='utf-8').splitlines()]
    second = [e for e in events if e['type'] == 'recording_started'][1]
    assert second['continued'] is True and second['from_seconds'] == 7.0


async def test_the_transcript_survives_a_restart_and_can_be_continued(patient):
    first = make_manager()
    first.attach(patient, DATE)
    await record(first, 7)
    before = first.snapshot_text(patient, DATE)
    assert before.startswith('【診間對話逐字稿 · 截至 00:07')

    second = make_manager()                                        # a new process: nothing in memory
    second.attach(patient, DATE)
    assert second.snapshot_text(patient, DATE) == before           # restored, byte for byte
    await record(second, 5)
    after = second.snapshot_text(patient, DATE)
    assert after.startswith('【診間對話逐字稿 · 截至 00:12') and '語音#1 ' in after and '語音#3 ' in after
    assert sample_count(voice_paths(patient, DATE).audio) == 12 * RATE


async def test_the_snapshot_is_empty_without_a_transcript_and_follows_the_requested_session(patient, tmp_path):
    manager = make_manager()
    assert manager.snapshot_text(patient, DATE) == ''             # nothing recorded yet (and the session gets loaded by the ask)
    assert manager.session.folder == patient
    await record(manager, 7)
    assert manager.snapshot_text(patient, DATE) != ''
    other = str(tmp_path / 'patient_data' / 'B_1990-02-02_李')
    assert manager.snapshot_text(other, DATE) == ''               # another session has its own (here: no) transcript
    assert manager.session.folder == other
    assert manager.snapshot_text(patient, DATE) != ''             # and the first one is restored from its files when asked again


async def test_a_snapshot_never_comes_from_another_session_while_a_recording_runs(patient, tmp_path):
    manager = make_manager(seconds_per_recording=60)
    manager.attach(patient, DATE)
    await manager.start()
    try:
        other = str(tmp_path / 'patient_data' / 'B_1990-02-02_李')
        assert manager.snapshot_text(other, DATE) == ''           # refused to switch: nothing from the other session, no crash
        assert manager.session.folder == patient
    finally:
        await manager.stop()


async def test_the_recording_clock_only_grows_and_says_paused_between_recordings(patient):
    manager = make_manager()
    assert manager.recording_position(patient, DATE) == (None, False)         # no session loaded yet
    manager.attach(patient, DATE)
    assert manager.recording_position(patient, DATE) == (None, False)         # loaded, never recorded
    await manager.start()
    seconds, paused = manager.recording_position(patient, DATE)
    assert seconds is not None and paused is False                            # recording: the live clock
    await asyncio.wait_for(manager.session._tasks[0], 30)
    await manager.stop()
    assert manager.recording_position(patient, DATE) == (7.0, True)           # between recordings: where the file ends, paused
    assert manager.recording_position('somewhere/else', DATE) == (None, False)
    await record(manager, 3)
    assert manager.recording_position(patient, DATE) == (10.0, True)


async def test_a_running_recording_keeps_the_session_it_belongs_to(patient, tmp_path):
    manager = make_manager(seconds_per_recording=60)
    manager.attach(patient, DATE)
    await manager.start()
    try:
        assert manager.busy and manager.recording
        manager.attach(patient, DATE)                                          # the same session: fine
        with pytest.raises(VoiceError, match='停止錄製'):
            manager.attach(str(tmp_path / 'other'), DATE)
        with pytest.raises(VoiceError, match='停止錄製'):
            manager.detach()
        with pytest.raises(VoiceError):
            await manager.start()                                              # not twice
    finally:
        await manager.stop()
    assert not manager.busy
    manager.detach()
    assert manager.session is None


async def test_no_models_means_a_clear_refusal_and_nothing_else_is_affected(patient, tmp_path):
    settings = make_settings()
    settings.asr.asr_model_dir = str(tmp_path / 'no-asr-model')
    manager = VoiceManager(load_config=lambda: {'voice_config': settings.to_dict()}, asr=FakeASR(),
                           behavior_log=lambda *a, **k: None)
    problem = manager.models_problem()
    assert NO_MODELS_MESSAGE in problem and 'install.cmd' in problem
    manager.attach(patient, DATE)
    with pytest.raises(VoiceError, match='目前無下載語音模型'):
        await manager.start()
    assert manager.session.state == 'idle'
    await manager.preload()
    assert manager.asr_status == '未安裝語音模型（不預載）'              # nothing to load, nothing loaded


async def test_preload_loads_the_models_when_they_are_there(patient):
    manager = make_manager()
    await manager.preload()
    assert manager.asr_status == '就緒（fake）' and manager.asr.loaded


async def test_a_broken_voice_config_is_reported_not_raised_into_the_pages(patient):
    manager = VoiceManager(load_config=lambda: {'voice_config': {'asr': {'window_seconds': 1}}}, asr=FakeASR(),
                           behavior_log=lambda *a, **k: None)
    assert '語音設定有問題' in manager.models_problem()
    assert manager.safe_settings().asr.window_seconds == 6.0                   # display code falls back to the defaults


# ---------------------------------------------------------------------------------------------------------------------
async def test_a_remote_recording_records_transcribes_and_stops_cleanly(patient):
    manager = make_manager()
    manager.attach(patient, DATE)
    await manager.start(remote=True, device_label='  測試麥克風\n（USB）  ')
    token = manager.remote_token
    source = manager.session.source
    assert isinstance(source, RemoteAudioSource) and token and REMOTE_SOURCES[token] is source
    ws, task = await open_socket(source)
    stream = blocks(7)
    for sequence, block in enumerate(stream):
        ws.audio(sequence, block)
    await wait_for(lambda: len(ws.acks()) == len(stream), timeout=15)
    await wait_for(lambda: manager.session.pipeline.segments, timeout=15)       # ASR + correction ran on the remote audio

    stopping = asyncio.create_task(manager.stop())
    await wait_for(lambda: 'stop' in ws.types(), timeout=15)                    # the browser is asked to stop
    ws.say(type='stop')
    await asyncio.wait_for(stopping, 30)
    await asyncio.wait_for(task, 5)
    assert token not in REMOTE_SOURCES and manager.remote_token == ''
    assert sample_count(voice_paths(patient, DATE).audio) == 7 * RATE
    events = [json.loads(line) for line in voice_paths(patient, DATE).events.read_text(encoding='utf-8').splitlines()]
    started = next(e for e in events if e['type'] == 'recording_started')
    assert started['audio_source'] == 'remote' and started['device'] == '遠端瀏覽器麥克風 · 測試麥克風 （USB）'


async def test_a_local_recording_has_no_remote_token(patient):
    manager = make_manager()
    manager.attach(patient, DATE)
    await record(manager, 2)
    assert manager.remote_token == '' and not REMOTE_SOURCES


async def test_a_remote_recording_whose_browser_never_connects_can_still_be_stopped(patient):
    manager = make_manager()
    manager.attach(patient, DATE)
    await manager.start(remote=True)
    source = manager.session.source
    await asyncio.wait_for(manager.stop(), 30)
    assert source.done.is_set() and '沒有收到任何遠端音訊' in manager.session.error
    assert manager.session.state == 'idle'


# ---------------------------------------------------------------------------------------------------------------------
async def test_replay_serves_only_the_loaded_sessions_recording(patient, tmp_path):
    manager = make_manager()
    manager.attach(patient, DATE)
    await record(manager, 7)
    token = manager.session.replay_token
    assert replay_response(manager, Request(), 0.0, token).status_code == 200
    assert replay_response(manager, Request(), 0.0, 'wrong').status_code == 409
    other_patient = tmp_path / 'patient_data' / 'B_1990-02-02_李'
    other_patient.mkdir()
    manager.detach()
    manager.attach(str(other_patient), DATE)
    assert manager.session.replay_token != token
    assert replay_response(manager, Request(), 0.0, token).status_code == 409   # a page left over from the first patient
    manager.detach()
    assert replay_response(manager, Request(), 0.0, token).status_code == 409


async def test_a_damaged_transcript_file_means_an_empty_transcript_not_a_crash(patient):
    paths = voice_paths(patient, DATE)
    paths.log_dir.mkdir(parents=True)
    paths.transcript_json.write_text('{ this is not json', encoding='utf-8')
    manager = make_manager()
    session = manager.attach(patient, DATE)
    assert session.pipeline is None and manager.snapshot_text(patient, DATE) == ''
    await record(manager, 7)                                                    # and recording works as if it were new
    assert len(manager.session.pipeline.segments) == 2


async def test_the_settings_snapshot_for_the_audit_never_holds_the_api_key():
    data = VoiceSettings().public_dict()
    assert 'api_key' not in data['corrector']
    assert SimpleNamespace(**data['asr']).window_seconds == 6.0


# ---------------------------------------------------------------------------------------------------------------------
# The voice window of one page (browser tab) against the manager every page shares
# ---------------------------------------------------------------------------------------------------------------------
def make_page(manager, folder=None, date=None):
    from unittest.mock import MagicMock
    from ui_app.controllers.voice_controller import VoiceController
    state = {'selected_patient_folder': folder, 'selected_session_date': date}
    page = VoiceController(ui=MagicMock(), app_state=state, manager=manager)
    page.w = MagicMock()
    page.w.source.value = 'local'
    return page, state


async def test_a_page_without_a_patient_does_not_unload_the_session_another_page_works_on(patient):
    manager = make_manager()
    working, _ = make_page(manager, patient, DATE)
    idle, _ = make_page(manager)                     # a second tab, nothing selected
    working.open_panel = idle.open_panel = True
    working.tick()
    session = manager.session
    assert session is not None and session.folder == patient
    for _ in range(3):                               # the two pages tick in turns, as they do in the browser
        idle.tick()
        working.tick()
    assert manager.session is session                # the same object: never detached, never rebuilt
    assert idle.sync_session() is False              # nothing to work on from the empty page while nothing records


async def test_a_page_loaded_in_the_middle_of_a_recording_can_still_open_the_window_and_stop_it(patient):
    manager = make_manager(seconds_per_recording=30)
    manager.attach(patient, DATE)
    await manager.start()
    try:
        fresh, _ = make_page(manager)                # the browser was reloaded: no patient selected on this page
        assert fresh.sync_session() is True
        fresh.toggle()
        assert fresh.open_panel is True              # the window opens on the running recording
        await fresh.stop()
        assert manager.session.state == 'idle'
    finally:
        await manager.shutdown()


# ---------------------------------------------------------------------------------------------------------------------
# Review findings: the recording clock of a session that is not loaded, and a continued recording's speaker settings
# ---------------------------------------------------------------------------------------------------------------------
async def test_the_recording_position_of_a_session_that_is_not_loaded_comes_from_its_file(patient):
    from pathlib import Path
    first = make_manager()
    first.attach(patient, DATE)
    await record(first, 7)
    restarted = make_manager()                       # the program was restarted, or a page reloaded: nothing is attached yet
    assert restarted.session is None
    assert restarted.recording_position(patient, DATE) == (7.0, True)
    assert restarted.recording_position(patient, '2026-01-01') == (None, False)       # no recording that day
    assert restarted.recording_position('', DATE) == (None, False)
    other = Path(patient).with_name('B456_1990-02-02_李')
    other.mkdir()
    restarted.attach(str(other), DATE)               # another session is loaded: the answer is still about the one asked for
    assert restarted.recording_position(patient, DATE) == (7.0, True)
    assert restarted.recording_position(str(other), DATE) == (None, False)


async def test_a_continued_recording_follows_the_speaker_settings_in_force_now(patient):
    from tests.speaker_fakes import FakeEmbedder, bases
    settings = make_settings()
    settings.speaker.enabled = True
    manager = make_manager(settings=settings)
    manager.speaker_embedder = lambda path: FakeEmbedder(bases(2))
    manager.load_config = lambda: {'voice_config': settings.to_dict()}
    manager.attach(patient, DATE)
    await record(manager, 4)
    first = manager.session.speaker
    assert first is not None and first.settings.unknown_percentile == 15.0
    # unchanged marking settings: the same service goes on (its voice groups are still there), with the model endpoint now in force
    settings.corrector.api_url = 'http://other.invalid/v1'
    await record(manager, 4)
    assert manager.session.speaker is first and first._endpoint().api_url == 'http://other.invalid/v1'
    # changed settings: a new service with the new numbers, used by the session and by the transcript
    settings.speaker.unknown_percentile = 40.0
    settings.speaker.use_in_jobs = False
    await record(manager, 4)
    second = manager.session.speaker
    assert second is not first and second.settings.unknown_percentile == 40.0 and second.settings.use_in_jobs is False
    assert manager.session.pipeline.speaker is second
    # switched off: the old service keeps what it marked but marks nothing more
    settings.speaker.enabled = False
    await record(manager, 4)
    assert second.frozen and manager.session.pipeline.speaker is second and manager.session.speaker is second
    # and on again: a new service takes over, the earlier marks are carried along
    settings.speaker.enabled = True
    await record(manager, 4)
    third = manager.session.speaker
    assert third is not second and not third.frozen and manager.session.pipeline.speaker is third


async def test_switching_marking_off_also_follows_use_in_jobs_and_a_failed_model_load_retires_the_old_service(patient):
    from tests.speaker_fakes import FakeEmbedder, bases
    settings = make_settings()
    settings.speaker.enabled = True
    settings.speaker.use_in_jobs = True
    embedder = FakeEmbedder(bases(2))
    manager = make_manager(settings=settings)
    manager.speaker_embedder = lambda path: embedder
    manager.load_config = lambda: {'voice_config': settings.to_dict()}
    manager.attach(patient, DATE)
    await record(manager, 4)
    first = manager.session.speaker
    assert first.settings.use_in_jobs is True and embedder.calls > 0       # it took the first run's segments
    # marking off and "use in jobs" off together: the marks stay on screen, but the agents are not given them any more
    settings.speaker.enabled = False
    settings.speaker.use_in_jobs = False
    await record(manager, 4)
    assert first.frozen and first.settings.use_in_jobs is False and manager.session.pipeline.speaker is first
    assert manager.session.pipeline.snapshot().marked is False
    # on again with another model that cannot be loaded: the old (display-only) service stays, takes nothing, a note says why
    settings.speaker.enabled = True
    settings.speaker.use_in_jobs = True
    await record(manager, 4)
    second = manager.session.speaker
    assert second is not first and not second.frozen
    calls = embedder.calls
    settings.speaker.unknown_percentile = 30.0              # settings changed...

    def broken(path):
        raise OSError('model file is gone')
    manager.speaker_embedder = broken                         # ...and the voiceprint model cannot be loaded
    second_embedder_calls = embedder.calls
    await record(manager, 4)
    assert second.frozen and manager.session.speaker is second and manager.session.pipeline.speaker is second
    assert '說話者標記未啟用' in manager.session.speaker_note and 'model file is gone' in manager.session.speaker_note
    assert embedder.calls == second_embedder_calls            # the retired service was not fed this run's segments


async def test_a_display_only_speaker_service_follows_use_in_jobs_every_time_it_is_changed(patient):
    from tests.speaker_fakes import FakeEmbedder, bases
    settings = make_settings()
    settings.speaker.enabled = True
    settings.speaker.use_in_jobs = True
    manager = make_manager(settings=settings)
    manager.speaker_embedder = lambda path: FakeEmbedder(bases(2))
    manager.load_config = lambda: {'voice_config': settings.to_dict()}
    manager.attach(patient, DATE)
    await record(manager, 4)
    first = manager.session.speaker
    settings.speaker.enabled = False                         # off, "use in jobs" still on: the marks stay in what the agents get
    await record(manager, 4)
    assert first.frozen and first.settings.use_in_jobs is True
    settings.speaker.use_in_jobs = False                     # changed later, while it is already display-only
    await record(manager, 4)
    assert first.frozen and first.settings.use_in_jobs is False and manager.session.pipeline.snapshot().marked is False
    settings.speaker.use_in_jobs = True
    await record(manager, 4)
    assert first.settings.use_in_jobs is True


# ---------------------------------------------------------------------------------------------------------------------
# The recording clock is where the transcript on the screen ends, not the length of the audio
# ---------------------------------------------------------------------------------------------------------------------
async def test_the_clock_is_where_the_visible_transcript_ends_not_the_length_of_the_audio(patient):
    manager = make_manager()
    manager.attach(patient, DATE)
    await record(manager, 7)
    pipeline = manager.session.pipeline
    assert sample_count(voice_paths(patient, DATE).audio) == 7 * RATE
    assert pipeline.upto == 7.0 and manager.recording_position(patient, DATE) == (7.0, True)
    last = pipeline.segments.pop()                              # 7 s of audio, but the last window is not on the screen yet
    assert last.end == 7.0 and 0 < pipeline.upto < 7.0
    assert manager.recording_position(patient, DATE) == (pipeline.upto, True)
    assert pipeline.snapshot().upto == pipeline.upto            # the header of a frozen snapshot ("截至 mm:ss") says the same
    pipeline.segments.clear()                                   # a recording, but nothing transcribed: 0, not "never recorded"
    assert manager.recording_position(patient, DATE) == (0.0, True)
    manager.session.pipeline = None
    assert manager.recording_position(patient, DATE) == (0.0, True)


async def test_while_recording_the_clock_follows_the_transcript_not_the_microphone(patient):
    manager = make_manager(seconds_per_recording=60)
    manager.attach(patient, DATE)
    await manager.start()
    try:
        pipeline = manager.session.pipeline
        seconds, paused = manager.recording_position(patient, DATE)
        assert paused is False and seconds == (pipeline.upto if pipeline is not None else 0.0)
        assert seconds <= manager.session.source.audio_time
    finally:
        await manager.stop()


async def test_the_position_of_a_session_that_is_not_loaded_follows_its_saved_transcript(patient):
    first = make_manager()
    first.attach(patient, DATE)
    await record(first, 7)
    paths = voice_paths(patient, DATE)
    saved = json.loads(paths.transcript_json.read_text(encoding='utf-8'))
    first_end = saved['segments'][0]['end']
    saved['segments'].pop()
    paths.transcript_json.write_text(json.dumps(saved), encoding='utf-8')
    restarted = make_manager()
    assert restarted.session is None
    assert restarted.recording_position(patient, DATE) == (first_end, True) and first_end < 7.0
    paths.transcript_json.write_text('not json', encoding='utf-8')               # a damaged file: a recording with no transcript
    assert restarted.recording_position(patient, DATE) == (0.0, True)
    paths.transcript_json.unlink()
    assert restarted.recording_position(patient, DATE) == (0.0, True)


async def test_the_clock_of_an_agent_turn_stays_where_the_frozen_transcript_ends(patient, tmp_path):
    from pathlib import Path
    manager = make_manager()
    manager.attach(patient, DATE)
    await record(manager, 7)
    clock = manager.frozen_clock(patient, DATE)                 # taken where the turn takes its transcript
    assert clock() == (7.0, True)
    await record(manager, 4)                                    # the recording goes on while the agent works
    assert manager.recording_position(patient, DATE) == (11.0, True) and clock() == (7.0, True)
    other = Path(patient).with_name('B456_1990-02-02_李')
    other.mkdir()
    manager.attach(str(other), DATE)
    before_recording = manager.frozen_clock(str(other), DATE)
    assert before_recording() == (None, False)
    await record(manager, 3)                                    # a recording that began during the turn is not "before" its versions
    assert before_recording()[0] is None and manager.recording_position(str(other), DATE)[0] == 3.0


# ---------------------------------------------------------------------------------------------------------------------
# A segment without text is not on the screen: the clock stops at the last one that has
# ---------------------------------------------------------------------------------------------------------------------
def segment_of(index, start, end, text):
    from voice.pipeline import Segment
    return Segment(index, 'speech', start, end, start, text, text, text, text)


async def test_the_clock_ignores_segments_that_have_no_text_yet(patient):
    manager = make_manager()
    manager.attach(patient, DATE)
    await record(manager, 7)
    pipeline = manager.session.pipeline
    pipeline.segments[:] = [segment_of(1, 0.0, 4.0, '我頭痛三天'), segment_of(2, 4.0, 7.0, '')]       # the 3 s tail is still empty
    assert pipeline.upto == 4.0 and pipeline.snapshot().upto == 4.0 and '截至 00:04' in pipeline.snapshot().text
    assert manager.recording_position(patient, DATE) == (4.0, True)
    pipeline.segments[:] = [segment_of(1, 0.0, 9.0, '')]                  # a window that added nothing: no text at all
    assert manager.snapshot_text(patient, DATE) == ''                    # the agents get no transcript ...
    assert manager.recording_position(patient, DATE) == (0.0, True)       # ... and the clock does not claim 9 s of it
    pipeline.segments[:] = [segment_of(1, 0.0, 4.0, '我頭痛三天'), segment_of(2, 4.0, 7.0, '')]
    pipeline.segments[1].corrected = '還有發燒'                           # the correction fills the tail in later: it lies after 4 s
    assert pipeline.upto == 7.0 and manager.recording_position(patient, DATE) == (7.0, True)


async def test_a_session_gives_the_same_position_loaded_or_not_loaded(patient):
    first = make_manager()
    first.attach(patient, DATE)
    await record(first, 7)
    rows = [segment_of(1, 0.0, 4.0, '我頭痛三天'), segment_of(2, 4.0, 7.0, '')]
    first.session.pipeline.segments[:] = rows
    first.session.flush()
    loaded = first.recording_position(patient, DATE)
    restarted = make_manager()                                            # nothing attached: the saved file answers
    assert restarted.session is None
    assert restarted.recording_position(patient, DATE) == loaded == (4.0, True)
    restarted.attach(patient, DATE)                                       # now loaded from the same file
    assert restarted.recording_position(patient, DATE) == loaded


# ---------------------------------------------------------------------------------------------------------------------
# A segment of nothing but background voices is not in what the agents are given: the clock stops before it
# ---------------------------------------------------------------------------------------------------------------------
class OnlyBackground:
    """A speaker service that filters the way the real one does: the listed segments hold nothing but background voices."""
    failed = ''
    marks_in_text = False
    settings = SimpleNamespace(use_in_jobs=True)

    def __init__(self, background):
        self.background = set(background)

    def body(self, segment, *, marks, keep_background=False):
        return '' if segment.index in self.background and not keep_background else None


async def test_the_clock_stops_before_a_segment_of_background_voices_and_moves_on_when_it_becomes_foreground(patient):
    manager = make_manager()
    manager.attach(patient, DATE)
    await record(manager, 7)
    pipeline = manager.session.pipeline
    pipeline.segments[:] = [segment_of(1, 0.0, 4.0, '我頭痛三天'), segment_of(2, 4.0, 7.0, '隔壁房間的說話聲')]
    pipeline.speaker = OnlyBackground({2})
    assert '隔壁' not in pipeline.snapshot().text                          # the agents are not given it ...
    assert pipeline.upto == 4.0 and manager.recording_position(patient, DATE) == (4.0, True)       # ... and the clock does not reach it
    pipeline.speaker = OnlyBackground(set())                              # judged again: it was a foreground voice after all
    assert '隔壁' in pipeline.snapshot().text and pipeline.upto == 7.0
    assert manager.recording_position(patient, DATE) == (7.0, True)       # a version made now is after it; one made before it was not


async def test_the_clock_of_an_agent_turn_reads_pauses_without_working_out_the_transcript(patient):
    manager = make_manager()
    manager.attach(patient, DATE)
    await record(manager, 7)
    clock = manager.frozen_clock(patient, DATE)
    calls = []
    original = manager.session.pipeline.__class__.upto.fget
    manager.session.pipeline.__class__.upto = property(lambda self: calls.append(1) or original(self))
    try:
        assert clock() == (7.0, True) and calls == []                     # only the pause is read again, never the transcript
    finally:
        manager.session.pipeline.__class__.upto = property(original)



async def test_the_boundary_is_where_the_text_stops_a_later_change_in_the_middle_does_not_move_it(patient):
    """Accepted limit: the text, the speaker marks and the background judgement are guesses that get revised. A segment in the middle that is
    judged again later still lies before the boundary; the physician reads the record at the end."""
    manager = make_manager()
    manager.attach(patient, DATE)
    await record(manager, 7)
    pipeline = manager.session.pipeline
    pipeline.segments[:] = [segment_of(1, 0.0, 4.0, '我頭痛三天'), segment_of(2, 4.0, 7.0, '胸痛'), segment_of(3, 7.0, 9.0, '大概三天')]
    pipeline.speaker = OnlyBackground({2})
    assert manager.recording_position(patient, DATE) == (9.0, True)       # the text handed out stops at 9 s
    pipeline.speaker = OnlyBackground(set())
    assert manager.recording_position(patient, DATE) == (9.0, True)       # #2 comes back; the boundary stays where it was
