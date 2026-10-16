"""Remote microphone (browser audio over a WebSocket): the source and the socket protocol.
(A whole recording that uses it is tested in test_voice_service.py.)"""
import asyncio
import json
import struct
import wave

import numpy as np
import pytest

import voice.remote as remote
from voice.config import ASRSettings
from voice.remote import REMOTE_SOURCES, RemoteAudioSource, remote_audio_socket
from voice.wavfile import RATE

BROWSER_RATE = 48000
BLOCK = 2048


def make_source(tmp_path=None, **kwargs) -> RemoteAudioSource:
    path = (tmp_path / 'audio.wav') if tmp_path is not None else None
    settings = ASRSettings()
    source = RemoteAudioSource(window_seconds=settings.window_seconds, overlap_seconds=settings.overlap_seconds,
                               recording_path=path, **kwargs)
    source.start()
    return source


@pytest.fixture(autouse=True)
def clean_registry():
    yield
    REMOTE_SOURCES.clear()


def blocks(seconds: float, *, level: float = 0.1) -> list[np.ndarray]:
    """`seconds` of a tone as the browser would send it: float32 blocks at the browser's sample rate."""
    total = int(seconds * BROWSER_RATE)
    tone = (np.sin(2 * np.pi * 220 * np.arange(total) / BROWSER_RATE) * level).astype('<f4')
    return [tone[i:i + BLOCK] for i in range(0, total, BLOCK)]


def packet(sequence: int, block: np.ndarray) -> bytes:
    return struct.pack('<I', sequence) + block.astype('<f4').tobytes()


async def wait_for(condition, timeout=5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError('condition not reached in time')
        await asyncio.sleep(0.01)


class FakeWebSocket:
    """What the server sees of one browser connection; the test plays the browser by pushing messages."""

    def __init__(self, origin='https://mini.example', host='mini.example'):
        self.headers = {'origin': origin, 'host': host}
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.sent: list[dict] = []
        self.accepted = False
        self.close_code = None

    async def accept(self):
        self.accepted = True

    async def receive_text(self):
        return (await self.inbox.get())['text']

    async def receive(self):
        return await self.inbox.get()

    async def send_json(self, value):
        self.sent.append(value)

    async def close(self, code=1000):
        self.close_code = code

    # -- the browser's side ------------------------------------------------------
    def say(self, **control):
        self.inbox.put_nowait({'type': 'websocket.receive', 'text': json.dumps(control)})

    def audio(self, sequence, block):
        self.inbox.put_nowait({'type': 'websocket.receive', 'bytes': packet(sequence, block)})

    def drop(self):
        self.inbox.put_nowait({'type': 'websocket.disconnect'})

    def types(self) -> list[str]:
        return [message['type'] for message in self.sent]

    def acks(self) -> list[int]:
        return [message['sequence'] for message in self.sent if message['type'] == 'ack']


def serve(ws: FakeWebSocket, source: RemoteAudioSource) -> asyncio.Task:
    return asyncio.create_task(remote_audio_socket(ws, source.token))


async def open_socket(source, *, session_id='session-0001', hello='start', rate=BROWSER_RATE, with_format=True):
    ws = FakeWebSocket()
    task = serve(ws, source)
    ws.say(type=hello, session_id=session_id)
    await wait_for(lambda: ws.sent)
    if with_format:
        ws.say(type='format', sample_rate=rate)
    return ws, task


# ============================ the source ====================================================
def test_source_resamples_and_records_a_continuous_wav(tmp_path):
    source = make_source(tmp_path)
    source.connect(BROWSER_RATE)
    for block in blocks(2):
        source.receive(block.tobytes())
    source.finish()
    assert source.done.is_set() and source.error == '' and source.status_text == '已結束'
    assert source.recorded_samples == RATE * 2 and source.audio_time == 2.0
    with wave.open(str(tmp_path / 'audio.wav'), 'rb') as saved:
        assert (saved.getframerate(), saved.getnchannels(), saved.getsampwidth(), saved.getnframes()) == (RATE, 1, 2, RATE * 2)
    chunk = source.queue.get_nowait()
    assert chunk.reason == 'stop' and chunk.duration == 2.0            # the unfinished tail window is queued at the end


def test_the_device_label_is_kept_as_one_short_line_for_the_audit_trail():
    assert make_source().label == '遠端瀏覽器麥克風'                          # the browser gave no name
    named = make_source(device_label='Realtek(R) Audio')
    assert named.label == '遠端瀏覽器麥克風 · Realtek(R) Audio' and named.device_label == 'Realtek(R) Audio'
    messy = make_source(device_label='a\r\nb\t\tc ' + 'x' * 500)              # text from the browser: one line, bounded
    assert '\n' not in messy.label and '\r' not in messy.label and '\t' not in messy.label
    assert messy.device_label.startswith('a b c x') and len(messy.device_label) == 120


def test_full_windows_are_queued_while_the_audio_arrives():
    source = make_source()
    source.connect(BROWSER_RATE)
    for block in blocks(7):
        source.receive(block.tobytes())
    first = source.queue.get_nowait()
    assert first.reason == 'window' and first.start == 0.0 and first.duration == 6.0
    assert source.level > 0
    source.finish()
    assert source.level == 0.0


def test_source_rejects_invalid_rates_and_packets():
    source = make_source()
    for bad in (1000, 500000, '48000', 48000.0, True):
        with pytest.raises(ValueError, match='取樣率'):
            source.connect(bad)
    with pytest.raises(ValueError, match='封包'):
        source.receive(b'x' * 8)                                    # not connected yet
    source.connect(BROWSER_RATE)
    with pytest.raises(ValueError, match='取樣率'):
        source.connect(BROWSER_RATE)                                # only once
    for payload in (b'bad', b'', b'\0' * 65540):
        with pytest.raises(ValueError, match='封包'):
            source.receive(payload)
    with pytest.raises(ValueError, match='數值'):
        source.receive(np.array([np.nan], dtype='<f4').tobytes())
    with pytest.raises(ValueError, match='缺少序號'):
        source.receive_packet(b'\0\0\0')


def test_packets_are_numbered_acknowledged_and_never_written_twice():
    source = make_source()
    source.connect(BROWSER_RATE)
    first, second, third = blocks(1)[:3]
    assert source.receive_packet(packet(0, first)) == 0
    assert source.receive_packet(packet(1, second)) == 1
    written = source.recorded_samples
    assert source.receive_packet(packet(0, first)) == 1               # replay after a lost ACK: acknowledged again
    assert source.receive_packet(packet(1, second)) == 1
    assert source.recorded_samples == written                          # ... but not written again
    with pytest.raises(ValueError, match='不連續'):
        source.receive_packet(packet(5, third))                        # a hole is an error, never silently skipped
    assert source.receive_packet(packet(2, third)) == 2 and source.next_sequence == 3


def test_stop_without_a_connected_browser_closes_the_recording_at_once():
    never = make_source()
    never.request_stop()
    assert never.done.is_set() and '沒有收到任何遠端音訊' in never.error

    dropped = make_source()
    dropped.connect(BROWSER_RATE)
    dropped.receive(blocks(1)[0].tobytes())
    dropped.request_stop()
    assert dropped.done.is_set() and '已斷線' in dropped.error and dropped.recorded_samples > 0


def test_stop_waits_for_the_tail_while_a_browser_is_connected():
    source = make_source()
    source.attach('session-0001', object())
    source.request_stop()
    assert source.stop.is_set() and not source.done.is_set()          # the socket loop asks the browser for its tail


async def test_a_lost_connection_is_kept_open_for_the_reconnect_window_then_closed():
    source = make_source(reconnect_seconds=0.15)
    owner = object()
    source.attach('session-0001', owner)
    source.connect(BROWSER_RATE)
    source.suspend(owner)
    assert '暫時中斷' in source.status_text and not source.done.is_set()
    await wait_for(source.done.is_set)
    assert '沒有重新連線' in source.error and source.reconnect_task is None

    kept = make_source(reconnect_seconds=0.15)
    first, second = object(), object()
    kept.attach('session-0001', first)
    kept.suspend(first)
    assert kept.attach('session-0001', second) == 0                    # back in time: the expiry is cancelled
    await asyncio.sleep(0.3)
    assert not kept.done.is_set() and kept.error == ''


def test_a_stale_socket_cannot_suspend_the_connection_that_replaced_it():
    source = make_source()
    old, new = object(), object()
    source.attach('session-0001', old)
    source.attach('session-0001', new)
    source.suspend(old)
    assert source.socket_owner is new and source.reconnect_task is None


def test_attach_rejects_another_session_and_a_finished_source():
    source = make_source()
    source.attach('session-0001', object())
    with pytest.raises(ValueError, match='另一個瀏覽器分頁'):
        source.attach('session-9999', object())
    source.finish()
    with pytest.raises(ValueError, match='已結束'):
        source.attach('session-0001', object())


async def test_close_releases_the_token_and_the_wav(tmp_path):
    source = make_source(tmp_path)
    assert REMOTE_SOURCES[source.token] is source
    await source.close()
    assert source.done.is_set() and source.cancelled.is_set() and source.token not in REMOTE_SOURCES
    with wave.open(str(tmp_path / 'audio.wav'), 'rb'):                  # a valid (empty) file, not a held-open handle
        pass


# ============================ the socket protocol ===========================================
@pytest.mark.parametrize('origin', [None, 'https://elsewhere.example'])
async def test_socket_rejects_a_missing_or_foreign_origin(origin):
    source = make_source()
    ws = FakeWebSocket()
    if origin is None:
        ws.headers.pop('origin')
    else:
        ws.headers['origin'] = origin
    await remote_audio_socket(ws, source.token)
    assert not ws.accepted and ws.close_code == 1008


async def test_socket_rejects_an_unknown_token():
    ws = FakeWebSocket()
    await remote_audio_socket(ws, 'not-a-token')
    assert not ws.accepted and ws.close_code == 1008


async def test_socket_streams_acknowledges_and_finishes_on_the_browsers_stop(tmp_path):
    source = make_source(tmp_path)
    ws, task = await open_socket(source)
    assert ws.accepted and ws.sent[0] == {'type': 'ready', 'next_sequence': 0}
    for sequence, block in enumerate(blocks(1)):
        ws.audio(sequence, block)
    count = len(blocks(1))
    await wait_for(lambda: len(ws.acks()) == count)
    assert ws.acks() == list(range(count)) and source.connected and source.status_text == '錄音中'
    ws.say(type='stop')
    await asyncio.wait_for(task, 5)
    assert ws.sent[-1] == {'type': 'finished'} and source.done.is_set() and source.error == ''
    assert source.recorded_samples == RATE and ws.close_code == 1000
    with wave.open(str(tmp_path / 'audio.wav'), 'rb') as saved:
        assert saved.getnframes() == RATE


async def test_socket_answers_heartbeats():
    source = make_source()
    ws, task = await open_socket(source)
    ws.say(type='ping')
    await wait_for(lambda: 'pong' in ws.types())
    ws.say(type='stop')
    await asyncio.wait_for(task, 5)


async def test_when_the_visit_ends_the_browser_is_asked_to_stop_and_its_tail_is_kept():
    source = make_source()
    ws, task = await open_socket(source)
    stream = blocks(1)
    for sequence, block in enumerate(stream[:-1]):
        ws.audio(sequence, block)
    await wait_for(lambda: len(ws.acks()) == len(stream) - 1)
    source.request_stop()                                              # what the visit's 結束並存檔 does
    await wait_for(lambda: 'stop' in ws.types())
    assert not source.done.is_set()                                    # still waiting for the browser's tail
    ws.audio(len(stream) - 1, stream[-1])                              # the browser flushes its last packet ...
    ws.say(type='stop')                                                # ... then confirms
    await asyncio.wait_for(task, 5)
    assert source.done.is_set() and source.error == ''
    assert abs(source.recorded_samples - RATE) <= 2                    # nothing was lost at the end


async def test_a_missing_tail_after_the_stop_request_is_recorded_as_an_error(monkeypatch):
    monkeypatch.setattr(remote, 'STOP_ACK_SECONDS', 0.3)
    source = make_source()
    ws, task = await open_socket(source)
    source.request_stop()
    await asyncio.wait_for(task, 5)
    assert source.done.is_set() and '沒有收到瀏覽器的音訊尾段' in source.error


async def test_resume_replays_only_the_missing_packets_into_the_same_recording(tmp_path):
    source = make_source(tmp_path, reconnect_seconds=5)
    stream = blocks(1)
    first, task = await open_socket(source)
    for sequence in range(4):
        first.audio(sequence, stream[sequence])
    await wait_for(lambda: len(first.acks()) == 4)
    first.drop()                                                       # the Wi-Fi hiccup
    await asyncio.wait_for(task, 5)
    assert source.socket_owner is None and not source.done.is_set() and '暫時中斷' in source.status_text
    written = source.recorded_samples

    second, task = await open_socket(source, hello='resume', with_format=False)
    assert second.sent[0] == {'type': 'resumed', 'next_sequence': 4}
    second.say(type='format', sample_rate=BROWSER_RATE)
    for sequence in range(2, len(stream)):                              # the browser replays from its last unacked packet
        second.audio(sequence, stream[sequence])
    # packets 2 and 3 were already stored: acknowledged again, not written twice
    await wait_for(lambda: len(second.acks()) == len(stream) - 2)
    second.say(type='stop')
    await asyncio.wait_for(task, 5)
    assert source.error == '' and source.recorded_samples > written
    assert abs(source.recorded_samples - RATE) <= 2                    # exactly one second, no duplicate, no hole
    with wave.open(str(tmp_path / 'audio.wav'), 'rb') as saved:
        assert abs(saved.getnframes() - RATE) <= 2


async def test_resume_of_an_unknown_session_is_refused():
    source = make_source()
    ws, task = await open_socket(source, session_id='session-0001')
    ws.drop()
    await asyncio.wait_for(task, 5)
    other, task = await open_socket(source, session_id='session-9999', hello='resume', with_format=False)
    await asyncio.wait_for(task, 5)
    assert other.sent[0]['type'] == 'error' and '找不到可恢復' in other.sent[0]['message']
    assert not source.done.is_set()                                    # the real recording is untouched


async def test_a_socket_that_goes_quiet_is_treated_as_lost(monkeypatch):
    monkeypatch.setattr(remote, 'STALE_SOCKET_SECONDS', 0.3)
    source = make_source(reconnect_seconds=5)
    ws, task = await open_socket(source)
    await asyncio.wait_for(task, 5)                                    # no packets, no heartbeat
    assert source.socket_owner is None and not source.done.is_set() and '暫時中斷' in source.status_text


async def test_a_hole_in_the_packets_ends_the_recording_with_an_error():
    source = make_source()
    stream = blocks(1)
    ws, task = await open_socket(source)
    ws.audio(0, stream[0])
    ws.audio(3, stream[3])
    await asyncio.wait_for(task, 5)
    assert source.done.is_set() and '不連續' in source.error
    assert ws.sent[-1]['type'] == 'error'


async def test_the_browsers_own_error_report_closes_the_recording_with_it():
    source = make_source()
    ws, task = await open_socket(source)
    ws.say(type='error', message='麥克風已中斷')
    await asyncio.wait_for(task, 5)
    assert source.done.is_set() and '麥克風已中斷' in source.error
    assert ws.sent[-1] == {'type': 'finished'}


@pytest.mark.parametrize('hello', [{'type': 'start'}, {'type': 'start', 'session_id': 'short'},
                                   {'type': 'play', 'session_id': 'session-0001'}])
async def test_an_invalid_hello_is_refused_without_touching_the_recording(hello):
    source = make_source()
    ws = FakeWebSocket()
    task = serve(ws, source)
    ws.say(**hello)
    await asyncio.wait_for(task, 5)
    assert ws.sent[0]['type'] == 'error' and not source.done.is_set() and source.session_id == ''


async def test_a_second_browser_cannot_take_over_a_running_recording():
    source = make_source()
    ws, task = await open_socket(source, session_id='session-0001')
    other, other_task = await open_socket(source, session_id='session-9999', with_format=False)
    await asyncio.wait_for(other_task, 5)
    assert other.sent[0]['type'] == 'error' and '另一個瀏覽器分頁' in other.sent[0]['message']
    assert source.socket_owner is not None and not source.done.is_set()
    ws.say(type='stop')
    await asyncio.wait_for(task, 5)


def test_routes_are_mounted_on_a_fastapi_app():
    from fastapi import FastAPI
    app = FastAPI()
    remote.register_routes(app)
    assert any(getattr(route, 'path', '') == '/tcm-remote-audio/{token}' for route in app.routes)
