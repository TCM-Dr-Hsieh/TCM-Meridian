"""Remote microphone: the browser captures the audio and streams it to this computer over a same-origin WebSocket.

Adapted from voice_to_text (remote_audio.py / remote_capture.js) through TCM-Meridian-mini. The browser sends 32-bit float PCM at its own
sample rate in numbered packets; the server acknowledges them, resamples to 16 kHz mono and feeds the same WAV
recorder and window chunker the local microphone uses, so everything after the audio source (ASR, correction, the
audit trail) is unchanged. Differences from voice_to_text: one global recording session instead of a page per recording (the
token is created with the recording and only the browser that started it receives it), an unbounded chunk queue (the
recording never stops because processing is slow), and no ASR start-up handshake (audio is queued until the model
is ready, exactly as with the local microphone).

Protocol (JSON text frames unless noted; `session_id` is chosen by the browser):
    browser -> {type: start|resume, session_id}        server -> {type: ready|resumed, next_sequence}
    browser -> {type: format, sample_rate}             (once the AudioContext exists)
    browser -> binary: <uint32 LE sequence><float32 LE samples>      server -> {type: ack, sequence}
    browser -> {type: ping}                            server -> {type: pong}
    server  -> {type: stop}   (the recording is being stopped)    browser -> remaining packets, then {type: stop}
    server  -> {type: finished|cancel|error}
A short outage is survived: the browser keeps capturing, buffers unacknowledged packets, reconnects with `resume` and
replays only what the server has not stored. Duplicates are never written twice.
"""
from __future__ import annotations

import asyncio
import json
import secrets
import struct
import time
from pathlib import Path
from urllib.parse import urlsplit

import av
import numpy as np
from starlette.websockets import WebSocket, WebSocketDisconnect

from .audio import AudioSource
from .wavfile import RATE, WavRecorder

WS_PATH = '/tcm-remote-audio'
REMOTE_SOURCES: dict[str, 'RemoteAudioSource'] = {}      # token -> the recording's remote source
HELLO_SECONDS = 15.0
STOP_ACK_SECONDS = 8.0           # how long the server waits for the browser's tail after asking it to stop
RECONNECT_SECONDS = 30.0         # how long a lost connection may take to come back
STALE_SOCKET_SECONDS = 20.0      # no audio and no heartbeat for this long = the old socket is dead
MAX_PACKET_BYTES = 65536


class RemoteAudioSource(AudioSource):
    """Push-based audio source (the base class pulls from an iterator on a thread; this one is fed by a socket)."""

    kind = 'remote'

    def __init__(self, *, window_seconds: float, overlap_seconds: float, recording_path: Path | None,
                 token: str | None = None, reconnect_seconds: float = RECONNECT_SECONDS, device_label: str = '',
                 start_samples: int = 0, append: bool = False):
        super().__init__(None, window_seconds=window_seconds, overlap_seconds=overlap_seconds,
                         recording_path=recording_path, start_samples=start_samples, append=append)
        # What the audit trail calls the device: the browser's own name for the microphone it opened (text from the
        # browser, so one short line only), never its device id (which is not stable).
        self.device_label = ' '.join(str(device_label).split())[:120]
        self.label = '遠端瀏覽器麥克風' + (f' · {self.device_label}' if self.device_label else '')
        self.token = token or secrets.token_urlsafe(24)
        self.reconnect_seconds = reconnect_seconds
        self.session_id = ''
        self.status_text = '等待瀏覽器連線'
        self.connected = False                       # the browser reported its sample rate
        self.sample_rate = 0
        self.next_sequence = 0
        self.socket_owner = None
        self.reconnect_task: asyncio.Task | None = None
        self._resampler = None
        self._recording = None
        self._last_patch = time.monotonic()
        self.created_at = time.monotonic()

    # -- lifecycle ------------------------------------------------------------
    def start(self):
        if self.recording_path:
            self._recording = WavRecorder(self.recording_path, append=self.append)
        REMOTE_SOURCES[self.token] = self

    def request_stop(self):
        """The recording is being stopped. With a live connection the socket loop asks the browser for its tail; without one there
        is nothing to wait for, so close the recording now instead of waiting out the reconnect window."""
        self.stop.set()
        if self.done.is_set() or self.socket_owner is not None:
            return
        if self.recorded_samples == 0:
            self.finish('沒有收到任何遠端音訊：瀏覽器沒有連上。')
        else:
            self.finish('停止錄音時瀏覽器已斷線，斷線期間尚未補傳的聲音沒有收到。')

    async def close(self):
        if self.reconnect_task is not None:
            self.reconnect_task.cancel()
            self.reconnect_task = None
        self.socket_owner = None
        self.cancelled.set()
        self.stop.set()
        self._close_recording()
        self.level = 0.0
        self.done.set()
        REMOTE_SOURCES.pop(self.token, None)

    def _close_recording(self):
        if self._recording is not None:
            try:
                self._recording.close()
            finally:
                self._recording = None

    # -- connection ownership ---------------------------------------------------
    def attach(self, session_id: str, owner) -> int:
        """Bind a socket to this source (first connection, or a reconnect of the same recording) and return the next
        packet sequence the browser has to send."""
        if self.done.is_set() or self.cancelled.is_set():
            raise ValueError('遠端音訊已結束，無法再連線。')
        if self.session_id and session_id != self.session_id:
            raise ValueError('這次錄音已經由另一個瀏覽器分頁在進行。')
        self.session_id = session_id
        if self.reconnect_task is not None:
            self.reconnect_task.cancel()
            self.reconnect_task = None
        self.socket_owner = owner
        self.status_text = '錄音中' if self.connected else '已連線，準備中'
        return self.next_sequence

    def suspend(self, owner):
        """A socket dropped: keep the recording open for `reconnect_seconds`, then close it as incomplete."""
        if self.socket_owner is not owner:
            return
        self.socket_owner = None
        if self.done.is_set() or self.cancelled.is_set() or self.reconnect_task is not None:
            return
        self.status_text = f'連線暫時中斷，等待瀏覽器在 {self.reconnect_seconds:.0f} 秒內重新連線'

        async def expire():
            task = asyncio.current_task()
            try:
                await asyncio.sleep(self.reconnect_seconds)
                if self.socket_owner is None and not self.done.is_set():
                    self.finish(f'遠端音訊超過 {self.reconnect_seconds:.0f} 秒沒有重新連線；已收到的聲音保留，之後的聲音沒有錄到。')
            except asyncio.CancelledError:
                pass
            finally:
                if self.reconnect_task is task:
                    self.reconnect_task = None

        self.reconnect_task = asyncio.create_task(expire())

    # -- audio -------------------------------------------------------------------
    def connect(self, sample_rate):
        if self.connected or not isinstance(sample_rate, int) or isinstance(sample_rate, bool) \
                or not 8000 <= sample_rate <= 192000:
            raise ValueError('遠端音訊取樣率不合法或來源已連線。')
        self._resampler = av.AudioResampler(format='fltp', layout='mono', rate=RATE)
        self.sample_rate = sample_rate
        self.connected = True
        self.status_text = '錄音中'

    def receive(self, payload: bytes):
        if self.done.is_set() or self.cancelled.is_set():
            return
        if not self.connected or not payload or len(payload) > MAX_PACKET_BYTES or len(payload) % 4:
            raise ValueError('遠端音訊封包格式或長度不合法。')
        samples = np.frombuffer(payload, dtype='<f4')
        if not np.isfinite(samples).all():
            raise ValueError('遠端音訊包含不合法數值。')
        frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format='flt', layout='mono')
        frame.sample_rate = self.sample_rate
        for converted in self._resampler.resample(frame):
            self._accept(converted.to_ndarray())

    def receive_packet(self, payload: bytes) -> int:
        """One numbered packet; returns the sequence to acknowledge. A replayed packet (our ACK was lost) is
        acknowledged again but never written twice; a gap is an error."""
        if len(payload) < 8:
            raise ValueError('遠端音訊封包缺少序號或聲音資料。')
        sequence = struct.unpack_from('<I', payload)[0]
        if sequence < self.next_sequence:
            return self.next_sequence - 1
        if sequence != self.next_sequence:
            raise ValueError('遠端音訊封包順序不連續。')
        self.receive(payload[4:])
        self.next_sequence += 1
        return sequence

    def _accept(self, samples):
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not len(samples):
            return
        if self._recording is not None:
            self._recording.write(samples)
            if time.monotonic() - self._last_patch > 1:
                self._recording.patch_header()           # a crash still leaves a playable file
                self._last_patch = time.monotonic()
        self.recorded_samples += len(samples)
        self.level = min(1.0, float(np.sqrt(np.mean(samples ** 2))) * 5)
        for chunk in self.chunker.feed(samples):
            self.queue.put(chunk)

    def finish(self, error: str = ''):
        """Close the recording: flush the resampler, queue the unfinished tail window, finalize the WAV."""
        if self.done.is_set():
            return
        if self.reconnect_task is not None:
            self.reconnect_task.cancel()
            self.reconnect_task = None
        self.socket_owner = None
        if error:
            self.error = error
        try:
            if not self.cancelled.is_set():
                if self._resampler is not None:
                    for converted in self._resampler.resample(None):
                        self._accept(converted.to_ndarray())
                tail = self.chunker.flush()
                if tail is not None:
                    self.queue.put(tail)
        except Exception as exc:
            self.error = f'遠端音訊收尾失敗：{exc}'
        finally:
            self._close_recording()
            self.level = 0.0
            self.status_text = '已結束'
            self.done.set()


def _same_origin(websocket: WebSocket) -> bool:
    origin = websocket.headers.get('origin', '')
    return bool(origin) and urlsplit(origin).netloc == websocket.headers.get('host')


async def remote_audio_socket(websocket: WebSocket, token: str):
    """The browser's side of the protocol (see the module docstring)."""
    source = REMOTE_SOURCES.get(token)
    if source is None or not _same_origin(websocket):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    owned = False
    try:
        hello = json.loads(await asyncio.wait_for(websocket.receive_text(), HELLO_SECONDS))
        kind, session_id = hello.get('type'), hello.get('session_id')
        if kind not in ('start', 'resume') or not isinstance(session_id, str) or not 8 <= len(session_id) <= 80:
            raise ValueError('遠端音訊控制訊息不合法。')
        if source.done.is_set():
            await websocket.send_json({'type': 'finished'})
            return
        if kind == 'resume' and session_id != source.session_id:
            raise ValueError('找不到可恢復的遠端音訊。')
        next_sequence = source.attach(session_id, websocket)
        owned = True
        await websocket.send_json({'type': 'resumed' if kind == 'resume' else 'ready', 'next_sequence': next_sequence})

        stop_sent_at = None
        ended_normally = False
        source_error = ''
        disconnected = False
        last_received = time.monotonic()
        while not source.cancelled.is_set() and not source.done.is_set():
            if source.socket_owner is not websocket:
                break                                    # a newer socket took over this recording
            if time.monotonic() - last_received > STALE_SOCKET_SECONDS:
                disconnected = True
                break
            if source.stop.is_set() and stop_sent_at is None:
                try:
                    await websocket.send_json({'type': 'stop'})
                except (OSError, RuntimeError, WebSocketDisconnect):
                    disconnected = True
                    break
                stop_sent_at = time.monotonic()
            if stop_sent_at is not None and time.monotonic() - stop_sent_at > STOP_ACK_SECONDS:
                source_error = '停止錄音時沒有收到瀏覽器的音訊尾段；最後一小段聲音可能沒有錄到。'
                break
            try:
                message = await asyncio.wait_for(websocket.receive(), .25)
            except asyncio.TimeoutError:
                continue
            except (OSError, RuntimeError, WebSocketDisconnect):
                disconnected = True
                break
            if message['type'] == 'websocket.disconnect':
                disconnected = True
                break
            if source.socket_owner is not websocket:
                break
            last_received = time.monotonic()
            if message.get('bytes') is not None:
                sequence = source.receive_packet(message['bytes'])
                if stop_sent_at is not None:
                    stop_sent_at = time.monotonic()      # the tail is still arriving
                try:
                    await websocket.send_json({'type': 'ack', 'sequence': sequence})
                except (OSError, RuntimeError, WebSocketDisconnect):
                    disconnected = True
                    break
                continue
            if message.get('text') is None:
                continue
            control = json.loads(message['text'])
            kind = control.get('type')
            if kind == 'format':
                sample_rate = control.get('sample_rate')
                if not source.connected:
                    source.connect(sample_rate)
                elif sample_rate != source.sample_rate:
                    raise ValueError('遠端音訊取樣率在重連後改變。')
            elif kind == 'stop':
                ended_normally = True
                break
            elif kind == 'ping':
                try:
                    await websocket.send_json({'type': 'pong'})
                except (OSError, RuntimeError, WebSocketDisconnect):
                    disconnected = True
                    break
            elif kind == 'error':
                source_error = '遠端音訊錯誤：' + str(control.get('message', '未知錯誤'))[:200]
                break
            else:
                raise ValueError('遠端音訊控制訊息不合法。')
        if source.socket_owner is not websocket:
            pass
        elif source.cancelled.is_set():
            try:
                await websocket.send_json({'type': 'cancel'})
            except (OSError, RuntimeError, WebSocketDisconnect):
                pass
        elif disconnected:
            source.suspend(websocket)
        else:
            source.finish('' if ended_normally else source_error)
            try:
                await websocket.send_json({'type': 'finished'})
            except (OSError, RuntimeError, WebSocketDisconnect):
                pass
    except Exception as exc:
        message = f'遠端音訊失敗：{exc}'
        if owned and source.socket_owner is websocket:
            source.finish(message)
        try:
            await websocket.send_json({'type': 'error', 'message': message})
        except Exception:
            pass
    finally:
        if owned and not source.done.is_set() and not source.cancelled.is_set():
            source.suspend(websocket)
        try:
            await websocket.close()
        except Exception:
            pass


_ROUTED_APPS: set[int] = set()


def register_routes(app) -> None:
    """Mount the WebSocket on a FastAPI-style app (NiceGUI's `app`, or a plain FastAPI app in tests). Once per app: NiceGUI's script
    mode runs the main script again for every browser connection."""
    if id(app) in _ROUTED_APPS:
        return
    _ROUTED_APPS.add(id(app))
    app.websocket(WS_PATH + '/{token}')(remote_audio_socket)
