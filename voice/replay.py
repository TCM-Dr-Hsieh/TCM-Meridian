"""Replay of the recording of the session that is loaded: a short WAV clip cut out of the recording, also while it is still growing.

The physician clicks `#n mm:ss` in the transcript to listen again to what was said. The browser asks for the clip that
starts at that audio time (`start`, seconds into the recording, the same clock the transcript shows) and plays it in a
floating window.

The clip is cut here, not streamed from the file, for two reasons: the recording is still being written while it runs (its
header only holds the length as of the last patch, so a browser given the file itself would show a wrong, stale duration),
and a clip of its own length gives the player's time bar the right range. The sample count is therefore taken from the file
size, never from the header, and a clip that reaches past what has been recorded so far is simply shorter (no silence is added).

Only the recording of the session that is loaded can be asked for, by time only: there is no path in the request. The request
must also carry that session's random `replay_token` (the transcript rows hold it). The server answers from whatever session is
loaded when the request arrives, so without the token a page left over from patient A, while patient B's session is loaded,
would be handed B's recording at the time it clicked in A's transcript. A stale page is refused instead.
"""
from __future__ import annotations

import io
import math
import secrets
import wave
from pathlib import Path

from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

from .wavfile import SAMPLE_BYTES, WavError, data_offset

REPLAY_PATH = '/tcm-voice-replay.wav'
CLIP_SECONDS = 30.0


class ReplayError(Exception):
    """A request that cannot be answered; `status` is the HTTP status to send, the message is shown to the physician."""

    def __init__(self, message: str, status: int = 404):
        super().__init__(message)
        self.status = status


def clip_wav(path: Path, start: float, seconds: float = CLIP_SECONDS) -> bytes:
    """A complete WAV of up to `seconds` of the recording from `start` (seconds into the recording)."""
    if not math.isfinite(start) or start < 0:
        raise ReplayError('起點時間不合法。', 400)
    try:
        with open(path, 'rb') as handle:
            try:
                offset, rate = data_offset(handle)
            except WavError as exc:
                raise ReplayError(str(exc), 500 if '格式' in str(exc) else 404) from None
            handle.seek(0, 2)
            available = max(0, handle.tell() - offset) // SAMPLE_BYTES
            first = int(start * rate)
            if first >= available:
                raise ReplayError('這個時間點的聲音還沒有錄到。')
            count = min(int(seconds * rate), available - first)
            handle.seek(offset + first * SAMPLE_BYTES)
            pcm = handle.read(count * SAMPLE_BYTES)
    except FileNotFoundError:
        raise ReplayError('錄音檔還不存在。') from None
    out = io.BytesIO()
    with wave.open(out, 'wb') as clip:
        clip.setnchannels(1)
        clip.setsampwidth(SAMPLE_BYTES)
        clip.setframerate(rate)
        clip.writeframes(pcm[:len(pcm) // SAMPLE_BYTES * SAMPLE_BYTES])
    return out.getvalue()


def replay_response(manager, request: Request, start: float, token: str = '') -> Response:
    """The clip of the loaded session's recording for `manager` (a VoiceManager), as an HTTP response. `token` must be that
    session's replay token. The `Sec-Fetch-Site` check only rejects what the browser itself labels cross-site; the token is what
    keeps other pages (and stale ones) out."""
    if request.headers.get('sec-fetch-site') == 'cross-site':
        return PlainTextResponse('不允許跨網站讀取。', status_code=403)
    session = manager.session
    headers = {'Cache-Control': 'no-store'}
    if session is None:
        return PlainTextResponse('目前沒有載入任何 session。', status_code=409, headers=headers)
    expected = getattr(session, 'replay_token', '')
    if not expected or not secrets.compare_digest(str(token).encode(), expected.encode()):
        return PlainTextResponse('這個畫面屬於另一個 session（或已過期），無法回聽；請重新整理頁面。', status_code=409,
                                 headers=headers)
    try:
        body = clip_wav(session.audio_path, start)
    except ReplayError as exc:
        return PlainTextResponse(str(exc), status_code=exc.status, headers=headers)
    return Response(body, media_type='audio/wav', headers=headers)


_ROUTED_APPS: set[int] = set()


def register_replay_routes(app, manager) -> None:
    """Mount the clip route on a FastAPI-style app (NiceGUI's `app`, or a plain FastAPI app in tests). Once per app (NiceGUI's script
    mode runs the main script again for every browser connection): the route serves the `manager` of the first call, which is the
    process-wide one (`voice.service.shared_manager`)."""
    if id(app) in _ROUTED_APPS:
        return
    _ROUTED_APPS.add(id(app))

    @app.get(REPLAY_PATH)
    def replay(request: Request, start: float = 0.0, token: str = ''):
        return replay_response(manager, request, start, token)
