"""The voice recording of the session that is loaded: start, stop, continue, restore, and the transcript for the agents.

One session (patient + date) has ONE recording and ONE transcript. `VoiceManager` is the process-wide owner; it follows the
loaded session (`attach` / `detach`) and runs the recording in the background (an audio thread plus asyncio tasks on the web
server's loop), so nothing the physician does in the pages stops it: the pop-up window is only a view.

    start()   begin recording; when the session already has a recording the WAV is continued, the numbering of the
              transcript goes on and the clock starts where the audio ends
    stop()    stop recording, let the audio already recorded finish being transcribed, write the files
    attach()  follow another session; the transcript saved for it is restored (and can be continued by `start()`)

Files, in the session's log folder (`log/<date>-log/`):
    <date>-voice.wav                16 kHz mono 16-bit, grows with every recording
    <date>-voice-transcript.json    the transcript (segments, corrections, speaker marks) -- what `attach` restores
    <date>-voice-transcript.txt     the same as plain lines (`語音#N mm:ss–mm:ss …`)
    <date>-voice-speaker.json       the speaker-marking role map and statistics (no voiceprints, ever)
    <date>-voice-llm.jsonl          every LLM call of the transcript correction (full input and output)
    <date>-voice-events.jsonl       what happened, with the recording clock
"""
from __future__ import annotations

import asyncio
import json
import secrets
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from agent_behavior_log import append_behavior_event

from .asr import ASRError, LocalASR, validate_model_dir, worker_python
from .audio import make_source
from .config import VoiceSettings, resolve_path
from .corrector import TranscriptCorrector
from .fileio import atomic_write_json, atomic_write_text
from .llm import LLMCaller, LLMClient
from .pipeline import TranscriptPipeline
from .remote import RemoteAudioSource
from .speaker.embedder import load_embedder
from .speaker.service import SpeakerService
from .wavfile import RATE, sample_count

BEHAVIOR_AGENT = 'voice_subagent'
FLUSH_SECONDS = 5.0
STOP_DRAIN_SECONDS = 600.0           # how long a stop may wait for the audio already recorded to finish being transcribed
NO_MODELS_MESSAGE = '目前無下載語音模型，無法啟動語音功能。'

# Events that also appear in the agents' behaviour timeline (the rest stays in the voice events file): label, severity.
TIMELINE_EVENTS = {
    'recording_started': ('開始錄音', 'normal'),
    'recording_stopped': ('停止錄音', 'normal'),
    'mic_error': ('錄音來源錯誤', 'error'),
    'transcript_gap': ('音訊缺漏', 'warning'),
    'transcript_pipeline_error': ('逐字稿管線錯誤', 'error'),
    'speaker_unavailable': ('說話者標記未啟用', 'warning'),
    'speaker_error': ('說話者標記錯誤', 'error'),
}


class VoiceError(RuntimeError):
    """A refusal with a message for the physician."""


@dataclass(frozen=True)
class VoicePaths:
    log_dir: Path
    audio: Path
    transcript_json: Path
    transcript_txt: Path
    speaker_json: Path
    llm_log: Path
    events: Path


def voice_paths(folder: str | Path, date: str) -> VoicePaths:
    log_dir = Path(folder) / 'log' / f'{date}-log'
    return VoicePaths(log_dir, log_dir / f'{date}-voice.wav', log_dir / f'{date}-voice-transcript.json',
                      log_dir / f'{date}-voice-transcript.txt', log_dir / f'{date}-voice-speaker.json',
                      log_dir / f'{date}-voice-llm.jsonl', log_dir / f'{date}-voice-events.jsonl')


def _now() -> str:
    return datetime.now().isoformat(timespec='seconds')


def _saved_transcript_end(path: Path) -> float:
    """Where the transcript saved for a session ends (0.0 when there is none or the file is unreadable)."""
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return max((float(row['end']) for row in data.get('segments') or []), default=0.0)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return 0.0


class VoiceSession:
    """The voice state of one loaded session."""

    def __init__(self, manager: 'VoiceManager', folder: str, date: str):
        self.manager = manager
        self.folder = str(folder)
        self.date = date
        self.paths = voice_paths(folder, date)
        # Random and per loaded session: the transcript rows carry it and the replay route serves audio only against it.
        self.replay_token = secrets.token_urlsafe(16)
        self.state = 'idle'                       # idle | starting | recording | stopping
        self.pipeline: TranscriptPipeline | None = None
        self.source = None
        self.speaker: SpeakerService | None = None
        self.speaker_note = ''                    # why marking is not running although it is switched on
        self.settings: VoiceSettings | None = None
        self.error = ''                           # the last problem worth showing
        self._tasks: list[asyncio.Task] = []
        self._mic_error_logged = ''
        self._last_flush = 0.0
        self._flushed_version = -1
        self._stop_lock = asyncio.Lock()

    # -- basics ------------------------------------------------------------------
    @property
    def audio_path(self) -> Path:
        return self.paths.audio

    def audio_t(self) -> float | None:
        if self.source is not None:
            return self.source.audio_time
        samples = sample_count(self.paths.audio)
        return samples / RATE if samples else None

    @property
    def has_recording(self) -> bool:
        return sample_count(self.paths.audio) > 0

    @property
    def segment_count(self) -> int:
        return len(self.pipeline.segments) if self.pipeline is not None else 0

    def emit(self, type: str, **fields):
        """Append to the voice events file (and mirror the important events into the agent behaviour timeline)."""
        try:
            event = {'ts': _now(), 't': self.audio_t(), 'type': type, **fields}
            self.paths.log_dir.mkdir(parents=True, exist_ok=True)
            with open(self.paths.events, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(event, ensure_ascii=False, default=str) + '\n')
            mirrored = TIMELINE_EVENTS.get(type)
            if mirrored is not None:
                label, severity = mirrored
                detail = fields.get('error') or fields.get('reason') or json.dumps(fields, ensure_ascii=False, default=str)
                self.manager.behavior_log(self.folder, self.date, agent=BEHAVIOR_AGENT, event_type=f'voice_{type}',
                                          label=label, title=f'語音辨識：{label}', content=str(detail), severity=severity)
        except Exception:                          # logging must never break the recording
            pass

    def _log_llm(self, record: dict):
        try:
            self.paths.log_dir.mkdir(parents=True, exist_ok=True)
            with open(self.paths.llm_log, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps({'ts': _now(), 't': self.audio_t(), **record}, ensure_ascii=False, default=str) + '\n')
        except Exception:
            pass

    def _changed(self):
        if self.pipeline is not None:
            self.pipeline.version += 1
        self.manager.bump()

    # -- restoring ----------------------------------------------------------------
    def load(self):
        """Restore the transcript saved for this session (if any). Never raises: a damaged file only means an empty transcript."""
        try:
            data = json.loads(self.paths.transcript_json.read_text(encoding='utf-8'))
            if not isinstance(data, dict) or not data.get('segments'):
                return
        except (OSError, ValueError):
            return
        settings = self.manager.safe_settings()
        pipeline = TranscriptPipeline(asr=self.manager.asr, asr_settings=settings.asr, corrector=None, source=None,
                                      emit=self.emit)
        try:
            pipeline.restore(data)
        except Exception as exc:                    # a transcript from another version: keep the file, start empty
            self.emit('transcript_restore_failed', error=f'{type(exc).__name__}: {exc}')
            return
        self.pipeline = pipeline
        saved = data.get('speaker') if isinstance(data.get('speaker'), dict) else {}
        if saved.get('labels'):
            self.speaker = self._frozen_speaker(settings, saved)
            pipeline.speaker = self.speaker
        self.emit('transcript_restored', segments=len(pipeline.segments))

    def _frozen_speaker(self, settings: VoiceSettings, saved: dict) -> SpeakerService:
        """A service that only shows the marks of an earlier run (it never takes new text)."""
        service = SpeakerService(settings=settings.speaker, embed=lambda audio: None,
                                 caller=LLMCaller(self.manager.llm), endpoint=lambda: settings.corrector,
                                 segments=lambda: self.pipeline.segments if self.pipeline is not None else [],
                                 emit=self.emit, on_change=self._changed, audio_t=self.audio_t)
        service.frozen = True
        service.restore_labels(saved.get('labels') or {}, bool(saved.get('roles_trusted')))
        return service

    # -- recording ------------------------------------------------------------------
    async def start_recording(self, settings: VoiceSettings, *, remote: bool, device_label: str):
        self.paths.log_dir.mkdir(parents=True, exist_ok=True)
        start_samples = sample_count(self.paths.audio)
        append = self.paths.audio.is_file()
        asr = settings.asr
        if remote:
            source = RemoteAudioSource(window_seconds=asr.window_seconds, overlap_seconds=asr.overlap_seconds,
                                       recording_path=self.paths.audio, device_label=device_label,
                                       start_samples=start_samples, append=append)
        elif self.manager.source_factory is not None:
            source = self.manager.source_factory(self.paths.audio, start_samples=start_samples, append=append)
        else:
            source = make_source(asr, self.paths.audio, start_samples=start_samples, append=append)
        self.settings = settings
        caller = LLMCaller(self.manager.llm, timeout=lambda: settings.timeout_seconds, retries=lambda: settings.retries,
                           logger=self._log_llm, backoff=self.manager.llm_backoff)
        corrector = (TranscriptCorrector(caller, lambda: settings.corrector, lambda: asr.correction_vocabulary)
                     if settings.corrector.configured else None)
        source.start()
        self.source = source
        self._mic_error_logged = ''
        remote_kind = getattr(source, 'kind', 'local') == 'remote'
        self.emit('recording_started', audio_source='remote' if remote_kind else 'local',
                  device=source.label if remote_kind else (asr.microphone_id or '系統預設麥克風'),
                  continued=start_samples > 0, from_seconds=round(start_samples / RATE, 2),
                  window_seconds=asr.window_seconds, overlap_seconds=asr.overlap_seconds,
                  corrector=bool(corrector), settings=settings.public_dict())
        speaker = await self._make_speaker(settings, caller)
        if self.pipeline is None:
            self.pipeline = TranscriptPipeline(asr=self.manager.asr, asr_settings=asr, corrector=corrector, source=source,
                                               emit=self.emit, speaker=speaker)
        else:
            self.pipeline.settings = asr
            self.pipeline.corrector = corrector
            self.pipeline.begin_run(source, speaker)
        self._tasks = [asyncio.create_task(self.pipeline.run(), name='voice-transcript'),
                       asyncio.create_task(self._watch(), name='voice-watch')]
        if speaker is not None:
            speaker.start()
        self.state = 'recording'

    async def _make_speaker(self, settings: VoiceSettings, caller: LLMCaller) -> SpeakerService | None:
        """The speaker-marking service for this recording, or None when it is switched off or cannot run (the recording goes on)."""
        config = settings.speaker
        self.speaker_note = ''
        previous = self.speaker
        if not config.enabled:
            # Marking was switched off since the last run of this transcript: the marks it made stay (shown, handed to the
            # prompts as they were, unless "use in jobs" is off now), but it must not mark anything new.
            await self._retire_speaker(previous, config)
            return None
        if previous is not None and not previous.frozen and not previous.failed and previous.settings == config:
            # Stop, then start in the same run with the same settings: the voice groups and role map are still there. The model
            # endpoint, its timeout and retries may have changed in between, so the service gets the new caller.
            previous.rebind(caller, lambda: settings.corrector)
            previous.resume()
            return previous
        if previous is not None and not previous.frozen:
            await previous.stop()                   # settings changed: the old service ends; its labels carry over, its voice groups do not
        try:
            embed = await asyncio.to_thread(self.manager.speaker_embedder, resolve_path(config.model_path))
        except Exception as exc:
            self.speaker_note = f'說話者標記未啟用：{exc}'
            self.emit('speaker_unavailable', reason=str(exc))
            await self._retire_speaker(previous, config)         # the old service must not keep taking this run's segments
            return None
        self.emit('speaker_started', model=config.model_path, unknown_percentile=config.unknown_percentile,
                  text_fill=config.text_fill, text_fill2=config.text_fill2, use_in_jobs=config.use_in_jobs,
                  tuning=config.advanced)
        service = SpeakerService(
            settings=config, embed=embed, caller=caller, endpoint=lambda: settings.corrector,
            segments=lambda: self.pipeline.segments if self.pipeline is not None else [],
            emit=self.emit, on_change=self._changed, audio_t=self.audio_t)
        if previous is not None:                    # an earlier run's marks (before a restart) stay as they were
            service.restore_labels(previous.labels_doc(), previous.marks_in_text)
        self.speaker = service
        return service

    @staticmethod
    async def _retire_speaker(previous: SpeakerService | None, config) -> None:
        """An earlier run's speaker service that no new run continues becomes display-only: it shows its marks (and hands them to the
        prompts per the settings now in force) and takes no new text."""
        if previous is None:
            return
        if not previous.frozen:
            await previous.stop()
            previous.frozen = True
        previous.settings = config              # every time: "use in jobs" may have changed since it was frozen

    async def _watch(self):
        """Persist the transcript periodically, surface microphone errors, keep the screen's clock moving."""
        while True:
            await asyncio.sleep(1.0)
            source = self.source
            if source is not None and source.error and source.error != self._mic_error_logged:
                self._mic_error_logged = source.error
                self.error = source.error
                self.emit('mic_error', error=source.error)
                self.manager.bump()
            if (self.pipeline is not None and self.pipeline.version != self._flushed_version
                    and time.monotonic() - self._last_flush >= FLUSH_SECONDS):
                self.flush()

    async def stop_recording(self):
        """Stop the audio, let what is already recorded finish being transcribed, write everything."""
        async with self._stop_lock:
            if self.state not in ('recording', 'stopping'):
                return
            self.state = 'stopping'
            self.manager.bump()
            source = self.source
            if source is not None:
                source.request_stop()
            if self._tasks:
                try:
                    outcome = (await asyncio.wait_for(asyncio.gather(self._tasks[0], return_exceptions=True),
                                                      STOP_DRAIN_SECONDS))[0]
                except asyncio.TimeoutError:
                    outcome = TimeoutError(f'逐字稿在 {STOP_DRAIN_SECONDS:g} 秒內沒有處理完，已先存檔。')
                    self._tasks[0].cancel()
                if isinstance(outcome, BaseException):
                    self.emit('transcript_pipeline_error', error=f'{type(outcome).__name__}: {outcome}')
                for task in self._tasks[1:]:
                    task.cancel()
                await asyncio.gather(*self._tasks[1:], return_exceptions=True)
                self._tasks = []
            if self.speaker is not None and not self.speaker.frozen:
                await self.speaker.finish()         # decide the last sentences, one last role / text-fill pass
            if source is not None:
                if source.error:
                    self.error = source.error
                await source.close()
            self.source = None
            self.flush()
            self.emit('recording_stopped', segments=self.segment_count, seconds=self.audio_t())
            self.state = 'idle'
            self.manager.bump()

    async def abort(self):
        """Best-effort shutdown (the program is ending): keep what exists."""
        if self.state == 'idle':
            self.flush()
            return
        try:
            await asyncio.wait_for(self.stop_recording(), 90)
        except Exception:
            if self.source is not None:
                self.source.cancelled.set()
                try:
                    await self.source.close()
                except Exception:
                    pass
            for task in self._tasks:
                task.cancel()
            self.flush()
            self.state = 'idle'

    # -- files -----------------------------------------------------------------------
    def flush(self):
        pipeline = self.pipeline
        if pipeline is None:
            return
        self._flushed_version = pipeline.version
        self._last_flush = time.monotonic()
        try:
            atomic_write_json(self.paths.transcript_json, pipeline.to_json())
            atomic_write_text(self.paths.transcript_txt, pipeline.transcript_txt())
            if self.speaker is not None:
                atomic_write_json(self.paths.speaker_json, self.speaker.map_doc())
        except OSError as exc:
            self.error = f'逐字稿存檔失敗：{exc}'
            self.manager.bump()


class VoiceManager:
    def __init__(self, *, load_config: Callable[[], dict], asr: LocalASR | None = None, llm_client: LLMClient | None = None,
                 on_change: Callable[[], None] | None = None, source_factory: Callable | None = None,
                 speaker_embedder: Callable | None = None, llm_backoff: float = 1.0, behavior_log: Callable | None = None):
        self.load_config = load_config
        self.asr = asr or LocalASR()
        self.llm = llm_client or LLMClient()
        self.source_factory = source_factory
        self.speaker_embedder = speaker_embedder or load_embedder
        self.llm_backoff = llm_backoff
        self.behavior_log = behavior_log or append_behavior_event
        self._on_change = on_change or (lambda: None)
        self.session: VoiceSession | None = None
        self.version = 0                    # moves on every change worth redrawing
        self.asr_status = ''
        self.notices: list[tuple[str, str]] = []
        self._preload_task: asyncio.Task | None = None
        self._transition = False

    # -- change tracking --------------------------------------------------------------
    def bump(self):
        self.version += 1
        try:
            self._on_change()
        except Exception:
            pass

    def notify(self, message: str, level: str = 'info'):
        self.notices.append((level, message))

    # -- settings ----------------------------------------------------------------------
    def settings(self) -> VoiceSettings:
        try:
            return VoiceSettings.from_config(self.load_config())
        except ValueError as exc:
            raise VoiceError(f'語音設定有問題：{exc}') from exc

    def safe_settings(self) -> VoiceSettings:
        try:
            return self.settings()
        except VoiceError:
            return VoiceSettings()

    def models_problem(self, settings: VoiceSettings | None = None) -> str:
        """'' when the speech models and the ASR environment are in place, else the reason (never raises)."""
        try:
            settings = settings or self.settings()
        except VoiceError as exc:
            return str(exc)
        try:
            validate_model_dir(settings.asr.asr_model_dir)
            validate_model_dir(settings.asr.aligner_model_dir, aligner=True)
        except ASRError as exc:
            return (f'{NO_MODELS_MESSAGE}{exc}（請重新執行 install.cmd 並選擇安裝語音功能，或執行 '
                    '.venv-asr\\Scripts\\python.exe tools\\download_models.py。）')
        python = worker_python(settings.asr)
        if not python.is_file():
            return f'找不到語音辨識環境：{python}。請執行 install.cmd 建立 .venv-asr。'
        return ''

    @property
    def models_ready(self) -> bool:
        return not self.models_problem()

    # -- the loaded session ---------------------------------------------------------------
    @property
    def recording(self) -> bool:
        return self.session is not None and self.session.state == 'recording'

    @property
    def busy(self) -> bool:
        """A recording is running, starting or stopping: the session cannot be left."""
        return self.session is not None and self.session.state != 'idle'

    def attach(self, folder: str, date: str) -> VoiceSession:
        current = self.session
        if current is not None and current.folder == str(folder) and current.date == date:
            return current
        if current is not None and current.state != 'idle':
            raise VoiceError('錄音進行中，請先停止錄製。')
        if current is not None:
            current.flush()
        session = VoiceSession(self, folder, date)
        session.load()
        self.session = session
        self.bump()
        return session

    def detach(self):
        current = self.session
        if current is None:
            return
        if current.state != 'idle':
            raise VoiceError('錄音進行中，請先停止錄製。')
        current.flush()
        self.session = None
        self.bump()

    def matches(self, folder: str | None, date: str | None) -> bool:
        return (self.session is not None and bool(folder) and bool(date)
                and self.session.folder == str(folder) and self.session.date == date)

    # -- start / stop ------------------------------------------------------------------------
    async def start(self, *, remote: bool = False, device_label: str = '') -> VoiceSession:
        session = self.session
        if session is None:
            raise VoiceError('請先選取患者與就診日期。')
        if session.state != 'idle' or self._transition:
            raise VoiceError('已經在錄音，或正在處理上一個動作。' if session.state == 'recording' else '正在處理上一個動作，請稍候。')
        settings = self.settings()
        problem = self.models_problem(settings)
        if problem:
            raise VoiceError(problem)
        self._transition = True
        session.state = 'starting'
        session.error = ''
        self.bump()
        try:
            await session.start_recording(settings, remote=remote, device_label=device_label)
        except BaseException as exc:
            session.state = 'idle'
            if session.source is not None:
                try:
                    session.source.cancelled.set()
                    await session.source.close()
                except Exception:
                    pass
                session.source = None
            self.bump()
            if isinstance(exc, VoiceError):
                raise
            if isinstance(exc, Exception):
                raise VoiceError(f'無法開始錄音：{exc}') from exc
            raise
        finally:
            self._transition = False
        self.schedule_preload()
        self.bump()
        return session

    async def stop(self):
        session = self.session
        if session is None or session.state == 'idle':
            return
        await session.stop_recording()

    # -- the transcript for the agents --------------------------------------------------------------
    def snapshot_text(self, folder: str | None, date: str | None) -> str:
        """The transcript of this session as it stands now, ready for a prompt ('' when there is none). Call it on the web loop
        (the pipeline is single-threaded); the text is a frozen copy."""
        if folder and date:
            try:
                self.attach(folder, date)            # idempotent; refused only while a recording of another session runs
            except VoiceError:
                pass
        if not self.matches(folder, date):
            return ''
        pipeline = self.session.pipeline
        if pipeline is None or not any(s.visible for s in pipeline.segments):
            return ''
        return pipeline.snapshot().text

    def recording_position(self, folder: str | None = None, date: str | None = None) -> tuple[float | None, bool]:
        """(where the transcript that can be seen now ends, paused), in seconds of the recording: the `截至 mm:ss` of a snapshot.
        It is not the length of the audio: the windows still being recognized and corrected are not on the screen, and a physician
        who edits the record has read only what is. `None` seconds = this session has no recording; 0.0 = it has one but nothing is
        transcribed yet. The clock only grows, so it orders a manual edit against the transcript lines: every line that ends at or
        before it was visible to the physician, every later one was not."""
        if folder is not None and not self.matches(folder, date):
            # Not the loaded session (after a restart, or a page reloaded before its voice window was opened): the recording of that
            # session is still its file. Without this a version made now would get "no recording yet", and a reviewer could take
            # the earlier transcript lines for evidence that appeared after the edit.
            if not folder or not date:
                return None, False
            paths = voice_paths(folder, date)
            return (_saved_transcript_end(paths.transcript_json), True) if sample_count(paths.audio) else (None, False)
        session = self.session
        if session is None:
            return None, False
        pipeline = session.pipeline
        if session.state != 'idle' and session.source is not None:
            return (pipeline.upto if pipeline is not None else 0.0), False
        if not sample_count(session.paths.audio):
            return None, False
        return (pipeline.upto if pipeline is not None else 0.0), True

    def frozen_clock(self, folder: str | None, date: str | None) -> Callable[[], tuple[float | None, bool]]:
        """The clock for the versions an agent writes during one turn. Call it where `snapshot_text` is called: the position stays where
        that frozen transcript ends. Lines that arrive while the agent works were seen neither by the physician who asked nor by the
        agent, so they are not "before" what the agent writes. Whether the recording is paused is still read live."""
        seconds = self.recording_position(folder, date)[0]
        return lambda: (seconds, self.recording_position(folder, date)[1])

    @property
    def remote_token(self) -> str:
        source = self.session.source if self.session is not None else None
        return source.token if isinstance(source, RemoteAudioSource) else ''

    def set_speaker_roles(self, mapping: dict[int, str]):
        session = self.session
        if session is None or session.speaker is None or session.speaker.frozen:
            return
        session.speaker.set_group_roles(mapping)

    def set_speaker_lock(self, locked: bool):
        session = self.session
        if session is None or session.speaker is None or session.speaker.frozen:
            return
        session.speaker.set_lock(locked)

    # -- ASR model ------------------------------------------------------------------------------------
    def schedule_preload(self):
        if self._preload_task and not self._preload_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._preload_task = loop.create_task(self.preload())

    async def preload(self):
        """Load the speech models into the GPU memory now (at program start, and again after a settings change). Without the
        models there is nothing to load: say so and leave everything else alone."""
        try:
            settings = self.settings()
        except VoiceError as exc:
            self.asr_status = str(exc)
            self.bump()
            return
        problem = self.models_problem(settings)
        if problem:
            self.asr_status = '未安裝語音模型（不預載）' if NO_MODELS_MESSAGE in problem else '語音環境未就緒（不預載）'
            self.bump()
            return
        self.asr_status = '載入中…'
        self.bump()
        try:
            info = await self.asr.load(settings.asr)
            self.asr_status = f'就緒（{info.get("device", "?")}）'
        except Exception as exc:
            self.asr_status = f'載入失敗：{exc}'
        self.bump()

    # -- shutdown -------------------------------------------------------------------------------------------
    async def shutdown(self):
        if self._preload_task and not self._preload_task.done():
            self._preload_task.cancel()
        if self.session is not None:
            await self.session.abort()
        try:
            await self.asr.close()
        except Exception:
            pass


_SHARED: VoiceManager | None = None


def shared_manager(*, load_config: Callable[[], dict]) -> VoiceManager:
    """The one VoiceManager of the process.

    NiceGUI's script mode runs the whole main script again for every browser connection (a page load, a second tab), so a manager
    made in the script would exist once per page load, each with its own speech-recognition process and its own idea of which
    session is recording. The modules of this package are imported once, so the manager lives here and every run of the script
    gets the same one. Tests build their own `VoiceManager`."""
    global _SHARED
    if _SHARED is None:
        _SHARED = VoiceManager(load_config=load_config)
    return _SHARED
