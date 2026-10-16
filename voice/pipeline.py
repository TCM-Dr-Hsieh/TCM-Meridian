"""Continuous transcript pipeline: window -> ASR -> align -> OpenCC -> LLM correction -> commit.

Adapted from voice_to_text Session.process_chunk. Differences: never stops the recording,
failed windows become numbered gap segments (auto-retry once), text is converted to
Traditional right after alignment, and a frozen snapshot can be taken at any time.

Differences from TCM-Meridian-mini: the transcript of a session outlives one recording. It can be stopped and continued
(`begin_run` takes a new audio source; the numbering and the clock go on) and it can be restored from `to_json()` after the
program was restarted (`restore`). The segments of an earlier run are locked: the rolling correction never reaches back into them.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Callable

from .config import ASRSettings
from .textutil import format_clock, join_texts, script
from .alignment import AlignmentError, map_alignment, select_new_window
from .asr import ASRError, LocalASR
from .audio import AudioSource, Chunk
from .speaker.service import MARKS_NOTE
from .corrector import RollingResult, TranscriptCorrector
from .editing import changes


@dataclass
class Segment:
    index: int
    kind: str                  # 'speech' | 'gap'
    start: float               # window range (audio seconds)
    end: float
    new_start: float           # start of the audio this segment newly covers
    raw_asr: str               # ASR original (often Simplified)
    raw: str                   # raw_asr converted to Traditional
    added: str                 # aligned new text of this window (Traditional)
    corrected: str             # current text (LLM-corrected when possible)
    elapsed: float = 0.0
    warning: str = ''
    reason: str = ''
    corrected_by: str = 'none'
    alignment: list = field(default_factory=list)
    edits: list = field(default_factory=list)
    error: str = ''
    added_asr: str = ''        # `added` before OpenCC (ASR original script)

    @property
    def visible(self) -> bool:
        return bool(self.corrected.strip())


@dataclass
class Revision:
    index: int
    segments: list[int]
    before: list[str]
    after: list[str]
    edits: list
    reason: str = ''


@dataclass
class TranscriptSnapshot:
    text: str            # header + lines, ready for a prompt
    lines: str           # lines only
    upto: float
    last_index: int      # highest segment index that has visible text
    max_index: int       # highest segment index that exists
    unlocked: list[int]
    pending: int
    marked: bool = False   # the lines carry speaker marks


def load_segments(rows) -> list[Segment]:
    """Segments from the rows `to_json()` saved (fields this version does not know are ignored)."""
    names = {item.name for item in fields(Segment)}
    return [Segment(**{k: v for k, v in row.items() if k in names}) for row in rows or []]


def visible_end(segments) -> float:
    """Where a transcript reaches: the end of its latest segment that has text to show. A segment still without text (the window held only
    the overlap that was transcribed before, or the recognizer returned nothing) is not on the screen and not in what the agents receive,
    and its text may still come with the rolling correction: lines it gets later must count as appearing after the point reached now."""
    return max((s.end for s in segments if s.visible), default=0.0)


def gap_marker(start: float, end: float) -> str:
    return f'【音訊可能缺漏 {format_clock(start)}–{format_clock(end)}】'


class TranscriptPipeline:
    def __init__(self, *, asr: LocalASR, asr_settings: ASRSettings, corrector: TranscriptCorrector | None,
                 source: AudioSource | None = None, emit: Callable[..., None] | None = None, speaker=None):
        self.asr = asr
        self.settings = asr_settings
        self.corrector = corrector
        self.source = source
        self.speaker = speaker           # a SpeakerService, or None when speaker marking is off for this visit
        self._emit = emit or (lambda *a, **k: None)
        self.segments: list[Segment] = []
        self.revisions: list[Revision] = []
        self.rolling_start = 0
        self.committed_end = -1.0
        self.version = 0                 # bumped on every visible change (UI refresh key)
        self.finished = source is None            # a pipeline without a source has nothing running
        self.processing = False
        self.status = '等待音訊'
        self.last_error = ''
        self._last_recognized: dict | None = None

    # -- continuing a transcript ----------------------------------------
    def begin_run(self, source: AudioSource, speaker=None):
        """The recording is started again: new audio source, the same transcript. Numbering and clock go on; everything
        transcribed so far stays as it is (locked)."""
        self.source = source
        if speaker is not None:
            self.speaker = speaker
        self.rolling_start = len(self.segments)
        self.finished = False
        self.processing = False
        self.status = '等待音訊'
        self.version += 1

    def restore(self, data: dict):
        """Load a transcript saved by `to_json()` (after a restart). The result is a finished transcript that `begin_run` can continue."""
        self.segments = load_segments(data.get('segments', []))
        revision_names = {item.name for item in fields(Revision)}
        self.revisions = [Revision(**{k: v for k, v in row.items() if k in revision_names}) for row in data.get('revisions', [])]
        self.rolling_start = len(self.segments)
        ends = [s.end for s in self.segments]
        saved = data.get('committed_end')
        self.committed_end = float(saved) if isinstance(saved, (int, float)) else (max(ends) if ends else -1.0)
        self.finished = True
        self.version += 1

    # -- state ---------------------------------------------------------
    @property
    def pending(self) -> int:
        queued = self.source.queue.qsize() if self.source is not None else 0
        return queued + (1 if self.processing else 0)

    def unlocked_indexes(self) -> list[int]:
        if self.finished:
            return []
        start = max(self.rolling_start, len(self.segments) - 2)
        return [s.index for s in self.segments[start:] if s.kind == 'speech']

    @property
    def upto(self) -> float:
        """Where the transcript reaches (the `截至 mm:ss` of a snapshot): the end of its latest segment that has text. The recording clock of a
        record version is this, so what the clock says was on the screen is what a snapshot taken then would hold."""
        return visible_end(self.segments)

    def snapshot(self) -> TranscriptSnapshot:
        marked = self.speaker is not None and self.speaker.marks_in_text and self.speaker.settings.use_in_jobs
        written = {s.index: self.line(s, marked) for s in self.segments if s.visible}
        lines = '\n'.join(line for line in written.values() if line)         # a segment of nothing but background voices has no line
        upto = self.upto
        unlocked = [i for i in self.unlocked_indexes() if written.get(i, True)]
        pending = self.pending if not self.finished else 0
        header = f'【診間對話逐字稿 · 截至 {format_clock(upto)}'
        if marked:
            header += f' · {MARKS_NOTE}'
        if unlocked:
            # "Unlocked" only means the rolling corrector may still touch these segments. The model must read them
            # as valid sources (they are usually the newest, most relevant lines); the old wording, "尚未鎖定", made
            # the writer and the reviewer invent contradictory rules about citing them.
            header += ' · 最新的 ' + '、'.join(f'#{i}' for i in unlocked) + ' 仍在校稿中（文字日後可能微調，仍是有效來源，可以引用）'
        if pending:
            header += f' · 另有 {pending} 個音訊視窗尚在處理，未納入'
        header += '】'
        last = max((s.index for s in self.segments if s.visible), default=0)
        return TranscriptSnapshot(f'{header}\n{lines}' if lines else f'{header}\n（尚無逐字稿內容）',
                                  lines, upto, last, len(self.segments), unlocked, pending, marked)

    def line(self, segment: Segment, marked: bool = False, keep_background: bool = False) -> str:
        """One transcript line; with `marked`, the text carries the speaker marks (`醫師: … -> 患者或家屬: …`) when it has any. Background
        voices (another room) are left out whether it is marked or not, so the line is '' when nothing else is left; `keep_background` keeps them
        (as `背景: …` in a marked line)."""
        body = self.speaker.body(segment, marks=marked, keep_background=keep_background) if self.speaker is not None and segment.kind == 'speech' else None
        if body == '':
            return ''
        text = segment.corrected if body is None else body
        return f'語音#{segment.index} {format_clock(segment.new_start)}–{format_clock(segment.end)} {text}'

    def transcript_txt(self) -> str:
        marked = self.speaker is not None and self.speaker.marks_in_text
        return '\n'.join(self.line(s, marked, keep_background=True) for s in self.segments if s.visible) + '\n'

    def to_json(self) -> dict:
        data = {'finished': self.finished, 'committed_end': self.committed_end,
                'segments': [asdict(s) for s in self.segments],
                'revisions': [asdict(r) for r in self.revisions]}
        if self.speaker is not None:
            data['speaker'] = {'status': self.speaker.status, 'roles_trusted': self.speaker.marks_in_text,
                               'labels': self.speaker.labels_doc()}
        return data

    # -- processing ----------------------------------------------------
    async def run(self):
        if self.source is None:
            return
        try:
            while True:
                chunk = await self.source.next_chunk()
                if chunk is None:
                    break
                self.processing = True
                try:
                    await self._handle(chunk)
                finally:
                    self.processing = False
            self.status = '音訊來源已結束，逐字稿處理完成'
        finally:
            self.finished = True
            self.version += 1
            self._emit('transcript_finished', segments=len(self.segments))

    async def _recognize(self, chunk: Chunk):
        last: Exception | None = None
        for attempt in (1, 2):
            try:
                recognized = await self.asr.recognize(self.settings, chunk.samples)
                self._last_recognized = recognized
                units = map_alignment(recognized['text'], recognized['items'], chunk.start, chunk.duration)
                return recognized, units
            except (ASRError, AlignmentError) as exc:
                last = exc
                self._emit('asr_error', chunk=chunk.index, attempt=attempt, error=str(exc))
        raise last  # type: ignore[misc]

    async def _handle(self, chunk: Chunk):
        started = time.monotonic()
        end = chunk.start + chunk.duration
        self.status = f'辨識音訊視窗 {format_clock(chunk.start)}–{format_clock(end)}'
        self._last_recognized = None
        try:
            recognized, units = await self._recognize(chunk)
        except (ASRError, AlignmentError) as exc:
            self.last_error = str(exc)
            self._add_gap(chunk, str(exc))
            return
        raw_asr = recognized['text']
        if not raw_asr.strip():
            self.committed_end = end          # silence: nothing to add
            self.status = '等待語音'
            return
        added_asr, selected = select_new_window(raw_asr, units, self.committed_end)
        raw, added = script.to_traditional(raw_asr), script.to_traditional(added_asr)
        new_start = chunk.start if self.committed_end < 0 else max(chunk.start, self.committed_end)
        candidate = Segment(len(self.segments) + 1, 'speech', chunk.start, end, new_start, raw_asr, raw, added,
                            added, 0.0, '', chunk.reason, 'none', [asdict(u) for u in units], added_asr=added_asr)
        recent_start = max(self.rolling_start, len(self.segments) - 2)
        recent = [s for s in self.segments[recent_start:] if s.kind == 'speech'] + [candidate]
        before = [s.corrected for s in recent]
        result = RollingResult(before)
        if self.corrector is not None and any(s.added for s in recent):
            self.status = '逐字稿校稿中'
            frozen = join_texts(s.corrected for s in self.segments[:recent_start])
            rows = [{'index': s.index, 'start': s.start, 'end': s.end, 'reason': s.reason, 'added': s.added,
                     'added_asr': s.added_asr, 'current': s.corrected, 'raw': s.raw} for s in recent]
            result = await self.corrector.revise_recent(
                frozen[-self.settings.context_chars:], rows, protect_start=recent_start > 0,
                protect_end=chunk.reason == 'window')
        candidate.warning = result.warning
        candidate.elapsed = time.monotonic() - started
        # No awaits below: commit the window, the revision and the watermark together.
        self.segments.append(candidate)
        if result.texts != before:
            for segment, revised in zip(recent, result.texts):
                segment.corrected = revised
                segment.corrected_by = 'llm'
                segment.edits = changes(segment.added, revised)
            revision = Revision(len(self.revisions) + 1, [s.index for s in recent], before, result.texts,
                                result.edits, '依後續 ASR 補齊接縫' if result.seam_repaired else '')
            self.revisions.append(revision)
            self._emit('transcript_revised', **asdict(revision))
        elif self.corrector is not None and candidate.added:
            candidate.corrected_by = 'llm' if not result.warning else 'none'
        self.committed_end = end
        self.version += 1
        self.status = '等待語音'
        self._emit('transcript_segment', index=candidate.index, start=candidate.start, end=candidate.end,
                   new_start=candidate.new_start, raw_asr=candidate.raw_asr, raw=candidate.raw,
                   added=candidate.added, corrected=candidate.corrected, elapsed=candidate.elapsed,
                   warning=candidate.warning)
        if self.speaker is not None and selected:
            await self.speaker.feed_segment(chunk, candidate, selected, added_asr)     # never raises: marking must not stop the transcript

    def _add_gap(self, chunk: Chunk, error: str):
        end = chunk.start + chunk.duration
        possible_start = max(chunk.start, self.committed_end)
        marker = gap_marker(possible_start, end)
        raw_asr = (self._last_recognized or {}).get('text', '')
        gap = Segment(len(self.segments) + 1, 'gap', chunk.start, end, possible_start, raw_asr,
                      script.to_traditional(raw_asr), '', marker, 0.0, error, chunk.reason, 'none', [], [], error)
        self.segments.append(gap)
        self.committed_end = end
        self.rolling_start = len(self.segments)     # never revise across a gap
        self.version += 1
        self.status = '音訊片段辨識失敗，已標記缺漏並繼續'
        self._emit('transcript_gap', index=gap.index, start=possible_start, end=end, error=error, raw_asr=raw_asr)

    async def wait_finished(self, timeout: float | None = None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.finished:
            if deadline is not None and time.monotonic() > deadline:
                raise asyncio.TimeoutError
            await asyncio.sleep(0.1)
