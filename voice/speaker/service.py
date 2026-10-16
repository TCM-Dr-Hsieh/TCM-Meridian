"""One visit's speaker marking: the tracker, the LLM role layer and text fill, and the views the transcript pipeline and the UI read.

The transcript pipeline only calls `feed` (after each committed window), `body` (to print a segment with its marks) and `finish`.
Nothing here may stop or slow the transcript: errors turn the marking off for the rest of the visit and are logged.
"""
from __future__ import annotations

import asyncio
from typing import Callable

from ..config import Endpoint, SpeakerSettings
from ..llm import CallFailed, LLMCaller, PRIORITY_BACKGROUND
from . import BACKGROUND, DOCTOR, LABEL_NAMES, OTHER, TEXT_SOURCES, UNKNOWN, fill, roles, tuning
from .render import project, prompt_body, segment_runs
from .tracker import Params, SpeakerTracker, group_name
from .units import Token, UnitParams, ends_sentence

FINAL_PASS_SECONDS = 60.0         # the last role / fill pass at the end of a visit may not hold the visit open longer than this
MARKS_NOTE = '說話者標記由程式自動判斷，可能有誤；「*」表示由上下文推測、較不可靠'


def build_params(values: dict) -> Params:
    """The tracker's numbers from the value of every knob (`tuning.effective`)."""
    return Params(
        unknown_percentile=float(values['unknown_percentile']), refit_seconds=float(values['refit_seconds']),
        first_fit_units=int(values['first_fit_units']), min_group_share=float(values['min_group_share']),
        min_group_units=int(values['min_group_units']), first_split_max_cos=float(values['first_split_max_cos']),
        new_group_max_cos=float(values['new_group_max_cos']), new_group_share=float(values['new_group_share']),
        new_group_window=float(values['new_group_window']), max_groups=int(values['max_groups']), far_cos=float(values['far_cos']),
        clip=float(values['clip']), background_rel_db=float(values['background_rel_db']),
        background_max_cos=float(values['background_max_cos']), quiet_gate_db=float(values['quiet_gate_db']),
        quiet_gate_percentile=float(values['quiet_gate_percentile']), min_history=int(values['min_history']),
        fill_after=int(values['fill_wait']), fill_batch=int(values['fill_batch']), fill_max_age=float(values['fill_max_age']),
        fill2_batch=int(values['fill2_batch']), fill2_refits=int(values['fill2_refits']),
        units=UnitParams(target=float(values['target']), pause=float(values['pause']), min_unit=float(values['min_unit']),
                         max_merged=float(values['max_merged']), sent_pause=float(values['sent_pause'])))


def window_tokens(index: int, selected, added_asr: str, added: str) -> list[Token]:
    """The timed pieces of one segment's new text, as character ranges of `added` (the Traditional text)."""
    if not selected:
        return []
    origin = selected[0].begin
    starts = [u.begin - origin for u in selected] + [len(added_asr)]
    cuts = starts if len(added) == len(added_asr) else project(added_asr, added, starts)
    return [Token(index, cuts[k], cuts[k + 1], u.start, u.end,
                  ends_sentence(added_asr[u.begin - origin:u.finish - origin])) for k, u in enumerate(selected)]


class SpeakerService:
    def __init__(self, *, settings: SpeakerSettings, embed, caller: LLMCaller, endpoint: Callable[[], Endpoint],
                 segments: Callable[[], list], emit: Callable[..., object], on_change: Callable[[], None],
                 audio_t: Callable[[], float | None] = lambda: None):
        self.settings = settings
        values = tuning.effective(settings.advanced, settings.unknown_percentile)
        self.tracker = SpeakerTracker(embed, build_params(values))
        self._max_lines = int(values['max_lines'])
        self._fill_chunk, self._fill_before, self._fill_after = int(values['fill_chunk']), int(values['fill_before']), int(values['fill_after'])
        self.caller = caller
        self._endpoint = endpoint
        self._segments = segments
        self._emit_event = emit
        self._on_change = on_change
        self._audio_t = audio_t
        self.voter = roles.RoleVoter(lock_after=int(values['lock_after']))
        self.role_versions: list[dict] = []
        self.manual_events: list[dict] = []
        self.failed = ''
        self.fill_stats = {'asked': 0, 'accepted': 0, 'rejected': 0, 'failed_calls': 0}
        self.fill2_stats = {'asked': 0, 'accepted': 0, 'rejected': 0, 'failed_calls': 0, 'accepted_agreeing': 0, 'accepted_differing': 0}
        self._asked_roles_at: tuple[int, int] = (0, 0)
        self._seen_groups = 0
        self._finished = False
        self._wake = asyncio.Event()
        self._settle_lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._prompts: dict[str, str] = {}
        # Labels of an earlier run of the same transcript (the program was restarted since): shown and handed to the prompts as
        # they were. The voice groups themselves are not stored (voiceprints never leave the memory), so a new run starts groups anew.
        self._restored: dict[int, tuple] = {}
        self._restored_trusted = False
        self.frozen = False             # a service that only shows an earlier run's marks: it takes no new text and asks no model

    # ------------------------------------------------------------------ plumbing
    def emit(self, type: str, **fields):
        try:
            self._emit_event(type, **fields)
        except Exception:                     # logging must never break the transcript
            pass

    def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name='speaker-llm')

    async def stop(self):
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def _prompt(self, name: str, loader) -> str:
        if name not in self._prompts:
            self._prompts[name] = loader()
        return self._prompts[name]

    def _fail(self, exc: BaseException, where: str):
        self.failed = f'{type(exc).__name__}: {exc}'
        self.emit('speaker_error', where=where, error=self.failed)
        self._on_change()

    # ------------------------------------------------------------------ views
    @property
    def marks_visible(self) -> bool:
        """True while the screen shows the marks (even in the first minute, before anybody is named: those lines say 不明)."""
        return not self.failed

    @property
    def trusted(self) -> bool:
        """The role map was confirmed (a second agreeing answer) or set by the physician."""
        return self.voter.trusted

    @property
    def _live_marks_ok(self) -> bool:
        """The running tracker's marks may go into text: a doctor is named AND the map is trusted (see `marks_in_text`)."""
        return self.tracker.doctor_known and self.voter.trusted

    @property
    def marks_in_text(self) -> bool:
        """True when the transcript text for prompts and files carries marks. Two conditions: a doctor is named (before that every
        line would only say 不明, which is no information) AND the map is trusted: the first answer alone could have named the
        wrong group, which turns every mark around for the writer and the reviewer at once. The screen shows marks earlier.
        Marks restored from an earlier run were trusted when they were saved; the segments of the current run follow `_live_marks_ok`."""
        return not self.failed and (self._live_marks_ok or (bool(self._restored) and self._restored_trusted))

    def _labels_for(self, seg: int, *, for_text: bool = False) -> tuple | None:
        """The labels of a segment: the running tracker's, else those restored from an earlier run. For text (prompts, files) the
        tracker's marks of the current run count only once they are trusted; until then the segment reads as unknown."""
        live = self.tracker.labels_for(seg)
        if live:
            if for_text and not self._live_marks_ok:
                return tuple((c0, c1, UNKNOWN, '') for c0, c1, *_ in live)
            return live
        return self._restored.get(seg)

    @property
    def status(self) -> str:
        if self.failed:
            return f'說話者標記已停止（{self.failed}）'
        if self.tracker.group_count == 0:
            return '說話者標記：建立聲音群中（約需 1 分鐘）'
        if not self.tracker.doctor_known:
            return '說話者標記：判定誰是醫師中'
        if not self.voter.trusted:
            return f'說話者標記：暫定，待下一次角色判定確認後才給病歷作業（{self.tracker.group_count} 個聲音群）'
        stats = self.tracker.stats()
        share = f'，不明 {stats["unknown"] * 100 // max(1, stats["decided"])}%' if stats['decided'] else ''
        share += f'，背景 {stats["background"]} 句' if stats['background'] else ''
        waiting = len(self.voter.pending_groups)
        pending = f'，{waiting} 群待確認（確認前畫面與病歷作業都視為不明）' if waiting > 0 else ''
        return f'說話者標記：已啟用（{stats["groups"]} 個聲音群{share}{pending}）'

    def body(self, segment, *, marks: bool, keep_background: bool = False) -> str | None:
        """The segment's text for a prompt or a file; None when it is to be used as it is (no timed text: a gap, or marking is off; or no marks
        wanted and no background voices in it). With `marks`: `醫師: … -> 患者或家屬: …`. Background voices (another room) are left out whether
        marks are wanted or not (an empty string when nothing else is left), or kept as `背景: …` with `keep_background` (marks only)."""
        if self.failed:
            return None
        labels = self._labels_for(segment.index, for_text=True)
        if not labels or not segment.corrected.strip():
            return None
        runs = segment_runs(segment.added, segment.corrected, labels)
        if marks:
            return prompt_body(runs, keep_background)
        if keep_background or not any(r.label == BACKGROUND for r in runs):
            return None
        return ' '.join(r.text for r in runs if r.label != BACKGROUND)

    def runs(self, segment):
        """The segment as runs of one speaker, for the screen; None when it has no timed text."""
        labels = None if self.failed else self._labels_for(segment.index)
        return segment_runs(segment.added, segment.corrected, labels) if labels else None

    def groups(self) -> list[dict]:
        """What the speaker-groups window shows: per group its role, speech share and two example lines."""
        out = []
        for info in self.tracker.group_info():
            examples = [self._sentence_text(i) for i in info.examples]
            out.append({'gid': info.gid, 'name': info.name, 'role': info.role, 'units': info.units,
                        'seconds': info.seconds, 'share': info.share, 'examples': [e for e in examples if e]})
        return out

    def effective_tuning(self) -> dict:
        """The value of every knob as the running objects hold it (not as the settings say): the log and the tests read what really ran."""
        p, u = self.tracker.params, self.tracker.params.units
        return {'target': u.target, 'pause': u.pause, 'min_unit': u.min_unit, 'max_merged': u.max_merged, 'sent_pause': u.sent_pause,
                'refit_seconds': p.refit_seconds, 'quiet_gate_db': p.quiet_gate_db, 'quiet_gate_percentile': p.quiet_gate_percentile,
                'first_fit_units': p.first_fit_units, 'min_group_share': p.min_group_share, 'min_group_units': p.min_group_units,
                'first_split_max_cos': p.first_split_max_cos, 'new_group_max_cos': p.new_group_max_cos, 'new_group_share': p.new_group_share,
                'new_group_window': p.new_group_window, 'max_groups': p.max_groups, 'far_cos': p.far_cos, 'clip': p.clip,
                'unknown_percentile': p.unknown_percentile, 'min_history': p.min_history,
                'lock_after': self.voter.lock_after, 'max_lines': self._max_lines,
                'background_rel_db': p.background_rel_db, 'background_max_cos': p.background_max_cos,
                'fill_wait': p.fill_after, 'fill_batch': p.fill_batch, 'fill_max_age': p.fill_max_age,
                'fill_chunk': self._fill_chunk, 'fill_before': self._fill_before, 'fill_after': self._fill_after,
                'fill2_batch': p.fill2_batch, 'fill2_refits': p.fill2_refits}

    def map_doc(self) -> dict:
        """`speaker/map.json`: no voiceprints, only the role maps, per-minute counts and the physician's manual actions."""
        return {'status': self.status, 'failed': self.failed, 'settings': {
                    'unknown_percentile': self.settings.unknown_percentile, 'text_fill': self.settings.text_fill,
                    'text_fill2': self.settings.text_fill2,
                    'use_in_jobs': self.settings.use_in_jobs, 'tuning': tuning.changed(self.effective_tuning())},
                'role_versions': self.role_versions, 'manual_events': self.manual_events,
                'minutes': self.tracker.minutes, 'stats': self.tracker.stats(), 'fill': self.fill_stats, 'fill2': self.fill2_stats,
                'groups': [{'name': g['name'], 'role': g['role'], 'units': g['units'], 'seconds': round(g['seconds'], 1),
                            'share': round(g['share'], 3)} for g in self.groups()]}

    def restore_labels(self, labels: dict, trusted: bool):
        """Take the labels saved by `labels_doc()` in an earlier run (`{"seg": [[c0, c1, label, source, group], ...]}`)."""
        restored: dict[int, tuple] = {}
        for seg, rows in (labels or {}).items():
            try:
                restored[int(seg)] = tuple((int(r[0]), int(r[1]), str(r[2]), str(r[3])) for r in rows)
            except (TypeError, ValueError, IndexError):
                continue
        self._restored = restored
        self._restored_trusted = bool(trusted)

    def rebind(self, caller: LLMCaller, endpoint: Callable[[], Endpoint]):
        """Use the model caller and endpoint of the run that starts now (they follow the settings in force)."""
        self.caller = caller
        self._endpoint = endpoint

    def resume(self):
        """Recording starts again in the same run (stop, then start): marking goes on with the voice groups and the role map it had."""
        if self.failed or self.frozen:
            return
        self._finished = False
        self.tracker.reopen()
        self.start()

    def labels_doc(self) -> dict:
        """Per segment, one entry per timed piece: [c0, c1, label, source, voice group] over the segment's `added` text. The group
        is the anonymous voice group the sentence was decided for (null while undecided); the label is what the screen shows, so a
        group whose role is not confirmed yet is stored as unknown with its group number kept. With the role map versions in
        `speaker/map.json` the FINAL marks can be rebuilt under any role map; the state at an earlier moment cannot, because an
        unknown sentence may later be labelled (re-scoring, text fill) and the time of that change is not stored."""
        groups = self.tracker.all_groups()
        doc = {str(seg): [[*item, None] for item in labels] for seg, labels in sorted(self._restored.items())}
        # What is saved is what the prompts see: marks of the current run that are not trusted yet are saved as unknown, so a
        # later restore cannot mistake a tentative guess for a confirmed role.
        for seg, labels in sorted(self.tracker.all_labels().items()):
            shown = labels if self._live_marks_ok else tuple((c0, c1, UNKNOWN, '') for c0, c1, *_ in labels)
            doc[str(seg)] = [[*item, gid] for item, gid in zip(shown, groups.get(seg, ()))]
        return doc

    def role_summary(self) -> dict:
        """The role map in force, for the log of a job that starts now."""
        return {'roles': {group_name(g): LABEL_NAMES.get(r, '不明') for g, r in sorted(self.tracker.roles.items())},
                'trusted': self.voter.trusted, 'locked_by_physician': self.voter.manual, 'in_text': self.marks_in_text,
                'unconfirmed_groups': [group_name(g) for g in self.voter.pending_groups]}

    # ------------------------------------------------------------------ feeding (called by the transcript pipeline)
    async def feed_segment(self, chunk, segment, selected, added_asr: str):
        """Called by the pipeline after it committed a segment: `selected` are the aligned units of the segment's new text."""
        if self.failed or self.frozen:
            return
        try:
            tokens = window_tokens(segment.index, selected, added_asr, segment.added)
        except Exception as exc:
            self._fail(exc, 'tokens')
            return
        await self.feed(chunk.start, chunk.samples, tokens)

    async def feed(self, chunk_start: float, samples, tokens: list[Token]):
        if self.failed:
            return
        try:
            changed = await asyncio.to_thread(self._process, chunk_start, samples, tokens)
        except Exception as exc:
            self._fail(exc, 'feed')
            return
        self._after_tracker(changed)

    def _process(self, chunk_start, samples, tokens):
        self.tracker.add_audio(chunk_start, samples)
        return self.tracker.add_tokens(tokens)

    def _after_tracker(self, changed: set[int]):
        for event in self.tracker.events:
            self.emit(event.pop('event'), **event)
        self.tracker.events.clear()
        if self.tracker.groups_version != self._seen_groups:
            self._seen_groups = self.tracker.groups_version
            self.voter.groups_changed(self.tracker.group_count)
        if changed:
            self._on_change()
        self._wake.set()


    # ------------------------------------------------------------------ LLM work, in the background
    async def _loop(self):
        while True:
            await self._wake.wait()
            self._wake.clear()
            try:
                await self.settle()
            except asyncio.CancelledError:
                raise
            except Exception as exc:              # a failing role / fill step must not end the loop
                self.emit('speaker_error', where='llm', error=f'{type(exc).__name__}: {exc}')

    async def settle(self, *, final: bool = False):
        """One round of LLM work: ask who the doctor is (when due), then fill what the voice left unknown."""
        async with self._settle_lock:
            if self.failed:
                return
            if self._roles_due(final):
                await self._ask_roles()
            if self.settings.text_fill and self.tracker.doctor_known:
                await self._fill(final)
                if self.settings.text_fill2:
                    await self._fill2(final)

    def _roles_due(self, final: bool) -> bool:
        tracker = self.tracker
        if tracker.group_count < 2 or not self.voter.wants_answer:
            return False
        now = (len(tracker.minutes), tracker.groups_version)
        return final and (not self.voter.trusted or bool(self.voter.pending_groups)) or now != self._asked_roles_at

    async def _ask_roles(self):
        tracker = self.tracker
        self._asked_roles_at = (len(tracker.minutes), tracker.groups_version)
        groups = list(range(tracker.group_count))
        lines = roles.dialogue_lines(self._segments(), tracker, limit=self._max_lines)
        if not lines:
            return
        base = self._prompt('roles', roles.load_prompt)
        user = roles.request_payload(groups, lines)

        def build(last_error):
            return [{'role': 'system', 'content': base + (roles.RETRY_NOTE if last_error else '')},
                    {'role': 'user', 'content': user}]

        try:
            outcome = await self.caller.call(agent='speaker_roles', endpoint=self._endpoint(), messages=build,
                                             priority=PRIORITY_BACKGROUND, retries=1, max_tokens=800,
                                             validator=lambda text: roles.parse_answer(text, groups))
        except CallFailed as exc:
            self.emit('speaker_roles_failed', error=exc.last_error)
            return
        if self.voter.feed(outcome.value):
            self._apply_roles('llm')
        else:
            self.emit('speaker_roles_answer', answer={group_name(g): r for g, r in outcome.value.items()},
                      confirmed=self.voter.confirmed, locked=self.voter.locked)

    def _apply_roles(self, source: str):
        self.tracker.set_roles(dict(self.voter.current))
        self._sync_confirmed()
        names = {group_name(g): LABEL_NAMES.get(r, '不明') for g, r in sorted(self.voter.current.items())}
        self.role_versions.append({'t': self._audio_t(), 'source': source, 'roles': names, 'confirmed': self.voter.confirmed,
                                   'confirmed_groups': sorted(group_name(g) for g in self.voter.confirmed_roles)})
        self.emit('speaker_map_set', source=source, roles=names, confirmed=self.voter.confirmed)
        self._on_change()                             # also when only the confirmation changed: what the screen and the files show depends on it
        self._wake.set()                              # sentences waiting for the map are decided now; some may need the fill

    def _sync_confirmed(self):
        """Tell the tracker which groups are confirmed: once the map is trusted, the others are unknown on the screen, in the files,
        in the counts and in the prompts alike (before that the screen shows the tentative roles)."""
        self.tracker.set_confirmed(frozenset(self.voter.confirmed_roles) if self.voter.trusted else None)

    # -- text fill -------------------------------------------------------
    def _pieces(self, seg_index: int) -> dict[int, str]:
        """Global token index -> the piece of the segment's CURRENT text that belongs to it."""
        segment = self._segments()[seg_index - 1]
        indexes = self.tracker.segment_token_indexes(seg_index)
        starts = [self.tracker.tokens[i].c0 for i in indexes] + [len(segment.added)]
        cuts = project(segment.added, segment.corrected, starts)
        cuts[0], cuts[-1] = 0, len(segment.corrected)
        return {i: segment.corrected[cuts[k]:cuts[k + 1]] for k, i in enumerate(indexes)}

    def _sentence_text(self, index: int) -> str:
        tracker = self.tracker
        first, last = tracker.sentence_range(index)
        cache: dict[int, dict[int, str]] = {}
        parts = []
        for i in range(first, last + 1):
            seg = tracker.tokens[i].seg
            if seg not in cache:
                cache[seg] = self._pieces(seg)
            parts.append(cache[seg][i])
        return ''.join(parts).strip()

    async def _fill(self, final: bool):
        """First text fill: the answer is used only when it agrees with the sentence's weak voice evidence."""
        tracker = self.tracker
        await self._fill_stage(final, wanted=tracker.fill_candidates(final), due=tracker.fill_due, second=False)

    async def _fill2(self, final: bool):
        """Second text fill: sentences still unknown are asked again, blind (the first answer is not shown) and with the neighbours'
        text-filled labels in view; the sentence takes the role it names, whatever the first answer or the voice said. The final pass takes whatever is
        left, however few."""
        tracker = self.tracker
        await self._fill_stage(final, wanted=tracker.fill2_candidates(), due=tracker.fill2_due, second=True)

    async def _fill_stage(self, final: bool, *, wanted: list[int], due, second: bool):
        tracker = self.tracker
        if not wanted or not (final or due(wanted)):
            return
        agent, stats = ('speaker_fill2', self.fill2_stats) if second else ('speaker_fill', self.fill_stats)
        mark = tracker.mark_asked2 if second else tracker.mark_asked
        apply = tracker.apply_fill2 if second else tracker.apply_fill
        base = self._prompt('fill2' if second else 'fill', fill.load_prompt2 if second else fill.load_prompt)
        for ask in fill.chunks(wanted, self._fill_chunk):
            if self.failed:
                return
            mark(ask)                                 # asked once: a failed call leaves its sentences unknown rather than retrying forever
            shown = fill.window(ask, tracker.stats()['decided'], self._fill_chunk, self._fill_before, self._fill_after)
            rows, previous_end = [], None
            for i in shown:
                sentence = tracker.sentences[i]
                if sentence.background:                   # somebody else's voice: not part of the dialogue the model reads
                    continue
                if i > 0 and previous_end is None:
                    previous_end = tracker.sentences[i - 1].t1
                if second:                            # every label is shown; the ones read from the text carry a star
                    label = tracker.role_of(sentence.gid) if sentence.gid is not None else UNKNOWN
                    inferred = sentence.source in TEXT_SOURCES
                else:
                    label = tracker.role_of(sentence.gid) if sentence.source == 'voice' else UNKNOWN
                    inferred = False
                rows.append(fill.row(i, sentence.t0, previous_end, label, self._sentence_text(i), inferred))
                previous_end = sentence.t1
            user = fill.request_payload(ask, rows)

            def build(last_error, user=user):
                return [{'role': 'system', 'content': base + (fill.RETRY_NOTE if last_error else '')},
                        {'role': 'user', 'content': user}]

            try:
                outcome = await self.caller.call(agent=agent, endpoint=self._endpoint(), messages=build,
                                                 priority=PRIORITY_BACKGROUND, retries=1,
                                                 max_tokens=min(3000, 80 * len(ask) + 300),
                                                 validator=lambda text, ask=ask: fill.parse_answers(text, ask))
            except asyncio.CancelledError:
                mark(ask, False)                      # cancelled (the visit is ending): the final pass asks these again
                raise
            except CallFailed as exc:
                stats['failed_calls'] += 1
                self.emit(f'{agent}_failed', error=exc.last_error, sentences=len(ask))
                continue
            taken = [i for i, role in outcome.value.items() if apply(i, role)]
            accepted = len(taken)
            stats['asked'] += len(ask)
            stats['accepted'] += accepted
            stats['rejected'] += len(outcome.value) - accepted
            extra = {}
            if second:                                # the second answer equal to the first one, or not: counted apart, for checking later
                differing = sum(1 for i in taken if tracker.sentences[i].said != outcome.value[i])
                stats['accepted_differing'] += differing
                stats['accepted_agreeing'] += accepted - differing
                extra = {'agreeing': accepted - differing, 'differing': differing}
            self.emit(agent, asked=len(ask), answered=len(outcome.value), accepted=accepted, **extra)
            if accepted:
                self._on_change()

    # ------------------------------------------------------------------ the physician
    def set_group_roles(self, mapping: dict[int, str]):
        """The physician names the role of each voice group. It stands over the model until released."""
        valid = {g: r for g, r in mapping.items() if 0 <= g < self.tracker.group_count and r in (DOCTOR, OTHER)}
        if not valid:
            return
        if self.tracker.group_count == 2 and len(valid) == 1:          # with two voices, naming one names the other
            (chosen, role), = valid.items()
            valid[1 - chosen] = OTHER if role == DOCTOR else DOCTOR
        self.voter.set_manual({**self.voter.current, **valid})
        self.manual_events.append({'t': self._audio_t(), 'event': 'roles', 'roles': {group_name(g): r for g, r in valid.items()}})
        self._apply_roles('manual')

    def set_lock(self, locked: bool):
        """Lock = keep the current map and stop asking the model; unlock = let it be asked again."""
        if locked:
            self.voter.set_manual(dict(self.voter.current))
        else:
            self.voter.release()
            self._wake.set()
        self._sync_confirmed()
        self.manual_events.append({'t': self._audio_t(), 'event': 'lock' if locked else 'unlock'})
        self.emit('speaker_lock', locked=locked)
        self._on_change()

    @property
    def locked(self) -> bool:
        return self.voter.manual

    # ------------------------------------------------------------------ the end of the visit
    async def finish(self):
        await self.stop()
        if self.failed or self.frozen or self._finished:                 # a retried "end and save" comes here again: nothing more to do
            return
        self._finished = True
        try:
            changed = await asyncio.to_thread(self.tracker.finish)
            self._after_tracker(changed)
            await asyncio.wait_for(self.settle(final=True), FINAL_PASS_SECONDS)
        except asyncio.TimeoutError:
            self.emit('speaker_error', where='final_pass', error=f'最後一輪角色判定與補標超過 {FINAL_PASS_SECONDS:g} 秒，已略過。')
        except Exception as exc:
            self._fail(exc, 'finish')
        self.emit('speaker_finished', **self.tracker.stats(), fill=self.fill_stats, fill2=self.fill2_stats)
