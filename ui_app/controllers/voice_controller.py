from __future__ import annotations

import asyncio
import html
import json
import time
from types import SimpleNamespace
from typing import Any

from voice.service import VoiceError, VoiceManager
from voice.speaker import BACKGROUND, DOCTOR, LABEL_NAMES, OTHER, TEXT_SOURCES, UNKNOWN
from voice.textutil import format_clock

BACKLOG_WARNING_WINDOWS = 6              # more windows waiting than this: say the transcript is behind
WAIT_FOR_BROWSER_SECONDS = 15            # a remote-microphone recording whose browser has not connected after this long is flagged
ROLE_OPTIONS = {DOCTOR: '醫師', OTHER: '患者或家屬'}      # a group nobody named shows no value; it cannot be set back to that


class VoiceController:
    """The 語音功能 pop-up: a floating window (no backdrop, draggable) over every page, and the button that opens it.

    The recording itself belongs to `VoiceManager` and runs in the background; this class only shows it. Closing the window
    does not stop anything, and the window can be opened again at any time to see what has been transcribed so far.
    """

    def __init__(self, *, ui: Any, app_state: dict[str, Any], manager: VoiceManager):
        self.ui = ui
        self.app_state = app_state
        self.manager = manager
        self.w = SimpleNamespace()
        self.button = None
        self.open_panel = False
        self._signature = None
        self._shown_token: str | None = None
        self._remote_started_at = 0.0
        app_state['_voice_manager'] = manager
        app_state['_voice_controller'] = self

    # ------------------------------------------------------------------ building
    def build(self):
        """Create the pop-up (hidden) and the timer that keeps it and the button current. Called once, outside any tab panel,
        so the window survives switching pages."""
        ui, w = self.ui, self.w
        with ui.element('div').classes('float-panel') as w.panel:
            w.panel.props('data-default-w=560 data-default-h=620')
            with ui.element('div').classes('float-head').tooltip('拖曳標題列移動；連按兩下回到預設位置；右下角可調整大小'):
                ui.label('🎙 語音功能').classes('float-title')
                w.state_chip = ui.label('').classes('hint')
                ui.element('div').style('flex:1 1 auto')
                ui.button(icon='close', on_click=self.close).props('dense flat round') \
                    .tooltip('關閉視窗（錄音與辨識會繼續在背景進行）').mark('voice-close')
            with ui.element('div').classes('voice-body'):
                with ui.row().classes('w-full items-center gap-2 no-wrap'):
                    w.btn_start = ui.button('開始錄製', icon='fiber_manual_record', color='red', on_click=self.start_local) \
                        .props('dense no-caps').mark('voice-start')
                    # remote microphone: the click itself must reach the browser's microphone prompt, so it is handled in the browser
                    w.btn_start_remote = ui.button('開始錄製', icon='fiber_manual_record', color='red') \
                        .props('dense no-caps').on('click', js_handler='() => window.tcmRemote.begin()').mark('voice-start-remote')
                    w.btn_stop = ui.button('停止錄製', icon='stop', color='grey-8', on_click=self.stop) \
                        .props('dense no-caps').mark('voice-stop')
                    ui.element('div').style('flex:1 1 auto')
                    w.btn_speakers = ui.button('說話者群', icon='groups', on_click=self.open_speaker_dialog) \
                        .props('dense flat no-caps').tooltip('啟用說話者標記且錄音中才能使用').mark('voice-speakers')
                with ui.row().classes('w-full items-center gap-2 no-wrap'):
                    w.source = ui.toggle({'local': '本機麥克風', 'remote': '此瀏覽器麥克風'}, value='local',
                                         on_change=self.on_source_change).props('dense no-caps color=primary').mark('voice-source')
                    w.mic_select = ui.select({'': '預設麥克風'}, value='', label='這台電腦的麥克風',
                                             on_change=self.on_mic_change).props('dense outlined options-dense') \
                        .classes('grow').mark('voice-mic')
                    w.btn_mic_refresh = ui.button(icon='refresh', on_click=self.refresh_mics).props('dense flat round') \
                        .tooltip('重新整理這台電腦的麥克風清單（會要求麥克風權限）').mark('voice-mic-refresh')
                with ui.row().classes('w-full items-center gap-2 no-wrap'):
                    w.level = ui.linear_progress(value=0, show_value=False).props('rounded size=6px color=primary').classes('grow')
                    w.info = ui.label('').classes('hint')
                w.status = ui.label('').classes('voice-status').mark('voice-status')
                with ui.element('div').classes('voice-scroll') as w.scroll:
                    w.html = ui.html('', sanitize=False)
        w.panel.set_visibility(False)
        ui.on('tcm_remote', self.on_remote_event)
        ui.timer(0.5, self.tick)
        self.render_transcript()
        self.update_controls()

    def bind_button(self, button):
        """The 語音功能 button of the main page's left column."""
        self.button = button
        button.on_click(self.toggle)
        self.update_button()

    # ------------------------------------------------------------------ session
    def current_target(self) -> tuple[str | None, str | None]:
        return self.app_state.get('selected_patient_folder'), self.app_state.get('selected_session_date')

    def sync_session(self) -> bool:
        """Make the manager follow the session this page has loaded. Returns whether the voice window has a session to work on.

        The manager is shared by every page (browser tab) of the program, so this never detaches it: a page that has no patient
        selected must not unload the session another page works on (two pages would undo each other every half second). With no
        selection here the manager is left as it is, and only a running recording can still be operated from this page, so a
        page loaded again in the middle of a recording can still stop it."""
        folder, date = self.current_target()
        if folder and date:
            try:
                self.manager.attach(folder, date)
            except VoiceError:
                pass                                # a recording is running: the pages cannot leave its session (the busy guard sees to it)
            return self.manager.session is not None
        return self.manager.busy

    @property
    def session(self):
        return self.manager.session

    # ------------------------------------------------------------------ open / close
    def toggle(self):
        if self.open_panel:
            self.close()
            return
        folder, date = self.current_target()
        if not (folder and date) and not self.manager.busy:
            self.ui.notify('請先在左欄選取並確認患者與就診日期。', type='warning')
            return
        problem = self.manager.models_problem()
        if problem:
            self.ui.notify(problem, type='negative', multi_line=True, close_button='關閉', timeout=0)
            return
        self.sync_session()
        self.open_panel = True
        self.w.panel.set_visibility(True)
        self.ui.run_javascript(f"window.tcmFloat.show('c{self.w.panel.id}')")
        self._signature = None
        self.update_controls()
        self.render_transcript()
        self.update_button()
        if not self.manager.asr_status:
            self.manager.schedule_preload()

    def close(self):
        self.open_panel = False
        self.w.panel.set_visibility(False)
        self.update_button()

    # ------------------------------------------------------------------ recording
    def _guard(self):
        guard = self.app_state.get('_session_busy_guard')
        if guard:
            guard.sync_navigation()

    async def start_local(self):
        if not self.sync_session():
            self.ui.notify('請先在左欄選取並確認患者與就診日期。', type='warning')
            return
        try:
            await self.manager.start()
        except VoiceError as exc:
            self.ui.notify(str(exc), type='negative', multi_line=True, close_button='關閉', timeout=0)
            self.update_controls()
            return
        self._guard()
        self.ui.notify('開始錄音（這台電腦的麥克風）。', type='positive')
        self.update_controls()

    async def stop(self):
        if not self.manager.busy:
            return
        self.ui.notify('正在停止錄音，並把已錄下的聲音辨識完…', type='info')
        self.update_controls()
        await self.manager.stop()
        self._guard()
        self.update_controls()
        self.render_transcript()
        self.ui.notify('錄音已停止，逐字稿與錄音檔已存檔。', type='positive')

    # -- remote microphone (this browser's microphone) -----------------------------------------
    @property
    def remote_mode(self) -> bool:
        return self.w.source.value == 'remote'

    async def on_source_change(self):
        if self.manager.busy:
            self.update_controls()
            return
        if self.remote_mode:
            secure = await self.ui.run_javascript('window.isSecureContext && !!navigator.mediaDevices', timeout=5)
            if not secure:
                self.ui.notify('這個網址不是 HTTPS，瀏覽器不會開放麥克風。請改用 https:// 網址（例如 Cloudflare），'
                               '或在執行程式的電腦上用 localhost 開啟；否則請改選「本機麥克風」。',
                               type='warning', multi_line=True, close_button='關閉', timeout=0)
        self.update_controls()

    def on_mic_change(self):
        self.ui.run_javascript(f'window.tcmRemote.deviceId = {json.dumps(self.w.mic_select.value or "")}')

    async def refresh_mics(self):
        try:
            devices = await self.ui.run_javascript('window.tcmRemote.listDevices(true)', timeout=60)
        except Exception as exc:                      # permission refused, no microphone, timeout
            self.ui.notify(f'無法取得這台電腦的麥克風清單：{exc}', type='warning', multi_line=True)
            return
        options = {'': '預設麥克風', **{d['id']: d['label'] for d in devices or []}}
        current = self.w.mic_select.value
        self.w.mic_select.set_options(options, value=current if current in options else '')
        self.ui.notify(f'找到 {len(options) - 1} 支麥克風。', type='positive' if len(options) > 1 else 'warning')

    async def on_remote_event(self, event):
        """Reports from window.tcmRemote (voice/web/remote_capture.js)."""
        args = event.args
        payload = args if isinstance(args, dict) else (args[0] if args else {})
        kind, message = payload.get('type'), payload.get('message', '')
        if kind == 'prepared':                        # the microphone is open: start the recording and hand over the token
            if not self.remote_mode:                  # switched back to the local microphone while the prompt was open
                self.ui.run_javascript('window.tcmRemote.abort()')
                self.ui.notify('已切回本機麥克風，取消這次遠端麥克風的錄音。', type='info')
                return
            await self.start_remote(str(payload.get('label') or ''))
        elif kind == 'error':
            self.ui.notify(message, type='negative', multi_line=True, close_button='關閉', timeout=0)
        elif kind == 'reconnecting':
            self.ui.notify('遠端麥克風連線暫時中斷，正在重新連線（30 秒內會自動補傳）…', type='warning')
        elif kind == 'reconnected':
            self.ui.notify('遠端麥克風已重新連線，中斷期間的聲音已補傳。', type='positive')
        elif kind == 'recording':
            self.ui.notify('遠端麥克風已連線，開始錄音。', type='positive')

    async def start_remote(self, device_label: str = ''):
        if not self.sync_session():
            self.ui.run_javascript('window.tcmRemote.abort()')
            self.ui.notify('請先在左欄選取並確認患者與就診日期。', type='warning')
            return
        try:
            await self.manager.start(remote=True, device_label=device_label)
        except VoiceError as exc:
            self.ui.run_javascript('window.tcmRemote.abort()')
            self.ui.notify(str(exc), type='negative', multi_line=True, close_button='關閉', timeout=0)
            self.update_controls()
            return
        except Exception as exc:
            self.ui.run_javascript('window.tcmRemote.abort()')
            self.ui.notify(f'無法開始錄音：{exc}', type='negative', multi_line=True)
            self.update_controls()
            return
        self._remote_started_at = time.monotonic()
        self.ui.run_javascript(f'window.tcmRemote.connect({json.dumps(self.manager.remote_token)})')
        self._guard()
        self.ui.notify('開始錄音（此瀏覽器的麥克風）。', type='positive')
        self.update_controls()

    # ------------------------------------------------------------------ speaker groups
    def open_speaker_dialog(self):
        manager, ui = self.manager, self.ui
        session = manager.session
        service = session.speaker if session is not None else None
        if service is None or service.frozen or not manager.busy:
            ui.notify('錄音中且啟用說話者標記時才能使用。', type='warning')
            return
        w = SimpleNamespace()
        rows: dict[int, SimpleNamespace] = {}
        shape: list = [None]
        programmatic = [False]                         # set while the window itself changes a widget (no click to answer)
        dialog = ui.dialog()
        with dialog, ui.card().classes('w-[760px] max-w-full gap-2').style('max-height:90vh; overflow:auto'):
            ui.label('說話者群').classes('text-xl font-semibold')
            ui.label('程式依聲音把說話的人分成幾群（S1、S2…），再請 LLM 判斷哪一群是醫師。判斷錯了可以在這裡改；'
                     '你改過之後以你的為準，逐字稿已標好的句子會跟著更新。患者與陪診家屬不必分開，都選「患者或家屬」。').classes('hint')
            w.status = ui.label('').classes('text-sm').mark('speaker-status')
            w.busy = ui.label('AI 主治醫師執行中，暫時不能修改對照（它用的是這一輪開始時的逐字稿）；執行結束後再改。') \
                .classes('text-sm text-amber-900').mark('speaker-busy')
            w.body = ui.element('div').classes('w-full gap-2')
            w.lock = ui.switch('鎖定目前的對照（不再請 LLM 重新判斷）', value=service.locked,
                               on_change=lambda e: toggle_lock(bool(e.value))).mark('speaker-lock')
            with ui.row().classes('w-full justify-end'):
                ui.button('關閉', on_click=dialog.close)

        def agent_busy() -> bool:
            return bool(self.app_state.get('ui_state', {}).get('agent_running'))

        def put_back(gid: int):
            programmatic[0] = True
            try:
                row = rows.get(gid)
                if row is not None:
                    role = service.tracker.role_of(gid)
                    row.select.set_value(role if role != UNKNOWN else None)
            finally:
                programmatic[0] = False

        def choose(gid: int, role: str):
            if programmatic[0] or role not in (DOCTOR, OTHER):
                return
            if agent_busy():
                ui.notify('AI 主治醫師執行中，暫時無法修改說話者對照。', type='warning')
                put_back(gid)
                return
            manager.set_speaker_roles({gid: role})

        def toggle_lock(locked: bool):
            if programmatic[0] or locked == service.locked:
                return
            if agent_busy():
                ui.notify('AI 主治醫師執行中，暫時無法變更鎖定。', type='warning')
                programmatic[0] = True
                try:
                    w.lock.set_value(service.locked)
                finally:
                    programmatic[0] = False
                return
            manager.set_speaker_lock(locked)

        def build(groups):
            rows.clear()
            w.body.clear()
            with w.body:
                if not groups:
                    ui.label('聲音群還沒有建立（錄音開始後約 1 分鐘，且兩個人都要有說話）。').classes('hint')
                for g in groups:
                    row = SimpleNamespace(examples=None, shown=None)
                    with ui.element('div').classes('w-full border rounded p-2 gap-1').mark(f'speaker-group-{g["gid"]}'):
                        with ui.row().classes('w-full items-center gap-3 no-wrap'):
                            ui.label(g['name']).classes('font-semibold')
                            row.select = ui.select(ROLE_OPTIONS, value=g['role'] or None, label='這一群是',
                                                   on_change=lambda e, gid=g['gid']: choose(gid, e.value)) \
                                .props('dense outlined').classes('w-44').mark(f'speaker-role-{g["gid"]}')
                            row.stats = ui.label('').classes('text-sm')
                        row.examples = ui.element('div').classes('w-full')
                    rows[g['gid']] = row

        def refresh():
            if not dialog.value or manager.session is not session or not manager.busy or session.speaker is not service:
                timer.cancel()
                if dialog.value:
                    dialog.close()
                return
            groups = service.groups()
            ids = tuple(g['gid'] for g in groups)
            if shape[0] != ids:
                build(groups)
                shape[0] = ids
            w.status.set_text(service.status)
            busy = agent_busy()
            w.busy.set_visibility(busy)
            w.lock.set_enabled(not busy)
            programmatic[0] = True
            try:
                if w.lock.value != service.locked:
                    w.lock.set_value(service.locked)
                for g in groups:
                    row = rows[g['gid']]
                    row.select.set_enabled(not busy)
                    row.stats.set_text(f'說話約 {g["seconds"]:.0f} 秒（佔 {g["share"] * 100:.0f}%），{g["units"]} 個片段')
                    if row.select.value != (g['role'] or None):
                        row.select.set_value(g['role'] or None)
                    shown = tuple(g['examples'])
                    if row.shown != shown:
                        row.shown = shown
                        row.examples.clear()
                        with row.examples:
                            for example in g['examples']:
                                ui.label('例：' + example).classes('text-sm hint')
            finally:
                programmatic[0] = False

        timer = ui.timer(0.5, refresh, immediate=False)
        dialog.open()
        refresh()

    # ------------------------------------------------------------------ periodic
    def tick(self):
        manager = self.manager
        if self.open_panel:
            self.sync_session()                     # only the page whose window is open drives the shared manager
        while manager.notices:
            level, message = manager.notices.pop(0)
            self.ui.notify(message, type={'warn': 'warning', 'error': 'negative'}.get(level, level))
        self.update_button()
        if not self.open_panel:
            return
        session = self.session
        pipeline = session.pipeline if session is not None else None
        signature = (id(session), session.state if session else None, manager.version,
                     pipeline.version if pipeline else -1, pipeline.finished if pipeline else None,
                     pipeline.pending if pipeline else 0, session.error if session else '',
                     manager.asr_status, self.manager.busy and int((session.audio_t() or 0)))
        if signature != self._signature:
            self._signature = signature
            self.render_transcript()
        self.update_controls()

    def update_button(self):
        button = self.button
        if button is None:
            return
        session = self.session
        state = session.state if session is not None else 'idle'
        if state == 'recording':
            text = f'⏺ 語音功能（錄音中 {format_clock(session.audio_t())}）'
            color = 'red'
        elif state in ('starting', 'stopping'):
            text = '⏳ 語音功能（處理中）'
            color = 'orange'
        else:
            text = '🎙 語音功能'
            color = 'blue-grey'
        if getattr(button, '_voice_text', None) != (text, color):
            button._voice_text = (text, color)
            button.set_text(text)
            button.props(f'color={color}')

    def update_controls(self):
        w, manager = self.w, self.manager
        session = self.session
        state = session.state if session is not None else 'idle'
        recording = state == 'recording'
        busy = state != 'idle'
        remote = self.remote_mode
        can_start = session is not None and not busy
        w.btn_start.set_visibility(not remote)
        w.btn_start.set_enabled(can_start)
        w.btn_start_remote.set_visibility(remote)
        w.btn_start_remote.set_enabled(can_start)
        w.btn_stop.set_enabled(recording)
        w.source.set_enabled(not busy)
        w.mic_select.set_visibility(remote)
        w.mic_select.set_enabled(not busy)
        w.btn_mic_refresh.set_visibility(remote)
        w.btn_mic_refresh.set_enabled(not busy)
        w.btn_speakers.set_enabled(bool(recording and session.speaker is not None and not session.speaker.frozen))
        if session is not None and session.source is not None:       # a running recording shows the source it really uses
            actual = 'remote' if getattr(session.source, 'kind', 'local') == 'remote' else 'local'
            if w.source.value != actual:
                w.source.set_value(actual)
        w.level.set_value(session.source.level if session is not None and session.source is not None and recording else 0)
        chip = {'idle': '已停止', 'starting': '啟動中…', 'recording': '錄音中', 'stopping': '收尾中…'}[state]
        if session is None:
            chip = '未載入 session'
        elif state == 'idle' and not session.has_recording:
            chip = '尚未錄音'
        w.state_chip.set_text(chip)
        w.info.set_text(self._info_text())
        text, warn = self._status_text()
        w.status.set_text(text)
        w.status.classes(add='warn' if warn else '', remove='' if warn else 'warn')

    def _info_text(self) -> str:
        session = self.session
        if session is None:
            return ''
        seconds = session.audio_t()
        clock = format_clock(seconds) if seconds else '00:00'
        return f'{session.segment_count} 段 · 錄音 {clock}'

    def _status_text(self) -> tuple[str, bool]:
        manager, session = self.manager, self.session
        if session is None:
            return '請先在左欄選取並確認患者與就診日期；語音錄音綁定在這個 session。', False
        parts: list[str] = []
        warn = False
        pipeline = session.pipeline
        if session.state == 'stopping':
            pending = pipeline.pending if pipeline else 0
            parts.append(f'收尾中：等待已錄下的聲音辨識完成（待處理 {pending}）並寫入檔案…')
        elif session.state == 'idle':
            parts.append('已停止。按「開始錄製」會接續錄在同一個錄音檔，逐字稿編號也會接下去。' if session.has_recording
                         else '按「開始錄製」開始。這個 session 的錄音與逐字稿會存在患者資料夾的 log 資料夾。')
        if pipeline is not None and session.state != 'idle':
            behind = pipeline.pending if not pipeline.finished else 0
            parts.append(f'逐字稿：{pipeline.status}' + (f'（待處理 {behind}）' if behind else ''))
            if behind >= BACKLOG_WARNING_WINDOWS:
                parts.append(f'⚠ 逐字稿已落後約 {behind * 3} 秒（錄音與音檔不受影響；停止時需等它處理完）')
                warn = True
        if session.speaker is not None and not session.speaker.frozen:
            parts.append(session.speaker.status)
        elif session.speaker_note:
            parts.append(f'⚠ {session.speaker_note}')
            warn = True
        source = session.source
        if source is not None and getattr(source, 'kind', '') == 'remote' and source.status_text:
            parts.append(f'遠端麥克風：{source.status_text}')
            if (source.status_text == '等待瀏覽器連線' and not source.done.is_set()
                    and time.monotonic() - source.created_at > WAIT_FOR_BROWSER_SECONDS):
                parts.append('⚠ 瀏覽器一直沒有連上遠端音訊（尚未錄到任何聲音）：請按「停止錄製」後改用本機麥克風，'
                             '或確認網址是 HTTPS、代理沒有改寫 Host 後重新開始')
                warn = True
        if session.error:
            parts.append(f'⚠ {session.error.splitlines()[0]}')
            warn = True
        if manager.asr_status:
            parts.append(f'ASR：{manager.asr_status}')
        return ' · '.join(parts), warn

    # ------------------------------------------------------------------ transcript
    def run_html(self, run) -> str:
        """One stretch of one speaker: `醫師：…`. A star marks a label read from the text, not heard (less reliable)."""
        if run.label == BACKGROUND:                     # grey, and not sent to the agents: the physician can still read it
            return (f'<span class="who background" title="疑似隔壁房間或環境的人聲，不會送進病歷書寫">{LABEL_NAMES[BACKGROUND]}：</span>'
                    f'<span class="bg-text">{html.escape(run.text)}</span>')
        guessed = run.source in TEXT_SOURCES and run.label != UNKNOWN
        name = LABEL_NAMES[run.label] + ('*' if guessed else '')
        title = ' title="由上下文推測，沒有聲音佐證，可信度較低"' if guessed else ''
        return f'<span class="who {run.label}"{title}>{name}：</span>{html.escape(run.text)}'

    def render_transcript(self):
        w = self.w
        session = self.session
        pipeline = session.pipeline if session is not None else None
        if pipeline is None or not pipeline.segments:
            w.html.set_content('<div class="hint" style="padding:8px">'
                               + ('（等待語音…）' if session is not None and session.state != 'idle'
                                  else '尚無逐字稿。按「開始錄製」後，語音片段與辨識文字會顯示在這裡。') + '</div>')
            if self._shown_token is not None:
                self._shown_token = None
                self.ui.run_javascript('window.tcmReplay && window.tcmReplay.close()')     # the replay window only exists for a transcript
            return
        if self._shown_token is not None and self._shown_token != session.replay_token:
            self.ui.run_javascript('window.tcmReplay && window.tcmReplay.close()')          # another session's transcript replaced it
        self._shown_token = session.replay_token
        unlocked = set(pipeline.unlocked_indexes())
        speaker = session.speaker if session.speaker is not None and session.speaker.marks_visible else None
        token = html.escape(session.replay_token)
        rows = []
        for seg in pipeline.segments:
            if not seg.visible:
                continue
            cls = 'seg gap' if seg.kind == 'gap' else ('seg unlocked' if seg.index in unlocked else 'seg')
            runs = speaker.runs(seg) if speaker is not None and seg.kind == 'speech' else None
            body = ' '.join(self.run_html(run) for run in runs) if runs else html.escape(seg.corrected)
            # `#n mm:ss` is the click target of the replay window (voice/web/replay.js); data-start is the exact start second,
            # data-token the session the row belongs to (the clip request carries it, see voice/replay.py).
            start = f'{seg.new_start:.2f}'
            rows.append(f'<div class="{cls}" data-idx="{seg.index}" data-start="{start}">'
                        f'<span class="seg-no" data-idx="{seg.index}" data-start="{start}" data-token="{token}" '
                        f'title="點擊回聽這一段（從這裡起 30 秒）">#{seg.index} {format_clock(seg.new_start)}</span>{body}</div>')
        w.html.set_content(''.join(rows) if rows else '<div class="hint" style="padding:8px">（等待語音…）</div>')
        self.ui.run_javascript(f"window.tcmScrollToEnd('c{w.scroll.id}')")
