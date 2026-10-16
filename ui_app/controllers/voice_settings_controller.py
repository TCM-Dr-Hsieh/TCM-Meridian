from __future__ import annotations

import asyncio
import os
from copy import deepcopy
from typing import Any

from voice.asr import ASRError, validate_model_dir
from voice.config import (
    MAX_VOCABULARY_CHARS,
    ASRSettings,
    Endpoint,
    SpeakerSettings,
    VoiceSettings,
    resolve_path,
)
from voice.speaker import tuning
from voice.speaker.embedder import availability

CARD_STYLE = "border: 1px solid var(--border); border-radius: 12px;"
TITLE_STYLE = "font-weight: 700; font-size: 16px; color: var(--primary-dark);"


def _num(control: Any, default: float | int) -> float | int:
    raw = control.value
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    return raw


def tuning_summary(advanced: dict, unknown_percentile: float) -> str:
    changed = len(advanced) + (1 if unknown_percentile != tuning.DEFAULTS['unknown_percentile'] else 0)
    return f'目前有 {changed} 項參數不是預設值。' if changed else '目前全部使用預設值。'


def open_tuning_dialog(ui: Any, state: dict, pct_field: Any, note: Any):
    """The pop-up with every tunable number of speaker marking: value (default shown), what it does, what raising / lowering it does.

    `state['advanced']` holds the overrides being edited; "套用" checks the values the way saving does and writes them back (and
    `unknown_percentile` into its own field of the settings page), "取消" throws the changes away. Nothing is saved until the settings
    page's own 儲存設定."""
    current = tuning.effective(state['advanced'], pct_field.value if pct_field.value is not None else tuning.DEFAULTS['unknown_percentile'])
    inputs: dict = {}
    dialog = ui.dialog().props('persistent')
    with dialog, ui.card().classes('w-[1180px] max-w-full gap-2').style('max-height:92vh'):
        ui.label('說話者標記的進階參數').classes('text-xl font-semibold')
        ui.label('★ 為較重要的參數。每個欄位旁標示預設值（目前程式使用的值）；沒有改動的欄位不會寫進設定檔。多數參數互相牽連，建議一次只改一項，'
                 '改完用一場錄音驗證。預設值是在少數家人或朋友模擬看診的錄音上調的，樣本數非常少，僅供參考。').classes('hint')
        with ui.scroll_area().classes('w-full').style('height:62vh'):
            for stage in tuning.STAGES:
                knobs = [k for k in tuning.KNOBS if k.stage == stage]
                ui.label(stage).classes('text-base font-semibold mt-3')
                with ui.grid(columns='230px 150px 1.2fr 1fr 1fr').classes('w-full gap-x-3 gap-y-2 items-start'):
                    for head in ('參數', '數值（預設）', '功能', '調高', '調低'):
                        ui.label(head).classes('hint font-medium')
                    for knob in knobs:
                        ui.label(('★ ' if knob.star else '') + knob.label).classes('font-medium' + (' text-amber-800' if knob.star else ''))
                        with ui.column().classes('gap-0'):
                            inputs[knob.key] = ui.number(value=current[knob.key], min=knob.low, max=knob.high, step=knob.step, format='%g') \
                                .props('dense outlined').classes('w-32').mark(f'tuning-{knob.key}')
                            ui.label(f'預設 {knob.default:g}（{knob.low:g}–{knob.high:g}）').classes('hint')
                        ui.label(knob.what).classes('text-sm')
                        ui.label(knob.higher).classes('text-sm')
                        ui.label(knob.lower).classes('text-sm')
        error = ui.label('').classes('text-red-700 text-sm')

        def reset():
            for key, field in inputs.items():
                field.set_value(tuning.DEFAULTS[key])
            error.set_text('')

        def apply():
            try:
                missing = [tuning.BY_KEY[key].label for key, field in inputs.items() if field.value is None]
                if missing:
                    raise ValueError('這些欄位不可留空：' + '、'.join(missing))
                raw = {key: field.value for key, field in inputs.items() if key != 'unknown_percentile'}
                candidate = SpeakerSettings(unknown_percentile=inputs['unknown_percentile'].value, advanced=raw)
                candidate.validate()                                 # the same check the settings save makes
            except (ValueError, TypeError) as exc:
                error.set_text(str(exc))
                return
            state['advanced'] = candidate.advanced
            pct_field.set_value(candidate.unknown_percentile)
            note.set_text(tuning_summary(candidate.advanced, candidate.unknown_percentile))
            dialog.close()

        with ui.row().classes('w-full justify-between'):
            ui.button('全部還原成預設值', on_click=reset).props('flat').mark('tuning-reset')
            with ui.row().classes('gap-2'):
                ui.button('取消', on_click=dialog.close).props('flat').mark('tuning-cancel')
                ui.button('套用', on_click=apply).mark('tuning-apply')
    dialog.open()


class VoiceSettingsForm:
    """The 語音功能 part of the model settings page: speech recognition, the transcript-correction LLM, speaker marking."""

    def __init__(self, ui: Any, cfg: dict, manager: Any = None):
        self.ui = ui
        self.manager = manager
        draft = VoiceSettings.from_config(cfg) if not self._bad(cfg) else VoiceSettings()
        self.draft = draft
        self.tuning_state = {'advanced': dict(draft.speaker.advanced)}
        self.w: dict[str, Any] = {}
        self._build(draft)

    @staticmethod
    def _bad(cfg: dict) -> bool:
        try:
            VoiceSettings.from_config(cfg)
            return False
        except ValueError:
            return True

    # ------------------------------------------------------------------ build
    def _build(self, draft: VoiceSettings):
        ui, w = self.ui, self.w
        ui.label('🎙 語音功能').classes('section-title').style('margin-top: 8px;')
        ui.label('醫療系統主介面左欄的「語音功能」會用這裡的設定錄音並辨識。模型與環境由 install.cmd 安裝；'
                 '只能在沒有載入患者時儲存。').style('color: #666; font-size: 14px; margin-bottom: 8px;')

        # ---- speech recognition -----------------------------------------------------------
        with ui.card().classes('w-full q-pa-md q-mb-lg').style(CARD_STYLE):
            ui.label('🗣 語音辨識 (ASR)').style(TITLE_STYLE)
            w['models_state'] = ui.label('').style('font-size: 12px;')
            w['asr_dir'] = ui.input('ASR 模型資料夾（相對路徑以專案目錄為基準）', value=draft.asr.asr_model_dir).classes('w-full')
            w['aligner_dir'] = ui.input('ForcedAligner 模型資料夾', value=draft.asr.aligner_model_dir).classes('w-full')
            w['python_path'] = ui.input('ASR Python 路徑（留空＝專案內 .venv-asr）', value=draft.asr.python_path).classes('w-full')
            with ui.row().classes('w-full gap-4'):
                w['device'] = ui.select({'auto': '自動（優先 CUDA）', 'cuda': 'NVIDIA GPU（CUDA）', 'cpu': 'CPU'},
                                        value=draft.asr.device, label='運算裝置').classes('flex-1')
                w['mic'] = ui.select({'': '系統預設麥克風'}, value=draft.asr.microphone_id, label='麥克風（執行程式的這台電腦）') \
                    .classes('flex-1')
            with ui.row().classes('w-full gap-4'):
                w['window'] = ui.number('音訊視窗長度 L（秒）', value=draft.asr.window_seconds, min=5, max=30, step=1).classes('flex-1')
                w['overlap'] = ui.number('重疊 Z（秒）', value=draft.asr.overlap_seconds, min=0, max=27, step=0.5).classes('flex-1')
                w['context'] = ui.number('校稿前文長度（字）', value=draft.asr.context_chars, min=100, max=20000, step=100).classes('flex-1')
            ui.label('每隔 L−Z 秒產生一段辨識；Z ≤ L−3。').style('color: #999; font-size: 12px;')
            w['vocabulary'] = ui.textarea(f'ASR 專有詞（只給語音辨識當提示；最多 {MAX_VOCABULARY_CHARS:,} 字元）',
                                          value=draft.asr.vocabulary).classes('w-full')
            ui.label('校稿 LLM 專有詞只給「逐字稿校稿」的 LLM 當拼寫參考，不會送給語音辨識；校稿的每次 LLM 呼叫都會帶上它，'
                     '詞表越長越慢，建議只放真正容易被辨識錯的詞。').style('color: #999; font-size: 12px;')
            w['correction_vocabulary'] = ui.textarea(
                f'校稿 LLM 專有詞（只給逐字稿校稿的 LLM；最多 {MAX_VOCABULARY_CHARS:,} 字元）',
                value=draft.asr.correction_vocabulary).classes('w-full')
            w['asr_result'] = ui.label('').style('font-size: 13px;')
            if os.environ.get('TCM_FAKE_AUDIO'):
                ui.label(f'⚠ 開發模式：音訊來源為檔案 {os.environ["TCM_FAKE_AUDIO"]}（TCM_FAKE_AUDIO）').style(
                    'color: #9a4b00; font-size: 13px;')
            with ui.row().classes('gap-3'):
                ui.button('測試載入 ASR', on_click=self.test_load).props('outline no-caps')
                ui.button('重新整理麥克風清單', on_click=self.load_mics).props('outline no-caps')
            ui.timer(0.2, self.load_mics, once=True)

        # ---- transcript correction LLM --------------------------------------------------------------
        with ui.card().classes('w-full q-pa-md q-mb-lg').style(CARD_STYLE):
            ui.label('✍️ 逐字稿校稿 (Transcript Corrector)').style(TITLE_STYLE)
            ui.label('即時校正辨識出的字，並供說話者標記的角色判定與補標使用。Model Name 留空＝不校稿（逐字稿就是辨識原稿，說話者標記的 LLM 步驟也不會執行）。'
                     ).style('color: #999; font-size: 12px; margin-bottom: 8px;')
            c = draft.corrector
            w['c_url'] = ui.input(label='API URL', value=c.api_url).classes('w-full')
            w['c_key'] = ui.input(label='API Key', value=c.api_key).classes('w-full')
            w['c_model'] = ui.input(label='Model Name', value=c.model_name).classes('w-full')
            with ui.row().classes('w-full gap-4'):
                w['c_max_tokens'] = ui.number(label='Max Tokens（0＝依每次請求自動決定）', value=c.max_tokens, min=0, max=128000, step=100) \
                    .classes('flex-1')
                w['c_temperature'] = ui.number(label='Temperature', value=c.temperature, min=0.0, max=2.0, step=0.05, format='%.2f') \
                    .classes('flex-1')
                w['c_context'] = ui.number(label='context 上限（tokens，0＝不檢查）', value=c.context_tokens, min=0, step=1024) \
                    .classes('flex-1')
            with ui.row().classes('w-full gap-4'):
                w['timeout'] = ui.number(label='單次呼叫逾時（秒）', value=draft.timeout_seconds, min=5, max=1800).classes('flex-1')
                w['retries'] = ui.number(label='失敗重試次數', value=draft.retries, min=0, max=10, step=1).classes('flex-1')
            ui.label('每個辨識視窗會呼叫一次，與 AI 主治醫師各子 Agent 並行執行，請確認 LLM 伺服器可同時處理多個請求。'
                     ).style('color: #999; font-size: 12px;')

        # ---- speaker marking ----------------------------------------------------------------------------------
        with ui.card().classes('w-full q-pa-md q-mb-lg').style(CARD_STYLE):
            ui.label('👥 說話者標記 (Speaker Marking)').style(TITLE_STYLE)
            ui.label('把逐字稿標成「醫師：／患者或家屬：／不明：」。程式用聲音分群，再請「逐字稿校稿」接口的 LLM 判斷哪一群是醫師；'
                     '標記少部分可能有誤，聲音太相近或太短的句子會標「不明」，不會硬猜。錄音中不能更改這些設定。預設關閉。'
                     ).style('color: #999; font-size: 12px; margin-bottom: 8px;')
            s = draft.speaker
            w['speaker_on'] = ui.switch('啟用說話者標記', value=s.enabled)
            w['speaker_model'] = ui.input('聲紋模型檔（相對路徑以專案目錄為基準）', value=s.model_path).classes('w-full')
            w['speaker_jobs'] = ui.switch('把說話者標記一起給 AI 主治醫師與各子 Agent 的 prompt（判為背景人聲的句子不論此項都不會給）',
                                          value=s.use_in_jobs)
            w['speaker_fill'] = ui.switch('用 LLM 依上下文補標「不明」的句子（只在與聲音傾向一致時採用，標示 *）', value=s.text_fill)
            w['speaker_fill2'] = ui.switch('二階補標：對一階補標後仍不明的句子再問 LLM 一次（不告訴它第一次的答案，但讓它看到鄰句的 * 標記），'
                                           '它答出醫師或患者或家屬就採用，標示 *（需先開啟上一項）', value=s.text_fill2)
            w['speaker_fill2'].bind_enabled_from(w['speaker_fill'], 'value')
            w['speaker_pct'] = ui.number('「不明」百分位（0–50，預設 15；越大越多「不明」、越少標錯）', value=s.unknown_percentile,
                                         min=0, max=50, step=1).classes('w-96')
            w['speaker_result'] = ui.label('').style('font-size: 13px;')
            with ui.row().classes('items-center gap-3'):
                ui.button('進階參數…', icon='tune',
                          on_click=lambda: open_tuning_dialog(self.ui, self.tuning_state, w['speaker_pct'], w['tuning_note'])) \
                    .props('outline no-caps')
                w['tuning_note'] = ui.label(tuning_summary(s.advanced, s.unknown_percentile)).style('color: #999; font-size: 12px;')
                w['speaker_pct'].on_value_change(lambda e: w['tuning_note'].set_text(tuning_summary(
                    self.tuning_state['advanced'], e.value if e.value is not None else tuning.DEFAULTS['unknown_percentile'])))
            ui.button('檢查聲紋模型', on_click=self.check_speaker).props('outline no-caps')
            ui.label('聲紋只存在記憶體，錄音結束就消失；存檔只有角色對照表、各分鐘的統計與你的手動操作（不含聲紋）。'
                     ).style('color: #999; font-size: 12px;')
        self.refresh_models_state()

    # ------------------------------------------------------------------ actions
    def refresh_models_state(self):
        label = self.w['models_state']
        manager = self.manager
        problem = manager.models_problem() if manager is not None else ''
        if manager is None:
            label.set_text('')
        elif problem:
            label.set_text('⚠ ' + problem)
            label.style('color: #9a4b00;')
        else:
            label.set_text('✅ 語音模型與辨識環境都已就緒。')
            label.style('color: var(--primary);')

    async def load_mics(self):
        w = self.w
        try:
            from voice.audio import list_microphones
            options = await asyncio.to_thread(list_microphones)
            current = w['mic'].value
            w['mic'].set_options({'': '系統預設麥克風', **options}, value=current if current in options else '')
        except Exception as exc:
            w['asr_result'].set_text(f'無法列出麥克風：{exc}')

    async def test_load(self):
        w = self.w
        try:
            trial = deepcopy(self.draft.asr)
            trial.asr_model_dir, trial.aligner_model_dir = w['asr_dir'].value or '', w['aligner_dir'].value or ''
            trial.device, trial.python_path = w['device'].value, w['python_path'].value or ''
            validate_model_dir(trial.asr_model_dir)
            validate_model_dir(trial.aligner_model_dir, aligner=True)
            if self.manager is None:
                raise ASRError('語音服務尚未啟動。')
            w['asr_result'].set_text('正在載入本機模型…首次載入可能需要數十秒。')
            info = await self.manager.asr.load(trial)
            w['asr_result'].set_text(f'ASR 與對齊模型載入成功 · {info.get("device")}')
        except Exception as exc:
            w['asr_result'].set_text(f'載入失敗：{exc}')

    def check_speaker(self):
        w = self.w
        reason = availability(resolve_path(w['speaker_model'].value or ''))
        w['speaker_result'].set_text(reason or '聲紋模型與所需套件都可用。')

    # ------------------------------------------------------------------ read
    def read(self) -> dict:
        """The `voice_config` section for config.json, checked. Raises ValueError with a message for the physician."""
        w = self.w
        asr = ASRSettings(
            asr_model_dir=w['asr_dir'].value or '', aligner_model_dir=w['aligner_dir'].value or '',
            device=w['device'].value, window_seconds=_num(w['window'], 6.0), overlap_seconds=_num(w['overlap'], 3.0),
            context_chars=_num(w['context'], 1000), vocabulary=w['vocabulary'].value or '',
            correction_vocabulary=w['correction_vocabulary'].value or '', microphone_id=w['mic'].value or '',
            python_path=w['python_path'].value or '')
        speaker = SpeakerSettings(
            enabled=bool(w['speaker_on'].value), model_path=w['speaker_model'].value or '',
            use_in_jobs=bool(w['speaker_jobs'].value), text_fill=bool(w['speaker_fill'].value),
            text_fill2=bool(w['speaker_fill2'].value) and bool(w['speaker_fill'].value),
            unknown_percentile=_num(w['speaker_pct'], 15.0), advanced=dict(self.tuning_state['advanced']))
        corrector = Endpoint(api_url=w['c_url'].value or '', api_key=w['c_key'].value or '', model_name=w['c_model'].value or '',
                             temperature=_num(w['c_temperature'], 0.2), max_tokens=_num(w['c_max_tokens'], 0),
                             context_tokens=_num(w['c_context'], 0))
        settings = VoiceSettings(asr=asr, speaker=speaker, corrector=corrector,
                                 timeout_seconds=_num(w['timeout'], 180.0), retries=_num(w['retries'], 3))
        settings.validate()
        return settings.to_dict()
