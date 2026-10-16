"""Voice settings: dataclasses, defaults and validation (the `voice_config` section of config.json).

Ported from TCM-Meridian-mini (`mini/config.py`): the same ASR / speaker-marking numbers and checks, with one difference of
shape. In this project every agent keeps its own `{api_url, api_key, model_name, ...}` block in config.json; the transcript
corrector (and the LLM calls of the speaker marking) use the block `voice_config.corrector`, and an empty model name means
"do not correct" (the transcript is then the raw ASR text), like the other optional agents.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from urllib.parse import urlsplit

from .speaker.tuning import validate_advanced

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = Path(__file__).resolve().parent / 'prompts'

DEFAULT_URL = 'http://localhost:1234/v1'
DEFAULT_KEY = 'lm-studio'
MAX_VOCABULARY_CHARS = 25_000
DEFAULT_CONTEXT_TOKENS = 125_000
DEFAULT_ASR_DIR = r'models\Qwen3-ASR-1.7B'
DEFAULT_ALIGNER_DIR = r'models\Qwen3-ForcedAligner-0.6B'
DEFAULT_SPEAKER_MODEL = r'models\speaker\3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx'


def resolve_path(value) -> Path:
    """Expand ~ and resolve relative paths against the project root (never the process CWD)."""
    path = Path(str(value).strip().strip('"')).expanduser()
    return path if path.is_absolute() else ROOT / path


def _number(value, name, low=None, high=None, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} 須為數值。')
    if integer and float(value) != int(value):
        raise ValueError(f'{name} 須為整數。')
    if low is not None and value < low:
        raise ValueError(f'{name} 不可小於 {low:g}。')
    if high is not None and value > high:
        raise ValueError(f'{name} 不可大於 {high:g}。')
    return int(value) if integer else float(value)


@dataclass
class Endpoint:
    api_url: str = DEFAULT_URL
    api_key: str = DEFAULT_KEY
    model_name: str = ''
    temperature: float = 0.2
    max_tokens: int = 0                     # 0 = chosen per request
    # The model's context window in tokens (prompt + reply). Prompts that would not fit fail early with a clear message
    # instead of failing at the server. 0 = do not check.
    context_tokens: int = DEFAULT_CONTEXT_TOKENS

    @property
    def configured(self) -> bool:
        return bool(self.model_name.strip())

    def validate(self, label: str = 'LLM'):
        self.api_url = str(self.api_url).strip().rstrip('/，, ')
        self.model_name = str(self.model_name).strip()
        if not self.configured:
            return                          # an unused endpoint needs no address
        url = urlsplit(self.api_url)
        if url.scheme not in ('http', 'https') or not url.hostname or url.query or url.fragment:
            raise ValueError(f'{label}：API URL 必須是完整的 http(s) 網址，不含查詢或片段。')
        self.temperature = _number(self.temperature, f'{label} temperature', 0, 2)
        self.max_tokens = _number(self.max_tokens, f'{label} max_tokens', 0, 200_000, integer=True)
        self.context_tokens = _number(self.context_tokens, f'{label} context 上限', 0, 4_000_000, integer=True)


@dataclass
class ASRSettings:
    asr_model_dir: str = DEFAULT_ASR_DIR            # relative paths are resolved against the project folder
    aligner_model_dir: str = DEFAULT_ALIGNER_DIR
    device: str = 'auto'
    window_seconds: float = 6.0
    overlap_seconds: float = 3.0
    context_chars: int = 1000
    vocabulary: str = ''                  # hint for the speech recognizer only
    correction_vocabulary: str = ''       # spelling reference for the transcript-correction LLM only
    microphone_id: str = ''
    python_path: str = ''

    def validate(self):
        self.asr_model_dir = str(self.asr_model_dir).strip().strip('"')
        self.aligner_model_dir = str(self.aligner_model_dir).strip().strip('"')
        if not self.asr_model_dir:
            raise ValueError('ASR 模型資料夾不可留空。')
        if not self.aligner_model_dir:
            raise ValueError('ForcedAligner 模型資料夾不可留空。')
        if self.device not in ('auto', 'cuda', 'cpu'):
            raise ValueError('ASR 運算裝置須為 auto、cuda 或 cpu。')
        self.window_seconds = _number(self.window_seconds, '音訊視窗長度', 5, 30)
        self.overlap_seconds = _number(self.overlap_seconds, '重疊秒數', 0)
        if self.overlap_seconds > self.window_seconds - 3:
            raise ValueError('重疊須小於等於「視窗長度 − 3」，至少保留 3 秒新音訊。')
        self.context_chars = _number(self.context_chars, '校稿前文長度', 100, 20000, integer=True)
        if len(self.vocabulary) > MAX_VOCABULARY_CHARS:
            raise ValueError(f'ASR 專有詞最多 {MAX_VOCABULARY_CHARS} 字元。')
        if len(self.correction_vocabulary) > MAX_VOCABULARY_CHARS:
            raise ValueError(f'校稿 LLM 專有詞最多 {MAX_VOCABULARY_CHARS} 字元。')
        self.microphone_id = str(self.microphone_id or '')
        self.python_path = str(self.python_path).strip().strip('"')


@dataclass
class SpeakerSettings:
    """Speaker marking (SPEC 4.3): label the transcript 醫師 / 患者或家屬 / 不明 from the voices."""
    enabled: bool = False
    model_path: str = DEFAULT_SPEAKER_MODEL          # relative paths are resolved against the project folder
    use_in_jobs: bool = True                         # the agents' prompts get the speaker marks
    text_fill: bool = True                           # an LLM decides some of the sentences the voice could not
    text_fill2: bool = False                         # ... and a second, blind pass over what is still unknown: two equal answers decide
    unknown_percentile: float = 15.0                 # higher = more "不明", fewer mistakes
    advanced: dict = field(default_factory=dict)     # the other tunable numbers (`speaker/tuning.py`): only those that differ from the default

    def validate(self):
        self.enabled = bool(self.enabled)
        self.use_in_jobs = bool(self.use_in_jobs)
        self.text_fill = bool(self.text_fill)
        self.text_fill2 = bool(self.text_fill2) and self.text_fill         # the second pass cannot work without the first
        self.model_path = str(self.model_path).strip().strip('"')
        if not self.model_path:
            raise ValueError('說話者聲紋模型路徑不可留空。')
        self.unknown_percentile = _number(self.unknown_percentile, '「不明」百分位', 0, 50)
        self.advanced = validate_advanced(self.advanced, _number)


def default_corrector() -> Endpoint:
    return Endpoint(temperature=0.2, max_tokens=0)


@dataclass
class VoiceSettings:
    asr: ASRSettings = field(default_factory=ASRSettings)
    speaker: SpeakerSettings = field(default_factory=SpeakerSettings)
    corrector: Endpoint = field(default_factory=default_corrector)
    # LLM calls of the voice pipeline (the corrector and the speaker marking's role / text-fill calls)
    timeout_seconds: float = 180.0
    retries: int = 3

    def validate(self):
        self.asr.validate()
        self.speaker.validate()
        self.corrector.validate('逐字稿校稿')
        self.timeout_seconds = _number(self.timeout_seconds, 'LLM 逾時', 5, 1800)
        self.retries = _number(self.retries, 'LLM 重試次數', 0, 10, integer=True)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data) -> 'VoiceSettings':
        data = data if isinstance(data, dict) else {}

        def build(klass, raw):
            names = {item.name for item in fields(klass)}
            return klass(**{k: v for k, v in (raw if isinstance(raw, dict) else {}).items() if k in names})

        settings = cls()
        settings.asr = build(ASRSettings, data.get('asr'))
        settings.speaker = build(SpeakerSettings, data.get('speaker'))
        settings.corrector = build(Endpoint, data.get('corrector')) if isinstance(data.get('corrector'), dict) else default_corrector()
        for key in ('timeout_seconds', 'retries'):
            if key in data:
                setattr(settings, key, data[key])
        settings.validate()
        return settings

    @classmethod
    def from_config(cls, config: dict | None) -> 'VoiceSettings':
        """The settings in a loaded config.json (or the defaults when it has no `voice_config`)."""
        raw = (config or {}).get('voice_config') if isinstance(config, dict) else None
        return cls.from_dict(raw)

    def public_dict(self) -> dict:
        """A snapshot for audit files: never includes API keys."""
        data = self.to_dict()
        data['corrector'].pop('api_key', None)
        return data
