"""Voiceprint model: 3D-Speaker CAM++ (Chinese + English, "advanced"), run by onnxruntime on the CPU.

Front end as validated in the experiments: 80-dim Kaldi fbank, no dither, per-utterance mean removal; 192-dim output.
Audio is float32 in [-1, 1] at 16 kHz (the same scale the experiments used).
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

RATE = 16000
MIN_SAMPLES = 1600           # 0.1 s: shorter audio gives too few frames
DIMENSION = 192


class SpeakerModelError(RuntimeError):
    pass


def availability(model_path: Path) -> str:
    """'' when the voiceprint model can run here, else the reason in plain words."""
    missing = []
    for module in ('onnxruntime', 'kaldi_native_fbank'):
        try:
            __import__(module)
        except Exception:
            missing.append(module)
    if missing:
        return f'缺少套件 {"、".join(missing)}（請重新執行 setup.ps1 安裝）。'
    if not model_path.is_file():
        return f'找不到說話者聲紋模型：{model_path}（請執行 tools\\download_models.py 下載）。'
    return ''


class OnnxEmbedder:
    """One voiceprint per audio snippet. Thread-safe; single-threaded inference (it must not compete with the ASR for the CPU)."""

    def __init__(self, model_path: Path):
        reason = availability(model_path)
        if reason:
            raise SpeakerModelError(reason)
        import kaldi_native_fbank as knf
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._knf = knf
        self._session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
        self._input = self._session.get_inputs()[0].name
        self._lock = threading.Lock()

    def embed(self, samples: np.ndarray) -> np.ndarray | None:
        """The voiceprint of this audio (192 floats), or None when it is too short to measure."""
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if len(samples) < MIN_SAMPLES:
            return None
        knf = self._knf
        options = knf.FbankOptions()
        options.frame_opts.dither = 0
        options.frame_opts.snip_edges = True
        options.frame_opts.samp_freq = RATE
        options.mel_opts.num_bins = 80
        fbank = knf.OnlineFbank(options)
        fbank.accept_waveform(RATE, samples.tolist())
        fbank.input_finished()
        if fbank.num_frames_ready < 10:
            return None
        frames = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
        features = (frames - frames.mean(axis=0, keepdims=True))[None].astype(np.float32)
        with self._lock:
            output = self._session.run(None, {self._input: features})[0]
        return np.asarray(output, dtype=np.float64).reshape(-1)


def load_embedder(model_path):
    """The voiceprint function for `model_path`; raises SpeakerModelError with the reason when it cannot run here."""
    return OnnxEmbedder(Path(model_path)).embed
