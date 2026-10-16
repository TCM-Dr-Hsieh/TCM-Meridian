"""Async local Qwen3-ASR client: a private worker subprocess speaking JSON lines."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from .config import ROOT, ASRSettings, resolve_path

ASR_TIMEOUT = 120.0
LOAD_TIMEOUT = 300.0


class ASRError(RuntimeError):
    pass


def validate_model_dir(value: str, *, aligner: bool = False) -> Path:
    folder = resolve_path(value).resolve()
    role = 'ForcedAligner' if aligner else 'ASR'
    if not folder.is_dir():
        raise ASRError(f'{role} 模型資料夾不存在：{folder}')
    for name in ('config.json', 'preprocessor_config.json', 'tokenizer_config.json'):
        if not (folder / name).is_file():
            raise ASRError(f'{role} 模型資料夾缺少 {name}。請選擇完整的模型資料夾。')
    try:
        config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
        if config.get('model_type') != 'qwen3_asr':
            raise ASRError(f'{role} 模型資料夾不是 Qwen3-ASR Transformers 模型（需要 model_type=qwen3_asr）。')
        if aligner != ('timestamp_token_id' in config):
            raise ASRError('請選擇 ForcedAligner 模型。' if aligner else '這是 ForcedAligner，請選擇語音辨識模型。')
        index = folder / 'model.safetensors.index.json'
        weights = (set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values())
                   if index.is_file() else {'model.safetensors'})
        if not weights:
            raise ASRError(f'{role} 模型權重索引為空。')
        for name in weights:
            path = (folder / name).resolve()
            if not path.is_relative_to(folder) or not path.is_file() or path.stat().st_size == 0:
                raise ASRError(f'{role} 模型資料夾缺少或有不合法的權重：{name}')
        if not (folder / 'tokenizer.json').is_file() and not all((folder / n).is_file() for n in ('vocab.json', 'merges.txt')):
            raise ASRError(f'{role} 模型資料夾缺少 tokenizer.json 或 vocab.json／merges.txt。')
    except (ValueError, KeyError, TypeError) as exc:
        raise ASRError(f'{role} 模型設定或權重索引無法解析：{exc}') from exc
    return folder


def worker_python(settings: ASRSettings) -> Path:
    if settings.python_path:
        return resolve_path(settings.python_path)
    return ROOT / '.venv-asr' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


class LocalASR:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.process = None
        self.identity = None
        self.info: dict | None = None
        self.log_path = ROOT / 'data' / 'asr-worker.log'

    @property
    def ready(self) -> bool:
        return bool(self.process and self.process.returncode is None and self.info)

    async def _stop(self):
        process, self.process = self.process, None
        self.identity = self.info = None
        if process and process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()

    def _diagnostic(self) -> str:
        try:
            with self.log_path.open('rb') as stream:
                stream.seek(max(0, self.log_path.stat().st_size - 2500))
                return stream.read().decode('utf-8', errors='replace').strip()
        except OSError:
            return ''

    async def _read(self) -> dict:
        line = await self.process.stdout.readline()
        if not line:
            raise ASRError('本機 ASR 程序已結束。' + self._diagnostic())
        try:
            result = json.loads(line)
        except ValueError as exc:
            raise ASRError('本機 ASR 回應格式錯誤。' + self._diagnostic()) from exc
        if 'error' in result:
            raise ASRError(result['error'])
        return result

    async def _ensure(self, settings: ASRSettings) -> dict:
        folder = validate_model_dir(settings.asr_model_dir)
        aligner = validate_model_dir(settings.aligner_model_dir, aligner=True)
        python = worker_python(settings)
        identity = (str(folder), str(aligner), settings.device, str(python))
        if self.process and self.process.returncode is None and identity == self.identity:
            return self.info
        await self._stop()
        if not python.is_file():
            raise ASRError(f'找不到 ASR 執行環境：{python}。請執行 install.cmd（或 setup.ps1）建立 ASR 環境，或在模型設定指定 ASR Python 路徑。')
        self.log_path.parent.mkdir(exist_ok=True)
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                   PYTHONIOENCODING='utf-8', TOKENIZERS_PARALLELISM='false')
        with self.log_path.open('wb') as log:
            self.process = await asyncio.create_subprocess_exec(
                str(python), '-u', str(Path(__file__).with_name('asr_worker.py')),
                '--model-dir', str(folder), '--device', settings.device, '--aligner-dir', str(aligner),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log,
                cwd=str(ROOT), env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0, limit=1024 * 1024)
        try:
            info = await asyncio.wait_for(self._read(), timeout=LOAD_TIMEOUT)
            if info.get('type') != 'ready':
                raise ASRError('ASR 未回報載入完成。')
            self.info, self.identity = info, identity
            return info
        except BaseException:
            await self._stop()
            raise

    async def load(self, settings: ASRSettings) -> dict:
        async with self.lock:
            try:
                return await self._ensure(settings)
            except asyncio.TimeoutError as exc:
                raise ASRError(f'ASR 載入超過 {LOAD_TIMEOUT:g} 秒，請查看 data/asr-worker.log。') from exc

    async def recognize(self, settings: ASRSettings, samples) -> dict:
        """Return {'text', 'items', 'language'}; `items` carries ForcedAligner timestamps."""
        audio = np.asarray(samples, dtype='<f4').reshape(-1)
        if not len(audio):
            return {'text': '', 'items': [], 'language': ''}
        if not np.isfinite(audio).all():
            raise ASRError('音訊包含不合法數值。')
        async with self.lock:
            try:
                await self._ensure(settings)
                request = json.dumps({'audio': base64.b64encode(audio.tobytes()).decode('ascii'),
                                      'align': True, 'context': settings.vocabulary}) + '\n'
                self.process.stdin.write(request.encode('utf-8'))
                await self.process.stdin.drain()
                result = await asyncio.wait_for(self._read(), timeout=ASR_TIMEOUT)
                if not isinstance(result.get('text'), str):
                    raise ASRError('ASR 沒有回傳辨識文字。')
                result['text'] = result['text'].strip()
                if not isinstance(result.get('items'), list):
                    raise ASRError('ForcedAligner 沒有回傳時間資訊。')
                return result
            except BaseException as exc:
                await self._stop()
                if isinstance(exc, asyncio.TimeoutError):
                    raise ASRError('本機 ASR 辨識逾時，程序已停止；可重試該片段。') from exc
                raise

    async def close(self):
        async with self.lock:
            await self._stop()
