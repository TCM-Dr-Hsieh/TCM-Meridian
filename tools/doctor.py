"""Environment health check for TCM-Meridian: Python, packages, project files, config.json, data folders, port, and the
voice feature (speech-recognition environment, models, voiceprint model, microphone).

    .venv/Scripts/python.exe tools/doctor.py [--voice] [--smoke] [--verify-models]

Exit code 1 when something required is broken (FAIL). WARN items do not block using the app.
The voice feature is optional: without its models it is reported as INFO and everything else works. With --voice (setup.ps1 passes it
when you chose to install the voice feature) missing models or an unusable GPU are failures. --smoke loads the speech models and
recognises a few seconds of silence; --verify-models compares every model file's content hash (reads ~6 GB).
This check never calls any LLM / embedding server and never prints API keys.
"""
import argparse
import asyncio
import importlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

ASR_PROBE = r'''
import json, sys
out = {"python": sys.version.split()[0]}
try:
    import torch
    out.update(torch=torch.__version__, cuda_build=torch.version.cuda, cuda=bool(torch.cuda.is_available()))
    if torch.cuda.is_available():
        out.update(gpu=torch.cuda.get_device_name(0), vram_gb=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
except Exception as exc:
    out["torch_error"] = repr(exc)
try:
    import qwen_asr, transformers
    out.update(qwen_asr="ok", transformers=transformers.__version__)
except Exception as exc:
    out["qwen_error"] = repr(exc)
print(json.dumps(out))
'''

results: list[tuple[str, str, str]] = []

REQUIRED_FILES = (
    'TCM_Meridian_main.py', 'Main_Agent.py', 'Professor.py', 'Record_Template.txt',
    'prompt_main_agent.txt', 'prompt_record_update.txt', 'prompt_hallucination_check.txt',
    'prompt_information_collection_subagent.txt', 'prompt_low_confidence_check.txt', 'prompt_note_review.txt',
    'professor-Template/Description.txt', 'professor-Template/prompt_system.txt',
    'professor-Template/prompt_3_prefix.txt', 'professor-Template/prompt_rerank.txt',
)

# (顯示名稱, config.json 內的路徑)
ENDPOINTS = (
    ('主 Agent', ('main_agent',)),
    ('歷史摘要', ('main_agent', 'history_summary')),
    ('摘要並退出', ('main_agent', 'summary_exit')),
    ('病歷登載', ('record_subagent',)),
    ('幻覺審查', ('hallucination_subagent',)),
    ('問診助理', ('ic_subagent',)),
    ('低信心標註', ('lc_subagent',)),
    ('病歷檢查員', ('nr_subagent',)),
    ('教授回答', ('professor_config', 'answer')),
    ('教授 embedding', ('professor_config', 'embedding')),
    ('教授前綴分類', ('professor_config', 'prefix')),
    ('教授 rerank', ('professor_config', 'rerank')),
)
PLACEHOLDER_KEYS = {'', 'KEY'}


def report(status: str, name: str, detail: str = ''):
    results.append((status, name, detail))
    mark = {'OK': '[ OK ]', 'INFO': '[INFO]', 'WARN': '[WARN]', 'FAIL': '[FAIL]'}[status]
    print(f'{mark} {name}' + (f' — {detail}' if detail else ''), flush=True)


def check_python():
    inside = Path(sys.prefix).resolve() == (ROOT / '.venv').resolve()
    report('OK' if inside else 'WARN', f'Python {sys.version.split()[0]}',
           str(sys.executable) if inside else f'不是專案的 .venv（{sys.prefix}）；建議用 .venv\\Scripts\\python.exe 執行')
    if sys.version_info[:2] != (3, 12):
        report('WARN', 'Python 版本', '建議使用 3.12（套件鎖定檔以 3.12 為準）')


def check_packages():
    for module in ('nicegui', 'openai', 'chromadb', 'numpy', 'requests', 'langchain', 'langchain_community',
                   'soundcard', 'av', 'opencc', 'onnxruntime', 'kaldi_native_fbank'):
        try:
            imported = importlib.import_module(module)
            report('OK', f'套件 {module}', getattr(imported, '__version__', ''))
        except Exception as exc:
            report('FAIL', f'套件 {module}', f'{type(exc).__name__}: {exc}')
    for label, statement in (('Chroma 向量庫', 'from langchain_community.vectorstores import Chroma'),
                             ('LangChain Document', 'from langchain.docstore.document import Document')):
        try:
            exec(statement, {})
            report('OK', label)
        except Exception as exc:
            report('FAIL', label, f'{type(exc).__name__}: {exc}')
    for module in ('Main_Agent', 'Professor', 'voice.service'):
        try:
            importlib.import_module(module)
            report('OK', f'模組 {module}')
        except Exception as exc:
            report('FAIL', f'模組 {module}', f'{type(exc).__name__}: {exc}')


def check_project_files():
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
    if missing:
        report('FAIL', '專案檔案', '缺少：' + '、'.join(missing))
    else:
        report('OK', '專案檔案', f'{len(REQUIRED_FILES)} 個必要檔案齊全')


def _dig(data: dict, path: tuple):
    for key in path:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data if isinstance(data, dict) else None


def check_config():
    path = ROOT / 'config.json'
    if not path.is_file():
        report('WARN', 'config.json', '不存在，程式會使用內建預設值（LM Studio localhost）；可由 config.example.json 複製')
        return
    if path.read_bytes().startswith(b'\xef\xbb\xbf'):
        # The app reads config.json with plain utf-8, so a BOM makes it silently fall back to the built-in defaults.
        report('FAIL', 'config.json', '檔案開頭有 UTF-8 BOM（常見於用記事本以「含 BOM」存檔），程式讀不了而會改用內建預設值；'
                                      '請改存為「UTF-8（無 BOM）」')
        return
    try:
        config = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        report('FAIL', 'config.json', f'無法解析：{exc}（程式會回落到預設值，請修正或刪除後重新複製 config.example.json）')
        return
    if not isinstance(config, dict):
        report('FAIL', 'config.json', '內容不是 JSON 物件')
        return
    report('OK', 'config.json', '格式正確')
    by_problem: dict[str, list[str]] = {}
    for label, keys in ENDPOINTS:
        section = _dig(config, keys)
        if section is None:
            by_problem.setdefault('缺少設定', []).append(label)
            continue
        if not str(section.get('model_name', '')).strip():
            by_problem.setdefault('模型名稱未填', []).append(label)
        if str(section.get('api_key', '')).strip() in PLACEHOLDER_KEYS:
            by_problem.setdefault('API 金鑰未填（仍是佔位符）', []).append(label)
    if by_problem:
        parts = [f'{problem}：' + ('全部 %d 個' % len(labels) if len(labels) == len(ENDPOINTS) else '、'.join(labels))
                 for problem, labels in by_problem.items()]
        report('WARN', '模型設定尚待填寫', '；'.join(parts) + '。啟動後到「模型設定」「教授設定」分頁填寫')
    else:
        report('OK', '模型設定', '各 Agent 的模型與金鑰都已填寫（未實際連線測試）')


def check_data_dirs():
    data_root = ROOT / 'patient_data'
    try:
        data_root.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=data_root, prefix='.doctor-', delete=True):
            pass
        report('OK', 'patient_data', '可寫入')
    except Exception as exc:
        report('FAIL', 'patient_data', f'無法寫入：{exc}')


def check_professors():
    folders = sorted(p for p in ROOT.glob('professor_*') if p.is_dir())
    if not folders:
        report('INFO', '教授', '沒有任何教授資料夾（可在「教授設定」分頁新增）')
        return
    for folder in folders:
        built = (folder / 'chroma_doc_index').is_dir() and (folder / 'parent_map.jsonl').is_file()
        docs = len(list((folder / 'doc').glob('*.txt'))) if (folder / 'doc').is_dir() else 0
        if built:
            report('OK', folder.name, f'知識庫文件 {docs} 個，向量索引已建立')
        else:
            report('INFO', folder.name, f'知識庫文件 {docs} 個，尚未建立向量索引'
                                        '（設好 embedding 模型後，在「教授設定」分頁按「建立資料庫」）')


def _voice_settings():
    """The voice settings of config.json (defaults when the file or the section is missing); None when they are invalid."""
    from voice.config import VoiceSettings
    path = ROOT / 'config.json'
    config = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding='utf-8-sig'))
            config = loaded if isinstance(loaded, dict) else {}
        except ValueError:
            config = {}
    try:
        return VoiceSettings.from_config(config)
    except ValueError as exc:
        report('FAIL', '語音設定', f'config.json 的 voice_config 有誤：{exc}')
        return None


def check_voice_environment(settings, expected: bool):
    """The speech-recognition environment (.venv-asr). Always installed by setup.ps1; without models it is not used."""
    from voice.asr import worker_python
    python = worker_python(settings.asr)
    miss = 'FAIL' if expected else 'WARN'
    if not python.is_file():
        report(miss, '語音辨識環境', f'找不到 {python}；請執行 install.cmd（語音功能無法使用，其他功能不受影響）')
        return False
    try:
        completed = subprocess.run([str(python), '-c', ASR_PROBE], capture_output=True, text=True, timeout=180)
        info = json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception as exc:
        report(miss, '語音辨識環境', f'無法執行 {python}：{exc}')
        return False
    report('OK', f'語音辨識 Python {info.get("python")}', str(python))
    ok = True
    if 'torch_error' in info:
        report(miss, 'PyTorch', info['torch_error'])
        ok = False
    else:
        report('OK', 'PyTorch', f'{info["torch"]}（CUDA 建置 {info["cuda_build"]}）')
        if info.get('cuda'):
            report('OK', 'GPU', f'{info["gpu"]}，顯存 {info["vram_gb"]} GB')
        else:
            report(miss if expected else 'INFO', 'GPU', 'CUDA 不可用：語音辨識需要 NVIDIA 顯卡與驅動（建議 ≥ 570）；CPU 太慢，不建議使用')
            ok = ok and not expected
    if 'qwen_error' in info:
        report(miss, 'qwen-asr', info['qwen_error'])
        ok = False
    else:
        report('OK', 'qwen-asr', f'transformers {info.get("transformers")}')
    return ok


def check_voice_models(settings, expected: bool, deep: bool) -> bool:
    """Whether the two speech models are in place. Not installed = INFO (the feature is off), unless it was asked for."""
    from voice.asr import ASRError, validate_model_dir
    import modelcheck
    present = True
    for label, value, aligner, repo in (('ASR 模型', settings.asr.asr_model_dir, False, modelcheck.REPOS['asr']),
                                        ('ForcedAligner 模型', settings.asr.aligner_model_dir, True, modelcheck.REPOS['aligner'])):
        try:
            folder = validate_model_dir(value, aligner=aligner)
        except ASRError as exc:
            present = False
            report('FAIL' if expected else 'INFO', label,
                   f'{exc}（未安裝語音模型時，「語音功能」按鈕會顯示「目前無下載語音模型」，其他功能不受影響；'
                   '要安裝請重新執行 install.cmd 並選擇安裝語音功能）')
            continue
        drift = modelcheck.problems(folder, repo, deep=deep)
        if drift:                                    # a different version may still work; it just was not verified
            report('WARN', label, f'{folder} 與本程式驗證的固定版本不同：{modelcheck.summarize(drift)}'
                                  '（可能是其他版本，相容性與辨識品質未經驗證）')
        else:
            report('OK', label, f'{folder}（與固定版本相符{"，已核對全部檔案的內容雜湊" if deep else ""}）')
    return present


def check_speaker(settings, deep: bool):
    """The voiceprint model of the speaker marking: packages, the model file, its version, and one real run."""
    from voice.config import resolve_path
    from voice.speaker.embedder import DIMENSION, OnnxEmbedder, availability
    import modelcheck
    path = resolve_path(settings.speaker.model_path)
    on = settings.speaker.enabled
    reason = availability(path)
    if reason:                                   # a recording goes on without marking when it cannot run
        report('WARN' if on else 'INFO', '說話者標記', reason if on else f'未啟用（{reason}）')
        return
    drift = modelcheck.problems(path.parent, modelcheck.REPOS['speaker'], deep=deep)
    try:
        import numpy as np
        vector = OnnxEmbedder(path).embed(np.random.default_rng(0).normal(0, 0.1, 12800).astype(np.float32))
        assert vector is not None and vector.shape == (DIMENSION,)
    except Exception as exc:
        report('WARN' if on else 'INFO', '說話者聲紋模型', f'{path} 無法執行：{type(exc).__name__}: {exc}')
        return
    state = '已啟用' if on else '未啟用（模型設定 → 說話者標記）'
    if drift:
        report('WARN', '說話者聲紋模型', f'{path} 與本程式驗證的固定版本不同：{modelcheck.summarize(drift)}；{state}')
    else:
        report('OK', '說話者聲紋模型', f'{path}（與固定版本相符{"，已核對內容雜湊" if deep else ""}，實際跑通一次）；{state}')


def check_microphone():
    try:
        from voice.audio import list_microphones
        devices = list_microphones()
        if devices:
            report('OK', '麥克風', '、'.join(devices.values())[:120])
        else:
            report('INFO', '麥克風', '找不到錄音裝置；可改用「此瀏覽器麥克風」，或檢查 Windows 隱私權設定與裝置')
    except Exception as exc:
        report('INFO', '麥克風', f'無法列出裝置：{exc}')


async def smoke(settings):
    """Load the worker and recognise a few seconds of silence: proves the GPU path end to end."""
    import numpy as np
    from voice.asr import LocalASR
    asr = LocalASR()
    try:
        started = time.monotonic()
        info = await asr.load(settings.asr)
        loaded = time.monotonic() - started
        started = time.monotonic()
        await asr.recognize(settings.asr, np.zeros(16000 * 4, dtype=np.float32))
        report('OK', '語音辨識實測', f'裝置 {info.get("device")}；載入 {loaded:.0f} 秒，4 秒音訊辨識 {time.monotonic() - started:.1f} 秒')
    except Exception as exc:
        report('FAIL', '語音辨識實測', f'{type(exc).__name__}: {exc}')
    finally:
        await asr.close()


def check_voice(args):
    settings = _voice_settings()
    if settings is None:
        return
    check_voice_environment(settings, args.voice)
    present = check_voice_models(settings, args.voice, args.verify_models)
    if present:
        check_speaker(settings, args.verify_models)
        check_microphone()
        if not settings.corrector.configured:
            report('INFO', '逐字稿校稿 LLM', '尚未設定（模型設定 → 逐字稿校稿）：逐字稿就是辨識原稿，說話者標記的 LLM 步驟也不會執行')
        if args.smoke and not any(s == 'FAIL' for s, _, _ in results):
            asyncio.run(smoke(settings))


def check_port():
    port = int(os.environ.get('TCM_PORT', '8080'))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        in_use = sock.connect_ex(('127.0.0.1', port)) == 0
    if in_use:
        report('WARN', f'埠 {port}', '已被占用（可能 TCM-Meridian 已在執行）；可用環境變數 TCM_PORT 改埠')
    else:
        report('OK', f'埠 {port}', '可使用')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--voice', action='store_true', help='the voice feature was asked for: missing models / GPU are failures')
    parser.add_argument('--smoke', action='store_true', help='also load the speech models and run one recognition')
    parser.add_argument('--verify-models', action='store_true',
                        help='also verify every model file content hash against the pinned revision (reads ~6 GB)')
    args = parser.parse_args()
    print(f'TCM-Meridian 健檢（專案：{ROOT}）')
    check_python()
    check_packages()
    check_project_files()
    check_config()
    check_data_dirs()
    check_professors()
    check_voice(args)
    check_port()
    fails = [r for r in results if r[0] == 'FAIL']
    warns = [r for r in results if r[0] == 'WARN']
    print()
    print(f'結果：{len(results) - len(fails) - len(warns)} 項正常/資訊，{len(warns)} 項注意，{len(fails)} 項失敗。')
    return 1 if fails else 0


if __name__ == '__main__':
    raise SystemExit(main())
