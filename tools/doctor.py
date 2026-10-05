"""Environment health check for TCM-Meridian: Python, packages, project files, config.json, data folders, port.

    .venv/Scripts/python.exe tools/doctor.py

Exit code 1 when something required is broken (FAIL). WARN items do not block using the app.
This check never calls any LLM / embedding server and never prints API keys.
"""
import importlib
import json
import os
import socket
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

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
    for module in ('nicegui', 'openai', 'chromadb', 'numpy', 'requests', 'langchain', 'langchain_community'):
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
    for module in ('Main_Agent', 'Professor'):
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
    print(f'TCM-Meridian 健檢（專案：{ROOT}）')
    check_python()
    check_packages()
    check_project_files()
    check_config()
    check_data_dirs()
    check_professors()
    check_port()
    fails = [r for r in results if r[0] == 'FAIL']
    warns = [r for r in results if r[0] == 'WARN']
    print()
    print(f'結果：{len(results) - len(fails) - len(warns)} 項正常/資訊，{len(warns)} 項注意，{len(fails)} 項失敗。')
    return 1 if fails else 0


if __name__ == '__main__':
    raise SystemExit(main())
