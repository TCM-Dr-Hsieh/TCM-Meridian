"""Compare a model folder with tools/model_manifest.json (the pinned Hugging Face revision).

Sizes are always checked (instant). Content hashes are checked on request (`deep=True`; reads ~6 GB, takes
seconds to minutes depending on the disk): SHA-256 for the Git LFS weights, the git blob id (SHA-1 of
"blob <size>\\0<content>", what Hugging Face lists for every non-LFS file) for everything else. With both, a deep
check covers every file in the manifest, so a match means the folder holds exactly the pinned revision's files.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST_PATH = Path(__file__).with_name('model_manifest.json')
REPOS = {'asr': 'Qwen/Qwen3-ASR-1.7B', 'aligner': 'Qwen/Qwen3-ForcedAligner-0.6B',
         'speaker': 'csukuangfj/speaker-embedding-models'}


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def sha256_of(path: Path, chunk: int = 8 * 2 ** 20) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path, chunk: int = 8 * 2 ** 20) -> str:
    """The git object id of a file's content, as `git hash-object` (and the Hugging Face API) report it."""
    digest = hashlib.sha1(b'blob %d\0' % path.stat().st_size)
    with path.open('rb') as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def problems(folder: Path, repo: str, *, deep: bool = False, manifest: dict | None = None) -> list[str]:
    """Differences between `folder` and the pinned revision of `repo`; empty when it matches.

    Extra files in the folder are ignored. An unknown repo or unreadable manifest is reported as a problem
    rather than raised, so callers can treat the result uniformly.
    """
    try:
        spec = (manifest if manifest is not None else load_manifest())['models'][repo]
    except (OSError, ValueError, KeyError) as exc:
        return [f'無法讀取 {repo} 的版本清單：{type(exc).__name__}']
    found: list[str] = []
    for name, info in spec['files'].items():
        path = folder / name
        if not path.is_file():
            found.append(f'缺少 {name}')
            continue
        size = path.stat().st_size
        if size != info['size']:
            found.append(f'{name} 大小不符（{size} ≠ {info["size"]}）')
        elif deep and info.get('sha256') and sha256_of(path) != info['sha256']:
            found.append(f'{name} 的 SHA-256 不符')
        elif deep and info.get('git_sha1') and git_blob_sha1(path) != info['git_sha1']:
            found.append(f'{name} 的內容雜湊（git）不符')
    return found


def summarize(items: list[str], limit: int = 3) -> str:
    shown = '；'.join(items[:limit])
    return shown + (f'…等共 {len(items)} 項' if len(items) > limit else '')
