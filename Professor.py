"""
Professor.py - 中醫教授諮詢模組（RAG 管線）
所有教授共用此模組，各自載入不同的 doc 資料夾與 prompt 檔案。

功能：
1. ProfessorInstance — 代表一位教授的完整 RAG 實例
2. LMStudioEmbeddings — Embedding 呼叫
3. load_all_professors — 掃描所有教授資料夾
4. check_professor_files — 檢查檔案完整性
5. build_professor_index — 建立向量索引
"""
from __future__ import annotations
import gc
import os
import re
import json
import time
import glob
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from collections import Counter

import numpy as np
import requests
from openai import OpenAI
from multimodal_utils import inject_images_into_messages
from agent_behavior_log import append_behavior_event


# ════════════════════════════════════════════════════════════════
# RAG 管線參數（寫死，使用者須打開程式碼修改）
# ════════════════════════════════════════════════════════════════
RAG_TOPK_CAND = 250        # Dense argpartition 候選數
RAG_TOPK_PREFIX = 45       # 三前綴子集 RRF 後取 Top-K
RAG_TOPK_FULL = 30         # 全庫 RRF 後取 Top-K
RAG_RRF_K = 60             # RRF 融合常數
RAG_RERANK_FLOOR = 0.2     # Rerank 最低門檻
RAG_RERANK_AUTO = 0.7      # Rerank 自動選取門檻
RAG_MIN_PARENTS = 10       # 最少父段數（不足時 backfill）
RAG_MAX_PARENTS = 18       # 最多父段數
EMB_NORMALIZE = True       # 是否 L2 正規化 embedding

# 建立索引用的切塊參數
CHUNK_SIZE = 400
CHUNK_OVER = 120
CHUNK_SIZE_CASE = 280
CHUNK_OVER_CASE = 70

# 專案根目錄
_PROJECT_DIR = Path(__file__).parent


# ════════════════════════════════════════════════════════════════
# LMStudioEmbeddings
# ════════════════════════════════════════════════════════════════
class LMStudioEmbeddings:
    """LM Studio Embedding 呼叫封裝"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        batch_size: int = 32,
        timeout: int = 120,
        query_instruction: str = "Instruct: Given a user question, retrieve passages that directly answer it.\nQuery: ",
        embed_instruction: str = "",
        max_chars_per_input: int = 900,
        verbose: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.query_instruction = query_instruction
        self.embed_instruction = embed_instruction
        self.max_chars_per_input = max_chars_per_input
        self.verbose = verbose
        self._session = requests.Session()
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post_embeddings(self, strings: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(strings), self.batch_size):
            batch = strings[i : i + self.batch_size]
            body = {"model": self.model, "input": batch}
            last_err = None
            for attempt in range(3):
                if self.verbose:
                    print(f"[LMStudioEmb] POST /embeddings n={len(batch)} attempt={attempt+1}")
                r = self._session.post(
                    f"{self.base_url}/embeddings",
                    headers=self._headers,
                    json=body,
                    timeout=self.timeout,
                )
                if r.status_code == 200:
                    data = r.json()
                    embs = [it["embedding"] for it in sorted(data["data"], key=lambda x: x.get("index", 0))]
                    out.extend(embs)
                    break
                last_err = f"{r.status_code}: {r.text[:300]}"
                time.sleep(1.5 * (attempt + 1))
            else:
                raise RuntimeError(f"LM Studio embeddings error {last_err}")
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        payload = [self.embed_instruction + (t or "") for t in texts]
        if self.max_chars_per_input:
            payload = [s[: self.max_chars_per_input] for s in payload]
        return self._post_embeddings(payload)

    def embed_query(self, text: str) -> list[float]:
        s = self.query_instruction + (text or "")
        if self.max_chars_per_input:
            s = s[: self.max_chars_per_input]
        return self._post_embeddings([s])[0]


# ════════════════════════════════════════════════════════════════
# 前綴正規化
# ════════════════════════════════════════════════════════════════
_CANON = {
    "case": "case", "formula": "formula", "herb": "herb",
    "acupuncture": "acupuncture", "diagnoses": "diagnoses",
    "treatment": "treatment",
    "disease-internal": "disease-Internal",
    "disease-obstetrics&gynecology": "disease-Obstetrics&Gynecology",
    "disease-pediatrics": "disease-Pediatrics",
    "disease-osteology&traumatology": "disease-Osteology&Traumatology",
    "disease-surgery": "disease-Surgery",
    "disease-dermatology": "disease-Dermatology",
    "disease-eye&ent": "disease-Eye&ENT",
    "theory": "theory", "classic": "classic",
    "others": "others", "norag": "NoRAG",
}

_PREFIX_REGEX = re.compile(
    r"^(case|formula|herb|acupuncture|diagnoses|treatment|"
    r"disease-Internal|disease-Obstetrics&Gynecology|disease-Pediatrics|"
    r"disease-Osteology&Traumatology|disease-Surgery|disease-Dermatology|"
    r"disease-Eye&ENT|theory|classic|others|NoRAG)\b", re.I
)


def _to_canonical(prefix: str) -> Optional[str]:
    if not prefix:
        return None
    return _CANON.get(prefix.strip().lower(), None)


# ════════════════════════════════════════════════════════════════
# RRF 融合
# ════════════════════════════════════════════════════════════════
def _rrf_fusion_multi(rank_dicts: List[Dict[int, int]], k: int = RAG_RRF_K) -> List[int]:
    """多路 RRF 融合"""
    all_docs: set = set()
    for rd in rank_dicts:
        all_docs |= set(rd.keys())
    scores: Dict[int, float] = {}
    for doc in all_docs:
        s = 0.0
        for rd in rank_dicts:
            if doc in rd:
                s += 1 / (k + rd[doc])
        scores[doc] = s
    return [doc for doc, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


# ════════════════════════════════════════════════════════════════
# ProfessorInstance
# ════════════════════════════════════════════════════════════════
class ProfessorInstance:
    """一位教授的完整 RAG 實例"""

    def __init__(self, professor_id: str, config: dict):
        """
        Args:
            professor_id: 如 "professor_01"
            config: 共用模型設定，結構：
                {
                    "answer": {"api_url", "api_key", "model_name", "max_tokens", "temperature"},
                    "embedding": {"api_url", "api_key", "model_name"},
                    "query_expansion": {"api_url", "api_key", "model_name"},
                    "prefix": {"api_url", "api_key", "model_name"},
                    "rerank": {"api_url", "api_key", "model_name"},
                }
        """
        self.professor_id = professor_id
        self.config = config
        self.prof_dir = _PROJECT_DIR / professor_id

        # 載入描述
        self.name = ""
        self.description_text = ""
        desc_path = self.prof_dir / "Description.txt"
        if desc_path.exists():
            try:
                desc = json.loads(desc_path.read_text(encoding="utf-8"))
                self.name = desc.get("name", "")
                self.description_text = desc.get("description", "")
            except Exception:
                pass

        # 載入 prompt 檔案
        self.prompt_system = self._load_prompt("prompt_system.txt")
        # 將 Description.txt 的內容注入 prompt_system 的 {description} 佔位符
        if self.description_text:
            self.prompt_system = self.prompt_system.replace("{description}", self.description_text)
        else:
            self.prompt_system = self.prompt_system.replace("{description}", "")
        self.prompt_3_prefix = self._load_prompt("prompt_3_prefix.txt")
        self.prompt_query_expansion = self._load_prompt("prompt_query_expansion.txt")
        self.prompt_rerank = self._load_prompt("prompt_rerank.txt")

        # LLM clients（惰性建立）
        self._answer_client: Optional[OpenAI] = None
        self._expansion_client: Optional[OpenAI] = None
        self._prefix_client: Optional[OpenAI] = None
        self._rerank_client: Optional[OpenAI] = None

        # Embedding + 向量索引（惰性載入）
        self._embedder: Optional[LMStudioEmbeddings] = None
        self._all_emb: Optional[np.ndarray] = None
        self._texts: List[str] = []
        self._meta: List[dict] = []
        self._role_to_idxs: Dict[str, List[int]] = {}
        self._parent_dict: Dict[str, dict] = {}
        self._index_loaded = False

    def _load_prompt(self, filename: str) -> str:
        p = self.prof_dir / filename
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
        return ""

    # ── LLM Client 惰性建立 ─────────────────────────────────
    def _get_answer_client(self) -> OpenAI:
        if self._answer_client is None:
            cfg = self.config.get("answer", {})
            self._answer_client = OpenAI(
                api_key=cfg.get("api_key", "lm-studio"),
                base_url=cfg.get("api_url", "http://localhost:1234/v1"),
            )
        return self._answer_client

    def _get_expansion_client(self) -> OpenAI:
        if self._expansion_client is None:
            cfg = self.config.get("query_expansion", {})
            self._expansion_client = OpenAI(
                api_key=cfg.get("api_key", "lm-studio"),
                base_url=cfg.get("api_url", "http://localhost:1234/v1"),
            )
        return self._expansion_client

    def _get_prefix_client(self) -> OpenAI:
        if self._prefix_client is None:
            cfg = self.config.get("prefix", {})
            self._prefix_client = OpenAI(
                api_key=cfg.get("api_key", "lm-studio"),
                base_url=cfg.get("api_url", "http://localhost:1234/v1"),
            )
        return self._prefix_client

    def _get_rerank_client(self) -> OpenAI:
        if self._rerank_client is None:
            cfg = self.config.get("rerank", {})
            self._rerank_client = OpenAI(
                api_key=cfg.get("api_key", "lm-studio"),
                base_url=cfg.get("api_url", "http://localhost:1234/v1"),
            )
        return self._rerank_client

    # ── 向量索引惰性載入 ─────────────────────────────────────
    def _ensure_index(self):
        """載入 Chroma 索引與 parent_map（僅首次）"""
        if self._index_loaded:
            return

        from langchain_community.vectorstores import Chroma

        emb_cfg = self.config.get("embedding", {})
        self._embedder = LMStudioEmbeddings(
            base_url=emb_cfg.get("api_url", "http://localhost:1234/v1"),
            api_key=emb_cfg.get("api_key", "lm-studio"),
            model=emb_cfg.get("model_name", ""),
            batch_size=32,
            max_chars_per_input=900,
        )

        # 載入 parent_map
        pmap_path = self.prof_dir / "parent_map.jsonl"
        if pmap_path.exists():
            with open(pmap_path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    self._parent_dict[rec["parent_id"]] = rec

        # 載入 Chroma
        chroma_dir = str(self.prof_dir / "chroma_doc_index")
        if os.path.isdir(chroma_dir):
            vectordb = Chroma(
                persist_directory=chroma_dir,
                embedding_function=self._embedder,
                collection_name="doc_blocks",
            )
            res = vectordb._collection.get(include=["embeddings", "documents", "metadatas"])
            self._texts = res["documents"]
            self._meta = res["metadatas"]
            self._role_to_idxs.clear()
            for idx, m in enumerate(self._meta):
                self._role_to_idxs.setdefault(m.get("role", "others"), []).append(idx)
            if res["embeddings"] is not None:
                self._all_emb = np.asarray(res["embeddings"], dtype=np.float32)
                if EMB_NORMALIZE:
                    norms = np.linalg.norm(self._all_emb, axis=1, keepdims=True) + 1e-8
                    self._all_emb /= norms
                print(f"[Professor {self.professor_id}] 已載入 {self._all_emb.shape[0]} 個嵌入向量")
            else:
                print(f"[Professor {self.professor_id}] ⚠️ 索引中沒有 embeddings")
        else:
            print(f"[Professor {self.professor_id}] ⚠️ 找不到 chroma_doc_index，請先建立資料庫")

        self._index_loaded = True

    # ════════════════════════════════════════════════════════════
    # RAG 管線
    # ════════════════════════════════════════════════════════════
    def answer(
        self,
        question: str,
        note_content: str = "",
        at_content: str = "",
        last_visit_block: str = "",
        history_summary: str = "",
        forum_history_text: str = "",
        loaded_files_block: str = "",
        image_files: list | None = None,
        log_callback: Optional[Callable[[str], None]] = None,
        behavior_context: dict | None = None,
    ) -> dict[str, Any]:
        """
        教授回答一個臨床問題。

        Args:
            question: AI 主治醫師的提問
            note_content: 今日病歷 NOTE
            at_content: 辨證論治 A&T
            last_visit_block: 上次就診病歷
            history_summary: 歷史病歷摘要
            forum_history_text: 目前的醫療問答討論區內容
            loaded_files_block: 當輪讀取檔案暫存區內容
            log_callback: 選用，日誌回呼

        Returns:
            {"response": str, "q_expand": str, "prefixes": list, "retr_doc": str, "error": str|None}
        """

        def _log(msg: str):
            if log_callback:
                log_callback(msg)
            print(msg)

        _log(f"[Professor {self.professor_id}] 開始處理提問...")

        # ── Step 0: 確保索引已載入 ──
        self._ensure_index()

        if self._all_emb is None or len(self._texts) == 0:
            _log(f"[Professor {self.professor_id}] ⚠️ 向量索引未載入，跳過 RAG 檢索")
            retr_doc = ""
            q_expand = question
            prefixes = []
        else:
            # ── Step 1: 組裝兩個 Query ──
            q_txt_origin = f"【今日病歷】\n{note_content}\n\n【提問】\n{question}" if note_content else question
            q_expand_input = f"## 【今日病歷(或當前編輯頁面的病歷)】\n{note_content or '（空白）'}\n\n## 【醫療問答討論區】\n{forum_history_text or '（空白）'}\n\n## 【提問】\n{question}"

            # ── Step 2: Query Expansion ──
            q_expand = self._query_expansion(q_expand_input, _log, behavior_context)

            # ── Step 3: 三前綴分類（用 expanded query）──
            prefixes = self._classify_prefixes(q_expand, _log, behavior_context)

            # ── Step 4~6: 檢索 + Rerank ──
            retr_doc = self._retrieve(q_txt_origin, q_expand, prefixes, _log, behavior_context)

        # ── Step 7: 答覆生成 ──
        response = self._generate_answer(
            question, note_content, at_content,
            last_visit_block, history_summary,
            retr_doc, forum_history_text,
            loaded_files_block,
            image_files=image_files,
            _log=_log,
            behavior_context=behavior_context,
        )

        return {
            "response": response,
            "q_expand": q_expand,
            "prefixes": prefixes,
            "retr_doc": retr_doc,
            "error": None,
        }

    # ── Query Expansion ──────────────────────────────────────
    def _query_expansion(self, query_input: str, _log: Callable, behavior_context: dict | None = None) -> str:
        if not self.prompt_query_expansion:
            _log(f"[Professor {self.professor_id}] ⚠️ prompt_query_expansion.txt 不存在，跳過擴展")
            return query_input

        exp_cfg = self.config.get("query_expansion", {})
        model = exp_cfg.get("model_name", "")
        if not model:
            _log(f"[Professor {self.professor_id}] ⚠️ Query Expansion model 未設定，跳過擴展")
            return query_input

        try:
            client = self._get_expansion_client()
            if behavior_context:
                append_behavior_event(
                    behavior_context.get("folder_path"),
                    behavior_context.get("date_str"),
                    agent="professor_subagent",
                    event_type="llm_input",
                    label="QE輸入",
                    title=f"{self.professor_id} Query Expansion 輸入",
                    content=query_input.strip(),
                )
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.prompt_query_expansion},
                    {"role": "user", "content": query_input.strip()},
                ],
            )
            expanded = (resp.choices[0].message.content or "").strip()
            if not expanded:
                expanded = query_input
            _log(f"[Professor {self.professor_id}] Query Expansion 完成 (len={len(expanded)})")
            _log(f"[Professor {self.professor_id}] =====擴展查詢=====\n{expanded}\n==========")
            if behavior_context:
                append_behavior_event(
                    behavior_context.get("folder_path"),
                    behavior_context.get("date_str"),
                    agent="professor_subagent",
                    event_type="llm_output",
                    label="QE輸出",
                    title=f"{self.professor_id} Query Expansion 輸出",
                    content=expanded,
                )
            return expanded
        except Exception as e:
            _log(f"[Professor {self.professor_id}] Query Expansion 失敗: {e}")
            return query_input

    # ── 三前綴分類 ────────────────────────────────────────────
    def _classify_prefixes(self, q_expand: str, _log: Callable, behavior_context: dict | None = None) -> List[str]:
        if not self.prompt_3_prefix:
            _log(f"[Professor {self.professor_id}] ⚠️ prompt_3_prefix.txt 不存在")
            return ["others", "others", "others"]

        pfx_cfg = self.config.get("prefix", {})
        model = pfx_cfg.get("model_name", "")
        if not model:
            return ["others", "others", "others"]

        try:
            client = self._get_prefix_client()
            prefix_input = f"【查詢文本】\n{q_expand.strip()}"
            if behavior_context:
                append_behavior_event(
                    behavior_context.get("folder_path"),
                    behavior_context.get("date_str"),
                    agent="professor_subagent",
                    event_type="llm_input",
                    label="Prefix輸入",
                    title=f"{self.professor_id} Prefix Classification 輸入",
                    content=prefix_input,
                )
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.prompt_3_prefix},
                    {"role": "user", "content": prefix_input},
                ],
            )
            text = (resp.choices[0].message.content or "").strip()
            if behavior_context:
                append_behavior_event(
                    behavior_context.get("folder_path"),
                    behavior_context.get("date_str"),
                    agent="professor_subagent",
                    event_type="llm_output",
                    label="Prefix輸出",
                    title=f"{self.professor_id} Prefix Classification 輸出",
                    content=text,
                )
            raw_tokens = re.split(r"\s+", text)
            picked: List[str] = []
            for tk in raw_tokens:
                c = _to_canonical(tk)
                if c and c not in picked:
                    picked.append(c)
                if len(picked) >= 3:
                    break
            while len(picked) < 3:
                picked.append("others")
            result = [p.lower() for p in picked[:3]]
            _log(f"[Professor {self.professor_id}] 三前綴分類: {result}")
            return result
        except Exception as e:
            _log(f"[Professor {self.professor_id}] 前綴分類失敗: {e}")
            return ["others", "others", "others"]

    # ── 雙路 Dense 檢索 + RRF + Parent Mapping + Rerank ──────
    def _retrieve(
        self,
        q_txt_origin: str,
        q_txt_expand: str,
        top3_prefixes: List[str],
        _log: Callable,
        behavior_context: dict | None = None,
    ) -> str:
        """完整的檢索管線，回傳檢索結果文字"""

        # NoRAG 檢查
        no_rag = any(pfx.lower() == "norag" for pfx in top3_prefixes)
        if no_rag or self._all_emb is None:
            _log(f"[Professor {self.professor_id}] 偵測到 NoRAG 或索引未初始化，跳過檢索")
            return ""

        from langchain.docstore.document import Document

        N = len(self._texts)

        # ── Dense 評分計算（全庫）──
        q_emb_orig = np.asarray(self._embedder.embed_query(q_txt_origin), dtype=np.float32)
        q_emb_exp = np.asarray(self._embedder.embed_query(q_txt_expand), dtype=np.float32)
        if EMB_NORMALIZE:
            q_emb_orig /= (np.linalg.norm(q_emb_orig) + 1e-8)
            q_emb_exp /= (np.linalg.norm(q_emb_exp) + 1e-8)
        dense_scores_orig = np.dot(self._all_emb, q_emb_orig)
        dense_scores_exp = np.dot(self._all_emb, q_emb_exp)

        _d_o = np.ascontiguousarray(dense_scores_orig, dtype=np.float32)
        _d_e = np.ascontiguousarray(dense_scores_exp, dtype=np.float32)

        # ── Path 1: Global 全庫檢索 (2-way Dense RRF → top-30) ──
        topk_cand = min(RAG_TOPK_CAND, N)
        _d_o_part_g = np.argpartition(_d_o, -topk_cand)[-topk_cand:]
        _d_e_part_g = np.argpartition(_d_e, -topk_cand)[-topk_cand:]
        _d_o_ord_g = np.lexsort((_d_o_part_g, -_d_o[_d_o_part_g]))
        _d_e_ord_g = np.lexsort((_d_e_part_g, -_d_e[_d_e_part_g]))
        dense_sorted_orig_g = _d_o_part_g[_d_o_ord_g].tolist()
        dense_sorted_exp_g = _d_e_part_g[_d_e_ord_g].tolist()
        ranks_orig_g = {int(i): rank for rank, i in enumerate(dense_sorted_orig_g, 1)}
        ranks_exp_g = {int(i): rank for rank, i in enumerate(dense_sorted_exp_g, 1)}

        fused_global = _rrf_fusion_multi([ranks_orig_g, ranks_exp_g])[:RAG_TOPK_FULL]
        _log(f"[Professor {self.professor_id}] [Global Path] 2-way Dense RRF top-{RAG_TOPK_FULL} = {len(fused_global)}")

        # ── Path 2: Prefix-Boosted 三前綴子集檢索 ──
        candidate_idxs = []
        for pfx in top3_prefixes:
            candidate_idxs.extend(self._role_to_idxs.get(pfx, []))
        candidate_idxs = sorted(set(candidate_idxs))
        if not candidate_idxs:
            candidate_idxs = list(range(N))
        _log(f"[Professor {self.professor_id}] [Prefix Path] 候選子塊數: {len(candidate_idxs)}")

        fused_prefix = []
        if candidate_idxs:
            cand_mask = np.zeros(N, dtype=bool)
            cand_mask[np.fromiter(candidate_idxs, dtype=np.int64)] = True

            _d_o_m = np.where(cand_mask, _d_o, -np.inf)
            _d_e_m = np.where(cand_mask, _d_e, -np.inf)
            prefix_topk = min(RAG_TOPK_CAND, len(candidate_idxs))
            _d_o_part_p = np.argpartition(_d_o_m, -prefix_topk)[-prefix_topk:]
            _d_e_part_p = np.argpartition(_d_e_m, -prefix_topk)[-prefix_topk:]
            _d_o_ord_p = np.lexsort((_d_o_part_p, -_d_o_m[_d_o_part_p]))
            _d_e_ord_p = np.lexsort((_d_e_part_p, -_d_e_m[_d_e_part_p]))
            ranks_orig_p = {int(i): rank for rank, i in enumerate(_d_o_part_p[_d_o_ord_p].tolist(), 1)}
            ranks_exp_p = {int(i): rank for rank, i in enumerate(_d_e_part_p[_d_e_ord_p].tolist(), 1)}

            fused_prefix = _rrf_fusion_multi([ranks_orig_p, ranks_exp_p])[:RAG_TOPK_PREFIX]
            _log(f"[Professor {self.professor_id}] [Prefix Path] 2-way Dense RRF top-{RAG_TOPK_PREFIX} = {len(fused_prefix)}")

        # ── Union + Parent Mapping ──
        final_child_idxs = list(set(fused_global) | set(fused_prefix))
        _log(f"[Professor {self.professor_id}] [Union] 聯集後子段數 = {len(final_child_idxs)}")

        parent_child_map: Dict[str, List[tuple]] = {}
        for idx in final_child_idxs:
            pid = self._meta[idx].get("parent_id")
            if pid and pid in self._parent_dict:
                dense_max = max(dense_scores_orig[idx], dense_scores_exp[idx])
                if pid not in parent_child_map:
                    parent_child_map[pid] = []
                parent_child_map[pid].append((idx, dense_max))

        parent_docs_with_dense: List[tuple] = []
        for pid, child_info in parent_child_map.items():
            rec = self._parent_dict[pid]
            max_dense = max(ds for _, ds in child_info)
            doc = Document(
                page_content=rec["text"],
                metadata={"parent_id": pid, "role": rec["role"], "source": rec["source"]},
            )
            parent_docs_with_dense.append((doc, max_dense))

        _log(f"[Professor {self.professor_id}] [Parent Mapping] 候選父段數 = {len(parent_docs_with_dense)}")

        # ── LLM Rerank ──
        scored_parents = self._llm_rerank(q_txt_expand, parent_docs_with_dense, _log, behavior_context)

        # ── 父段選取 ──
        filtered = [(r, d, doc) for r, d, doc in scored_parents if r > RAG_RERANK_FLOOR]
        primary = [(r, d, doc) for r, d, doc in filtered if r >= RAG_RERANK_AUTO]
        remaining = [(r, d, doc) for r, d, doc in filtered if r < RAG_RERANK_AUTO]

        selected = list(primary)
        if len(selected) < RAG_MIN_PARENTS and remaining:
            need = RAG_MIN_PARENTS - len(selected)
            selected.extend(remaining[:need])
        if len(selected) > RAG_MAX_PARENTS:
            selected = selected[:RAG_MAX_PARENTS]

        _log(f"[Professor {self.professor_id}] [父段選取] 最終父段數 = {len(selected)}")

        # 顯示 Rerank 結果明細
        _log(f"[Professor {self.professor_id}] [父段｜Scoring 後] -------------------------------")
        rerank_lines = []
        selected_doc_ids = {id(doc) for _, _, doc in selected}
        for idx, (rerank_score, dense_score, doc) in enumerate(scored_parents, 1):
            preview = doc.page_content.replace('\n', ' ')[:80]
            source = doc.metadata.get("source", "unknown")
            marker = "✓" if id(doc) in selected_doc_ids else " "
            line = (
                f"[Professor {self.professor_id}]  {marker} {idx:02d}. "
                f"{source} {preview}  "
                f"(rerank={rerank_score:.3f}, dense={dense_score:.4f})"
            )
            rerank_lines.append(line)
            _log(line)

        if behavior_context:
            rerank_content = "\n".join(rerank_lines) or "（無 Rerank 結果）"
            if rerank_lines:
                rerank_content = f"```text\n{rerank_content}\n```"
            append_behavior_event(
                behavior_context.get("folder_path"),
                behavior_context.get("date_str"),
                agent="professor_subagent",
                event_type="tool_event",
                label="Rerank結果",
                title=f"{self.professor_id} Rerank 統合結果",
                content=rerank_content,
            )

        top_parents = [doc for _, _, doc in selected]
        retr_doc = "\n\n".join(
            [d.page_content + f" (來源:{d.metadata['source']})" for d in top_parents]
        )
        if behavior_context:
            append_behavior_event(
                behavior_context.get("folder_path"),
                behavior_context.get("date_str"),
                agent="professor_subagent",
                event_type="rag_retrieval",
                label="RAG檢索資料",
                title=f"{self.professor_id} RAG 檢索資料",
                content=retr_doc or "（無檢索結果）",
            )
        return retr_doc

    # ── LLM Rerank ───────────────────────────────────────────
    def _llm_rerank(
        self,
        query: str,
        docs_with_dense: List[tuple],
        _log: Callable,
        behavior_context: dict | None = None,
    ) -> List[tuple]:
        """LLM Rerank 評分（雙排序：rerank + dense）"""
        rrk_cfg = self.config.get("rerank", {})
        model = rrk_cfg.get("model_name", "")
        if not model or not self.prompt_rerank:
            # 無 rerank model → 只用 dense 排序
            out = [(0.5, ds, doc) for doc, ds in docs_with_dense]
            out.sort(key=lambda x: x[1], reverse=True)
            return out

        client = self._get_rerank_client()
        out: List[tuple] = []
        t0 = time.perf_counter()

        for doc, dense_score in docs_with_dense:
            chunk_txt = doc.page_content.strip().replace("\n", " ")
            rerank_input = f"問題：{query}\n\n段落：{chunk_txt}\n分析相關度分數（0~1）："
            messages = [
                {"role": "system", "content": self.prompt_rerank},
                {"role": "user", "content": rerank_input},
            ]
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                gen_txt = (resp.choices[0].message.content or "").strip()
                m = re.search(r"\d*\.?\d+", gen_txt)
                rerank_score = float(m.group(0)) if m else 0.0
            except Exception:
                rerank_score = 0.0
            out.append((rerank_score, dense_score, doc))

        out.sort(key=lambda x: (x[0], x[1]), reverse=True)
        _log(f"[Professor {self.professor_id}] [LLM Rerank] 耗時 {time.perf_counter() - t0:.2f}s")
        return out

    # ── 答覆生成 ─────────────────────────────────────────────
    def _generate_answer(
        self,
        question: str,
        note_content: str,
        at_content: str,
        last_visit_block: str,
        history_summary: str,
        retr_doc: str,
        forum_history_text: str,
        loaded_files_block: str,
        image_files: list | None = None,
        _log: Callable = lambda x: None,
        behavior_context: dict | None = None,
    ) -> str:
        ans_cfg = self.config.get("answer", {})
        model = ans_cfg.get("model_name", "")
        if not model:
            return "⚠️ 教授 Answer LLM 模型未設定。"

        # 格式化 system prompt
        sys_prompt = self.prompt_system
        if not sys_prompt:
            sys_prompt = "你是一位中醫學教授，請根據提供的知識庫內容回答臨床問題。"

        sys_prompt = sys_prompt.replace("{last_visit_block}", last_visit_block or "（無上次就診紀錄）")
        sys_prompt = sys_prompt.replace("{history_summary}", history_summary or "（無歷史病歷）")
        sys_prompt = sys_prompt.replace("{retrieved_context}", retr_doc or "（無檢索結果）")

        # 建立 user prompt（動態資料放此處，利於 KV cache）
        user_parts = [f"【提問】\n{question}"]

        user_parts.append(f"## 【今日病歷(或當前編輯頁面的病歷) - NOTE】\n{note_content or '（空白）'}")
        user_parts.append(f"## 【今日病歷(或當前編輯頁面的病歷) - ASSESSMENT & TREATMENT】\n{at_content or '（空白）'}")
        user_parts.append(f"## 【醫療問答討論區】\n{forum_history_text if forum_history_text else '（空白）'}")
        user_parts.append(f"## 【當輪讀取檔案暫存區】\n{loaded_files_block if loaded_files_block else '（空白）'}")

        user_prompt = "\n\n".join(user_parts)

        _log(f"\n{'▼'*60}")
        _log(f"[Professor {self.professor_id}] ══ 送入 LLM 的 User Prompt ══")
        _log(user_prompt)
        _log(f"{'▼'*60}")
        if behavior_context:
            append_behavior_event(
                behavior_context.get("folder_path"),
                behavior_context.get("date_str"),
                agent="professor_subagent",
                event_type="llm_input",
                label="回答輸入",
                title=f"{self.professor_id} 教授回答輸入",
                content=user_prompt,
            )

        try:
            client = self._get_answer_client()
            _prof_messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]
            if image_files:
                _prof_messages = inject_images_into_messages(_prof_messages, image_files)
            resp = client.chat.completions.create(
                model=model,
                messages=_prof_messages,
                max_tokens=int(ans_cfg.get("max_tokens", 20000)),
                temperature=float(ans_cfg.get("temperature", 0.7)),
            )
            answer_text = (resp.choices[0].message.content or "").strip()
            _log(f"[Professor {self.professor_id}] 答覆生成完成 (len={len(answer_text)})")
            if behavior_context:
                append_behavior_event(
                    behavior_context.get("folder_path"),
                    behavior_context.get("date_str"),
                    agent="professor_subagent",
                    event_type="llm_output",
                    label="回答輸出",
                    title=f"{self.professor_id} 教授回答輸出",
                    content=answer_text,
                )
            return answer_text
        except Exception as e:
            _log(f"[Professor {self.professor_id}] 答覆生成失敗: {e}")
            return f"⚠️ 教授回答失敗: {e}"


# ════════════════════════════════════════════════════════════════
# 工具函數
# ════════════════════════════════════════════════════════════════

def load_all_professors() -> List[dict]:
    """
    掃描專案資料夾下所有 professor_xx 目錄，回傳教授清單。

    Returns:
        [{"id": "professor_01", "name": "學院派教授", "description": "..."}]
    """
    result = []
    for d in sorted(_PROJECT_DIR.glob("professor_*")):
        if not d.is_dir():
            continue
        prof_id = d.name
        desc_path = d / "Description.txt"
        name = ""
        description = ""
        if desc_path.exists():
            try:
                desc = json.loads(desc_path.read_text(encoding="utf-8"))
                name = desc.get("name", "")
                description = desc.get("description", "")
            except Exception:
                pass
        result.append({
            "id": prof_id,
            "name": name,
            "description": description,
        })
    return result


def check_professor_files(professor_id: str) -> dict:
    """
    檢查指定教授資料夾的檔案完整性。

    Returns:
        {"complete": bool, "missing": list[str], "existing": list[str]}
    """
    prof_dir = _PROJECT_DIR / professor_id
    required = [
        "doc/",
        "prompt_system.txt",
        "prompt_3_prefix.txt",
        "prompt_query_expansion.txt",
        "prompt_rerank.txt",
        "Description.txt",
    ]
    optional_db = [
        "chroma_doc_index/",
        "parent_map.jsonl",
    ]

    missing = []
    existing = []

    for item in required:
        path = prof_dir / item.rstrip("/")
        if item.endswith("/"):
            if path.is_dir() and any(path.iterdir()):
                existing.append(item)
            else:
                missing.append(item + "（資料夾不存在或為空）")
        else:
            if path.exists() and path.stat().st_size > 0:
                existing.append(item)
            else:
                missing.append(item)

    db_missing = []
    for item in optional_db:
        path = prof_dir / item.rstrip("/")
        if item.endswith("/"):
            if path.is_dir():
                existing.append(item)
            else:
                db_missing.append(item + "（請點擊『建立資料庫』）")
        else:
            if path.exists():
                existing.append(item)
            else:
                db_missing.append(item + "（請點擊『建立資料庫』）")

    return {
        "complete": len(missing) == 0 and len(db_missing) == 0,
        "missing": missing + db_missing,
        "existing": existing,
    }


def release_chroma_handles():
    """釋放 process 內快取的 Chroma 連線與 mmap 檔案。

    chromadb 會把 PersistentClient 以路徑為 key 快取在全域，SQLite 連線與
    HNSW mmap 檔案（data_level0.bin 等）會被本程序一直持有；Windows 上刪除
    或重建 chroma_doc_index 前必須先釋放，否則 rmtree 會撞 WinError 32。
    ProfessorInstance 載入索引時已把向量複製進 numpy，清快取不影響已載入的教授。
    """
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
    except ImportError:
        try:
            from chromadb.api.client import SharedSystemClient
        except ImportError:
            gc.collect()
            return
    for system in list(SharedSystemClient._identifier_to_system.values()):
        try:
            system.stop()
        except Exception:
            pass
    try:
        SharedSystemClient.clear_system_cache()
    except Exception:
        pass
    gc.collect()


def _clear_chroma_dir(chroma_dir: str) -> bool:
    """釋放控制代碼後刪除既有索引資料夾，含 Windows 檔案鎖重試。"""
    if not os.path.isdir(chroma_dir):
        return True
    release_chroma_handles()
    for _ in range(3):
        try:
            shutil.rmtree(chroma_dir)
            return True
        except PermissionError:
            time.sleep(0.5)
            release_chroma_handles()
    return False


def build_professor_index(
    professor_id: str,
    config: dict,
    log_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    為指定教授建立向量索引（Parent-Child 架構）。

    Args:
        professor_id: 教授 ID
        config: 共用模型設定（需要 embedding 區塊）
        log_callback: 日誌回呼

    Returns:
        {"success": bool, "message": str, "child_count": int, "parent_count": int}
    """
    from langchain_community.vectorstores import Chroma
    from langchain.docstore.document import Document

    def _log(msg: str):
        if log_callback:
            log_callback(msg)
        print(msg)

    prof_dir = _PROJECT_DIR / professor_id
    doc_dir = prof_dir / "doc"
    chroma_dir = str(prof_dir / "chroma_doc_index")
    pmap_path = str(prof_dir / "parent_map.jsonl")

    if not doc_dir.is_dir():
        return {"success": False, "message": f"找不到 {doc_dir}", "child_count": 0, "parent_count": 0}

    txt_files = glob.glob(os.path.join(str(doc_dir), "*.txt"))
    if not txt_files:
        return {"success": False, "message": "doc/ 資料夾內沒有 .txt 檔案", "child_count": 0, "parent_count": 0}

    # 建立 Embedder
    emb_cfg = config.get("embedding", {})
    emb_model = emb_cfg.get("model_name", "")
    if not emb_model:
        return {"success": False, "message": "Embedding 模型未設定", "child_count": 0, "parent_count": 0}

    embedder = LMStudioEmbeddings(
        base_url=emb_cfg.get("api_url", "http://localhost:1234/v1"),
        api_key=emb_cfg.get("api_key", "lm-studio"),
        model=emb_model,
        batch_size=32,
        max_chars_per_input=900,
    )

    _log(f"[建立索引] 使用 Embedding 模型: {emb_model}")

    # 重建前先清除舊索引：Chroma.from_documents 對既有 collection 是 append，
    # 不清除會讓 chunk 疊加重複、舊 parent_id 變成孤兒。
    if os.path.isdir(chroma_dir):
        if not _clear_chroma_dir(chroma_dir):
            return {
                "success": False,
                "message": "無法清除舊索引（檔案被占用），請關閉占用程式後重試",
                "child_count": 0,
                "parent_count": 0,
            }
        _log("[建立索引] 已清除舊索引，將重新建立")

    # 父段切割
    split_pat = re.compile(r"(?=(?:\r?\n){3,})")

    def strip_leading_role(text: str, role: str) -> str:
        if not role:
            return text
        pat = rf"^(?:{re.escape(role)})\b[\s:：]*"
        return re.sub(pat, "", text, count=1, flags=re.I)

    def split_overlap(text: str, size: int, over: int) -> List[str]:
        out, i = [], 0
        while i < len(text):
            out.append(text[i : i + size])
            i += size - over
        return [s for s in out if len(s.strip()) > 10]

    docs = []
    parent_map = []
    t0 = time.perf_counter()

    for fp in txt_files:
        t_file = time.perf_counter()
        with open(fp, encoding="utf-8") as f:
            raw = f.read()

        for blk in [b.strip() for b in split_pat.split(raw) if b.strip()]:
            m = _PREFIX_REGEX.match(blk)
            role = (m.group(1).lower() if m else "others")

            parent_id = str(uuid.uuid4())
            parent_map.append({
                "parent_id": parent_id,
                "role": role,
                "source": os.path.basename(fp),
                "text": blk,
            })

            if role == "case":
                size, over = CHUNK_SIZE_CASE, CHUNK_OVER_CASE
            else:
                size, over = CHUNK_SIZE, CHUNK_OVER

            blk_body = strip_leading_role(blk, role)
            for chunk in split_overlap(blk_body, size=size, over=over):
                docs.append(Document(
                    page_content=f"{role}: {chunk}",
                    metadata={"parent_id": parent_id, "role": role},
                ))

        _log(f"[建立索引] 處理完成: {os.path.basename(fp)} ({time.perf_counter() - t_file:.2f}s)")

    _log(f"[建立索引] 總計 子塊: {len(docs)}, 父段: {len(parent_map)}")

    # 寫 parent_map
    with open(pmap_path, "w", encoding="utf-8") as f:
        for r in parent_map:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 建立 Chroma 索引
    _log("[建立索引] 開始向量化入庫...")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedder,
        collection_name="doc_blocks",
        persist_directory=chroma_dir,
    )
    vectordb.persist()
    del vectordb
    # 建完立即釋放檔案鎖，否則 Windows 上後續的刪除教授 / 再次重建會撞 WinError 32
    release_chroma_handles()

    elapsed = time.perf_counter() - t0
    _log(f"[建立索引] ✅ 完成！耗時 {elapsed:.2f}s")

    return {
        "success": True,
        "message": f"索引建立完成，子塊 {len(docs)} 個，父段 {len(parent_map)} 個，耗時 {elapsed:.1f}s",
        "child_count": len(docs),
        "parent_count": len(parent_map),
    }
