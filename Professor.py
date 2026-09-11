"""
Professor.py - 中醫教授諮詢模組（ReAct + RAG tools）
所有教授共用此模組，各自載入不同的 doc 資料夾與 prompt 檔案。

功能：
1. ProfessorInstance — 代表一位教授的 ReAct 多輪實例
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
from multimodal_utils import IMAGE_EXTENSIONS, inject_images_into_messages
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

PROFESSOR_DEFAULT_MAX_ROUNDS = 15
PROFESSOR_DEFAULT_MAX_RETRIEVALS_PER_ANSWER = 3
PROFESSOR_DEFAULT_REACT_HISTORY_PROMPT_CHARS = 5000
PROFESSOR_DEFAULT_GRAFFITI_SUMMARIZE_THRESHOLD = 8000
PROFESSOR_MAX_JSON_RETRIES = 3
PROFESSOR_DEFAULT_ROLE_STYLE = "請維持資深中醫教授的客觀、嚴謹、臨床導向回答風格。"

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


def datetime_now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


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
    """一位教授的 ReAct + RAG 工具實例"""

    def __init__(self, professor_id: str, config: dict):
        """
        Args:
            professor_id: 如 "professor_01"
            config: 共用模型設定，結構：
                {
                    "max_rounds": 15,
                    "max_retrievals_per_answer": 3,
                    "react_history_prompt_chars": 5000,
                    "graffiti_summarize_threshold": 8000,
                    "answer": {"api_url", "api_key", "model_name", "max_tokens", "temperature"},
                    "embedding": {"api_url", "api_key", "model_name"},
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
        self.role_style = ""
        desc_path = self.prof_dir / "Description.txt"
        if desc_path.exists():
            try:
                desc = json.loads(desc_path.read_text(encoding="utf-8"))
                self.name = desc.get("name", "")
                self.description_text = desc.get("description", "")
                self.role_style = desc.get("role_style", "")
            except Exception:
                pass

        # 載入 prompt 檔案
        self.prompt_system = self._load_prompt("prompt_system.txt")
        # 將 Description.txt 的教授名稱、簡介與角色風格注入 prompt_system
        professor_display_name = self.name or self.professor_id
        self.prompt_system = self.prompt_system.replace("{name}", professor_display_name)
        if self.description_text:
            self.prompt_system = self.prompt_system.replace("{description}", self.description_text)
        else:
            self.prompt_system = self.prompt_system.replace("{description}", "")
        role_style = self.role_style.strip() if isinstance(self.role_style, str) else ""
        self.prompt_system = self.prompt_system.replace(
            "{role_style}",
            role_style or PROFESSOR_DEFAULT_ROLE_STYLE,
        )
        self.prompt_3_prefix = self._load_prompt("prompt_3_prefix.txt")
        self.prompt_rerank = self._load_prompt("prompt_rerank.txt")

        # 教授 ReAct session 記憶（跟 MainAgent session 綁定；MainAgent reset 時會清掉 instance）
        self.graffiti_wall = ""
        self.react_history: list[dict[str, Any]] = []
        self.last_react_history_truncated = False
        self.last_context_overflow = False
        self.max_rounds = self._parse_max_rounds(config)
        self.max_retrievals_per_answer = self._parse_max_retrievals(config)
        self.react_history_prompt_chars = self._parse_react_history_prompt_chars(config)
        self.graffiti_summarize_threshold = self._parse_graffiti_summarize_threshold(config)

        # LLM clients（惰性建立）
        self._answer_client: Optional[OpenAI] = None
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

    def _parse_max_rounds(self, config: dict) -> int:
        try:
            value = int(config.get("max_rounds", PROFESSOR_DEFAULT_MAX_ROUNDS) or PROFESSOR_DEFAULT_MAX_ROUNDS)
        except Exception:
            value = PROFESSOR_DEFAULT_MAX_ROUNDS
        return max(1, value)

    def _parse_max_retrievals(self, config: dict) -> int:
        try:
            raw = config.get(
                "max_retrievals_per_answer",
                PROFESSOR_DEFAULT_MAX_RETRIEVALS_PER_ANSWER,
            )
            if raw is None or raw == "":
                raw = PROFESSOR_DEFAULT_MAX_RETRIEVALS_PER_ANSWER
            value = int(raw)
        except Exception:
            value = PROFESSOR_DEFAULT_MAX_RETRIEVALS_PER_ANSWER
        return max(0, value)

    def _parse_react_history_prompt_chars(self, config: dict) -> int:
        return self._parse_positive_int_setting(
            config,
            "react_history_prompt_chars",
            PROFESSOR_DEFAULT_REACT_HISTORY_PROMPT_CHARS,
        )

    def _parse_graffiti_summarize_threshold(self, config: dict) -> int:
        return self._parse_positive_int_setting(
            config,
            "graffiti_summarize_threshold",
            PROFESSOR_DEFAULT_GRAFFITI_SUMMARIZE_THRESHOLD,
        )

    @staticmethod
    def _parse_positive_int_setting(config: dict, key: str, default: int) -> int:
        try:
            raw = config.get(key, default)
            if raw is None or raw == "":
                raw = default
            value = int(raw)
        except Exception:
            value = default
        return max(1, value)

    def export_memory(self) -> dict[str, Any]:
        return {
            "graffiti_wall": self.graffiti_wall,
            "react_history": self.react_history,
            "last_react_history_truncated": self.last_react_history_truncated,
            "last_context_overflow": self.last_context_overflow,
        }

    def restore_memory(self, memory: dict[str, Any] | None) -> None:
        if not isinstance(memory, dict):
            return
        self.graffiti_wall = str(memory.get("graffiti_wall", "") or "")
        react_history = memory.get("react_history", [])
        self.react_history = react_history if isinstance(react_history, list) else []
        self.last_react_history_truncated = bool(memory.get("last_react_history_truncated", False))
        self.last_context_overflow = bool(memory.get("last_context_overflow", False))

    # ── LLM Client 惰性建立 ─────────────────────────────────
    def _get_answer_client(self) -> OpenAI:
        if self._answer_client is None:
            cfg = self.config.get("answer", {})
            self._answer_client = OpenAI(
                api_key=cfg.get("api_key", "lm-studio"),
                base_url=cfg.get("api_url", "http://localhost:1234/v1"),
            )
        return self._answer_client

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
        show_forum_history: bool = False,
        loaded_files_block: str = "",
        image_files: list | None = None,
        patient_folder: str | None = None,
        manual_stop_event: Any | None = None,
        log_callback: Optional[Callable[[str], None]] = None,
        behavior_context: dict | None = None,
    ) -> dict[str, Any]:
        """
        教授以 ReAct 多輪工具迴圈回答一個臨床問題。

        Args:
            question: AI 主治醫師的提問
            note_content: 今日病歷 NOTE
            at_content: 辨證論治 A&T
            last_visit_block: 上次就診病歷
            history_summary: 歷史病歷摘要
            forum_history_text: 本次允許教授看見的醫療問答討論區內容
            show_forum_history: 是否向教授顯示既有醫療問答討論區；只控制討論區，不影響教授記憶
            loaded_files_block: 主 Agent 當輪讀取檔案暫存區內容
            image_files: 多模態圖片檔案（主 Agent 已讀）
            patient_folder: 患者資料夾，供教授 list/read patient files tool 使用
            manual_stop_event: 選用，MainAgent 合作式中斷事件
            log_callback: 選用，日誌回呼
            behavior_context: 選用，智能體互動行為 log 的患者/日期/輪次資訊

        Returns:
            {
                "response": str,
                "error": str|None,
                "forced": bool,
                "retrieval_records": list[dict],
                "react_history": list[dict],
                "graffiti_wall": str,
                "react_history_truncated": bool,
                "context_overflow": bool,
                "q_expand": str,   # 相容欄位：本次最後一次檢索 query
                "prefixes": list,  # 相容欄位：本次最後一次檢索 prefixes
                "retr_doc": str,   # 相容欄位：本次最後一次檢索原文
            }
        """

        def _log(msg: str):
            if log_callback:
                log_callback(msg)
            print(msg)

        # 直接呼叫 ProfessorInstance.answer() 時也採 fail-safe：只有真正的 bool True 才顯示討論區。
        show_forum_history = show_forum_history is True
        self.max_rounds = self._parse_max_rounds(self.config)
        self.max_retrievals_per_answer = self._parse_max_retrievals(self.config)
        self.react_history_prompt_chars = self._parse_react_history_prompt_chars(self.config)
        self.graffiti_summarize_threshold = self._parse_graffiti_summarize_threshold(self.config)
        self.last_react_history_truncated = False
        self.last_context_overflow = False

        _log(
            f"[Professor {self.professor_id}] 開始 ReAct 多輪處理提問"
            f"（max_rounds={self.max_rounds}, max_retrievals={self.max_retrievals_per_answer}, "
            f"react_history_prompt_chars={self.react_history_prompt_chars}, "
            f"graffiti_summarize_threshold={self.graffiti_summarize_threshold}）..."
        )
        self._behavior_event(
            behavior_context,
            event_type="professor_react_start",
            label="教授開始",
            title=f"{self.professor_id} ReAct 開始",
            content=question,
            meta={
                "max_rounds": self.max_rounds,
                "max_retrievals_per_answer": self.max_retrievals_per_answer,
                "react_history_prompt_chars": self.react_history_prompt_chars,
                "graffiti_summarize_threshold": self.graffiti_summarize_threshold,
                "show_forum_history": bool(show_forum_history),
                "forum_history_chars_available": (behavior_context or {}).get(
                    "forum_history_chars_available", len(forum_history_text or "")
                ),
                "forum_history_chars_exposed": len(forum_history_text or "") if show_forum_history else 0,
            },
        )

        if not self.config.get("answer", {}).get("model_name", ""):
            error_msg = "教授 Answer LLM 模型未設定"
            self._react_end_error(behavior_context, error_msg)
            return {
                "response": "⚠️ 教授 Answer LLM 模型未設定。",
                "q_expand": "",
                "prefixes": [],
                "retr_doc": "",
                "retrieval_records": [],
                "error": error_msg,
                "react_history": self.react_history,
                "graffiti_wall": self.graffiti_wall,
                "react_history_truncated": self.last_react_history_truncated,
                "context_overflow": self.last_context_overflow,
                "forced": False,
            }
        system_prompt = self._build_react_system_prompt(last_visit_block, history_summary)

        tool_state: dict[str, Any] = {
            "patient_file_list_cache": "",
            "professor_loaded_files": [],
            "retrieved_context": "",
            "last_retrieval_query": "",
            "last_retrieval_prefixes": [],
            "retrieval_records": [],
            "retrieval_count": 0,
        }

        for round_idx in range(1, self.max_rounds + 1):
            if self._stop_requested(manual_stop_event):
                return self._manual_stop_response(tool_state, behavior_context)
            round_label = f"P-{self.professor_id}-{round_idx}"
            final_round_notice = ""
            if round_idx == self.max_rounds:
                final_round_notice = "（系統提示：這是本次教授 ReAct 的最後一輪，除非完全無法回答，請使用 reply_to_forum 交付最終答案。）"

            react_history_text = self._format_react_history(limit=None)
            react_history_chars = len(react_history_text)
            react_history_limit = self._react_history_prompt_limit(react_history_chars)
            if react_history_limit is not None:
                self.last_react_history_truncated = True
                self._log_react_history_truncated(
                    behavior_context,
                    round_label=round_label,
                    round_meta={"round": round_idx},
                    react_history_chars=react_history_chars,
                )
            user_prompt = self._build_react_user_prompt(
                question=question,
                note_content=note_content,
                at_content=at_content,
                forum_history_text=forum_history_text,
                show_forum_history=show_forum_history,
                main_loaded_files_block=loaded_files_block,
                professor_loaded_files=tool_state["professor_loaded_files"],
                patient_file_list_cache=tool_state["patient_file_list_cache"],
                retrieved_context=tool_state["retrieved_context"],
                react_history_limit=react_history_limit,
                react_history_text_override=(
                    self._truncate_react_history_text(react_history_text, react_history_limit)
                    if react_history_limit is not None
                    else react_history_text
                ),
                overflow_notice=react_history_limit is not None,
                final_round_notice=final_round_notice,
            )
            image_files_current = (image_files or []) + tool_state["professor_loaded_files"]
            self._emit_react_llm_input(
                user_prompt=user_prompt,
                round_label=round_label,
                round_meta={"round": round_idx, "react_history_chars": react_history_chars},
                _log=_log,
                behavior_context=behavior_context,
            )
            parsed, error, context_overflow = self._call_react_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_files=image_files_current,
                round_label=round_label,
                _log=_log,
                behavior_context=behavior_context,
                manual_stop_event=manual_stop_event,
            )

            if parsed is None:
                if error == "manual_stop":
                    return self._manual_stop_response(tool_state, behavior_context)
                if context_overflow:
                    return self._context_overflow_failure(
                        tool_state,
                        behavior_context,
                        error=error,
                        round_label=round_label,
                        round_meta={"round": round_idx, "react_history_chars": react_history_chars},
                    )
                observation = f"LLM 輸出無法解析或呼叫失敗：{error or '未知錯誤'}"
                self.react_history.append({
                    "ts": datetime_now_text(),
                    "round": round_idx,
                    "thought_summary": "",
                    "action": "invalid_or_failed_llm_output",
                    "action_input": {},
                    "observation": observation,
                })
                self._behavior_event(
                    behavior_context,
                    event_type="professor_tool_error",
                    label="教授錯誤",
                    title=f"{self.professor_id} {round_label} LLM 失敗",
                    content=observation,
                    severity="warning",
                    meta={"round": round_idx},
                )
                continue

            thought_summary = self._sanitize_thought_summary(parsed.get("thought_summary", ""))
            action = str(parsed.get("action", "") or "").strip()
            action_input = parsed.get("action_input", "")

            observation, reply_text = self._execute_react_action(
                action=action,
                action_input=action_input,
                question=question,
                note_content=note_content,
                patient_folder=patient_folder,
                tool_state=tool_state,
                _log=_log,
                behavior_context=behavior_context,
                round_label=round_label,
                round_idx=round_idx,
                manual_stop_event=manual_stop_event,
            )
            if observation == "manual_stop":
                return self._manual_stop_response(tool_state, behavior_context)

            self.react_history.append({
                "ts": datetime_now_text(),
                "round": round_idx,
                "thought_summary": thought_summary,
                "action": action,
                "action_input": self._sanitize_action_input(action, action_input),
                "observation": self._sanitize_observation(action, observation),
            })

            if reply_text is not None:
                response = reply_text.strip()
                _log(f"[Professor {self.professor_id}] reply_to_forum 完成 (len={len(response)})")
                self._behavior_event(
                    behavior_context,
                    event_type="professor_react_end",
                    label="教授結束",
                    title=f"{self.professor_id} ReAct 結束",
                    content=response,
                    meta={
                        "round": round_idx,
                        "react_history_truncated": self.last_react_history_truncated,
                        "context_overflow": self.last_context_overflow,
                    },
                )
                return {
                    "response": response,
                    "q_expand": tool_state["last_retrieval_query"],
                    "prefixes": tool_state["last_retrieval_prefixes"],
                    "retr_doc": tool_state["retrieved_context"],
                    "retrieval_records": tool_state["retrieval_records"],
                    "error": None,
                    "react_history": self.react_history,
                    "graffiti_wall": self.graffiti_wall,
                    "react_history_truncated": self.last_react_history_truncated,
                    "context_overflow": self.last_context_overflow,
                    "forced": False,
                }

        _log(f"[Professor {self.professor_id}] ⚠️ 達到 ReAct 輪數上限 ({self.max_rounds})，強制整理目前資訊回答")
        self._behavior_event(
            behavior_context,
            event_type="professor_react_limit_reached",
            label="教授達上限",
            title=f"{self.professor_id} ReAct 達到輪數上限",
            content=f"已達 max_rounds={self.max_rounds}，要求教授整理目前資訊回答。",
            severity="warning",
            meta={"max_rounds": self.max_rounds},
        )
        response, force_error, forced = self._force_reply_to_forum(
            question=question,
            note_content=note_content,
            at_content=at_content,
            forum_history_text=forum_history_text,
            show_forum_history=show_forum_history,
            main_loaded_files_block=loaded_files_block,
            professor_loaded_files=tool_state["professor_loaded_files"],
            patient_file_list_cache=tool_state["patient_file_list_cache"],
            retrieved_context=tool_state["retrieved_context"],
            image_files=(image_files or []) + tool_state["professor_loaded_files"],
            system_prompt=system_prompt,
            _log=_log,
            behavior_context=behavior_context,
            manual_stop_event=manual_stop_event,
        )

        return {
            "response": response,
            "q_expand": tool_state["last_retrieval_query"],
            "prefixes": tool_state["last_retrieval_prefixes"],
            "retr_doc": tool_state["retrieved_context"],
            "retrieval_records": tool_state["retrieval_records"],
            "error": force_error,
            "react_history": self.react_history,
            "graffiti_wall": self.graffiti_wall,
            "react_history_truncated": self.last_react_history_truncated,
            "context_overflow": self.last_context_overflow,
            "forced": forced,
        }

    # ── ReAct 執行器輔助方法 ─────────────────────────────────
    def _build_react_system_prompt(self, last_visit_block: str, history_summary: str) -> str:
        sys_prompt = self.prompt_system or "你是一位中醫學教授，請以 ReAct 工具迴圈回答臨床問題。"
        sys_prompt = sys_prompt.replace("{last_visit_block}", last_visit_block or "（無上次就診紀錄）")
        sys_prompt = sys_prompt.replace("{history_summary}", history_summary or "（無歷史病歷）")
        sys_prompt = sys_prompt.replace("{retrieved_context}", "（知識庫檢索結果已移至 user prompt 最下方。）")

        if self.max_retrievals_per_answer <= 0:
            budget_text = (
                f"本次教授諮詢最多 {self.max_rounds} 輪；"
                "本次設定禁止呼叫 retrieve_knowledge，請依既有上下文、患者檔案與一般醫學知識回答。"
            )
        else:
            budget_text = (
                f"本次教授諮詢最多 {self.max_rounds} 輪，"
                f"最多可呼叫 retrieve_knowledge {self.max_retrievals_per_answer} 次；"
                "請合理分配檢索次數，避免重複查詢。"
            )

        runtime_contract = """

# 執行器輸出契約
你每一輪只能輸出一個 JSON 物件，不要輸出 JSON 以外的文字。格式如下：
{
  "thought_summary": "用可審計摘要描述本輪判斷，不要揭露冗長隱性思考",
  "action": "工具名稱",
  "action_input": {}
}

`action` 必須精確等於下列五個工具名稱之一，且只能填一個；不得輸出含 `|` 的複合字串、其他別名或說明文字：
- retrieve_knowledge
- update_graffiti_wall
- list_patient_files
- read_patient_file
- reply_to_forum

action_input 必須是 JSON 物件，並符合外置 prompt「可用工具」區塊定義的 schema。
reply_to_forum 是唯一正常結束方式；若尚未足以回答，請使用其他工具補足資訊。
""" + budget_text + "\n"
        return sys_prompt + runtime_contract

    def _behavior_event(
        self,
        behavior_context: dict | None,
        *,
        event_type: str,
        label: str,
        title: str,
        content: str,
        severity: str = "normal",
        meta: dict[str, Any] | None = None,
    ) -> None:
        if not behavior_context:
            return
        append_behavior_event(
            behavior_context.get("folder_path"),
            behavior_context.get("date_str"),
            agent="professor_subagent",
            event_type=event_type,
            label=label,
            title=title,
            content=content,
            turn=behavior_context.get("turn"),
            sub_turn=behavior_context.get("sub_turn"),
            severity=severity,
            meta=meta,
        )

    def _react_end_error(self, behavior_context: dict | None, error_msg: str, meta: dict[str, Any] | None = None) -> None:
        self._behavior_event(
            behavior_context,
            event_type="professor_react_end",
            label="教授結束",
            title=f"{self.professor_id} ReAct 錯誤結束",
            content=error_msg,
            severity="error",
            meta=meta,
        )

    def _stop_requested(self, manual_stop_event: Any | None) -> bool:
        try:
            return bool(manual_stop_event and manual_stop_event.is_set())
        except Exception:
            return False

    def _manual_stop_response(self, tool_state: dict[str, Any], behavior_context: dict | None) -> dict[str, Any]:
        error_msg = "manual_stop"
        self._react_end_error(behavior_context, "教授 ReAct 已收到手動中斷。", meta={"manual_stop": True})
        return {
            "response": "⚠️ 教授 ReAct 已手動中斷。",
            "q_expand": tool_state.get("last_retrieval_query", ""),
            "prefixes": tool_state.get("last_retrieval_prefixes", []),
            "retr_doc": tool_state.get("retrieved_context", ""),
            "retrieval_records": tool_state.get("retrieval_records", []),
            "error": error_msg,
            "react_history": self.react_history,
            "graffiti_wall": self.graffiti_wall,
            "react_history_truncated": self.last_react_history_truncated,
            "context_overflow": self.last_context_overflow,
            "forced": False,
        }

    def _build_react_user_prompt(
        self,
        *,
        question: str,
        note_content: str,
        at_content: str,
        forum_history_text: str,
        show_forum_history: bool,
        main_loaded_files_block: str,
        professor_loaded_files: list[dict],
        patient_file_list_cache: str,
        retrieved_context: str,
        react_history_limit: int | None,
        react_history_text_override: str | None = None,
        overflow_notice: bool = False,
        final_round_notice: str = "",
    ) -> str:
        if react_history_text_override is not None:
            react_history_text = react_history_text_override
        else:
            react_history_text = self._format_react_history(limit=react_history_limit)
        professor_files_block = self._format_loaded_files_block_from_list(professor_loaded_files)
        current_retrieved_context = retrieved_context or "（本次回答尚未檢索，或本次最新檢索無結果）"
        graffiti_char_count = len(self.graffiti_wall or "")
        graffiti_block = self.graffiti_wall if self.graffiti_wall else "（空白）"
        graffiti_stats = (
            f"（塗鴉牆字數統計：{graffiti_char_count} 字；"
            f"超過 {self.graffiti_summarize_threshold} 字時，"
            "請優先使用 update_graffiti_wall 的 summarize 模式精簡整理。）"
        )

        forum_block = (
            forum_history_text or "（目前尚無既有討論）"
            if show_forum_history
            else (
                "（本次 call_professor 設定 show_forum_history=false；"
                "既有討論區歷史已由系統隱藏。請依目前可見的提問與患者資料獨立分析，"
                "不要猜測或重建被隱藏的討論內容。）"
            )
        )

        parts = [
            f"【提問】\n{question}",
            f"## 【醫療問答討論區】\n{forum_block}",
            f"## 【今日病歷(或當前編輯頁面的病歷) - NOTE】\n{note_content or '（空白）'}",
            f"## 【今日病歷(或當前編輯頁面的病歷) - ASSESSMENT & TREATMENT】\n{at_content or '（空白）'}",
            f"## 【主 Agent 讀取後暫存區】\n{main_loaded_files_block if main_loaded_files_block else '（空白）'}",
            f"## 【醫學教授 subagent 患者檔案清單】\n{patient_file_list_cache if patient_file_list_cache else '（空白）'}",
            f"## 【醫學教授 subagent 讀取後暫存區】（僅存在於本次回答思考過程）\n{professor_files_block if professor_files_block else '（空白）'}",
            f"## 【醫學教授 subagent 塗鴉牆】（跨本 session 問答保留）\n{graffiti_block}\n\n{graffiti_stats}",
            f"## 【醫學教授 subagent ReAct 工作紀錄】（動作摘要；跨本 session 問答保留，過長時僅顯示最新片段）\n{react_history_text}",
        ]
        if overflow_notice:
            overflow_text = (
                "完整 react_history 超過系統設定上限；"
                f"本輪僅放入最新 {self.react_history_prompt_chars} 字。"
            )
            parts.append(
                f"## 【系統提示】\n{overflow_text}"
            )
        if final_round_notice:
            parts.append(f"## 【系統提示】\n{final_round_notice}")
        parts.append("請根據以上資訊，決定你的下一步動作。只能輸出一個 JSON 物件。")
        parts.append(f"## 【知識庫檢索結果】\n{current_retrieved_context}")
        return "\n\n".join(parts)

    def _format_react_history(self, limit: int | None = None) -> str:
        if not self.react_history:
            return "（空白）"
        blocks: list[str] = []
        for idx, entry in enumerate(self.react_history, 1):
            action_input = entry.get("action_input", "")
            if isinstance(action_input, (dict, list)):
                action_input_text = json.dumps(action_input, ensure_ascii=False, indent=2)
            else:
                action_input_text = str(action_input)
            blocks.append(
                "\n".join([
                    f"[{idx}] {entry.get('ts', '')} Round {entry.get('round', '?')}",
                    f"Thought summary: {entry.get('thought_summary', '')}",
                    f"Action: {entry.get('action', '')}",
                    f"Action input:\n{action_input_text}",
                    f"Observation:\n{entry.get('observation', '')}",
                ])
            )
        text = "\n\n".join(blocks)
        if limit and len(text) > limit:
            return f"（react_history 因長度上限截斷，以下僅保留最新 {limit} 字）\n{text[-limit:]}"
        return text

    def _truncate_react_history_text(self, text: str, limit: int | None) -> str:
        if limit and len(text or "") > limit:
            return f"（react_history 因長度上限截斷，以下僅保留最新 {limit} 字）\n{text[-limit:]}"
        return text

    def _format_loaded_files_block_from_list(self, loaded_files: list[dict]) -> str:
        if not loaded_files:
            return ""
        file_blocks = []
        for lf in loaded_files:
            if lf.get("type") == "text":
                file_blocks.append(f"### 📄 {lf.get('name', 'unknown')}\n{lf.get('content', '')}")
            elif lf.get("type") == "image":
                file_blocks.append(f"### 🖼️ {lf.get('name', 'unknown')}\n（圖片已載入，見多模態訊息）")
        return "\n\n".join(file_blocks)

    def _preview_text(self, text: Any, limit: int = 20) -> str:
        compact = " ".join(str(text or "").split())
        return compact[:limit]

    def _sanitize_thought_summary(self, text: Any, limit: int = 300) -> str:
        compact = " ".join(str(text or "").split())
        return compact[:limit]

    def _sanitize_action_input(self, action: str, action_input: Any) -> dict[str, Any]:
        action_key = (action or "").strip().lower()

        if action_key == "retrieve_knowledge":
            return {"query": self._extract_query(action_input)}
        if action_key == "update_graffiti_wall":
            mode, content = self._extract_graffiti_input(action_input)
            return {
                "mode": self._normalize_graffiti_mode(mode),
                "content_preview": self._preview_text(content),
                "content_chars": len(content or ""),
            }
        if action_key == "list_patient_files":
            return {"scope": self._extract_scope(action_input)}
        if action_key == "read_patient_file":
            return {"filenames": self._extract_filenames(action_input)}
        if action_key == "reply_to_forum":
            answer = self._extract_reply(action_input)
            return {
                "answer_preview": self._preview_text(answer),
                "answer_chars": len(answer or ""),
            }
        if isinstance(action_input, (dict, list)):
            text = json.dumps(action_input, ensure_ascii=False)
            return {
                "input_type": type(action_input).__name__,
                "input_preview": self._preview_text(text),
                "input_chars": len(text),
            }
        text = str(action_input or "")
        return {
            "input_preview": self._preview_text(text),
            "input_chars": len(text),
        }

    def _sanitize_observation(self, action: str, observation: str) -> str:
        action_key = (action or "").strip().lower()
        if action_key in {"retrieve_knowledge", "update_graffiti_wall", "reply_to_forum"}:
            return observation
        if action_key == "list_patient_files":
            lines = [line for line in (observation or "").splitlines() if line.strip()]
            file_count = sum(1 for line in lines if line.lstrip().startswith(("📄", "🖼️")))
            return f"list_patient_files: 已列出 {file_count} 個檔案/項目，詳情僅存在本次工具觀察。"
        if action_key == "read_patient_file":
            lines = [line for line in (observation or "").splitlines() if line.strip()]
            if not lines:
                return "read_patient_file: 無工具回饋。"
            if len(lines) == 1 and lines[0].startswith("read_patient_file:"):
                return observation
            failed = [line for line in lines if line.lstrip().startswith("❌")]
            ok_count = max(0, len(lines) - len(failed))
            out = [f"read_patient_file: 成功 {ok_count} 個，內容僅存在本次回答暫存區。"]
            if failed:
                out.append("失敗：\n" + "\n".join(failed))
            return "\n".join(out)
        return observation

    def _normalize_graffiti_mode(self, mode: str) -> str:
        mode_key = (mode or "append").strip().lower()
        if mode_key == "replace":
            return "summarize"
        if mode_key not in {"append", "summarize"}:
            return "append"
        return mode_key

    def _emit_react_llm_input(
        self,
        *,
        user_prompt: str,
        round_label: str,
        round_meta: dict[str, Any] | None,
        _log: Callable,
        behavior_context: dict | None,
    ) -> None:
        _log(f"\n{'▼'*60}")
        _log(f"[Professor {self.professor_id}] ══ {round_label} 送入 LLM 的 User Prompt ══")
        _log(user_prompt)
        _log(f"{'▼'*60}")
        meta = dict(round_meta or {})
        self._behavior_event(
            behavior_context,
            event_type="llm_input",
            label="教授輸入",
            title=f"{self.professor_id} {round_label} 輸入",
            content=user_prompt,
            meta=meta,
        )

    def _react_history_prompt_limit(self, react_history_chars: int) -> int | None:
        if react_history_chars > self.react_history_prompt_chars:
            return self.react_history_prompt_chars
        return None

    def _log_react_history_truncated(
        self,
        behavior_context: dict | None,
        *,
        round_label: str,
        round_meta: dict[str, Any] | None,
        react_history_chars: int,
    ) -> None:
        meta = dict(round_meta or {})
        meta.update({
            "reason": "react_history_chars_exceeded",
            "react_history_chars": react_history_chars,
            "react_history_prompt_chars": self.react_history_prompt_chars,
        })
        self._behavior_event(
            behavior_context,
            event_type="professor_react_history_truncated",
            label="歷史截斷",
            title=f"{self.professor_id} {round_label} react_history 截斷",
            content=(
                f"react_history 已達 {react_history_chars} 字，"
                f"本輪 prompt 僅放入最新 {self.react_history_prompt_chars} 字。"
            ),
            severity="normal",
            meta=meta,
        )

    def _context_overflow_failure(
        self,
        tool_state: dict[str, Any],
        behavior_context: dict | None,
        *,
        error: str | None,
        round_label: str,
        round_meta: dict[str, Any] | None = None,
        forced: bool = False,
    ) -> dict[str, Any]:
        self.last_context_overflow = True
        error_msg = (
            f"教授 Answer LLM context overflow：react_history 已按 {self.react_history_prompt_chars} 字上限截斷，"
            "但本輪 prompt 仍超過模型可接受大小。請減少討論區/讀檔/圖片內容後重試。"
        )
        meta = dict(round_meta or {})
        meta.update({"error": error or "", "forced": forced})
        self._behavior_event(
            behavior_context,
            event_type="professor_context_overflow",
            label="Context溢出",
            title=f"{self.professor_id} {round_label} context overflow fail-closed",
            content=error_msg,
            severity="error",
            meta=meta,
        )
        self._react_end_error(behavior_context, error_msg, meta=meta)
        return {
            "response": f"⚠️ {error_msg}",
            "q_expand": tool_state.get("last_retrieval_query", ""),
            "prefixes": tool_state.get("last_retrieval_prefixes", []),
            "retr_doc": tool_state.get("retrieved_context", ""),
            "retrieval_records": tool_state.get("retrieval_records", []),
            "error": error_msg,
            "react_history": self.react_history,
            "graffiti_wall": self.graffiti_wall,
            "react_history_truncated": self.last_react_history_truncated,
            "context_overflow": True,
            "forced": forced,
        }

    def _call_react_llm(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_files: list[dict],
        round_label: str,
        _log: Callable,
        behavior_context: dict | None,
        manual_stop_event: Any | None,
    ) -> tuple[dict[str, Any] | None, str | None, bool]:
        ans_cfg = self.config.get("answer", {})
        model = ans_cfg.get("model_name", "")
        if not model:
            return None, "教授 Answer LLM 模型未設定", False

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if image_files:
            messages = inject_images_into_messages(messages, image_files)

        client = self._get_answer_client()
        last_error: str | None = None

        for attempt in range(PROFESSOR_MAX_JSON_RETRIES + 1):
            if self._stop_requested(manual_stop_event):
                return None, "manual_stop", False
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=int(ans_cfg.get("max_tokens", 20000)),
                    temperature=float(ans_cfg.get("temperature", 0.7)),
                )
                output = (resp.choices[0].message.content or "").strip()
            except Exception as e:
                last_error = self._stringify_llm_error(e)
                _log(f"[Professor {self.professor_id}] {round_label} LLM 呼叫失敗: {last_error}")
                if self._is_context_overflow_error(e):
                    return None, last_error, True
                return None, last_error, False

            _log(f"\n[Professor {self.professor_id}] {round_label} ══ LLM 原始輸出 (attempt {attempt}) ══")
            _log(output)
            _log(f"{'─'*40}")
            self._behavior_event(
                behavior_context,
                event_type="llm_output",
                label="教授輸出",
                title=f"{self.professor_id} {round_label} 輸出",
                content=output,
                meta={"attempt": attempt},
            )

            parsed = self._parse_json_output(output)
            if parsed is not None:
                return parsed, None, False

            last_error = "JSON 解析失敗"
            if self._stop_requested(manual_stop_event):
                return None, "manual_stop", False
            if attempt < PROFESSOR_MAX_JSON_RETRIES:
                messages.append({"role": "assistant", "content": output or ""})
                messages.append({
                    "role": "user",
                    "content": "（系統提示：你剛才的輸出無法解析為合法 JSON。請只輸出一個完整 JSON 物件，不要加任何額外文字。）",
                })

        return None, last_error, False

    def _parse_json_output(self, output: str) -> dict[str, Any] | None:
        if not output:
            return None
        match = re.search(r"\{[\s\S]*\}", output)
        if not match:
            return None
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    def _stringify_llm_error(self, err: Any) -> str:
        texts = self._collect_error_texts(err)
        compact: list[str] = []
        seen: set[str] = set()
        for text in texts:
            text = " ".join(str(text or "").split())
            if text and text not in seen:
                seen.add(text)
                compact.append(text)
        return " | ".join(compact) or str(err)

    def _collect_error_texts(self, err: Any, depth: int = 0) -> list[str]:
        if err is None or depth > 4:
            return []
        texts: list[str] = []
        if isinstance(err, (str, int, float, bool)):
            return [str(err)]
        if isinstance(err, dict):
            for _key, value in err.items():
                texts.extend(self._collect_error_texts(value, depth + 1))
            return texts
        if isinstance(err, (list, tuple)):
            for item in err:
                texts.extend(self._collect_error_texts(item, depth + 1))
            return texts

        texts.append(str(err))
        for attr in ("status_code", "code", "type", "message", "body"):
            try:
                value = getattr(err, attr, None)
            except Exception:
                value = None
            if value is not None:
                texts.extend(self._collect_error_texts(value, depth + 1))

        try:
            response = getattr(err, "response", None)
        except Exception:
            response = None
        if response is not None:
            for attr in ("status_code", "text", "content"):
                try:
                    value = getattr(response, attr, None)
                except Exception:
                    value = None
                if value is not None:
                    texts.extend(self._collect_error_texts(value, depth + 1))
            try:
                texts.extend(self._collect_error_texts(response.json(), depth + 1))
            except Exception:
                pass
        return texts

    def _is_context_overflow_error(self, err: Any) -> bool:
        texts = self._collect_error_texts(err)
        text = " ".join(str(t or "") for t in texts).lower()
        status_markers = {str(t).strip() for t in texts}
        if "413" in status_markers:
            return True
        rate_markers = [
            "rate limit",
            "rate_limit",
            "too many requests",
            "insufficient_quota",
            "quota",
            "per minute",
            "per-minute",
            "tokens per minute",
        ]
        if "429" in status_markers or any(marker in text for marker in rate_markers):
            return False
        markers = [
            "context_length_exceeded",
            "context length",
            "context size",
            "context window",
            "maximum context",
            "maximum context length",
            "token limit",
            "too many tokens",
            "input is too long",
            "exceeds the context",
            "exceeded model context",
            "prompt is too long",
            "request too large",
            "payload too large",
            "tokens exceed",
            "reduce the length",
        ]
        return any(marker in text for marker in markers)

    def _execute_react_action(
        self,
        *,
        action: str,
        action_input: Any,
        question: str,
        note_content: str,
        patient_folder: str | None,
        tool_state: dict[str, Any],
        _log: Callable,
        behavior_context: dict | None,
        round_label: str,
        round_idx: int,
        manual_stop_event: Any | None,
    ) -> tuple[str, str | None]:
        if self._stop_requested(manual_stop_event):
            return "manual_stop", None
        action_key = (action or "").strip().lower()
        if action_key == "retrieve_knowledge":
            query = self._extract_query(action_input)
            if not query:
                observation = "retrieve_knowledge: 缺少 query。"
                self._log_professor_tool_error(behavior_context, round_label, observation, round_idx)
                return observation, None
            observation = self._tool_retrieve_knowledge(
                query=query,
                question=question,
                note_content=note_content,
                tool_state=tool_state,
                _log=_log,
                behavior_context=behavior_context,
                round_label=round_label,
                round_idx=round_idx,
                manual_stop_event=manual_stop_event,
            )
            return observation, None

        if action_key == "update_graffiti_wall":
            mode, content = self._extract_graffiti_input(action_input)
            observation = self._tool_update_graffiti_wall(mode, content, behavior_context, round_label, round_idx)
            return observation, None

        if action_key == "list_patient_files":
            scope = self._extract_scope(action_input)
            listing = self._list_patient_files(patient_folder, scope)
            tool_state["patient_file_list_cache"] = listing
            self._behavior_event(
                behavior_context,
                event_type="professor_list_patient_files",
                label="教授列檔",
                title=f"{self.professor_id} {round_label} list_patient_files",
                content=listing,
                meta={"round": round_idx, "scope": scope},
            )
            return listing, None

        if action_key == "read_patient_file":
            filenames = self._extract_filenames(action_input)
            result_text, loaded_files = self._read_patient_files(patient_folder, filenames)
            tool_state["professor_loaded_files"].extend(loaded_files)
            tool_state["patient_file_list_cache"] = ""
            self._behavior_event(
                behavior_context,
                event_type="professor_read_patient_file",
                label="教授讀檔",
                title=f"{self.professor_id} {round_label} read_patient_file",
                content=result_text,
                meta={"round": round_idx, "filenames": filenames},
            )
            return result_text, None

        if action_key == "reply_to_forum":
            reply_text = self._extract_reply(action_input)
            if not reply_text.strip():
                observation = "reply_to_forum: answer 為空，未提交；請下一輪提供具體回答。"
                self._log_professor_tool_error(behavior_context, round_label, observation, round_idx)
                return observation, None
            observation = f"reply_to_forum: 已提交最終答案（{len(reply_text)} 字）。"
            self._behavior_event(
                behavior_context,
                event_type="professor_reply_to_forum",
                label="教授回覆",
                title=f"{self.professor_id} {round_label} reply_to_forum",
                content=reply_text,
                meta={"round": round_idx},
            )
            return observation, reply_text

        observation = f"未知 action: {action}。可用 action: retrieve_knowledge, update_graffiti_wall, list_patient_files, read_patient_file, reply_to_forum。"
        self._log_professor_tool_error(behavior_context, round_label, observation, round_idx)
        return observation, None

    def _tool_retrieve_knowledge(
        self,
        *,
        query: str,
        question: str,
        note_content: str,
        tool_state: dict[str, Any],
        _log: Callable,
        behavior_context: dict | None,
        round_label: str,
        round_idx: int,
        manual_stop_event: Any | None,
    ) -> str:
        if self._stop_requested(manual_stop_event):
            return "manual_stop"

        if int(tool_state.get("retrieval_count", 0) or 0) >= self.max_retrievals_per_answer:
            observation = (
                "retrieve_knowledge: 本次回答的知識庫檢索次數已達上限 "
                f"({self.max_retrievals_per_answer})，請使用既有資訊回答，或改用 list/read patient files 補足患者資料。"
            )
            self._behavior_event(
                behavior_context,
                event_type="professor_retrieve_knowledge",
                label="教授檢索上限",
                title=f"{self.professor_id} {round_label} retrieve_knowledge 達上限",
                content=observation,
                severity="warning",
                meta={"round": round_idx, "query": query, "max_retrievals": self.max_retrievals_per_answer},
            )
            return observation

        tool_state["retrieval_count"] = int(tool_state.get("retrieval_count", 0) or 0) + 1
        self._ensure_index()
        if self._all_emb is None or len(self._texts) == 0:
            tool_state["last_retrieval_query"] = query
            tool_state["last_retrieval_prefixes"] = []
            tool_state["retrieved_context"] = ""
            tool_state["retrieval_records"].append({
                "round": round_idx,
                "query": query,
                "prefixes": [],
                "retr_doc": "",
                "source_summary": [],
                "error": "向量索引未載入或資料庫為空",
            })
            observation = "retrieve_knowledge: 向量索引未載入或資料庫為空，無檢索結果。"
            self._log_professor_tool_error(behavior_context, round_label, observation, round_idx)
            return observation

        prefixes = self._classify_prefixes(query, _log, behavior_context)
        q_txt_origin = f"【今日病歷】\n{note_content}\n\n【提問】\n{question}" if note_content else question
        if self._stop_requested(manual_stop_event):
            return "manual_stop"
        retr_doc = self._retrieve(q_txt_origin, query, prefixes, _log, behavior_context, manual_stop_event=manual_stop_event)
        if self._stop_requested(manual_stop_event):
            return "manual_stop"
        source_summary = self._extract_retrieval_sources(retr_doc)
        tool_state["last_retrieval_query"] = query
        tool_state["last_retrieval_prefixes"] = prefixes
        tool_state["retrieved_context"] = retr_doc
        tool_state["retrieval_records"].append({
            "round": round_idx,
            "query": query,
            "prefixes": prefixes,
            "retr_doc": retr_doc,
            "source_summary": source_summary,
            "error": None,
        })
        observation = (
            f"retrieve_knowledge: 檢索完成，query={query}\n"
            f"prefixes={', '.join(prefixes) if prefixes else '（無）'}\n"
            f"retrieved_chars={len(retr_doc or '')}\n"
            f"sources={', '.join(source_summary) if source_summary else '（無）'}\n"
            "檢索原文已更新至 user prompt 最下方的【知識庫檢索結果】，不寫入長期 ReAct 紀錄。"
        )
        self._behavior_event(
            behavior_context,
            event_type="professor_retrieve_knowledge",
            label="教授檢索",
            title=f"{self.professor_id} {round_label} retrieve_knowledge",
            content=observation,
            meta={"round": round_idx, "query": query, "prefixes": prefixes},
        )
        return observation

    def _tool_update_graffiti_wall(
        self,
        mode: str,
        content: str,
        behavior_context: dict | None,
        round_label: str,
        round_idx: int,
    ) -> str:
        mode_key = self._normalize_graffiti_mode(mode)
        if mode_key == "summarize":
            self.graffiti_wall = content.strip()
        else:
            if self.graffiti_wall.strip():
                self.graffiti_wall = f"{self.graffiti_wall.rstrip()}\n\n{content.strip()}"
            else:
                self.graffiti_wall = content.strip()
        observation = (
            f"update_graffiti_wall: 已以 {mode_key} 模式更新塗鴉牆"
            f"（目前 {len(self.graffiti_wall)} 字；"
            f"超過 {self.graffiti_summarize_threshold} 字時請使用 summarize 精簡整理）。"
        )
        self._behavior_event(
            behavior_context,
            event_type="professor_update_graffiti_wall",
            label="教授塗鴉牆",
            title=f"{self.professor_id} {round_label} update_graffiti_wall",
            content=f"{observation}\n\n## 目前塗鴉牆\n{self.graffiti_wall or '（空白）'}",
            meta={"round": round_idx, "mode": mode_key},
        )
        return observation

    def _extract_retrieval_sources(self, retr_doc: str) -> list[str]:
        if not retr_doc:
            return []
        sources: list[str] = []
        for match in re.finditer(r"\(來源:(.+?)\)(?=\n\n|\Z)", retr_doc, re.S):
            source = match.group(1).strip()
            if source and source not in sources:
                sources.append(source)
        return sources

    def _log_professor_tool_error(
        self,
        behavior_context: dict | None,
        round_label: str,
        observation: str,
        round_idx: int,
    ) -> None:
        self._behavior_event(
            behavior_context,
            event_type="professor_tool_error",
            label="教授工具錯誤",
            title=f"{self.professor_id} {round_label} tool error",
            content=observation,
            severity="warning",
            meta={"round": round_idx},
        )

    def _force_reply_to_forum(
        self,
        *,
        question: str,
        note_content: str,
        at_content: str,
        forum_history_text: str,
        show_forum_history: bool,
        main_loaded_files_block: str,
        professor_loaded_files: list[dict],
        patient_file_list_cache: str,
        retrieved_context: str,
        image_files: list[dict],
        system_prompt: str,
        _log: Callable,
        behavior_context: dict | None,
        manual_stop_event: Any | None,
    ) -> tuple[str, str | None, bool]:
        if self._stop_requested(manual_stop_event):
            self._react_end_error(behavior_context, "教授 ReAct 已收到手動中斷。", meta={"manual_stop": True, "forced": True})
            return "⚠️ 教授 ReAct 已手動中斷。", "manual_stop", True

        react_history_text = self._format_react_history(limit=None)
        react_history_chars = len(react_history_text)
        react_history_limit = self._react_history_prompt_limit(react_history_chars)
        if react_history_limit is not None:
            self.last_react_history_truncated = True
            self._log_react_history_truncated(
                behavior_context,
                round_label=f"P-{self.professor_id}-force-reply",
                round_meta={"forced": True},
                react_history_chars=react_history_chars,
            )

        user_prompt = self._build_react_user_prompt(
            question=question,
            note_content=note_content,
            at_content=at_content,
            forum_history_text=forum_history_text,
            show_forum_history=show_forum_history,
            main_loaded_files_block=main_loaded_files_block,
            professor_loaded_files=professor_loaded_files,
            patient_file_list_cache=patient_file_list_cache,
            retrieved_context=retrieved_context,
            react_history_limit=react_history_limit,
            react_history_text_override=(
                self._truncate_react_history_text(react_history_text, react_history_limit)
                if react_history_limit is not None
                else react_history_text
            ),
            overflow_notice=react_history_limit is not None,
            final_round_notice="已達 ReAct 輪數上限。現在必須使用 reply_to_forum，整理目前可用資訊、標示不確定處，交付最終答案。",
        )
        force_round_label = f"P-{self.professor_id}-force-reply"
        self._emit_react_llm_input(
            user_prompt=user_prompt,
            round_label=force_round_label,
            round_meta={"forced": True, "react_history_chars": react_history_chars},
            _log=_log,
            behavior_context=behavior_context,
        )
        parsed, error, context_overflow = self._call_react_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_files=image_files,
            round_label=force_round_label,
            _log=_log,
            behavior_context=behavior_context,
            manual_stop_event=manual_stop_event,
        )
        if context_overflow:
            self.last_context_overflow = True
            error_msg = (
                f"教授 Answer LLM context overflow：react_history 已按 {self.react_history_prompt_chars} 字上限截斷，"
                "但強制回覆 prompt 仍超過模型可接受大小。請減少討論區/讀檔/圖片內容後重試。"
            )
            meta = {
                "forced": True,
                "error": error or "",
                "react_history_chars": react_history_chars,
                "react_history_truncated": self.last_react_history_truncated,
                "context_overflow": True,
            }
            self._behavior_event(
                behavior_context,
                event_type="professor_context_overflow",
                label="Context溢出",
                title=f"{self.professor_id} force reply context overflow fail-closed",
                content=error_msg,
                severity="error",
                meta=meta,
            )
            self._react_end_error(behavior_context, error_msg, meta=meta)
            return f"⚠️ {error_msg}", error_msg, True

        if parsed and str(parsed.get("action", "")).strip().lower() == "reply_to_forum":
            thought_summary = self._sanitize_thought_summary(parsed.get("thought_summary", ""))
            action_input = parsed.get("action_input", "")
            reply_text = self._extract_reply(action_input).strip()
            if not reply_text:
                error_msg = "⚠️ 教授 ReAct 已達輪數上限，且強制 reply_to_forum 回傳空 answer。"
                self.react_history.append({
                    "ts": datetime_now_text(),
                    "round": "force",
                    "thought_summary": thought_summary,
                    "action": "reply_to_forum",
                    "action_input": self._sanitize_action_input("reply_to_forum", action_input),
                    "observation": "reply_to_forum: answer 為空，強制回覆失敗。",
                })
                self._behavior_event(
                    behavior_context,
                    event_type="professor_tool_error",
                    label="教授錯誤",
                    title=f"{self.professor_id} force reply empty",
                    content=error_msg,
                    severity="error",
                    meta={"forced": True},
                )
                self._react_end_error(behavior_context, error_msg, meta={"forced": True})
                return error_msg, error_msg, True
            self.react_history.append({
                "ts": datetime_now_text(),
                "round": "force",
                "thought_summary": thought_summary,
                "action": "reply_to_forum",
                "action_input": self._sanitize_action_input("reply_to_forum", action_input),
                "observation": f"reply_to_forum: 已在達上限後提交最終答案（{len(reply_text)} 字）。",
            })
            self._behavior_event(
                behavior_context,
                event_type="professor_reply_to_forum",
                label="教授回覆",
                title=f"{self.professor_id} force reply_to_forum",
                content=reply_text,
                meta={"forced": True},
            )
            self._behavior_event(
                behavior_context,
                event_type="professor_react_end",
                label="教授結束",
                title=f"{self.professor_id} ReAct 強制結束",
                content=reply_text,
                meta={
                    "forced": True,
                    "react_history_truncated": self.last_react_history_truncated,
                    "context_overflow": self.last_context_overflow,
                },
            )
            return reply_text, None, True

        if error == "manual_stop":
            stop_msg = "⚠️ 教授 ReAct 已手動中斷。"
            self._react_end_error(
                behavior_context,
                "教授 ReAct 已收到手動中斷。",
                meta={"manual_stop": True, "forced": True},
            )
            return stop_msg, "manual_stop", True

        fallback = f"⚠️ 教授 ReAct 已達輪數上限，且強制 reply_to_forum 失敗：{error or '未輸出 reply_to_forum'}"
        self._behavior_event(
            behavior_context,
            event_type="professor_tool_error",
            label="教授錯誤",
            title=f"{self.professor_id} force reply failed",
            content=fallback,
            severity="error",
        )
        self._react_end_error(behavior_context, fallback, meta={"forced": True})
        return fallback, fallback, True

    def _extract_query(self, action_input: Any) -> str:
        if isinstance(action_input, dict):
            return str(action_input.get("query", "") or "").strip()
        return str(action_input or "").strip()

    def _extract_graffiti_input(self, action_input: Any) -> tuple[str, str]:
        if isinstance(action_input, dict):
            return (
                str(action_input.get("mode", "append") or "append"),
                str(action_input.get("content", "") or ""),
            )
        return "append", str(action_input or "")

    def _extract_scope(self, action_input: Any) -> str:
        if isinstance(action_input, dict):
            return str(action_input.get("scope", "all") or "all").strip()
        value = str(action_input or "all").strip()
        return value or "all"

    def _extract_filenames(self, action_input: Any) -> list[str]:
        raw: Any = action_input
        if isinstance(action_input, dict):
            raw = action_input.get("filenames", action_input.get("filename", action_input.get("files", [])))
        if isinstance(raw, list):
            return [str(x).strip() for x in raw if str(x).strip()]
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass
            return [text]
        return []

    def _extract_reply(self, action_input: Any) -> str:
        if isinstance(action_input, dict):
            return str(action_input.get("answer", action_input.get("content", action_input.get("response", ""))) or "")
        return str(action_input or "")

    def _safe_patient_path(self, patient_folder: str | None) -> Path | None:
        if not patient_folder:
            return None
        try:
            pf = Path(patient_folder).resolve()
        except Exception:
            return None
        if not pf.is_dir():
            return None
        return pf

    def _list_patient_files(self, patient_folder: str | None, scope: str = "all") -> str:
        pf = self._safe_patient_path(patient_folder)
        if pf is None:
            return "list_patient_files: 尚未選取患者或患者資料夾不存在。"

        def _compact_summary(text: str, limit: int = 50) -> str:
            compact = " ".join((text or "").split())
            return compact if len(compact) <= limit else compact[:limit]

        patient_sessions = {}
        patient_info_file = pf / "patient_info.json"
        if patient_info_file.is_file():
            try:
                patient_sessions = json.loads(patient_info_file.read_text(encoding="utf-8")).get("sessions", {})
            except Exception:
                patient_sessions = {}

        def _record_summary_for(filename: str) -> str:
            for _date, session in patient_sessions.items():
                if filename == session.get("note_file", ""):
                    summary = session.get("note_summary", "")
                    if summary:
                        return _compact_summary(summary)
                if filename == session.get("assessment_treatment_file", ""):
                    summary = session.get("assessment_treatment_summary", "")
                    if summary:
                        return _compact_summary(summary)
            record_path = pf / filename
            try:
                return _compact_summary(record_path.read_text(encoding="utf-8"))
            except Exception:
                return ""

        folder_key = (scope or "all").lower().replace(" ", "_")
        result_lines: list[str] = []
        folders_to_scan: list[tuple[str, Path]] = []
        if folder_key in ("all", "picture_row", ""):
            folders_to_scan.append(("Picture_Row", pf / "Picture_Row"))
        if folder_key in ("all", "medical_information", ""):
            folders_to_scan.append(("Medical_information", pf / "Medical_information"))
        include_records = folder_key in ("all", "medical_records", "records", "record", "")

        for folder_label, folder_path in folders_to_scan:
            if not folder_path.is_dir():
                result_lines.append(f"[{folder_label}] 資料夾不存在")
                continue
            files = sorted(x.name for x in folder_path.iterdir() if x.is_file())
            if not files:
                result_lines.append(f"[{folder_label}] （空）")
            else:
                result_lines.append(f"[{folder_label}] 共 {len(files)} 個檔案：")
                for fn in files:
                    tag = "🖼️" if Path(fn).suffix.lower() in IMAGE_EXTENSIONS else "📄"
                    result_lines.append(f"  {tag} {fn}")

        if include_records:
            md_files = sorted(x.name for x in pf.iterdir() if x.is_file() and x.name.lower().endswith(".md"))
            if not md_files:
                result_lines.append("[Medical_Records] （空）")
            else:
                result_lines.append(f"[Medical_Records] 共 {len(md_files)} 個病歷檔")
                for fn in md_files:
                    summary = _record_summary_for(fn)
                    suffix = f"（{summary}）" if summary else "（空白）"
                    result_lines.append(f"  📄 {fn}{suffix}")

        return "\n".join(result_lines) or "list_patient_files: 無可列出的檔案。"

    def _read_patient_files(self, patient_folder: str | None, filenames: list[str]) -> tuple[str, list[dict]]:
        pf = self._safe_patient_path(patient_folder)
        if pf is None:
            return "read_patient_file: 尚未選取患者或患者資料夾不存在。", []
        if not filenames:
            return "read_patient_file: 未指定檔名。", []

        results: list[str] = []
        loaded_files: list[dict] = []
        for raw_name in filenames:
            fn = Path(str(raw_name).strip()).name
            if not fn:
                continue
            found_path: Path | None = None
            for sub in ["Picture_Row", "Medical_information", ""]:
                if not sub and not fn.lower().endswith(".md"):
                    continue
                candidate = (pf / sub / fn) if sub else (pf / fn)
                try:
                    resolved = candidate.resolve()
                    resolved.relative_to(pf)
                except Exception:
                    continue
                if resolved.is_file():
                    found_path = resolved
                    break

            if found_path is None:
                results.append(f"❌ 找不到檔案: {fn}")
                continue

            ext = found_path.suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                loaded_files.append({
                    "name": fn,
                    "type": "image",
                    "path": str(found_path),
                })
                results.append(f"🖼️ {fn}: 圖片已載入醫學教授 subagent 暫存區")
            else:
                try:
                    content = found_path.read_text(encoding="utf-8")
                except Exception as e:
                    results.append(f"❌ {fn}: 讀取失敗 ({e})")
                    continue
                loaded_files.append({
                    "name": fn,
                    "type": "text",
                    "content": content,
                })
                results.append(f"📄 {fn}: 已讀取 ({len(content)} 字)，內容見【醫學教授 subagent 讀取後暫存區】")

        return "\n".join(results), loaded_files

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
            self._behavior_event(
                behavior_context,
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
            self._behavior_event(
                behavior_context,
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
        manual_stop_event: Any | None = None,
    ) -> str:
        """完整的檢索管線，回傳檢索結果文字"""
        if self._stop_requested(manual_stop_event):
            return ""

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
        scored_parents = self._llm_rerank(q_txt_expand, parent_docs_with_dense, _log, behavior_context, manual_stop_event)

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

        rerank_content = "\n".join(rerank_lines) or "（無 Rerank 結果）"
        if rerank_lines:
            rerank_content = f"```text\n{rerank_content}\n```"
        self._behavior_event(
            behavior_context,
            event_type="tool_event",
            label="Rerank結果",
            title=f"{self.professor_id} Rerank 統合結果",
            content=rerank_content,
        )

        top_parents = [doc for _, _, doc in selected]
        retr_doc = "\n\n".join(
            [d.page_content + f" (來源:{d.metadata['source']})" for d in top_parents]
        )
        self._behavior_event(
            behavior_context,
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
        manual_stop_event: Any | None = None,
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
            if self._stop_requested(manual_stop_event):
                _log(f"[Professor {self.professor_id}] [LLM Rerank] 收到手動中斷")
                break
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

# ════════════════════════════════════════════════════════════════
# 工具函數
# ════════════════════════════════════════════════════════════════

def load_all_professors() -> List[dict]:
    """
    掃描專案資料夾下所有 professor_xx 目錄，回傳教授清單。

    Returns:
        [{"id": "professor_01", "name": "學院派教授", "description": "...", "role_style": "..."}]
    """
    result = []
    for d in sorted(_PROJECT_DIR.glob("professor_*")):
        if not d.is_dir():
            continue
        prof_id = d.name
        desc_path = d / "Description.txt"
        name = ""
        description = ""
        role_style = ""
        if desc_path.exists():
            try:
                desc = json.loads(desc_path.read_text(encoding="utf-8"))
                name = desc.get("name", "")
                description = desc.get("description", "")
                role_style = desc.get("role_style", "")
            except Exception:
                pass
        result.append({
            "id": prof_id,
            "name": name,
            "description": description,
            "role_style": role_style,
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
        "prompt_rerank.txt",
        "Description.txt",
    ]
    optional_prompt = [
        "prompt_query_expansion.txt",
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

    for item in optional_prompt:
        path = prof_dir / item
        if path.exists() and path.stat().st_size > 0:
            existing.append(item + "（相容舊版/遷移參考；新版不直接呼叫）")

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
