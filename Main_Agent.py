"""
Main_Agent.py - AI 主治醫師 Agent
採用 ReAct (Reasoning and Acting) 迴圈架構。
每次使用者提問為一個「主輪 (Main Turn)」，其中可包含多個「子輪 (Sub-turn)」。
"""
from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime
from typing import Any, Callable, Optional

from openai import OpenAI
from multimodal_utils import inject_images_into_messages
from agent_behavior_log import append_behavior_event
from deidentification_utils import format_patient_basic_info_for_llm

from Record_Subagent import RecordSubagent
from Low_Confidence_Subagent import LowConfidenceSubagent
from Note_Review_Subagent import NoteReviewSubagent
from Professor import ProfessorInstance, load_all_professors

# ════════════════════════════════════════════════════════════════
# Prompt 載入
# ════════════════════════════════════════════════════════════════
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_prompt(filename: str) -> str:
    path = os.path.join(CURRENT_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ════════════════════════════════════════════════════════════════
# Main Agent 類別
# ════════════════════════════════════════════════════════════════

class MainAgent:
    """
    AI 主治醫師 Agent。

    使用方式：
        agent = MainAgent(main_config, record_config)
        result = agent.process_message(
            user_message="請幫我更新 NOTE...",
            note_content="...",
            at_content="...",
            patient_info={...},
            session_date="2026-06-01",
        )
    """

    MAX_SUB_TURNS = 10
    MAX_JSON_RETRIES = 3

    def __init__(
        self,
        main_config: dict,
        record_config: dict,
        hallucination_config: dict | None = None,
        ic_config: dict | None = None,
        lc_config: dict | None = None,
        nr_config: dict | None = None,
        professor_config: dict | None = None,
    ):
        """
        Args:
            main_config: {
                "api_url": "http://localhost:1234/v1",
                "api_key": "lm-studio",
                "model_name": "model-name"
            }
            record_config: 同上，給 Record Subagent 用
            hallucination_config: 同上，給 Hallucination Subagent 用。
                額外支援 "detection_strength" 和 "max_review_rounds"。
                若為 None 則跳過幻覺審查。
            ic_config: 同上，給 Information Collection Subagent 用。
                額外支援 "max_collection_rounds"。
                若為 None 則無法啟動問診助理。
            lc_config: 同上，給 Low Confidence Subagent 用。
                額外支援 "max_scan_rounds" 和 "detection_strength"。
                若為 None 則無法執行低信心標註。
            nr_config: 同上，給 Note Review Subagent 用。
                若為 None 則無法執行病歷完整性掃描。
            professor_config: 教授共用模型設定（answer/embedding/query_expansion/prefix/rerank）。
                若為 None 則無法呼叫教授。
        """
        self.client = OpenAI(
            base_url=main_config.get("api_url", "http://localhost:1234/v1"),
            api_key=main_config.get("api_key", "lm-studio"),
        )
        self.model = main_config.get("model_name", "")
        self.max_tokens = int(main_config.get("max_tokens", 4000))
        self.temperature = float(main_config.get("temperature", 0.7))
        self.MAX_SUB_TURNS = int(main_config.get("max_sub_turns", 10))
        self.system_prompt_template = _load_prompt("prompt_main_agent.txt")

        # 載入標準病歷模板
        template_path = os.path.join(CURRENT_DIR, "Record_Template.txt")
        if os.path.isfile(template_path):
            with open(template_path, "r", encoding="utf-8") as f:
                self.record_template = f.read()
        else:
            self.record_template = "（尚未設定標準病歷模板）"

        # 建立 Record Subagent（內含 Hallucination Reviewer）
        self.record_subagent = RecordSubagent(record_config, hallucination_config)

        # 儲存 IC config（給 UI 建立 IC Subagent 使用）
        self.ic_config = ic_config or {}

        # 建立 Low Confidence Subagent
        if lc_config and lc_config.get("model_name"):
            self.lc_subagent = LowConfidenceSubagent(lc_config)
        else:
            self.lc_subagent = None

        # 建立 Note Review Subagent（病歷檢查員）
        if nr_config and nr_config.get("model_name"):
            self.nr_subagent = NoteReviewSubagent(nr_config)
        else:
            self.nr_subagent = None

        # 教授模組（惰性載入）
        self.professor_config = professor_config or {}
        self._professor_instances: dict[str, ProfessorInstance] = {}

        # 醫療問答討論區歷史
        self.forum_history: list[dict] = []

        # 主輪計數器
        self.turn_count = 0

        # 對話歷史（主輪級別，用於上下文累積）
        self.conversation_history: list[dict[str, str]] = []

        # 跨主輪行為紀錄（累積所有已完成主輪的行動，含人類醫師提問與 AI 回覆）
        self.turn_history: list[dict] = []
        # 格式: {"turn": int, "user_message": str, "sub_steps": [{thinking, action_summary, next_step, subagent_result?}], "reply": str}

        # 暫停狀態（IC Subagent 暫停/恢復用）
        self._suspended: dict | None = None

        # 患者檔案查詢狀態
        self._file_list_cache: str | None = None       # list_patient_files 的結果文字
        self._loaded_files: list[dict] = []             # 當輪讀取的檔案 [{name, content}]
        self._patient_folder: str | None = None         # 患者資料夾路徑

        # 即時步驟（供 UI polling 讀取，背景線程 append，主線程 read）
        self._live_steps: list[dict] = []
        self._active_turn: dict | None = None
        self._manual_stop_event = threading.Event()
        self._current_step_snapshot: dict | None = None
        self._stop_snapshot: dict | None = None

    # ── RAG 行為 Log ──

    def _write_rag_log(
        self,
        session_date: str,
        professor_id: str,
        professor_name: str,
        question: str,
        q_expand: str,
        prefixes: list[str],
        retr_doc: str,
        response: str,
    ):
        """將教授 RAG 完整行為寫入獨立 log 檔"""
        pf = self._patient_folder
        if not pf or not session_date:
            return
        log_dir = os.path.join(pf, "log", f"{session_date}-log")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{session_date}-RAG-full-behavior.txt")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        separator = "=" * 80

        entry_lines = [
            separator,
            f"[{timestamp}] 教授諮詢 RAG 紀錄  |  {professor_id} ({professor_name})",
            separator,
            "",
            "## 【主 Agent 的提問】",
            question,
            "",
            "## 【Query Expansion 擴展結果】",
            q_expand,
            "",
            "## 【三前綴分類結果】",
            ", ".join(prefixes) if prefixes else "（無）",
            "",
            "## 【RAG 檢索引入的資料庫原文】",
            retr_doc if retr_doc else "（無檢索結果 / NoRAG）",
            "",
            "## 【教授的回答】",
            response,
            "",
            "",
        ]

        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n".join(entry_lines))

    # ── 狀態序列化（Session 持久化用）──

    def export_state(self) -> dict:
        """匯出 Agent 內部狀態，供存檔使用"""
        return {
            "turn_count": self.turn_count,
            "turn_history": self.turn_history,
            "conversation_history": self.conversation_history,
            "_suspended": self._suspended,
            "forum_history": self.forum_history,
            "_file_list_cache": self._file_list_cache,
            "_patient_folder": self._patient_folder,
            "_active_turn": self._active_turn,
        }

    def restore_state(self, state: dict):
        """從存檔還原 Agent 內部狀態"""
        self.turn_count = state.get("turn_count", 0)
        self.turn_history = state.get("turn_history", [])
        self.conversation_history = state.get("conversation_history", [])
        self._suspended = state.get("_suspended", None)
        self.forum_history = state.get("forum_history", [])
        self._file_list_cache = state.get("_file_list_cache", None)
        self._patient_folder = state.get("_patient_folder", self._patient_folder)
        self._active_turn = state.get("_active_turn", None)
        self._manual_stop_event.clear()
        self._current_step_snapshot = None
        self._stop_snapshot = None

    def request_manual_stop(self) -> bool:
        """請求合作式中斷目前主 Agent 執行。"""
        if not self._active_turn and not self._suspended:
            return False

        snapshot = self._current_step_snapshot or {}
        active = self._active_turn or {}
        self._stop_snapshot = {
            "turn": snapshot.get("turn", active.get("turn", self.turn_count)),
            "user_message": snapshot.get("user_message", active.get("user_message", "")),
            "steps_len": snapshot.get("steps_len", len(self._live_steps)),
            "note": snapshot.get("note", active.get("note", "")),
            "at": snapshot.get("at", active.get("at", "")),
            "forum_len": snapshot.get("forum_len", len(self.forum_history)),
            "discarding_step": snapshot.get("discarding_step", ""),
        }
        self._manual_stop_event.set()
        return True

    def _finalize_turn_history(
        self,
        *,
        turn_num: int,
        user_message: str,
        steps: list[dict],
        reply_text: str,
    ):
        """將一個主輪收斂為 turn_history，供正常或異常結束共用。"""
        self.conversation_history.append({
            "role": "assistant",
            "content": reply_text,
        })

        sub_steps = self._build_sub_steps_summary(steps)
        self.turn_history.append({
            "turn": turn_num,
            "user_message": user_message,
            "sub_steps": sub_steps,
            "reply": reply_text,
        })
        self._active_turn = None

    def finalize_suspended_turn(
        self,
        *,
        reply_text: str,
        interrupted_step_result: str = "",
        note_content: str = "",
        at_content: str = "",
    ) -> dict[str, Any]:
        """將等待問診的暫停主輪以系統合成 reply 收斂，供中斷/異常時保存進度。"""
        if not self._suspended:
            return {
                "reply": reply_text,
                "steps": [],
                "note": note_content,
                "at": at_content,
                "note_changed": False,
                "at_changed": False,
                "turn_number": self.turn_count,
                "waiting_for_interview": False,
                "interview_guidelines": "",
                "error": "No suspended state",
            }

        s = self._suspended
        self._suspended = None
        steps = s.get("steps", [])

        if steps and interrupted_step_result:
            steps[-1]["result"] = interrupted_step_result
            steps[-1].pop("subagent_result", None)
            self._active_turn = {
                "turn": s["turn_num"],
                "user_message": s["user_message"],
                "steps": steps,
                "status": "interrupted",
            }

        self._finalize_turn_history(
            turn_num=s["turn_num"],
            user_message=s["user_message"],
            steps=steps,
            reply_text=reply_text,
        )

        return {
            "reply": reply_text,
            "steps": steps,
            "note": note_content,
            "at": at_content,
            "note_changed": note_content != s.get("original_note", note_content),
            "at_changed": at_content != s.get("original_at", at_content),
            "turn_number": s["turn_num"],
            "waiting_for_interview": False,
            "interview_guidelines": "",
            "error": reply_text,
        }

    def _finalize_manual_stop_turn(
        self,
        *,
        session_date: str,
        turn_num: int,
        user_message: str,
        steps: list[dict],
        current_note: str,
        current_at: str,
        original_note: str,
        original_at: str,
        on_step: Optional[Callable[[dict], None]] = None,
    ) -> dict[str, Any]:
        """以手動中斷方式收斂主輪，保留停止前已完成的子輪。"""
        reply_text = "服務已中斷(使用者手動停止)"
        snapshot = self._stop_snapshot or {}
        keep_len = snapshot.get("steps_len", len(steps))
        try:
            keep_len = max(0, min(int(keep_len), len(steps)))
        except Exception:
            keep_len = len(steps)

        if keep_len < len(steps):
            del steps[keep_len:]

        forum_len = snapshot.get("forum_len", len(self.forum_history))
        try:
            forum_len = max(0, min(int(forum_len), len(self.forum_history)))
        except Exception:
            forum_len = len(self.forum_history)
        if forum_len < len(self.forum_history):
            del self.forum_history[forum_len:]

        final_note = snapshot["note"] if "note" in snapshot else current_note
        final_at = snapshot["at"] if "at" in snapshot else current_at
        discarding_step = snapshot.get("discarding_step") or "目前執行中的子輪"
        content = (
            f"主 Agent 被使用者手動中斷。\n\n"
            f"- 保留已完成子輪數：{len(steps)}\n"
            f"- 丟棄/未採用內容：{discarding_step}\n"
            f"- 最終回覆：{reply_text}"
        )
        self._behavior_event(
            session_date,
            agent="main_agent",
            event_type="manual_stop",
            label="主Agent中斷",
            title="AI主治醫師被使用者手動中斷",
            content=content,
            turn=turn_num,
            severity="warning",
        )
        self._finalize_turn_history(
            turn_num=turn_num,
            user_message=user_message,
            steps=steps,
            reply_text=reply_text,
        )
        self._manual_stop_event.clear()
        self._current_step_snapshot = None
        self._stop_snapshot = None

        if on_step:
            on_step({
                "step_label": f"Turn {turn_num}-STOP",
                "thinking": "",
                "action": "manual_stop",
                "action_input": "",
                "next_step": "",
                "timestamp": datetime.now().isoformat(),
                "result": reply_text,
            })

        return {
            "reply": reply_text,
            "steps": steps,
            "note": final_note,
            "at": final_at,
            "note_changed": final_note != original_note,
            "at_changed": final_at != original_at,
            "turn_number": turn_num,
            "waiting_for_interview": False,
            "interview_guidelines": "",
            "error": reply_text,
            "manual_stopped": True,
        }

    def _behavior_event(
        self,
        session_date: str | None,
        *,
        agent: str,
        event_type: str,
        label: str,
        title: str,
        content: str,
        content_type: str = "markdown",
        turn: int | None = None,
        sub_turn: str | None = None,
        severity: str = "normal",
        target_agent: str | None = None,
        meta: dict[str, Any] | None = None,
    ):
        append_behavior_event(
            self._patient_folder,
            session_date,
            agent=agent,
            event_type=event_type,
            label=label,
            title=title,
            content=content,
            content_type=content_type,
            turn=turn,
            sub_turn=sub_turn,
            severity=severity,
            target_agent=target_agent,
            meta=meta,
        )

    def process_message(
        self,
        user_message: str,
        note_content: str,
        at_content: str,
        patient_folder: str = "",
        patient_info: Optional[dict] = None,
        session_date: str = "",
        conversation_history: str = "",
        last_visit_block: str = "",
        history_summary: str = "",
        interview_dialogue: str = "",
        on_step: Optional[Callable[[dict], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> dict[str, Any]:
        """
        處理一個主輪的使用者訊息。

        Returns:
            {
                "reply": str,              # 最終回覆（僅當 reply 動作時）
                "steps": list[dict],       # 所有子輪的詳細記錄
                "note": str,               # 最終 NOTE 內容
                "at": str,                 # 最終 A&T 內容
                "note_changed": bool,
                "at_changed": bool,
                "turn_number": int,
                "waiting_for_interview": bool,  # 是否暫停等待 IC
                "interview_guidelines": str,    # IC 方針（僅當暫停時）
                "error": str | None,
            }
        """

        def _log(msg: str):
            if log_callback:
                log_callback(msg)
            print(msg)

        if self._active_turn:
            active_turn = self._active_turn
            active_turn_num = active_turn.get("turn")
            already_finalized = any(
                th.get("turn") == active_turn_num for th in self.turn_history
            )
            if not already_finalized:
                self._finalize_turn_history(
                    turn_num=active_turn_num,
                    user_message=active_turn.get("user_message", ""),
                    steps=active_turn.get("steps", []),
                    reply_text="服務已中斷(模型呼叫失敗)",
                )
            else:
                self._active_turn = None
        if self._suspended:
            _log("[Main Agent] 清除孤兒問診暫停狀態")
            self._suspended = None

        self._manual_stop_event.clear()
        self._current_step_snapshot = None
        self._stop_snapshot = None

        self.turn_count += 1
        turn_num = self.turn_count
        _log(f"\n{'='*60}")
        _log(f"[Main Agent] === Turn {turn_num} 開始 ===")
        _log(f"[Main Agent] 使用者訊息:")
        _log(user_message)
        _log(f"{'='*60}\n")

        # 每個主輪開始時：清空當輪暫存檔案
        self._loaded_files.clear()
        self._file_list_cache = None
        self._patient_folder = patient_folder or self._patient_folder

        # 記錄使用者訊息到對話歷史
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        return self._run_react_loop(
            user_message=user_message,
            note_content=note_content,
            at_content=at_content,
            patient_info=patient_info,
            session_date=session_date,
            conversation_history=conversation_history,
            last_visit_block=last_visit_block,
            history_summary=history_summary,
            interview_dialogue=interview_dialogue,
            steps=[],
            sub_turn_start=0,
            turn_num=turn_num,
            original_note=note_content,
            original_at=at_content,
            on_step=on_step,
            log_callback=log_callback,
        )

    def continue_after_interview(
        self,
        interview_dialogue: str,
        interview_summary: str,
        note_content: str,
        at_content: str,
        conversation_history: str = "",
        on_step: Optional[Callable[[dict], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> dict[str, Any]:
        """
        IC Subagent 完成後，注入結果並繼續 ReAct 迴圈。

        Args:
            interview_dialogue: 完整問診對話紀錄
            interview_summary: 精簡摘要（如「問診助理完成 R1~R3 共3回合問診」）
            note_content: 當前最新 NOTE（可能在 IC 期間被手動修改）
            at_content: 當前最新 A&T
            conversation_history: 更新後的對話歷史文字
        """
        if not self._suspended:
            return {
                "reply": "⚠️ 系統錯誤：無暫停狀態可恢復",
                "steps": [],
                "note": note_content,
                "at": at_content,
                "note_changed": False,
                "at_changed": False,
                "turn_number": self.turn_count,
                "waiting_for_interview": False,
                "interview_guidelines": "",
                "error": "No suspended state",
            }

        s = self._suspended
        self._suspended = None

        def _log(msg: str):
            if log_callback:
                log_callback(msg)
            print(msg)

        _log(f"\n{'='*60}")
        _log(f"[Main Agent] === 問診助理完成，恢復 Turn {s['turn_num']} 迴圈 ===")
        _log(f"[Main Agent] 問診摘要: {interview_summary}")
        _log(f"{'='*60}\n")

        # Update the last step (the IC step) with the result summary
        steps = s["steps"]
        if steps:
            steps[-1]["result"] += f" → {interview_summary}"
            steps[-1]["subagent_result"] = interview_summary

        return self._run_react_loop(
            user_message=s["user_message"],
            note_content=note_content,
            at_content=at_content,
            patient_info=s["patient_info"],
            session_date=s["session_date"],
            conversation_history=conversation_history or s["conversation_history"],
            last_visit_block=s["last_visit_block"],
            history_summary=s["history_summary"],
            interview_dialogue=interview_dialogue,
            steps=steps,
            sub_turn_start=s["sub_turn"],
            turn_num=s["turn_num"],
            original_note=s["original_note"],
            original_at=s["original_at"],
            on_step=on_step,
            log_callback=log_callback,
        )

    # ────────────────────────────────────────────────────────────
    # 核心 ReAct 迴圈（process_message 和 continue_after_interview 共用）
    # ────────────────────────────────────────────────────────────

    def _run_react_loop(
        self,
        *,
        user_message: str,
        note_content: str,
        at_content: str,
        patient_info: Optional[dict],
        session_date: str,
        conversation_history: str,
        last_visit_block: str,
        history_summary: str,
        interview_dialogue: str,
        steps: list[dict],
        sub_turn_start: int,
        turn_num: int,
        original_note: str,
        original_at: str,
        on_step: Optional[Callable[[dict], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> dict[str, Any]:
        """核心 ReAct 迴圈，可從頭開始或從暫停處恢復。"""

        def _log(msg: str):
            if log_callback:
                log_callback(msg)
            print(msg)

        current_note = note_content
        current_at = at_content
        sub_turn = sub_turn_start
        record_snapshots: list[dict[str, Any]] = []

        # 暴露 steps 引用供 UI polling
        self._live_steps = steps
        self._active_turn = {
            "turn": turn_num,
            "user_message": user_message,
            "steps": steps,
            "status": "running",
            "note": current_note,
            "at": current_at,
        }

        def _mark_progress(discarding_step: str = ""):
            self._current_step_snapshot = {
                "turn": turn_num,
                "user_message": user_message,
                "steps_len": len(steps),
                "note": current_note,
                "at": current_at,
                "forum_len": len(self.forum_history),
                "discarding_step": discarding_step,
            }
            if self._active_turn:
                self._active_turn["steps"] = steps
                self._active_turn["note"] = current_note
                self._active_turn["at"] = current_at
                self._active_turn["forum_len"] = len(self.forum_history)
                self._active_turn["status"] = "running"

        _mark_progress()

        def _manual_stop_result() -> dict[str, Any]:
            return self._finalize_manual_stop_turn(
                session_date=session_date,
                turn_num=turn_num,
                user_message=user_message,
                steps=steps,
                current_note=current_note,
                current_at=current_at,
                original_note=original_note,
                original_at=original_at,
                on_step=on_step,
            )

        while sub_turn < self.MAX_SUB_TURNS:
            if self._manual_stop_event.is_set():
                return _manual_stop_result()

            sub_turn += 1
            step_label = f"Turn {turn_num}-{sub_turn}"
            _mark_progress(discarding_step=step_label)
            _log(f"\n--- {step_label} ---")

            # 建立 user prompt（包含上下文）
            context_prompt = self._build_context_prompt(
                user_message=user_message,
                note_content=current_note,
                at_content=current_at,
                patient_info=patient_info,
                session_date=session_date,
                steps=steps,
                interview_dialogue=interview_dialogue,
            )

            _log(f"\n{'─'*40}")
            _log(f"[Main Agent] {step_label} ══ 送入 LLM 的 User Prompt ══")
            _log(context_prompt)
            _log(f"{'─'*40}\n")
            self._behavior_event(
                session_date,
                agent="main_agent",
                event_type="llm_input",
                label="輸入",
                title=f"AI主治醫師 {step_label} 輸入",
                content=context_prompt,
                turn=turn_num,
                sub_turn=step_label,
            )

            # 建立 messages
            # 動態生成教授清單
            prof_list_text = "（無可用教授）"
            if self.professor_config:
                profs = load_all_professors()
                if profs:
                    lines = []
                    for p in profs:
                        name_str = p['name'] or '（未命名）'
                        desc_str = p['description'] or '（未設定風格描述）'
                        lines.append(f"  - {p['id']}：{name_str}（{desc_str}）")
                    prof_list_text = "\n".join(lines)

            system_prompt = self.system_prompt_template.replace(
                "{record_template}", self.record_template
            ).replace(
                "{last_visit_block}", last_visit_block or "（無上次就診紀錄）"
            ).replace(
                "{history_summary}", history_summary or "（無歷史病歷摘要）"
            ).replace(
                "{professor_list}", prof_list_text
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context_prompt},
            ]

            # 呼叫 LLM
            parsed = self._call_llm(messages, step_label, _log, session_date, turn_num)
            if self._manual_stop_event.is_set():
                return _manual_stop_result()
            if parsed is None:
                error_msg = f"{step_label}: LLM 回應解析失敗"
                reply_text = "服務已中斷(模型呼叫失敗)"
                _log(f"[Main Agent] ❌ {error_msg}")
                self._behavior_event(
                    session_date,
                    agent="main_agent",
                    event_type="model_error",
                    label="模型呼叫失敗",
                    title=f"AI主治醫師 {step_label} 模型呼叫失敗",
                    content=error_msg,
                    turn=turn_num,
                    sub_turn=step_label,
                    severity="error",
                )
                self._finalize_turn_history(
                    turn_num=turn_num,
                    user_message=user_message,
                    steps=steps,
                    reply_text=reply_text,
                )
                return {
                    "reply": reply_text,
                    "steps": steps,
                    "note": current_note,
                    "at": current_at,
                    "record_snapshots": record_snapshots,
                    "note_changed": current_note != original_note,
                    "at_changed": current_at != original_at,
                    "turn_number": turn_num,
                    "waiting_for_interview": False,
                    "interview_guidelines": "",
                    "error": error_msg,
                }

            # 提取欄位
            thinking = parsed.get("thinking", "")
            action = parsed.get("action", "")
            action_input = parsed.get("action_input", "")
            next_step_plan = parsed.get("next_step", "")

            step_record = {
                "step_label": step_label,
                "thinking": thinking,
                "action": action,
                "action_input": action_input if isinstance(action_input, str) else json.dumps(action_input, ensure_ascii=False),
                "next_step": next_step_plan,
                "timestamp": datetime.now().isoformat(),
                "result": None,
            }

            _log(f"[Main Agent] ══ LLM 解析結果 ══")
            _log(f"[Main Agent] thinking: {thinking}")
            _log(f"[Main Agent] action: {action}")
            _log(f"[Main Agent] action_input: {action_input if isinstance(action_input, str) else json.dumps(action_input, ensure_ascii=False)}")
            _log(f"[Main Agent] next_step: {next_step_plan}")

            # ── 處理動作 ──
            if action == "reply":
                reply_text = action_input if isinstance(action_input, str) else str(action_input)
                step_record["result"] = "回覆完成"
                steps.append(step_record)
                _mark_progress()

                if on_step:
                    on_step(step_record)

                _log(f"[Main Agent] ══ 最終回覆 ══")
                _log(reply_text)
                _log(f"[Main Agent] === Turn {turn_num} 結束 ===\n")

                self._finalize_turn_history(
                    turn_num=turn_num,
                    user_message=user_message,
                    steps=steps,
                    reply_text=reply_text,
                )

                return {
                    "reply": reply_text,
                    "steps": steps,
                    "note": current_note,
                    "at": current_at,
                    "record_snapshots": record_snapshots,
                    "note_changed": current_note != original_note,
                    "at_changed": current_at != original_at,
                    "turn_number": turn_num,
                    "waiting_for_interview": False,
                    "interview_guidelines": "",
                    "error": None,
                }

            elif action == "update_record":
                self._behavior_event(
                    session_date,
                    agent="main_agent",
                    event_type="tool_event",
                    label="→病歷登載",
                    title=f"{step_label} 呼叫病歷登載 Subagent",
                    content=action_input if isinstance(action_input, str) else json.dumps(action_input, ensure_ascii=False, indent=2),
                    turn=turn_num,
                    sub_turn=step_label,
                    target_agent="record_subagent",
                )
                # 解析 action_input
                if isinstance(action_input, str):
                    try:
                        action_input = json.loads(action_input)
                    except json.JSONDecodeError:
                        action_input = {"target_field": "note", "guidelines": action_input}

                target_field = action_input.get("target_field", "note")
                guidelines = action_input.get("guidelines", "")

                _log(f"[Main Agent] 呼叫 Record Subagent: field={target_field}")

                # 呼叫 Record Subagent
                sub_result = self.record_subagent.execute(
                    note_content=current_note,
                    at_content=current_at,
                    target_field=target_field,
                    guidelines=guidelines,
                    conversation_history=conversation_history,
                    last_visit_block=last_visit_block,
                    history_summary=history_summary,
                    interview_dialogue=interview_dialogue,
                    forum_history=self._format_forum_history(),
                    loaded_files_block=self._format_loaded_files_block(),
                    image_files=self._loaded_files,
                    log_callback=log_callback,
                    behavior_context={"folder_path": self._patient_folder, "date_str": session_date},
                )
                if self._manual_stop_event.is_set():
                    return _manual_stop_result()

                if sub_result["success"]:
                    before_note = current_note
                    before_at = current_at
                    current_note = sub_result["note"]
                    current_at = sub_result["at"]
                    ops_summary = f"成功執行 {len(sub_result['operations'])} 個操作"
                    review_info = f"(審查: {sub_result.get('review_result', '?')}, {sub_result.get('review_rounds', 0)}輪)"
                    step_record["result"] = f"{ops_summary} {review_info}"
                    if current_note != before_note or current_at != before_at:
                        record_snapshots.append({
                            "note": current_note,
                            "at": current_at,
                            "source": f"{step_label} update_record",
                            "target_field": target_field,
                            "result": step_record["result"],
                        })
                    _log(f"[Main Agent] Record Subagent: {ops_summary} {review_info}")
                    self._behavior_event(
                        session_date,
                        agent="record_subagent",
                        event_type="tool_event",
                        label="操作結果",
                        title=f"{step_label} 病歷登載操作結果",
                        content="\n".join(sub_result.get("op_logs", [])) or step_record["result"],
                        turn=turn_num,
                        sub_turn=step_label,
                    )
                else:
                    step_record["result"] = f"失敗: {sub_result.get('error', '未知錯誤')}"
                    _log(f"[Main Agent] Record Subagent 失敗: {sub_result.get('error')}")

                steps.append(step_record)
                _mark_progress()

                if on_step:
                    on_step(step_record)

                # 繼續下一個子輪
                continue

            elif action == "information_collection_subagent":
                self._behavior_event(
                    session_date,
                    agent="main_agent",
                    event_type="tool_event",
                    label="→問診助理",
                    title=f"{step_label} 呼叫問診助理 Subagent",
                    content=action_input if isinstance(action_input, str) else str(action_input),
                    turn=turn_num,
                    sub_turn=step_label,
                    target_agent="information_collection_subagent",
                )
                # 啟動問診助理：暫停迴圈，由 UI 執行問診流程
                if not self.ic_config.get("model_name"):
                    step_record["result"] = "問診助理 Subagent 未設定（model_name 為空）"
                    _log("[Main Agent] IC Subagent 未設定")
                    steps.append(step_record)
                    _mark_progress()
                    if on_step:
                        on_step(step_record)
                    continue

                ic_guidelines = action_input if isinstance(action_input, str) else str(action_input)
                step_record["result"] = f"啟動問診助理 (方針: {ic_guidelines[:200]}...)"
                steps.append(step_record)
                _mark_progress()

                if on_step:
                    on_step(step_record)

                _log(f"[Main Agent] 啟動問診助理 Subagent")
                _log(f"[Main Agent] 資訊蒐集方針: {ic_guidelines[:3000]}")

                # 暫停迴圈 — 儲存狀態供 continue_after_interview() 恢復
                self._suspended = {
                    "steps": steps,
                    "sub_turn": sub_turn,
                    "turn_num": turn_num,
                    "user_message": user_message,
                    "patient_info": patient_info,
                    "session_date": session_date,
                    "conversation_history": conversation_history,
                    "last_visit_block": last_visit_block,
                    "history_summary": history_summary,
                    "original_note": original_note,
                    "original_at": original_at,
                }
                if self._active_turn:
                    self._active_turn["status"] = "waiting_for_interview"

                return {
                    "waiting_for_interview": True,
                    "interview_guidelines": ic_guidelines,
                    "steps": steps,
                    "note": current_note,
                    "at": current_at,
                    "record_snapshots": record_snapshots,
                    "note_changed": current_note != original_note,
                    "at_changed": current_at != original_at,
                    "turn_number": turn_num,
                    "error": None,
                }

            elif action == "low_confidence_check":
                self._behavior_event(
                    session_date,
                    agent="main_agent",
                    event_type="tool_event",
                    label="→低信心",
                    title=f"{step_label} 呼叫低信心標註 Subagent",
                    content=action_input if isinstance(action_input, str) else str(action_input),
                    turn=turn_num,
                    sub_turn=step_label,
                    target_agent="low_confidence_subagent",
                )
                # 執行低信心標註掃描
                if self.lc_subagent is None:
                    step_record["result"] = "低信心標註 Subagent 未設定（model_name 為空）"
                    _log("[Main Agent] LC Subagent 未設定")
                else:
                    _log("[Main Agent] 執行 Low Confidence Check")
                    lc_result = self.lc_subagent.execute(
                        note_content=current_note,
                        interview_dialogue=interview_dialogue,
                        conversation_history=conversation_history,
                        last_visit_block=last_visit_block,
                        history_summary=history_summary,
                        loaded_files_block=self._format_loaded_files_block(),
                        image_files=self._loaded_files,
                        log_callback=log_callback,
                        behavior_context={"folder_path": self._patient_folder, "date_str": session_date},
                    )
                    if self._manual_stop_event.is_set():
                        return _manual_stop_result()
                    if lc_result["success"]:
                        current_note = lc_result["annotated_note"]
                        if lc_result.get("skipped_control_group"):
                            step_record["result"] = "低信心標註未執行（檢測強度0/對照組），NOTE 未變更"
                        else:
                            step_record["result"] = (
                                f"完成低信心標註，共 {lc_result['total_rounds']} 輪，"
                                f"標註 {lc_result['total_annotated']} 個片段"
                            )
                        _log(f"[Main Agent] LC: {step_record['result']}")
                    else:
                        step_record["result"] = (
                            f"低信心標註失敗: {lc_result.get('error', '?')}"
                        )
                        _log(f"[Main Agent] LC 失敗: {lc_result.get('error')}")

                steps.append(step_record)
                _mark_progress()
                if on_step:
                    on_step(step_record)
                continue

            elif action == "note_review_subagent":
                self._behavior_event(
                    session_date,
                    agent="main_agent",
                    event_type="tool_event",
                    label="→病歷檢查",
                    title=f"{step_label} 呼叫病歷檢查員 Subagent",
                    content=action_input if isinstance(action_input, str) else str(action_input),
                    turn=turn_num,
                    sub_turn=step_label,
                    target_agent="note_review_subagent",
                )
                # 啟動病歷檢查員 Subagent
                if self.nr_subagent is None:
                    step_record["result"] = "病歷檢查員 Subagent 未設定（model_name 為空）"
                    _log("[Main Agent] NR Subagent 未設定")
                    steps.append(step_record)
                    _mark_progress()
                    if on_step:
                        on_step(step_record)
                    continue

                _log("[Main Agent] 啟動病歷檢查員 Subagent")
                nr_result = self.nr_subagent.execute(
                    note_content=current_note,
                    interview_dialogue=interview_dialogue,
                    conversation_history=conversation_history,
                    loaded_files_block=self._format_loaded_files_block(),
                    log_callback=log_callback,
                    behavior_context={"folder_path": self._patient_folder, "date_str": session_date},
                )
                if self._manual_stop_event.is_set():
                    return _manual_stop_result()

                nr_thinking = nr_result.get("thinking", "")
                nr_guidelines = nr_result.get("guidelines", "")
                nr_update_reminder = nr_result.get("update_reminder", "")
                nr_summary = (
                    f"[thinking] {nr_thinking}\n"
                    f"[action_input] {nr_guidelines or '本次無需要蒐集的資訊'}"
                )
                if nr_update_reminder:
                    nr_summary += f"\n[update_reminder] {nr_update_reminder}"
                step_record["note_review_result"] = nr_summary

                if nr_update_reminder:
                    _log(f"[Main Agent] 病歷檢查員提醒：有已取得但尚未寫入 NOTE 的資訊")
                    _log(f"[Main Agent] update_reminder: {nr_update_reminder[:3000]}")

                if nr_result.get("needs_collection"):
                    # 有待補問項目 → 自動銜接啟動問診助理 (IC)
                    if not self.ic_config.get("model_name"):
                        step_record["result"] = (
                            "病歷檢查員發現待補問項目，但問診助理 Subagent 未設定（model_name 為空）"
                        )
                        _log("[Main Agent] NR 發現待補問項目，但 IC Subagent 未設定")
                        steps.append(step_record)
                        _mark_progress()
                        if on_step:
                            on_step(step_record)
                        continue

                    step_record["result"] = (
                        f"病歷檢查員發現待補問項目，自動啟動問診助理"
                    )
                    steps.append(step_record)
                    _mark_progress()
                    if on_step:
                        on_step(step_record)

                    _log(f"[Main Agent] 病歷檢查員發現待補問項目，自動啟動問診助理")
                    _log(f"[Main Agent] 資訊蒐集方針: {nr_guidelines[:3000]}")

                    # 暫停迴圈 — 與 information_collection_subagent 相同的邏輯
                    self._suspended = {
                        "steps": steps,
                        "sub_turn": sub_turn,
                        "turn_num": turn_num,
                        "user_message": user_message,
                        "patient_info": patient_info,
                        "session_date": session_date,
                        "conversation_history": conversation_history,
                        "last_visit_block": last_visit_block,
                        "history_summary": history_summary,
                        "original_note": original_note,
                        "original_at": original_at,
                    }
                    if self._active_turn:
                        self._active_turn["status"] = "waiting_for_interview"

                    return {
                        "waiting_for_interview": True,
                        "interview_guidelines": nr_guidelines,
                        "steps": steps,
                        "note": current_note,
                        "at": current_at,
                        "record_snapshots": record_snapshots,
                        "note_changed": current_note != original_note,
                        "at_changed": current_at != original_at,
                        "turn_number": turn_num,
                        "error": None,
                    }
                else:
                    # 所有項目已完成
                    if nr_update_reminder:
                        step_record["result"] = (
                            f"病歷檢查員回報所有項目已完成，"
                            f"但有已取得未登載的資訊需更新 NOTE"
                        )
                    else:
                        step_record["result"] = "病歷檢查員回報所有項目已完成"
                    steps.append(step_record)
                    _mark_progress()
                    if on_step:
                        on_step(step_record)
                    _log("[Main Agent] 病歷檢查員回報所有項目已完成")
                    continue

            elif action == "call_professor":
                self._behavior_event(
                    session_date,
                    agent="main_agent",
                    event_type="tool_event",
                    label="→教授",
                    title=f"{step_label} 呼叫醫學教授 Subagent",
                    content=action_input if isinstance(action_input, str) else json.dumps(action_input, ensure_ascii=False, indent=2),
                    turn=turn_num,
                    sub_turn=step_label,
                    target_agent="professor_subagent",
                )
                # 諮詢中醫教授
                try:
                    if isinstance(action_input, str):
                        prof_input = json.loads(action_input)
                    else:
                        prof_input = action_input
                    prof_id = prof_input.get("professor_id", "")
                    prof_question = prof_input.get("question", "")
                except (json.JSONDecodeError, AttributeError):
                    step_record["result"] = "call_professor: action_input 格式錯誤（需 JSON 物件）"
                    steps.append(step_record)
                    _mark_progress()
                    if on_step:
                        on_step(step_record)
                    continue

                if not prof_id or not prof_question:
                    step_record["result"] = "call_professor: 缺少 professor_id 或 question"
                    steps.append(step_record)
                    _mark_progress()
                    if on_step:
                        on_step(step_record)
                    continue

                if not self.professor_config:
                    step_record["result"] = "call_professor: 教授模型設定未配置"
                    steps.append(step_record)
                    _mark_progress()
                    if on_step:
                        on_step(step_record)
                    continue

                # 惰性建立 ProfessorInstance
                if prof_id not in self._professor_instances:
                    prof_dir = os.path.join(CURRENT_DIR, prof_id)
                    if not os.path.isdir(prof_dir):
                        step_record["result"] = f"call_professor: 找不到教授資料夾 {prof_id}"
                        steps.append(step_record)
                        _mark_progress()
                        if on_step:
                            on_step(step_record)
                        continue
                    self._professor_instances[prof_id] = ProfessorInstance(prof_id, self.professor_config)

                prof_inst = self._professor_instances[prof_id]
                prof_display_name = prof_inst.name or prof_id

                _log(f"[Main Agent] 呼叫教授 {prof_id} ({prof_display_name})")
                _log(f"[Main Agent] 提問: {prof_question[:500]}")

                # 格式化 forum_history 文字
                forum_text = self._format_forum_history()
                post_q_id = f"D{len(self.forum_history) + 1}"
                post_a_id = f"D{len(self.forum_history) + 2}"

                # 呼叫教授 RAG 管線
                prof_result = prof_inst.answer(
                    question=prof_question,
                    note_content=current_note,
                    at_content=current_at,
                    last_visit_block=last_visit_block,
                    history_summary=history_summary,
                    forum_history_text=forum_text,
                    loaded_files_block=self._format_loaded_files_block(),
                    image_files=self._loaded_files,
                    log_callback=log_callback,
                    behavior_context={"folder_path": self._patient_folder, "date_str": session_date},
                )
                if self._manual_stop_event.is_set():
                    return _manual_stop_result()

                prof_response = prof_result.get("response", "（無回應）")

                if prof_response.startswith("⚠️ 教授回答失敗") or prof_response.startswith("⚠️ 教授 Answer LLM 模型未設定"):
                    step_record["result"] = f"call_professor: 教授 {prof_display_name} 回答失敗"
                    _log(f"[Main Agent] 教授 {prof_display_name} 回答失敗，不寫入醫療問答討論區")
                    steps.append(step_record)
                    _mark_progress()
                    if on_step:
                        on_step(step_record)
                    continue

                # 教授回答完成後，將 Q/A 成對寫入 forum_history。
                self.forum_history.append({
                    "post_id": post_q_id,
                    "role": "main_agent",
                    "professor_id": prof_id,
                    "professor_name": prof_display_name,
                    "content": prof_question,
                })
                self.forum_history.append({
                    "post_id": post_a_id,
                    "role": "professor",
                    "professor_id": prof_id,
                    "professor_name": prof_display_name,
                    "content": prof_response,
                })

                step_record["result"] = (
                    f"教授 {prof_display_name} 已回答 ({post_q_id}-{post_a_id})，"
                    f"回答長度 {len(prof_response)} 字"
                )
                step_record["professor_response"] = prof_response
                step_record["professor_post_a_id"] = post_a_id
                step_record["professor_id_ref"] = prof_id
                step_record["professor_name_ref"] = prof_display_name
                _log(f"[Main Agent] 教授 {prof_display_name} 回答完成 (len={len(prof_response)})")
                _log(f"[Main Agent] 回答摘要: {prof_response[:300]}...")

                # 寫入 RAG 完整行為 log
                self._write_rag_log(
                    session_date=session_date,
                    professor_id=prof_id,
                    professor_name=prof_display_name,
                    question=prof_question,
                    q_expand=prof_result.get("q_expand", ""),
                    prefixes=prof_result.get("prefixes", []),
                    retr_doc=prof_result.get("retr_doc", ""),
                    response=prof_response,
                )

                steps.append(step_record)
                _mark_progress()
                if on_step:
                    on_step(step_record)
                continue

            elif action == "list_patient_files":
                self._behavior_event(
                    session_date,
                    agent="main_agent",
                    event_type="tool_event",
                    label="列檔",
                    title=f"{step_label} list_patient_files",
                    content=action_input if isinstance(action_input, str) else str(action_input),
                    turn=turn_num,
                    sub_turn=step_label,
                )
                # 列出患者檔案
                folder_arg = action_input.strip() if isinstance(action_input, str) else "all"
                folder_key = folder_arg.lower().replace(" ", "_")
                result_lines = []
                pf = self._patient_folder
                if not pf:
                    step_record["result"] = "list_patient_files: 尚未選取患者"
                    steps.append(step_record)
                    _mark_progress()
                    if on_step:
                        on_step(step_record)
                    continue

                valid_img_exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif"}

                def _compact_summary(text: str, limit: int = 50) -> str:
                    compact = " ".join((text or "").split())
                    if len(compact) <= limit:
                        return compact
                    return compact[:limit]

                patient_info_file = os.path.join(pf, "patient_info.json")
                patient_sessions = {}
                if os.path.isfile(patient_info_file):
                    try:
                        with open(patient_info_file, "r", encoding="utf-8") as f:
                            patient_sessions = json.load(f).get("sessions", {})
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
                    record_path = os.path.join(pf, filename)
                    try:
                        with open(record_path, "r", encoding="utf-8") as f:
                            return _compact_summary(f.read())
                    except Exception:
                        return ""

                folders_to_scan = []
                if folder_key in ("all", "picture_row", ""):
                    folders_to_scan.append(("Picture_Row", os.path.join(pf, "Picture_Row")))
                if folder_key in ("all", "medical_information", ""):
                    folders_to_scan.append(("Medical_information", os.path.join(pf, "Medical_information")))
                include_records = folder_key in ("all", "medical_records", "records", "record", "")

                for folder_label, folder_path in folders_to_scan:
                    if not os.path.isdir(folder_path):
                        result_lines.append(f"[{folder_label}] 資料夾不存在")
                        continue
                    files = sorted(os.listdir(folder_path))
                    files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]
                    if not files:
                        result_lines.append(f"[{folder_label}] （空）")
                    else:
                        result_lines.append(f"[{folder_label}] 共 {len(files)} 個檔案：")
                        for fn in files:
                            _, ext = os.path.splitext(fn)
                            tag = "🖼️" if ext.lower() in valid_img_exts else "📄"
                            result_lines.append(f"  {tag} {fn}")

                if include_records:
                    md_files = sorted(
                        f
                        for f in os.listdir(pf)
                        if os.path.isfile(os.path.join(pf, f)) and f.lower().endswith(".md")
                    )
                    if not md_files:
                        result_lines.append("[Medical_Records] （空）")
                    else:
                        result_lines.append(f"[Medical_Records] 共 {len(md_files)} 個病歷檔")
                        for fn in md_files:
                            summary = _record_summary_for(fn)
                            suffix = f"（{summary}）" if summary else "（空白）"
                            result_lines.append(f"  📄 {fn}{suffix}")

                listing_text = "\n".join(result_lines)
                if self._manual_stop_event.is_set():
                    return _manual_stop_result()
                self._file_list_cache = listing_text

                listing_summary = "；".join(
                    line.rstrip("：")
                    for line in result_lines
                    if line.startswith("[")
                )
                step_record["result"] = f"list_patient_files 完成：{listing_summary or '已列出患者檔案'}"
                _log(f"[Main Agent] list_patient_files:\n{listing_text}")
                self._behavior_event(
                    session_date,
                    agent="main_agent",
                    event_type="tool_event",
                    label="列檔結果",
                    title=f"{step_label} list_patient_files 結果",
                    content=listing_text,
                    turn=turn_num,
                    sub_turn=step_label,
                )
                steps.append(step_record)
                _mark_progress()
                if on_step:
                    on_step(step_record)
                continue

            elif action == "read_patient_file":
                self._behavior_event(
                    session_date,
                    agent="main_agent",
                    event_type="tool_event",
                    label="讀檔",
                    title=f"{step_label} read_patient_file",
                    content=action_input if isinstance(action_input, str) else json.dumps(action_input, ensure_ascii=False, indent=2),
                    turn=turn_num,
                    sub_turn=step_label,
                )
                # 讀取患者檔案
                pf = self._patient_folder
                if not pf:
                    step_record["result"] = "read_patient_file: 尚未選取患者"
                    steps.append(step_record)
                    _mark_progress()
                    if on_step:
                        on_step(step_record)
                    continue

                # 支援單檔（字串）或多檔（JSON 陣列）
                filenames = []
                if isinstance(action_input, list):
                    filenames = action_input
                elif isinstance(action_input, str):
                    try:
                        parsed_list = json.loads(action_input)
                        if isinstance(parsed_list, list):
                            filenames = parsed_list
                        else:
                            filenames = [action_input.strip()]
                    except (json.JSONDecodeError, ValueError):
                        filenames = [action_input.strip()]

                if not filenames:
                    step_record["result"] = "read_patient_file: 未指定檔名"
                    steps.append(step_record)
                    _mark_progress()
                    if on_step:
                        on_step(step_record)
                    continue

                valid_img_exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
                results = []
                for fn in filenames:
                    fn = fn.strip()
                    # 搜尋 Picture_Row、Medical_information 和患者根目錄病歷 MD
                    found_path = None
                    for sub in ["Picture_Row", "Medical_information", ""]:
                        if not sub and not fn.lower().endswith(".md"):
                            continue
                        candidate = os.path.join(pf, sub, fn) if sub else os.path.join(pf, fn)
                        if os.path.isfile(candidate):
                            found_path = candidate
                            break

                    if not found_path:
                        results.append(f"❌ 找不到檔案: {fn}")
                        continue

                    _, ext = os.path.splitext(fn)
                    if ext.lower() in valid_img_exts:
                        # 圖片檔 → 記錄路徑供多模態注入
                        self._loaded_files.append({
                            "name": fn,
                            "type": "image",
                            "path": found_path,
                        })
                        results.append(f"🖼️ {fn}: 圖片已載入預覽區")
                    else:
                        # 文字檔 → 讀取內容
                        try:
                            with open(found_path, "r", encoding="utf-8") as f:
                                content = f.read()
                            self._loaded_files.append({
                                "name": fn,
                                "type": "text",
                                "content": content,
                            })
                            results.append(f"📄 {fn}: 已讀取 ({len(content)} 字)，內容見【當輪讀取檔案暫存區】")
                        except Exception as e:
                            results.append(f"❌ {fn}: 讀取失敗 ({e})")

                result_text = "\n".join(results)
                if self._manual_stop_event.is_set():
                    return _manual_stop_result()
                step_record["result"] = f"read_patient_file 完成\n{result_text}"
                _log(f"[Main Agent] read_patient_file:\n{result_text}")
                self._behavior_event(
                    session_date,
                    agent="main_agent",
                    event_type="tool_event",
                    label="讀檔結果",
                    title=f"{step_label} read_patient_file 結果",
                    content=result_text,
                    turn=turn_num,
                    sub_turn=step_label,
                )

                # 檔案已讀取，立即收回檔案清單
                if self._file_list_cache is not None:
                    self._file_list_cache = None
                    _log("[Main Agent] 檔案清單已收回（read_patient_file 完成後自動移除）")

                steps.append(step_record)
                _mark_progress()
                if on_step:
                    on_step(step_record)
                continue

            else:
                # 未知動作，記錄並繼續
                _log(f"[Main Agent] ⚠️ 未知動作: {action}，嘗試繼續...")
                step_record["result"] = f"未知動作: {action}"
                steps.append(step_record)
                _mark_progress()
                if on_step:
                    on_step(step_record)
                continue

        # 達到子輪上限，強制回覆
        _log(f"[Main Agent] ⚠️ 達到子輪上限 ({self.MAX_SUB_TURNS})，強制結束")
        reply_text = "⚠️ 系統提示：已達到最大執行步驟數，請嘗試簡化指令。"
        self._finalize_turn_history(
            turn_num=turn_num,
            user_message=user_message,
            steps=steps,
            reply_text=reply_text,
        )
        return {
            "reply": reply_text,
            "steps": steps,
            "note": current_note,
            "at": current_at,
            "record_snapshots": record_snapshots,
            "note_changed": current_note != original_note,
            "at_changed": current_at != original_at,
            "turn_number": turn_num,
            "waiting_for_interview": False,
            "interview_guidelines": "",
            "error": "達到子輪上限",
        }

    # ────────────────────────────────────────────────────────────
    # 輔助方法
    # ────────────────────────────────────────────────────────────

    def _build_sub_steps_summary(self, steps: list[dict]) -> list[dict]:
        """從 steps 列表建立 turn_history 用的 sub_steps 摘要。"""
        sub_steps = []
        for s in steps:
            target_f = ""
            if s["action"] == "update_record":
                try:
                    ai = json.loads(s["action_input"]) if isinstance(s["action_input"], str) else s["action_input"]
                    target_f = ai.get("target_field", "?") if isinstance(ai, dict) else "?"
                except Exception:
                    target_f = "?"
                summary = f"update_record: 更新 {target_f}, {s.get('result', '?')}"
            elif s["action"] == "reply":
                summary = "reply: 回覆人類醫師"
            elif s["action"] == "information_collection_subagent":
                summary = f"information_collection_subagent: {s.get('result', '?')}"
            elif s["action"] == "note_review_subagent":
                summary = f"note_review_subagent: {s.get('result', '?')}"
            elif s["action"] == "call_professor":
                summary = f"call_professor: {s.get('result', '?')}"
            else:
                summary = f"{s['action']}: {s.get('result', '?')}"

            entry = {
                "step_label": s["step_label"],
                "thinking": s.get("thinking", ""),
                "action_summary": summary,
                "next_step": s.get("next_step", "") or "無",
            }
            # 保留 subagent_result（若有）
            if s.get("subagent_result"):
                entry["subagent_result"] = s["subagent_result"]
            # 保留 note_review_result（若有）
            if s.get("note_review_result"):
                entry["note_review_result"] = s["note_review_result"]
            # 保留 professor 相關欄位（若有）
            if s.get("professor_response"):
                entry["professor_response"] = s["professor_response"]
                entry["professor_post_a_id"] = s.get("professor_post_a_id", "")
                entry["professor_id_ref"] = s.get("professor_id_ref", "")
                entry["professor_name_ref"] = s.get("professor_name_ref", "")

            sub_steps.append(entry)
        return sub_steps

    def _format_loaded_files_block(self) -> str:
        """將當輪讀取的檔案格式化為文字區塊，供注入各 Subagent 的 prompt。"""
        if not self._loaded_files:
            return ""
        file_blocks = []
        for lf in self._loaded_files:
            if lf["type"] == "text":
                file_blocks.append(f"### 📄 {lf['name']}\n{lf['content']}")
            elif lf["type"] == "image":
                file_blocks.append(f"### 🖼️ {lf['name']}\n（圖片已載入，見多模態訊息）")
        return "\n\n".join(file_blocks)

    def _build_context_prompt(
        self,
        user_message: str,
        note_content: str,
        at_content: str,
        patient_info: Optional[dict],
        session_date: str,
        steps: list[dict],  # 本主輪的子輪步驟
        interview_dialogue: str = "",
    ) -> str:
        """建立包含完整上下文的 user prompt。"""

        parts: list[str] = []

        # 患者資訊
        if patient_info:
            bi = patient_info.get("basic_info", {})
            parts.append(f"""## 【患者資訊】
{format_patient_basic_info_for_llm(bi, session_date, include_visit_date=True)}""")

        # 人類醫師與 AI 主治醫師的完整互動過程（穩定區，利於 KV cache）
        interaction_text = ""

        # 歷屆已完成的主輪
        for th in self.turn_history:
            interaction_text += f"[Turn {th['turn']} 人類醫師的提問] {th['user_message']}\n"
            for ss in th["sub_steps"]:
                interaction_text += f"[{ss['step_label']}的thinking] {ss['thinking']}\n"
                interaction_text += f"[{ss['step_label']}已執行的action] {ss['action_summary']}\n"
                interaction_text += f"[{ss['step_label']}預計的next_step] {ss['next_step']}\n"
                # 問診助理蒐集結果（若有）
                if ss.get("subagent_result"):
                    interaction_text += f"[{ss['step_label']}問診助理蒐集結果] {ss['subagent_result']}\n"
                # 病歷檢查員掃描結果（若有）
                if ss.get("note_review_result"):
                    interaction_text += f"[{ss['step_label']}病歷檢查員掃描結果] {ss['note_review_result']}\n"
                # 教授回答（若有）— 僅放簡短索引
                if ss.get("professor_response"):
                    _pa_id = ss.get('professor_post_a_id', '?')
                    _p_id = ss.get('professor_id_ref', '?')
                    _p_name = ss.get('professor_name_ref', '?')
                    interaction_text += f"[{ss['step_label']}教授回答] <對話欄{_pa_id}，{_p_id} ({_p_name})的回答>\n"
            interaction_text += f"[Turn {th['turn']} AI主治醫師的回覆] {th['reply']}\n"

        # 上次異常中斷前已保存、但尚未收斂成 turn_history 的主輪
        if self._active_turn and self._active_turn.get("turn") != self.turn_count:
            active_turn = self._active_turn
            status = active_turn.get("status", "running")
            interaction_text += (
                f"[Turn {active_turn.get('turn', '?')} 人類醫師的提問] "
                f"{active_turn.get('user_message', '')}\n"
            )
            interaction_text += f"[Turn {active_turn.get('turn', '?')} 狀態] 服務已中斷前已保存（{status}）\n"
            for s in active_turn.get("steps", []):
                interaction_text += f"[{s['step_label']}的thinking] {s.get('thinking', '')}\n"
                interaction_text += f"[{s['step_label']}已執行的action] {s['action']}: {s.get('result', '?')}\n"
                interaction_text += f"[{s['step_label']}預計的next_step] {s.get('next_step', '') or '無'}\n"
                if s.get("subagent_result"):
                    interaction_text += f"[{s['step_label']}問診助理蒐集結果] {s['subagent_result']}\n"
                if s.get("note_review_result"):
                    interaction_text += f"[{s['step_label']}病歷檢查員掃描結果] {s['note_review_result']}\n"
                if s.get("professor_response"):
                    _pa_id = s.get('professor_post_a_id', '?')
                    _p_id = s.get('professor_id_ref', '?')
                    _p_name = s.get('professor_name_ref', '?')
                    interaction_text += f"[{s['step_label']}教授回答] <對話欄{_pa_id}，{_p_id} ({_p_name})的回答>\n"

        # 本主輪（當前進行中）
        interaction_text += f"[Turn {self.turn_count} 人類醫師的提問] {user_message}\n"
        if steps:
            for s in steps:
                interaction_text += f"[{s['step_label']}的thinking] {s.get('thinking', '')}\n"
                interaction_text += f"[{s['step_label']}已執行的action] {s['action']}: {s.get('result', '?')}\n"
                interaction_text += f"[{s['step_label']}預計的next_step] {s.get('next_step', '') or '無'}\n"
                # 問診助理蒐集結果（若有）
                if s.get("subagent_result"):
                    interaction_text += f"[{s['step_label']}問診助理蒐集結果] {s['subagent_result']}\n"
                # 病歷檢查員掃描結果（若有）
                if s.get("note_review_result"):
                    interaction_text += f"[{s['step_label']}病歷檢查員掃描結果] {s['note_review_result']}\n"
                # 教授回答（若有）— 僅放簡短索引，完整內容見【醫療問答討論區】
                if s.get("professor_response"):
                    _pa_id = s.get('professor_post_a_id', '?')
                    _p_id = s.get('professor_id_ref', '?')
                    _p_name = s.get('professor_name_ref', '?')
                    interaction_text += f"[{s['step_label']}教授回答] <對話欄{_pa_id}，{_p_id} ({_p_name})的回答>\n"

        parts.append(f"""## 【人類醫師與 AI 主治醫師的互動過程 (詳細版)】
{interaction_text}""")

        # 完整問診對話紀錄（介於互動過程與今日病歷之間）
        parts.append(f"""## 【完整問診對話紀錄】
{interview_dialogue if interview_dialogue else '（空白）'}""")

        # 醫療問答討論區
        forum_text = self._format_forum_history()
        parts.append(f"""## 【醫療問答討論區】
{forum_text if forum_text else '（空白）'}""")

        # 今日病歷（隨時變動區，放在後面避免破壞前面的 KV cache）
        parts.append(f"""## 【今日病歷(或當前編輯頁面的病歷) - NOTE】
{note_content if note_content else '（空白）'}""")

        parts.append(f"""## 【今日病歷(或當前編輯頁面的病歷) - ASSESSMENT & TREATMENT】
{at_content if at_content else '（空白）'}""")

        # 患者檔名清單（按需注入，read_patient_file 完成後自動收回）
        parts.append(f"""## 【患者檔案清單】（讀取檔案後此清單將自動移除）
{self._file_list_cache if self._file_list_cache else '（空白）'}""")

        # 當輪讀取檔案暫存區（每個主輪開始時清空）
        if self._loaded_files:
            file_blocks = []
            for lf in self._loaded_files:
                if lf["type"] == "text":
                    file_blocks.append(f"### 📄 {lf['name']}\n{lf['content']}")
                elif lf["type"] == "image":
                    file_blocks.append(f"### 🖼️ {lf['name']}\n（圖片已載入，見多模態訊息）")
            parts.append(f"""## 【當輪讀取檔案暫存區】（本區塊僅存在於本主輪）
{chr(10).join(file_blocks)}""")
        else:
            parts.append("""## 【當輪讀取檔案暫存區】（本區塊僅存在於本主輪）
（空白）""")

        parts.append("請根據以上資訊，決定你的下一步動作。輸出 JSON。")

        return "\n\n".join(parts)

    def _call_llm(
        self,
        messages: list[dict],
        step_label: str,
        _log: Callable,
        session_date: str,
        turn_num: int,
    ) -> Optional[dict]:
        """呼叫 LLM 並解析 JSON 回應，含重試機制。"""

        # 注入多模態圖片（若有載入圖片檔）
        messages = inject_images_into_messages(messages, self._loaded_files)

        output = None
        parsed = None

        for attempt in range(self.MAX_JSON_RETRIES + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=0.9,
                )
                output = (resp.choices[0].message.content or "").strip()
            except Exception as e:
                _log(f"[Main Agent] {step_label} LLM 呼叫失敗: {e}")
                return None

            _log(f"\n[Main Agent] {step_label} ══ LLM 原始輸出 (attempt {attempt}) ══")
            _log(output)
            _log(f"{'─'*40}")
            self._behavior_event(
                session_date,
                agent="main_agent",
                event_type="llm_output",
                label="輸出",
                title=f"AI主治醫師 {step_label} 輸出",
                content=output or "",
                turn=turn_num,
                sub_turn=step_label,
                meta={"attempt": attempt},
            )

            # 嘗試解析 JSON
            json_match = re.search(r"\{[\s\S]*\}", output)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    return parsed
                except json.JSONDecodeError:
                    pass

            # 重試
            if attempt < self.MAX_JSON_RETRIES:
                retry_msg = (
                    "（系統提示：你剛才的輸出無法被解析為合法的 JSON 格式。請重新輸出完整的 JSON 物件。）"
                )
                messages.append({"role": "assistant", "content": output or ""})
                messages.append({"role": "user", "content": retry_msg})
                _log(f"[Main Agent] {step_label} JSON 解析失敗，第 {attempt + 1} 次重試...")

        return None

    def _format_forum_history(self) -> str:
        """將 forum_history 格式化為文字。"""
        if not self.forum_history:
            return ""
        parts = []
        for post in self.forum_history:
            pid = post.get('post_id', '?')
            prof_name = post.get('professor_name', post.get('professor_id', '?'))
            prof_id = post.get('professor_id', '?')
            content = post.get('content', '')
            if post.get('role') == 'main_agent':
                parts.append(f"<對話欄{pid}，AI 主治醫師呼叫 {prof_id} ({prof_name})>\n{content}")
            else:
                parts.append(f"<對話欄{pid}，{prof_id} ({prof_name})的回答>\n{content}")
        return "\n\n".join(parts)

    def reset(self):
        """重置 Agent 狀態（清除對話歷史與輪次計數器）。"""
        self.turn_count = 0
        self.conversation_history.clear()
        self.turn_history.clear()
        self._suspended = None
        self.forum_history.clear()
        self._professor_instances.clear()
        self._file_list_cache = None
        self._loaded_files.clear()
        self._patient_folder = None
