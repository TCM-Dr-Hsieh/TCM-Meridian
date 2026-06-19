"""
Hallucination_Subagent.py - 幻覺與過度聲稱審查員 (Hallucination & Overclaim Reviewer)
負責審查 Record Subagent 即將寫入的病歷內容，檢查是否有幻覺、過度推論、來源標註錯誤等問題。
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Optional

from openai import OpenAI
from multimodal_utils import inject_images_into_messages
from agent_behavior_log import append_behavior_event

# ════════════════════════════════════════════════════════════════
# Prompt 載入
# ════════════════════════════════════════════════════════════════
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_prompt(filename: str) -> str:
    path = os.path.join(CURRENT_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ════════════════════════════════════════════════════════════════
# Hallucination Subagent 類別
# ════════════════════════════════════════════════════════════════

class HallucinationSubagent:
    """
    幻覺審查員：審查即將寫入的病歷內容，
    檢查是否有幻覺、過度推論、來源標註錯誤、關鍵遺漏等問題。

    使用方式：
        reviewer = HallucinationSubagent(config)
        result = reviewer.review(
            field="note",
            content="即將寫入的病歷內容...",
            conversation_history="人類醫師與 AI 的互動過程...",
        )
        # result = {"agree": "yes"|"no", "comment": "...", "thinking": "..."}
    """

    MAX_JSON_RETRIES = 2

    def __init__(self, config: dict):
        """
        Args:
            config: {
                "api_url": "http://localhost:1234/v1",
                "api_key": "lm-studio",
                "model_name": "model-name",
                "max_tokens": 8000,
                "temperature": 1.0,
            }
        """
        self.client = OpenAI(
            base_url=config.get("api_url", "http://localhost:1234/v1"),
            api_key=config.get("api_key", "lm-studio"),
        )
        self.model = config.get("model_name", "")
        self.max_tokens = int(config.get("max_tokens", 8000))
        self.temperature = float(config.get("temperature", 1.0))
        self.system_prompt_template = _load_prompt("prompt_hallucination_check.txt")


    def review(
        self,
        field: str,
        content: str,
        conversation_history: str = "",
        last_visit_block: str = "",
        history_summary: str = "",
        interview_dialogue: str = "",
        record_diff_context: str = "",
        forum_history: str = "",
        loaded_files_block: str = "",
        image_files: list | None = None,
        log_callback: Optional[Callable[[str], None]] = None,
        behavior_context: dict | None = None,
    ) -> dict[str, str]:
        """
        審查即將寫入的病歷內容。

        Args:
            field: 欄位名（"note" 或 "assessment_&_treatment"）
            content: 即將寫入的病歷內容
            conversation_history: 人類醫師與 AI 主治醫師的互動過程
            last_visit_block: 上次就診病歷內容
            history_summary: 歷史病歷摘要
            log_callback: 選用，日誌回呼函式

        Returns:
            {"agree": "yes"|"no", "comment": "...", "thinking": "..."}
        """

        def _log(msg: str):
            if log_callback:
                log_callback(msg)
            print(msg)

        # 建立 system prompt
        system_prompt = self.system_prompt_template.replace(
            "{last_visit_block}", last_visit_block or "（無上次就診紀錄）"
        ).replace(
            "{history_summary}", history_summary or "（無歷史病歷摘要）"
        )

        # 建立 user prompt
        conversation_section = f"""
## 【人類醫師與 AI 主治醫師的互動過程】
{conversation_history if conversation_history else '（空白）'}
"""

        interview_section = f"""
## 【完整問診對話紀錄】
{interview_dialogue if interview_dialogue else '（空白）'}
"""

        forum_section = f"""
## 【醫療問答討論區】
{forum_history if forum_history else '（空白）'}
"""

        loaded_files_section = f"""
## 【當輪讀取檔案暫存區】
{loaded_files_block if loaded_files_block else '（空白）'}
"""

        record_diff_text = record_diff_context or "## 【病歷修改 diff 過程】\n（無病歷版本歷史）"

        user_prompt = f"""## 【即將登載的病歷更新】
欄位：{field}
內容：
{content}
{record_diff_text}
{conversation_section}{interview_section}{forum_section}{loaded_files_section}
請根據以上資訊，進行幻覺與過度聲稱審查。輸出 JSON。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        _log(f"\n{'▼'*60}")
        _log(f"[Hallucination Reviewer] 審查輸入")
        _log(f"{'▼'*60}")
        _log(f"[Hallucination Reviewer] 欄位: {field}")
        _log(f"[Hallucination Reviewer] 內容長度: {len(content)} 字")
        _log(f"\n{'─'*40}")
        _log(f"[Hallucination Reviewer] ══ 送入 LLM 的 User Prompt ══")
        _log(user_prompt)
        _log(f"{'─'*40}")
        if behavior_context:
            append_behavior_event(
                behavior_context.get("folder_path"),
                behavior_context.get("date_str"),
                agent="hallucination_subagent",
                event_type="llm_input",
                label="輸入",
                title="幻覺檢查 Subagent 輸入",
                content=user_prompt,
            )

        # 注入多模態圖片
        if image_files:
            messages = inject_images_into_messages(messages, image_files)

        # 呼叫 LLM（含 JSON 解析重試）
        for attempt in range(self.MAX_JSON_RETRIES + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=0.95,
                )
                output = (resp.choices[0].message.content or "").strip()
            except Exception as e:
                error_msg = f"Hallucination Reviewer LLM 呼叫失敗：{e}"
                _log(f"[Hallucination Reviewer] ❌ {error_msg}")
                # Fail closed: if the safety reviewer cannot run, do not allow writing.
                return {
                    "agree": "no",
                    "comment": f"審查失敗（{e}），為安全起見拒絕寫入",
                    "thinking": "",
                    "error": str(e),
                }

            attempt_label = f" (重試 {attempt})" if attempt > 0 else ""
            _log(f"\n{'▲'*60}")
            _log(f"[Hallucination Reviewer] 審查輸出{attempt_label}")
            _log(f"{'▲'*60}")
            _log(output)
            _log(f"{'▲'*60}")
            if behavior_context:
                append_behavior_event(
                    behavior_context.get("folder_path"),
                    behavior_context.get("date_str"),
                    agent="hallucination_subagent",
                    event_type="llm_output",
                    label="輸出",
                    title="幻覺檢查 Subagent 輸出",
                    content=output or "",
                    meta={"attempt": attempt},
                )

            # 嘗試解析 JSON
            json_match = re.search(r'\{[\s\S]*\}', output)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    agree = str(result.get("agree", "")).strip().lower()
                    comment = result.get("comment", "")
                    thinking = result.get("thinking", "")

                    review_verdict = "✅ 通過" if agree == "yes" else "❌ 未通過"
                    _log(f"[Hallucination Reviewer] 審查結論：{review_verdict}")
                    _log(f"[Hallucination Reviewer] Comment：{comment}")

                    return {"agree": agree, "comment": comment, "thinking": thinking}
                except json.JSONDecodeError:
                    pass

            # JSON 解析失敗，重試
            if attempt < self.MAX_JSON_RETRIES:
                _log(f"[Hallucination Reviewer] JSON 解析失敗，第 {attempt + 1} 次重試...")
                retry_msg = (
                    "（系統提示：你剛才的輸出無法被解析為合法的 JSON 格式。請重新輸出。）\n"
                    '請確保輸出包含完整的 JSON 物件：{"thinking": "...", "agree": "yes 或 no", "comment": "..."}'
                )
                messages.append({"role": "assistant", "content": output or ""})
                messages.append({"role": "user", "content": retry_msg})
            else:
                _log(f"[Hallucination Reviewer] JSON 解析失敗，已達重試上限，拒絕寫入")

        # Fail closed: malformed reviewer output should not pass the gate.
        return {
            "agree": "no",
            "comment": "JSON 解析失敗，為安全起見拒絕寫入",
            "thinking": "",
            "error": "JSON parse failed",
        }
