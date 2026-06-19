"""
Note_Review_Subagent.py - 病歷檢查員 Subagent
對照標準病歷格式（甲～癸），逐項掃描 NOTE 欄位的完整性，
找出尚未完成的項目，並統整出「資訊蒐集方針」。
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Optional

from openai import OpenAI
from agent_behavior_log import append_behavior_event

# ════════════════════════════════════════════════════════════════
# Prompt 載入
# ════════════════════════════════════════════════════════════════
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_prompt(filename: str) -> str:
    path = os.path.join(CURRENT_DIR, filename)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


# ════════════════════════════════════════════════════════════════
# Note Review Subagent
# ════════════════════════════════════════════════════════════════

class NoteReviewSubagent:
    """
    病歷檢查員 Subagent。

    對照標準病歷格式（甲～癸），逐項掃描 NOTE 欄位的完整性，
    找出尚未完成的項目，並統整出「資訊蒐集方針」交給問診助理去補問。

    使用方式：
        nr = NoteReviewSubagent(config)
        result = nr.execute(
            note_content="...",
            interview_dialogue="...",
            conversation_history="...",
        )
    """

    MAX_JSON_RETRIES = 2

    def __init__(self, config: dict):
        """
        Args:
            config: {
                "api_url": "http://localhost:1234/v1",
                "api_key": "lm-studio",
                "model_name": "model-name",
                "max_tokens": 20000,
                "temperature": 1.0,
            }
        """
        self.client = OpenAI(
            base_url=config.get("api_url", "http://localhost:1234/v1"),
            api_key=config.get("api_key", "lm-studio"),
        )
        self.model = config.get("model_name", "")
        self.max_tokens = int(config.get("max_tokens", 20000))
        self.temperature = float(config.get("temperature", 1.0))

        # 載入 system prompt 模板
        self.system_prompt_template = _load_prompt("prompt_note_review.txt")

        # 載入標準病歷模板
        template_path = os.path.join(CURRENT_DIR, "Record_Template.txt")
        if os.path.isfile(template_path):
            with open(template_path, "r", encoding="utf-8") as f:
                self.record_template = f.read()
        else:
            self.record_template = "（尚未設定標準病歷模板）"

    # ════════════════════════════════════════════════════════════
    # 公開介面
    # ════════════════════════════════════════════════════════════

    def execute(
        self,
        note_content: str,
        interview_dialogue: str = "",
        conversation_history: str = "",
        record_diff_context: str = "",
        loaded_files_block: str = "",
        log_callback: Optional[Callable[[str], None]] = None,
        behavior_context: dict | None = None,
    ) -> dict[str, Any]:
        """
        掃描 NOTE 欄位的甲～癸完整性。

        Args:
            note_content: 當前 NOTE 內容
            interview_dialogue: 完整問診對話紀錄
            conversation_history: 人類醫師與 AI 主治醫師的互動過程
            log_callback: 選用，日誌回呼

        Returns:
            {
                "success": bool,
                "needs_collection": bool,   # True=有待補問, False=全部完成
                "guidelines": str,          # 資訊蒐集方針（needs_collection=True 時有值）
                "thinking": str,            # 掃描思考過程
                "error": str | None,
            }
        """
        def _log(msg: str):
            if log_callback:
                log_callback(msg)
            print(msg)

        if not self.model:
            return {
                "success": False,
                "needs_collection": False,
                "guidelines": "",
                "update_reminder": "",
                "thinking": "",
                "error": "NR Subagent model_name 未設定",
            }

        if not self.system_prompt_template:
            return {
                "success": False,
                "needs_collection": False,
                "guidelines": "",
                "update_reminder": "",
                "thinking": "",
                "error": "prompt_note_review.txt 不存在",
            }

        # 格式化 system prompt（注入標準病歷模板）
        system_prompt = self.system_prompt_template.replace(
            "{record_template}", self.record_template
        )

        # 組裝 user prompt
        record_diff_text = record_diff_context or "## 【病歷修改 diff 過程】\n（無病歷版本歷史）"
        user_prompt = f"""{record_diff_text}

## 【今日病歷(或當前編輯頁面的病歷) - NOTE】
{note_content or '（空白）'}

## 【人類醫師與 AI 主治醫師的互動過程】
{conversation_history or '（無互動過程）'}

## 【完整問診對話紀錄】
{interview_dialogue or '（無問診對話紀錄）'}

## 【當輪讀取檔案暫存區】
{loaded_files_block if loaded_files_block else '（空白）'}

請對照標準病歷格式，掃描上述【今日病歷(或當前編輯頁面的病歷) - NOTE】的完整性，找出待完成項目，以 JSON 格式輸出資訊蒐集方針。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        _log(f"\n{'◆'*60}")
        _log(f"[Note Review Subagent] 掃描輸入")
        _log(f"{'◆'*60}")
        _log(
            user_prompt[:50000] + "..."
            if len(user_prompt) > 50000
            else user_prompt
        )
        _log(f"{'◆'*60}\n")
        if behavior_context:
            append_behavior_event(
                behavior_context.get("folder_path"),
                behavior_context.get("date_str"),
                agent="note_review_subagent",
                event_type="llm_input",
                label="輸入",
                title="病歷檢查員 Subagent 輸入",
                content=user_prompt,
            )

        # 呼叫 LLM（含 JSON 解析重試）
        for attempt in range(self.MAX_JSON_RETRIES + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                raw_output = (resp.choices[0].message.content or "").strip()
            except Exception as e:
                _log(f"[Note Review Subagent] LLM 呼叫失敗: {e}")
                return {
                    "success": False,
                    "needs_collection": False,
                    "guidelines": "",
                    "update_reminder": "",
                    "thinking": f"LLM 呼叫失敗（{e}）",
                    "error": str(e),
                }

            attempt_label = f" (重試 {attempt})" if attempt > 0 else ""
            _log(f"\n{'◇'*60}")
            _log(f"[Note Review Subagent] 掃描輸出{attempt_label}")
            _log(f"{'◇'*60}")
            _log(
                raw_output[:50000] + "..."
                if len(raw_output) > 50000
                else raw_output
            )
            _log(f"{'◇'*60}\n")
            if behavior_context:
                append_behavior_event(
                    behavior_context.get("folder_path"),
                    behavior_context.get("date_str"),
                    agent="note_review_subagent",
                    event_type="llm_output",
                    label="輸出",
                    title="病歷檢查員 Subagent 輸出",
                    content=raw_output or "",
                    meta={"attempt": attempt},
                )

            # 解析 JSON
            json_match = re.search(r'\{[\s\S]*\}', raw_output)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    action_input = result.get("action_input", "")
                    thinking = result.get("thinking", "")
                    update_reminder = result.get("update_reminder", "")

                    # 判斷是否有待補問項目
                    needs_collection = (
                        action_input != "本次無需要蒐集的資訊"
                        and bool(action_input)
                    )

                    _log(
                        f"[Note Review Subagent] "
                        f"needs_collection={needs_collection}"
                    )
                    if update_reminder:
                        _log(
                            f"[Note Review Subagent] "
                            f"update_reminder={update_reminder[:500]}"
                        )

                    return {
                        "success": True,
                        "needs_collection": needs_collection,
                        "guidelines": action_input if needs_collection else "",
                        "update_reminder": update_reminder,
                        "thinking": thinking,
                        "error": None,
                    }
                except json.JSONDecodeError:
                    pass

            # JSON 解析失敗
            if attempt < self.MAX_JSON_RETRIES:
                _log(
                    f"[Note Review Subagent] JSON 解析失敗，"
                    f"第 {attempt + 1} 次重試..."
                )
                retry_msg = (
                    "（系統提示：你剛才的輸出無法被解析為合法的 JSON 格式。"
                    "請重新輸出完整的 JSON 物件。）"
                )
                messages.append({"role": "assistant", "content": raw_output or ""})
                messages.append({"role": "user", "content": retry_msg})
            else:
                _log(
                    "[Note Review Subagent] JSON 解析失敗，"
                    "已達重試上限，預設通過"
                )

        # 解析全部失敗，預設通過
        return {
            "success": True,
            "needs_collection": False,
            "guidelines": "",
            "update_reminder": "",
            "thinking": "JSON 解析失敗，預設通過",
            "error": None,
        }
