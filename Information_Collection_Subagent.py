"""
Information_Collection_Subagent.py - 問診助理 (Information Collection Subagent)
負責根據主治醫師的資訊蒐集方針，向患者進行多回合問診，蒐集完整臨床資訊。
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
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ════════════════════════════════════════════════════════════════
# Information Collection Subagent 類別
# ════════════════════════════════════════════════════════════════

class InformationCollectionSubagent:
    """
    問診助理 Subagent：根據主治醫師的資訊蒐集方針，
    向患者進行多回合問診，蒐集完整臨床資訊。

    使用方式：
        subagent = InformationCollectionSubagent(config)
        result = subagent.start_collection(
            guidelines="請詢問患者頭痛的發作頻率與伴隨症狀...",
            note_content="...",
            at_content="...",
        )
        # result = {"action": "ask_patient", "message": "...", "round": 1, ...}

        # 患者回答後
        result = subagent.receive_answer(
            patient_input="頭痛大概一個月兩三次...",
            note_content="...",
            at_content="...",
        )
    """

    MAX_JSON_RETRIES = 3

    def __init__(self, config: dict):
        """
        Args:
            config: {
                "api_url": "http://localhost:1234/v1",
                "api_key": "lm-studio",
                "model_name": "model-name",
                "max_tokens": 20000,
                "temperature": 0.7,
                "max_collection_rounds": 10,
            }
        """
        self.client = OpenAI(
            base_url=config.get("api_url", "http://localhost:1234/v1"),
            api_key=config.get("api_key", "lm-studio"),
        )
        self.model = config.get("model_name", "")
        self.max_tokens = int(config.get("max_tokens", 20000))
        self.temperature = float(config.get("temperature", 0.7))
        self.max_collection_rounds = int(config.get("max_collection_rounds", 10))

        # 載入 system prompt 模板
        self.system_prompt_template = _load_prompt(
            "prompt_information_collection_subagent.txt"
        )

        # 載入標準病歷模板
        template_path = os.path.join(CURRENT_DIR, "Record_Template.txt")
        if os.path.isfile(template_path):
            with open(template_path, "r", encoding="utf-8") as f:
                self.record_template = f.read()
        else:
            self.record_template = "（尚未設定標準病歷模板）"

        # ── 狀態初始化 ──
        # 當前循環的問診對話（start_collection 時重置）
        self.conversation: list[dict] = []
        # 跨循環累計的所有問診對話（僅 reset() 時重置）
        self.all_conversations: list[dict] = []
        # 跨循環連續的問診回合計數器
        self.dialogue_round: int = 0
        # 當前循環的 turn 計數器（每次 start_collection 重置）
        self.turn_count: int = 0
        # 各輪行為摘要紀錄（供 context 使用）
        self.turn_history: list[dict] = []
        # 主治醫師的資訊蒐集方針
        self.guidelines: str = ""
        # 格式化後的 system prompt
        self.system_prompt: str = ""
        self.interaction_history: str = ""
        self.forum_history: str = ""
        self.behavior_context: dict | None = None
        # 是否已完成問診
        self.finished: bool = False

    # ════════════════════════════════════════════════════════════
    # 公開介面
    # ════════════════════════════════════════════════════════════

    def start_collection(
        self,
        guidelines: str,
        interaction_history: str = "",
        note_content: str = "",
        at_content: str = "",
        last_visit_block: str = "",
        history_summary: str = "",
        forum_history: str = "",
        log_callback: Optional[Callable[[str], None]] = None,
        behavior_context: dict | None = None,
    ) -> dict:
        """
        啟動問診循環，產生第一個問題。

        Args:
            guidelines: 主治醫師給予的資訊蒐集方針
            interaction_history: 人類醫師與 AI 主治醫師的互動過程
            note_content: 當前 NOTE 內容
            at_content: 當前 ASSESSMENT & TREATMENT 內容
            last_visit_block: 上次就診病歷內容
            history_summary: 歷史病歷摘要
            forum_history: 醫療問答討論區的問答紀錄
            log_callback: 選用，日誌回呼函式

        Returns:
            {
                "action": "ask_patient" | "finish_collection",
                "message": str,
                "round": int,
                "thinking": str,
                "finished": bool,
            }
        """
        # 儲存方針
        self.guidelines = guidelines

        # 儲存互動過程與討論區紀錄（供 user prompt 使用）
        self.interaction_history = interaction_history
        self.forum_history = forum_history
        self.behavior_context = behavior_context

        # 重置當前循環狀態（跨循環狀態保留）
        self.conversation = []
        self.turn_count = 0
        self.turn_history = []
        self.finished = False

        # 建立 system prompt（注入歷史資訊與模板）
        self.system_prompt = self.system_prompt_template.replace(
            "{record_template}", self.record_template
        ).replace(
            "{last_visit_block}", last_visit_block or "（無上次就診紀錄）"
        ).replace(
            "{history_summary}", history_summary or "（無歷史病歷摘要）"
        ).replace(
            "{max_collection_rounds}", str(self.max_collection_rounds)
        )

        # 執行第一輪
        return self._execute_turn(note_content, at_content, log_callback)

    def receive_answer(
        self,
        patient_input: str,
        note_content: str = "",
        at_content: str = "",
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """
        接收患者回答，記錄到 conversation，然後呼叫 LLM 產生下一個問題或結束。

        Args:
            patient_input: 患者的回答內容
            note_content: 當前 NOTE 內容
            at_content: 當前 ASSESSMENT & TREATMENT 內容
            log_callback: 選用，日誌回呼函式

        Returns:
            {
                "action": "ask_patient" | "finish_collection",
                "message": str,
                "round": int,
                "thinking": str,
                "finished": bool,
            }
        """
        # 將患者回答記錄到對話歷史（使用當前的 dialogue_round）
        patient_msg = {
            "role": "patient",
            "content": patient_input,
            "round": self.dialogue_round,
        }
        self.conversation.append(patient_msg)
        self.all_conversations.append(patient_msg)
        if self.behavior_context:
            append_behavior_event(
                self.behavior_context.get("folder_path"),
                self.behavior_context.get("date_str"),
                agent="information_collection_subagent",
                event_type="tool_event",
                label=f"患者R{self.dialogue_round}",
                title=f"問診助理收到第 {self.dialogue_round} 回合回覆",
                content=patient_input,
            )

        # 將患者回應也記錄到上一輪的 turn_history
        if self.turn_history:
            self.turn_history[-1]["patient_response"] = patient_input

        # 執行下一輪
        return self._execute_turn(note_content, at_content, log_callback)

    def get_full_dialogue(self) -> str:
        """
        取得完整問診對話紀錄（格式化文字）。
        使用 self.all_conversations（跨循環累計）產出完整對話文字。

        Returns:
            格式化的完整問診對話紀錄字串
        """
        if not self.all_conversations:
            return "（無問診對話紀錄）"

        dialogue_text = ""
        for msg in self.all_conversations:
            r = msg.get("round", "?")
            if msg["role"] == "subagent":
                dialogue_text += f"＜第{r}回合(R{r})助理提問＞：{msg['content']}\n"
            elif msg["role"] == "patient":
                dialogue_text += f"＜第{r}回合(R{r})收到回應＞：{msg['content']}\n"

        return dialogue_text

    def get_dialogue_summary_for_history(self) -> str:
        """
        取得精簡版摘要（用於【互動過程】區塊的 step result）。
        只報告「本次啟動」的問診回合範圍，不含先前啟動的累積。

        Returns:
            精簡版問診摘要字串
        """
        if self.dialogue_round == 0:
            return "（問診助理尚未進行問診）"

        # 計算「本次啟動」涵蓋的回合範圍
        # 使用 self.conversation（當前這次 start_collection 的對話）
        rounds = [
            msg.get("round", 0)
            for msg in self.conversation
            if msg.get("round")
        ]
        if not rounds:
            return "（問診助理尚未進行問診）"

        min_r = min(rounds)
        max_r = max(rounds)
        total = max_r - min_r + 1

        return (
            f"問診助理完成 R{min_r}~R{max_r} "
            f"共{total}回合問診，詳見【完整問診對話紀錄】"
        )

    def reset(self):
        """重置所有狀態（新患者時呼叫）"""
        self.conversation = []
        self.all_conversations = []
        self.dialogue_round = 0
        self.turn_count = 0
        self.turn_history = []
        self.guidelines = ""
        self.system_prompt = ""
        self.finished = False

    # ── 狀態序列化（Session 持久化用）──

    def export_state(self) -> dict:
        """匯出問診助理狀態，供存檔使用"""
        return {
            "active": not self.finished,
            "finished": self.finished,
            "conversation": self.conversation,
            "all_conversations": self.all_conversations,
            "dialogue_round": self.dialogue_round,
            "turn_count": self.turn_count,
            "turn_history": self.turn_history,
            "guidelines": self.guidelines,
            "system_prompt": self.system_prompt,
            "interaction_history": self.interaction_history,
            "forum_history": self.forum_history,
        }

    def restore_state(self, state: dict):
        """從存檔還原問診助理狀態"""
        self.finished = bool(state.get("finished", False))
        self.conversation = state.get("conversation", state.get("all_conversations", []))
        self.all_conversations = state.get("all_conversations", [])
        self.dialogue_round = state.get("dialogue_round", 0)
        self.turn_count = state.get("turn_count", 0)
        self.turn_history = state.get("turn_history", [])
        self.guidelines = state.get("guidelines", "")
        self.system_prompt = state.get("system_prompt", "")
        self.interaction_history = state.get("interaction_history", "")
        self.forum_history = state.get("forum_history", "")

    # ════════════════════════════════════════════════════════════
    # 內部方法
    # ════════════════════════════════════════════════════════════

    def _execute_turn(
        self,
        note_content: str,
        at_content: str,
        log_callback: Optional[Callable[[str], None]],
    ) -> dict:
        """
        執行一輪問診助理 Subagent 的推理。

        流程：
        1. 組裝 system prompt 與 user prompt
        2. 呼叫 LLM（含 JSON 重試機制）
        3. 解析 action 並更新狀態

        Args:
            note_content: 當前 NOTE 內容
            at_content: 當前 ASSESSMENT & TREATMENT 內容
            log_callback: 選用，日誌回呼函式

        Returns:
            {
                "action": "ask_patient" | "finish_collection",
                "message": str,
                "round": int,
                "thinking": str,
                "finished": bool,
            }
        """

        def _log(msg: str):
            if log_callback:
                log_callback(msg)
            print(msg)

        self.turn_count += 1

        # ── 組裝 user prompt ──
        # 病歷內容區塊
        current_input = f"""## 【今日病歷(或當前編輯頁面的病歷)】
NOTE:
{note_content or '（空白）'}
ASSESSMENT & TREATMENT:
{at_content or '（空白）'}

## 【主治醫師的資訊蒐集方針】
{self.guidelines}

"""

        # 各輪行為紀錄區塊
        subagent_turn_summaries = ""
        for turn in self.turn_history:
            subagent_turn_summaries += (
                f"[第{turn['turn_num']}輪的thinking] {turn['thinking']}\n"
                f"[第{turn['turn_num']}輪已執行的action] {turn['action_summary']}\n"
                f"[第{turn['turn_num']}輪預計的next_step] {turn['next_step']}\n"
            )
            if turn.get("patient_response"):
                subagent_turn_summaries += (
                    f"[第{turn['turn_num']}輪患者回應] "
                    f"{turn['patient_response']}\n"
                )

        if subagent_turn_summaries:
            current_input += f"""## 【問診助理各輪行為紀錄】
{subagent_turn_summaries}
"""
        else:
            current_input += """## 【問診助理各輪行為紀錄】
（空白）
"""

        # 本次循環的問診對話過程區塊
        conversation_text = ""
        for msg in self.conversation:
            r = msg.get("round", "?")
            if msg["role"] == "subagent":
                conversation_text += f"＜第{r}回合(R{r})助理提問＞：{msg['content']}\n"
            elif msg["role"] == "patient":
                conversation_text += f"＜第{r}回合(R{r})收到回應＞：{msg['content']}\n"

        # 人類醫師與 AI 主治醫師的互動過程區塊
        current_input += f"""## 【人類醫師與 AI 主治醫師的互動過程】
{self.interaction_history if self.interaction_history else '（空白）'}
"""

        # 醫療問答討論區的問答紀錄區塊
        current_input += f"""## 【醫療問答討論區】
{self.forum_history if self.forum_history else '（空白）'}
"""

        current_input += f"""## 【本次循環的問診對話過程】
{conversation_text if conversation_text else '（空白）'}
"""

        current_input += "請根據以上資訊，決定下一步動作。輸出 JSON。"

        # 組裝 messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": current_input},
        ]

        # ── 日誌輸出 ──
        _log(f"\n{'@'*60}")
        _log(f"[Info Collection Subagent] Turn {self.turn_count} 輸入")
        _log(f"{'@'*60}")
        _log(
            current_input[:50000] + "..."
            if len(current_input) > 50000
            else current_input
        )
        _log(f"{'@'*60}\n")
        if self.behavior_context:
            append_behavior_event(
                self.behavior_context.get("folder_path"),
                self.behavior_context.get("date_str"),
                agent="information_collection_subagent",
                event_type="llm_input",
                label=f"輸入T{self.turn_count}",
                title=f"問診助理 Turn {self.turn_count} 輸入",
                content=current_input,
            )

        # ── 呼叫 LLM（含 JSON 解析重試） ──
        output: str | None = None

        for attempt in range(self.MAX_JSON_RETRIES + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=0.8,
                )
                output = (resp.choices[0].message.content or "").strip()
            except Exception as e:
                error_msg = f"Info Collection Subagent LLM 呼叫失敗：{e}"
                _log(f"[Info Collection Subagent] ❌ {error_msg}")
                self.finished = True
                return {
                    "action": "finish_collection",
                    "message": f"（問診助理 LLM 呼叫失敗：{e}）",
                    "round": self.dialogue_round,
                    "thinking": "",
                    "finished": True,
                }

            # 記錄輸出
            attempt_label = f" (重試 {attempt})" if attempt > 0 else ""
            _log(f"\n{'@'*60}")
            _log(
                f"[Info Collection Subagent] Turn {self.turn_count} "
                f"輸出{attempt_label}"
            )
            _log(f"{'@'*60}")
            _log(output)
            _log(f"{'@'*60}\n")
            if self.behavior_context:
                append_behavior_event(
                    self.behavior_context.get("folder_path"),
                    self.behavior_context.get("date_str"),
                    agent="information_collection_subagent",
                    event_type="llm_output",
                    label=f"輸出T{self.turn_count}",
                    title=f"問診助理 Turn {self.turn_count} 輸出",
                    content=output or "",
                    meta={"attempt": attempt},
                )

            # 嘗試解析 JSON
            json_match = re.search(r"\{[\s\S]*\}", output)
            if json_match:
                try:
                    json.loads(json_match.group())
                    break  # 解析成功，跳出重試迴圈
                except json.JSONDecodeError:
                    pass

            # JSON 解析失敗，重試
            if attempt < self.MAX_JSON_RETRIES:
                retry_msg = (
                    "（系統提示：你剛才的輸出因為被截斷或格式錯誤，"
                    "無法被解析為合法的 JSON 格式。請重新思考並完整輸出一次。）\n"
                    "請確保輸出包含完整的 JSON 物件，格式如下：\n"
                    '{"thinking": "...", "action": "ask_patient|finish_collection", '
                    '"action_input": "...", "next_step": "..."}'
                )
                # 不把損壞的 output 加入 messages，避免引發解析錯誤
                messages.append({"role": "user", "content": retry_msg})
                _log(
                    f"[Info Collection Subagent] JSON 解析失敗，"
                    f"第 {attempt + 1} 次重試..."
                )
            else:
                _log(
                    f"[Info Collection Subagent] JSON 解析失敗，已達重試上限"
                )

        # ── 解析 LLM 輸出 ──
        return self._parse_and_execute(output or "", _log)

    def _parse_and_execute(
        self,
        output: str,
        _log: Callable[[str], None],
    ) -> dict:
        """
        解析問診助理 LLM 輸出的 JSON 並執行對應動作。

        Args:
            output: LLM 原始輸出字串
            _log: 日誌函式

        Returns:
            標準結果 dict
        """
        # 嘗試提取 JSON
        try:
            json_match = re.search(r"\{[\s\S]*\}", output)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                raise ValueError("找不到 JSON")
        except Exception as e:
            _log(f"[Info Collection Subagent] JSON 解析失敗：{e}")
            # 解析失敗時結束問診循環，用原始輸出作為結果
            self.finished = True
            return {
                "action": "finish_collection",
                "message": f"（問診助理輸出解析失敗）\n{output[:2000]}",
                "round": self.dialogue_round,
                "thinking": "",
                "finished": True,
            }

        action = parsed.get("action", "")
        action_input = parsed.get("action_input", "")
        thinking = parsed.get("thinking", "")
        next_step = parsed.get("next_step", "")

        # 建立動作摘要
        if action == "ask_patient":
            action_summary = (
                f"ask_patient: 向患者問診「{action_input[:200]}...」"
                if len(action_input) > 200
                else f"ask_patient: 向患者問診「{action_input}」"
            )
        elif action == "finish_collection":
            action_summary = "finish_collection: 結束問診循環"
        else:
            action_summary = f"{action}: {action_input[:200]}..."

        # 儲存到 turn_history
        self.turn_history.append({
            "turn_num": self.turn_count,
            "thinking": (
                thinking[:9999] + "..."
                if len(thinking) > 9999
                else thinking
            ),
            "action_summary": action_summary,
            "next_step": (
                next_step[:9999] + "..."
                if len(next_step) > 9999
                else next_step
            ),
        })

        _log(f"[Info Collection Subagent] 動作: {action}")

        # ── 執行動作 ──
        if action == "ask_patient":
            # 遞增問診回合計數器（跨循環連續）
            self.dialogue_round += 1

            # 記錄到對話歷史
            subagent_msg = {
                "role": "subagent",
                "content": action_input,
                "round": self.dialogue_round,
            }
            self.conversation.append(subagent_msg)
            self.all_conversations.append(subagent_msg)

            _log(f"[Info Collection Subagent] 向患者問診（第 {self.dialogue_round} 回合）")
            _log(f"[Info Collection Subagent] 問題：{action_input}")

            return {
                "action": "ask_patient",
                "message": action_input,
                "round": self.dialogue_round,
                "thinking": thinking,
                "finished": False,
            }

        elif action == "finish_collection":
            # 結束問診循環
            self.finished = True

            _log(f"[Info Collection Subagent] 問診完成，共 {self.dialogue_round} 回合")

            return {
                "action": "finish_collection",
                "message": action_input or "結束問診",
                "round": self.dialogue_round,
                "thinking": thinking,
                "finished": True,
            }

        else:
            # 未知動作，當作 finish_collection 處理
            _log(
                f"[Info Collection Subagent] 未知動作: {action}，"
                f"視為 finish_collection"
            )
            self.finished = True
            return {
                "action": "finish_collection",
                "message": f"（未知動作 {action}）\n{action_input}",
                "round": self.dialogue_round,
                "thinking": thinking,
                "finished": True,
            }
