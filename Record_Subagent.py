"""
Record_Subagent.py - 病歷登載助理 (Record Subagent)
負責根據主 Agent 的方針，以行級修改 (line-by-line diff) 的方式精確更新病歷。
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Callable, Optional

from openai import OpenAI
from multimodal_utils import inject_images_into_messages

from agent_behavior_log import append_behavior_event
from Hallucination_Subagent import HallucinationSubagent

# ════════════════════════════════════════════════════════════════
# Prompt 載入
# ════════════════════════════════════════════════════════════════
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_prompt(filename: str) -> str:
    path = os.path.join(CURRENT_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ════════════════════════════════════════════════════════════════
# 行級修改引擎
# ════════════════════════════════════════════════════════════════

def apply_operations(
    text: str, operations: list[dict]
) -> tuple[bool, str, list[str], list[str]]:
    """
    對文字內容套用一系列行級修改操作。

    所有操作的 line 一律指向「原始內容」的行號（與顯示給模型、附行號的那一份
    一致），不需考慮先前操作造成的位移。先整批驗證，全部合法才一次重建；只要
    有任一操作不合法（未知 op、行號超出範圍、同一行衝突），整批退回不套用。

    Args:
        text: 原始文字內容
        operations: 操作列表，每個操作是 {"op": "insert"|"delete"|"replace", "line": N, "content": "..."}

    Returns:
        (是否全部合法, 修改後的文字, 操作日誌列表, 失敗原因列表)
        當 ok 為 False 時，回傳的文字為原始 text（未做任何修改）。
    """
    lines = text.split("\n") if text else [""]
    n = len(lines)

    logs: list[str] = []
    failures: list[str] = []
    normalized_ops: list[dict] = []

    # ── 階段一：整批驗證（全部對照原始行號 n） ──
    # 記錄每個原始行已被哪個「就地操作」(delete/replace) 指向，偵測同行衝突。
    inplace_targets: dict[int, str] = {}

    for i, op_data in enumerate(operations, 1):
        op = op_data.get("op", "")
        line_num = op_data.get("line", 0)
        normalized = dict(op_data)

        if op == "insert":
            # 可插在第 1 行之前 ~ 最後一行之後（n+1）
            if not isinstance(line_num, int) or line_num < 1:
                failures.append(
                    f"第{i}個操作 insert：行號 {line_num} 超出可插入範圍 1~{n + 1}（原文共 {n} 行）"
                )
            elif line_num > n + 1:
                normalized["line"] = n + 1
                logs.append(
                    f"[NORMALIZE] 第{i}個 insert 行號 {line_num} 超出原文範圍，"
                    f"視為文末追加 line={n + 1}"
                )
        elif op in ("delete", "replace"):
            if not isinstance(line_num, int) or not (1 <= line_num <= n):
                failures.append(
                    f"第{i}個操作 {op}：行號 {line_num} 超出範圍 1~{n}（原文共 {n} 行）"
                )
            elif line_num in inplace_targets:
                failures.append(
                    f"第{i}個操作 {op}：行 {line_num} 已被 {inplace_targets[line_num]} 指向，"
                    f"同一行不可重複 delete/replace"
                )
            else:
                inplace_targets[line_num] = op
        else:
            failures.append(f"第{i}個操作：未知 op「{op}」")
        normalized_ops.append(normalized)

    if failures:
        return False, text, logs, failures

    # ── 階段二：一次重建（依原始行號，不受位移影響） ──
    # 預先把 insert 依目標行號分組（同一行多個 insert 依陣列順序排列）。
    inserts_before: dict[int, list[str]] = {}
    replaced: dict[int, str] = {}
    deleted: set[int] = set()
    for op_data in normalized_ops:
        op = op_data["op"]
        line_num = op_data["line"]
        content = op_data.get("content", "")
        if op == "insert":
            inserts_before.setdefault(line_num, []).append(content)
        elif op == "replace":
            replaced[line_num] = content
        elif op == "delete":
            deleted.add(line_num)

    rebuilt: list[str] = []
    original_is_empty = text == ""
    if original_is_empty and set(inserts_before).issubset({1, n + 1}) and not replaced and not deleted:
        for ins in inserts_before.get(1, []):
            rebuilt.append(ins)
            logs.append(f"[INSERT] 空白文件開頭: \"{ins}\"")
    else:
        for line_num in range(1, n + 1):
            for ins in inserts_before.get(line_num, []):
                rebuilt.append(ins)
                logs.append(f"[INSERT] 行{line_num}前: \"{ins}\"")
            if line_num in deleted:
                logs.append(f"[DELETE] 行{line_num}: \"{lines[line_num - 1]}\"")
                continue
            if line_num in replaced:
                logs.append(f"[REPLACE] 行{line_num}: \"{lines[line_num - 1]}\" → \"{replaced[line_num]}\"")
                rebuilt.append(replaced[line_num])
            else:
                rebuilt.append(lines[line_num - 1])
    # 尾端追加（line == n+1）
    for ins in inserts_before.get(n + 1, []):
        rebuilt.append(ins)
        logs.append(f"[INSERT] 末尾: \"{ins}\"")

    return True, "\n".join(rebuilt), logs, failures


def format_content_with_line_numbers(text: str) -> str:
    """將文字內容加上行號，供 LLM 閱讀定位。"""
    lines = text.split("\n") if text else [""]
    numbered = [f"{i+1:3d} | {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered)


# ════════════════════════════════════════════════════════════════
# Record Subagent 類別
# ════════════════════════════════════════════════════════════════

class RecordSubagent:
    """
    病歷登載助理：根據主 Agent 的方針，以行級修改的方式更新病歷。

    使用方式：
        subagent = RecordSubagent(config)
        result = subagent.execute(
            note_content="...",
            at_content="...",
            target_field="note",
            guidelines="請在第3行補充...",
        )
    """

    MAX_JSON_RETRIES = 3

    def __init__(self, config: dict, hallucination_config: dict | None = None):
        """
        Args:
            config: {
                "api_url": "http://localhost:1234/v1",
                "api_key": "lm-studio",
                "model_name": "model-name"
            }
            hallucination_config: 同上，給 Hallucination Subagent 用。
                額外支援 "detection_strength" (累積次數) 和 "max_review_rounds" (最大輪次)。
                若為 None 則跳過幻覺審查。
        """
        self.client = OpenAI(
            base_url=config.get("api_url", "http://localhost:1234/v1"),
            api_key=config.get("api_key", "lm-studio"),
        )
        self.model = config.get("model_name", "")
        self.max_tokens = int(config.get("max_tokens", 8000))
        self.temperature = float(config.get("temperature", 0.7))
        self.system_prompt_template = _load_prompt("prompt_record_update.txt")

        # 載入標準病歷模板
        template_path = os.path.join(CURRENT_DIR, "Record_Template.txt")
        if os.path.isfile(template_path):
            with open(template_path, "r", encoding="utf-8") as f:
                self.record_template = f.read()
        else:
            self.record_template = "（尚未設定標準病歷模板）"

        # 建立 Hallucination Subagent
        if hallucination_config and hallucination_config.get("model_name"):
            self.hallucination_reviewer = HallucinationSubagent(hallucination_config)
            self.max_review_rounds = max(1, int(hallucination_config.get("max_review_rounds", 5) or 5))
            try:
                detection_strength = int(hallucination_config.get("detection_strength", 2))
            except (TypeError, ValueError):
                detection_strength = 2
            if detection_strength < 0:
                # 負數視為無效輸入，回到預設值（避免手改 config 打錯字無聲停用審查）
                detection_strength = 2
            # 明確的 0 = 不審查直接放行（研究對照組模式）；>0 則 clamp 進 [1, max_review_rounds]
            if detection_strength == 0:
                self.detection_strength = 0
            else:
                self.detection_strength = min(detection_strength, self.max_review_rounds)
        else:
            self.hallucination_reviewer = None
            self.detection_strength = 2
            self.max_review_rounds = 5

    def execute(
        self,
        note_content: str,
        at_content: str,
        target_field: str,
        guidelines: str,
        conversation_history: str = "",
        last_visit_block: str = "",
        history_summary: str = "",
        interview_dialogue: str = "",
        forum_history: str = "",
        loaded_files_block: str = "",
        image_files: list | None = None,
        log_callback: Optional[Callable[[str], None]] = None,
        behavior_context: dict | None = None,
    ) -> dict[str, Any]:
        """
        執行病歷更新。

        Args:
            note_content: 當前 NOTE 內容
            at_content: 當前 ASSESSMENT & TREATMENT 內容
            target_field: "note" 或 "assessment_&_treatment"
            guidelines: 主 Agent 給予的更新方針
            conversation_history: 人類醫師與主治醫師 Agent 的精簡版對話歷史
            last_visit_block: 上次就診病歷內容
            history_summary: 歷史病歷摘要
            interview_dialogue: 完整問診對話紀錄
            forum_history: 醫療問答討論區內容
            loaded_files_block: 當輪讀取檔案暫存區內容
            log_callback: 選用，日誌回呼函式

        Returns:
            {
                "success": bool,
                "note": str,           # 更新後的 NOTE
                "at": str,             # 更新後的 A&T
                "operations": list,    # 執行的操作列表
                "op_logs": list[str],  # 操作日誌
                "error": str | None,   # 錯誤訊息
            }
        """

        def _log(msg: str):
            if log_callback:
                log_callback(msg)
            print(msg)

        # 決定要編輯的內容
        if target_field == "note":
            target_content = note_content
        elif target_field == "assessment_&_treatment":
            target_content = at_content
        else:
            return {
                "success": False,
                "note": note_content,
                "at": at_content,
                "operations": [],
                "op_logs": [],
                "error": f"未知的 target_field: {target_field}",
                "review_result": "跳過",
                "review_rounds": 0,
                "review_comment": "",
            }

        # 建立帶行號的內容
        numbered_content = format_content_with_line_numbers(target_content)

        # 建立另一欄位的參考內容（唯讀）
        if target_field == "note":
            other_field_name = "ASSESSMENT & TREATMENT"
            other_content = at_content
        else:
            other_field_name = "NOTE"
            other_content = note_content

        # 建立 user prompt
        conversation_section = ""
        if conversation_history:
            conversation_section = f"""\n## 【人類醫師與 AI 主治醫師的互動過程】
{conversation_history}
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

        user_prompt = f"""## 【指定更新欄位】
{target_field}
{conversation_section}{interview_section}{forum_section}{loaded_files_section}
## 【今日病歷(或當前編輯頁面的病歷) - {other_field_name}（參考用，不可修改）】
{other_content if other_content else '（空白）'}

## 【今日病歷(或當前編輯頁面的病歷) - 待修改欄位（附行號）】
{numbered_content}

## 【主治醫師的更新方針】
{guidelines}

請根據以上方針，輸出你的修改操作。"""

        system_prompt = self.system_prompt_template.replace(
            "{record_template}", self.record_template
        ).replace(
            "{last_visit_block}", last_visit_block or "（無上次就診紀錄）"
        ).replace(
            "{history_summary}", history_summary or "（無歷史病歷摘要）"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        _log(f"\n{'='*60}")
        _log(f"[Record Subagent] 開始更新 {target_field}")
        _log(f"{'='*60}")
        _log(f"[Record Subagent] 方針:")
        _log(guidelines)
        _log(f"\n{'─'*40}")
        _log(f"[Record Subagent] ══ 送入 LLM 的 User Prompt ══")
        _log(user_prompt)
        _log(f"{'─'*40}")
        if behavior_context:
            append_behavior_event(
                behavior_context.get("folder_path"),
                behavior_context.get("date_str"),
                agent="record_subagent",
                event_type="llm_input",
                label="輸入",
                title="病歷登載 Subagent 輸入",
                content=user_prompt,
            )

        # 注入多模態圖片
        if image_files:
            messages = inject_images_into_messages(messages, image_files)

        # 呼叫 LLM（含 JSON 解析重試）
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
                error_msg = f"LLM 呼叫失敗：{e}"
                _log(f"[Record Subagent] ❌ {error_msg}")
                return {
                    "success": False,
                    "note": note_content,
                    "at": at_content,
                    "operations": [],
                    "op_logs": [],
                    "error": error_msg,
                    "review_result": "跳過",
                    "review_rounds": 0,
                    "review_comment": "",
                }

            _log(f"\n[Record Subagent] ══ LLM 原始輸出 (attempt {attempt}) ══")
            _log(output)
            _log(f"{'─'*40}")
            if behavior_context:
                append_behavior_event(
                    behavior_context.get("folder_path"),
                    behavior_context.get("date_str"),
                    agent="record_subagent",
                    event_type="llm_output",
                    label="輸出",
                    title="病歷登載 Subagent 輸出",
                    content=output or "",
                    meta={"attempt": attempt},
                )

            # 嘗試解析 JSON
            json_match = re.search(r"\{[\s\S]*\}", output)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    break
                except json.JSONDecodeError:
                    pass

            # 重試
            if attempt < self.MAX_JSON_RETRIES:
                retry_msg = (
                    "（系統提示：你剛才的輸出無法被解析為合法的 JSON 格式。請重新輸出。）\n"
                    "請確保輸出包含完整的 JSON 物件。"
                )
                messages.append({"role": "assistant", "content": output or ""})
                messages.append({"role": "user", "content": retry_msg})
                _log(f"[Record Subagent] JSON 解析失敗，第 {attempt + 1} 次重試...")
            else:
                _log(f"[Record Subagent] JSON 解析失敗，已達重試上限")

        if parsed is None:
            return {
                "success": False,
                "note": note_content,
                "at": at_content,
                "operations": [],
                "op_logs": [],
                "error": f"JSON 解析失敗。LLM 原始輸出: {output or ''}",
                "review_result": "跳過",
                "review_rounds": 0,
                "review_comment": "",
            }

        # 提取操作
        operations = parsed.get("operations", [])
        thinking = parsed.get("thinking", "")

        _log(f"[Record Subagent] ══ 解析結果 ══")
        _log(f"[Record Subagent] thinking: {thinking}")
        _log(f"[Record Subagent] 操作數量: {len(operations)}")
        if operations:
            _log(f"[Record Subagent] 操作列表: {json.dumps(operations, ensure_ascii=False, indent=2)}")

        if not operations:
            _log(f"[Record Subagent] 無需修改")
            return {
                "success": True,
                "note": note_content,
                "at": at_content,
                "operations": [],
                "op_logs": ["無需修改"],
                "error": None,
                "review_result": "跳過",
                "review_rounds": 0,
                "review_comment": "",
            }

        # 套用操作（嚴格驗證：任一操作不合法則整批退回，不寫入）
        ok, new_content, op_logs, op_failures = apply_operations(target_content, operations)

        if not ok:
            error_msg = "操作已退回未寫入：" + "；".join(op_failures)
            _log(f"[Record Subagent] ❌ {error_msg}")
            if behavior_context:
                append_behavior_event(
                    behavior_context.get("folder_path"),
                    behavior_context.get("date_str"),
                    agent="record_subagent",
                    event_type="model_error",
                    label="操作退回",
                    title="病歷登載 Subagent 操作驗證失敗，整批退回",
                    content=error_msg,
                    severity="warning",
                )
            return {
                "success": False,
                "note": note_content,
                "at": at_content,
                "operations": [],
                "op_logs": op_failures,
                "error": error_msg,
                "review_result": "跳過",
                "review_rounds": 0,
                "review_comment": "",
            }

        _log(f"\n[Record Subagent] ══ 操作日誌 ══")
        for log_line in op_logs:
            _log(f"  {log_line}")

        _log(f"\n[Record Subagent] ══ 更新後的內容 ({target_field}) ══")
        _log(new_content)
        _log(f"{'─'*40}")

        # 回傳結果
        new_note = new_content if target_field == "note" else note_content
        new_at = new_content if target_field == "assessment_&_treatment" else at_content

        # ════════════════════════════════════════
        # Hallucination Review Gate
        # ════════════════════════════════════════
        if self.hallucination_reviewer is None:
            _log(f"[Record Subagent] 跳過幻覺審查（未設定 Hallucination Reviewer）")
            return {
                "success": True,
                "note": new_note,
                "at": new_at,
                "operations": operations,
                "op_logs": op_logs,
                "error": None,
                "review_result": "跳過",
                "review_rounds": 0,
                "review_comment": "",
            }

        if self.detection_strength == 0:
            # 研究對照組模式：審查器有設定但檢測強度為 0 → 不審查直接放行
            _log(f"[Record Subagent] 檢測強度為 0，跳過幻覺審查直接放行（對照組模式）")
            if behavior_context:
                append_behavior_event(
                    behavior_context.get("folder_path"),
                    behavior_context.get("date_str"),
                    agent="record_subagent",
                    event_type="tool_event",
                    label="審查停用",
                    title="幻覺審查停用（檢測強度0，對照組模式），未審查直接放行",
                    content="本次病歷更新未經幻覺審查。",
                    severity="warning",
                )
            return {
                "success": True,
                "note": new_note,
                "at": new_at,
                "operations": operations,
                "op_logs": op_logs,
                "error": None,
                "review_result": "未審查（檢測強度0/對照組）",
                "review_rounds": 0,
                "review_comment": "",
            }

        _log(f"\n[Record Subagent] 開始 Hallucination Review 循環...")
        hallucination_feedbacks: list[dict] = []
        pass_count = 0
        review_content = new_content  # 當前要審查的內容

        for review_round in range(1, self.max_review_rounds + 1):
            _log(f"\n[Record Subagent] === Hallucination Review 第 {review_round} 輪 ===")

            review_result = self.hallucination_reviewer.review(
                field=target_field,
                content=review_content,
                conversation_history=conversation_history,
                last_visit_block=last_visit_block,
                history_summary=history_summary,
                interview_dialogue=interview_dialogue,
                forum_history=forum_history,
                loaded_files_block=loaded_files_block,
                image_files=image_files,
                log_callback=log_callback,
                behavior_context=behavior_context,
            )
            reviewer_comment = review_result.get("comment", "")

            if review_result.get("error"):
                _log(f"[Record Subagent] ❌ 審查服務異常，跳過重寫直接拒絕：{reviewer_comment}")
                return {
                    "success": False,
                    "note": note_content,
                    "at": at_content,
                    "operations": operations,
                    "op_logs": op_logs,
                    "error": f"幻覺審查服務異常，病歷未寫入：{reviewer_comment}",
                    "review_result": "審查失敗（服務異常）",
                    "review_rounds": review_round,
                    "review_comment": reviewer_comment,
                }

            if review_result.get("agree") == "yes":
                pass_count += 1
                _log(f"[Record Subagent] Hallucination Review 第 {review_round} 輪通過 "
                     f"(累積 {pass_count}/{self.detection_strength})")

                if pass_count >= self.detection_strength:
                    # 達到檢測強度，正式通過
                    _log(f"[Record Subagent] ✅ Hallucination Review 正式通過！")
                    new_note_final = review_content if target_field == "note" else note_content
                    new_at_final = review_content if target_field == "assessment_&_treatment" else at_content
                    return {
                        "success": True,
                        "note": new_note_final,
                        "at": new_at_final,
                        "operations": operations,
                        "op_logs": op_logs,
                        "error": None,
                        "review_result": "通過",
                        "review_rounds": review_round,
                        "review_comment": reviewer_comment,
                    }
                else:
                    # 尚未達到檢測強度，繼續下一輪審查（不重寫）
                    continue
            else:
                # 審查未通過，累計建議，重寫
                _log(f"[Record Subagent] ❌ Hallucination Review 第 {review_round} 輪未通過")
                _log(f"[Record Subagent] 建議：{reviewer_comment}")
                hallucination_feedbacks.append({"content": review_content, "comment": reviewer_comment})

                if review_round >= self.max_review_rounds:
                    # Fail closed: reviewer did not pass the proposed record content.
                    _log(f"[Record Subagent] 已達最大審查輪次 {self.max_review_rounds}，拒絕寫入")
                    return {
                        "success": False,
                        "note": note_content,
                        "at": at_content,
                        "operations": operations,
                        "op_logs": op_logs,
                        "error": "幻覺審查未通過，病歷未寫入",
                        "review_result": "未通過（達上限）",
                        "review_rounds": review_round,
                        "review_comment": reviewer_comment,
                    }

                # 重新撰寫：組裝 feedback 注入 user prompt 讓 Record Writer 重寫
                _log(f"[Record Subagent] 重新撰寫病歷（第 {review_round + 1} 輪）...")
                feedback_text = "\n## 【Hallucination Reviewer 歷次修改建議與被退件內容】\n"
                for idx, fb in enumerate(hallucination_feedbacks, 1):
                    feedback_text += f"[第{idx}次準備上傳的病歷內容]\n{fb['content']}\n"
                    feedback_text += f"[第{idx}次 Hallucination Reviewer 的修改建議]\n{fb['comment']}\n\n"

                # 重新呼叫 Record Writer LLM
                interview_rewrite_section = ""
                if interview_dialogue:
                    interview_rewrite_section = f"""\n## 【完整問診對話紀錄】
{interview_dialogue}
"""

                forum_rewrite_section = f"""
## 【醫療問答討論區】
{forum_history if forum_history else '（空白）'}
"""

                loaded_files_rewrite_section = f"""
## 【當輪讀取檔案暫存區】
{loaded_files_block if loaded_files_block else '（空白）'}
"""

                rewrite_user_prompt = f"""## 【指定更新欄位】
{target_field}

## 【人類醫師與 AI 主治醫師的互動過程】
{conversation_history or '（無）'}
{interview_rewrite_section}{forum_rewrite_section}{loaded_files_rewrite_section}
## 【今日病歷(或當前編輯頁面的病歷) - {other_field_name}（參考用，不可修改）】
{other_content if other_content else '（空白）'}

## 【今日病歷(或當前編輯頁面的病歷) - 待修改欄位（附行號）】
{format_content_with_line_numbers(review_content)}

## 【主治醫師的更新方針】
{guidelines}
{feedback_text}
請根據 Hallucination Reviewer 的建議修改病歷。輸出你的修改操作。"""

                rewrite_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": rewrite_user_prompt},
                ]

                _log(f"\n[Record Subagent] 重寫 User Prompt:")
                _log(rewrite_user_prompt)
                if behavior_context:
                    append_behavior_event(
                        behavior_context.get("folder_path"),
                        behavior_context.get("date_str"),
                        agent="record_subagent",
                        event_type="llm_input",
                        label="重寫輸入",
                        title=f"病歷登載 Subagent 重寫輸入 第 {review_round + 1} 輪",
                        content=rewrite_user_prompt,
                    )

                # 注入多模態圖片
                if image_files:
                    rewrite_messages = inject_images_into_messages(rewrite_messages, image_files)

                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=rewrite_messages,
                        max_completion_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_p=0.9,
                    )
                    rewrite_output = (resp.choices[0].message.content or "").strip()
                except Exception as e:
                    _log(f"[Record Subagent] ❌ 重寫 LLM 呼叫失敗：{e}")
                    break

                _log(f"\n[Record Subagent] 重寫 LLM 輸出:")
                _log(rewrite_output)
                if behavior_context:
                    append_behavior_event(
                        behavior_context.get("folder_path"),
                        behavior_context.get("date_str"),
                        agent="record_subagent",
                        event_type="llm_output",
                        label="重寫輸出",
                        title=f"病歷登載 Subagent 重寫輸出 第 {review_round + 1} 輪",
                        content=rewrite_output,
                    )

                # 解析重寫結果
                json_match = re.search(r"\{[\s\S]*\}", rewrite_output)
                if json_match:
                    try:
                        rewrite_parsed = json.loads(json_match.group())
                        rewrite_ops = rewrite_parsed.get("operations", [])
                        if rewrite_ops:
                            rw_ok, rw_content, rewrite_logs, rw_failures = apply_operations(
                                review_content, rewrite_ops
                            )
                            if rw_ok:
                                review_content = rw_content
                                op_logs.extend(rewrite_logs)
                                operations.extend(rewrite_ops)
                                _log(f"[Record Subagent] 重寫完成，執行 {len(rewrite_ops)} 個操作")
                            else:
                                _log(
                                    "[Record Subagent] 重寫操作驗證失敗，保留前一版內容："
                                    + "；".join(rw_failures)
                                )
                    except json.JSONDecodeError:
                        _log(f"[Record Subagent] 重寫 JSON 解析失敗")

        # Fail closed: the reviewer loop ended without reaching detection_strength.
        return {
            "success": False,
            "note": note_content,
            "at": at_content,
            "operations": operations,
            "op_logs": op_logs,
            "error": "幻覺審查未達通過門檻，病歷未寫入",
            "review_result": f"未通過（僅通過{pass_count}次）",
            "review_rounds": self.max_review_rounds,
            "review_comment": hallucination_feedbacks[-1]["comment"] if hallucination_feedbacks else "",
        }
