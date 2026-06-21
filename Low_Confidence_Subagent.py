"""
Low_Confidence_Subagent.py - 低信心標註 Subagent
掃描 NOTE 中證據支撐度偏低的臨床描述，以 **...（原因）** 標註。
採用迭代掃描機制，累積 N 次 pass 後結束。
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
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def _inside_bold_span(text: str, index: int) -> bool:
    before = text[:index]
    return before.count("**") % 2 == 1


def _citation_tag_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"\[[^\]\n]*\]", text or "")]


def _overlaps_citation_tag(text: str, start: int, end: int) -> bool:
    return any(start < tag_end and end > tag_start for tag_start, tag_end in _citation_tag_spans(text))


def _replace_first_unbolded(text: str, original: str, replacement: str) -> tuple[str, str]:
    if not original:
        return text, "not_found"

    start = 0
    found_in_bold = False
    found_in_citation_tag = False
    while True:
        index = text.find(original, start)
        if index < 0:
            if found_in_citation_tag:
                return text, "citation_tag"
            return text, "already_tagged" if found_in_bold else "not_found"
        if not _inside_bold_span(text, index):
            end = index + len(original)
            if _overlaps_citation_tag(text, index, end):
                found_in_citation_tag = True
                start = end
                continue
            return text[:index] + replacement + text[index + len(original):], "replaced"
        found_in_bold = True
        start = index + len(original)


# ════════════════════════════════════════════════════════════════
# Low Confidence Subagent
# ════════════════════════════════════════════════════════════════

class LowConfidenceSubagent:
    """
    低信心標註 Subagent。

    掃描 NOTE 中每一項臨床事實描述，判斷是否有充分的原始證據支撐。
    證據薄弱、推論過度或幻覺風險的片段會被標註為 **片段（原因）**。
    採用迭代掃描機制，累積 detection_strength 次 pass 後結束。

    使用方式：
        lc = LowConfidenceSubagent(config)
        result = lc.execute(
            note_content="...",
            interview_dialogue="...",
            last_visit_block="...",
            history_summary="...",
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
                "max_scan_rounds": 8,
                "detection_strength": 4,
            }
        """
        self.client = OpenAI(
            base_url=config.get("api_url", "http://localhost:1234/v1"),
            api_key=config.get("api_key", "lm-studio"),
        )
        self.model = config.get("model_name", "")
        self.max_tokens = int(config.get("max_tokens", 20000))
        self.temperature = float(config.get("temperature", 1.0))
        self.max_scan_rounds = max(1, int(config.get("max_scan_rounds", 8) or 8))
        try:
            detection_strength = int(config.get("detection_strength", 4))
        except (TypeError, ValueError):
            detection_strength = 4
        if detection_strength < 0:
            # 負數視為無效輸入，回到預設值（避免手改 config 打錯字無聲停用掃描）
            detection_strength = 4
        # 明確的 0 = 不掃描直接放行（研究對照組模式）；
        # >0 則 clamp 進 [1, max_scan_rounds]，否則「達到檢測強度完成」永遠不可達，
        # 結束條件會默默退化成「跑滿輪次」。
        if detection_strength == 0:
            self.detection_strength = 0
        else:
            self.detection_strength = min(detection_strength, self.max_scan_rounds)

        # 載入 system prompt 模板
        self.system_prompt_template = _load_prompt(
            "prompt_low_confidence_check.txt"
        )

    # ════════════════════════════════════════════════════════════
    # 公開介面
    # ════════════════════════════════════════════════════════════

    def execute(
        self,
        note_content: str,
        interview_dialogue: str = "",
        conversation_history: str = "",
        last_visit_block: str = "",
        history_summary: str = "",
        record_diff_context: str = "",
        loaded_files_block: str = "",
        image_files: list | None = None,
        log_callback: Optional[Callable[[str], None]] = None,
        behavior_context: dict | None = None,
    ) -> dict[str, Any]:
        """
        執行迭代掃描，標註 NOTE 中的低信心片段。

        Args:
            note_content: 當前 NOTE 內容
            interview_dialogue: 完整問診對話紀錄
            conversation_history: 人類醫師與 AI 主治醫師的互動過程
            last_visit_block: 上次就診病歷
            history_summary: 歷史病歷摘要
            log_callback: 選用，日誌回呼

        Returns:
            {
                "success": bool,
                "annotated_note": str,  # 標註後的 NOTE
                "total_rounds": int,
                "total_annotated": int,
                "all_phrases": list[dict],  # 所有輪次的低信心片段
                "error": str | None,
            }
        """
        def _log(msg: str):
            if log_callback:
                log_callback(msg)
            print(msg)

        if self.detection_strength == 0:
            # 研究對照組模式：掃描器有設定但檢測強度為 0 → 不掃描直接放行
            _log("[LC Subagent] 檢測強度為 0，跳過低信心掃描直接放行（對照組模式）")
            if behavior_context:
                append_behavior_event(
                    behavior_context.get("folder_path"),
                    behavior_context.get("date_str"),
                    agent="low_confidence_subagent",
                    event_type="tool_event",
                    label="掃描停用",
                    title="低信心標註停用（檢測強度0，對照組模式），未掃描直接放行",
                    content="本次未執行低信心標註掃描，NOTE 未變更。",
                    severity="warning",
                )
            return {
                "success": True,
                "annotated_note": note_content,
                "total_rounds": 0,
                "total_annotated": 0,
                "all_phrases": [],
                "error": None,
                "skipped_control_group": True,
            }

        if not self.model:
            return {
                "success": False,
                "annotated_note": note_content,
                "total_rounds": 0,
                "total_annotated": 0,
                "all_phrases": [],
                "error": "LC Subagent model_name 未設定",
            }

        if not self.system_prompt_template:
            return {
                "success": False,
                "annotated_note": note_content,
                "total_rounds": 0,
                "total_annotated": 0,
                "all_phrases": [],
                "error": "prompt_low_confidence_check.txt 不存在",
            }

        # 格式化 system prompt（注入歷史資訊）
        system_prompt = self.system_prompt_template.format(
            last_visit_block=last_visit_block or "（無上次就診紀錄）",
            history_summary=history_summary or "（無歷史病歷）",
        )

        # 準備問診對話文字
        if not interview_dialogue:
            interview_dialogue = "（無問診對話紀錄）"

        # 迭代掃描
        current_note = note_content
        all_phrases: list[dict] = []
        total_annotated = 0
        already_tagged_summary = ""
        failed_phrases_summary = ""
        citation_tag_summary = ""
        lc_pass_count = 0
        lc_round = 0

        _log(f"\n{'='*60}")
        _log(f"[LC Subagent] 開始迭代掃描（至多 {self.max_scan_rounds} 輪，檢測強度 {self.detection_strength}）")
        _log(f"{'='*60}")

        for lc_round in range(1, self.max_scan_rounds + 1):
            _log(f"\n--- Low Confidence Check 第 {lc_round}/{self.max_scan_rounds} 輪 ---")

            check_result = self._scan_round(
                current_note=current_note,
                interview_dialogue=interview_dialogue,
                conversation_history=conversation_history,
                system_prompt=system_prompt,
                round_num=lc_round,
                already_tagged_summary=already_tagged_summary,
                failed_phrases_summary=failed_phrases_summary,
                citation_tag_summary=citation_tag_summary,
                record_diff_context=record_diff_context,
                loaded_files_block=loaded_files_block,
                behavior_context=behavior_context,
                log_callback=log_callback,
            )

            intercept = check_result.get("intercept", "pass")
            round_phrases = check_result.get("phrases", [])

            # 標註替換
            round_annotated = 0
            replacement_results: dict[int, str] = {}
            for phrase_info in round_phrases:
                original = phrase_info.get("original_phrase", "")
                reason = phrase_info.get("reason", "")
                replacement_status = "not_found"
                if original:
                    annotated = f"**{original}（{reason}）**"
                    current_note, replacement_status = _replace_first_unbolded(current_note, original, annotated)
                if replacement_status == "replaced":
                    round_annotated += 1
                replacement_results[id(phrase_info)] = replacement_status

            total_annotated += round_annotated

            # 收集片段紀錄
            for phrase_info in round_phrases:
                orig = phrase_info.get("original_phrase", "")
                replacement_status = replacement_results.get(id(phrase_info), "not_found")
                all_phrases.append({
                    "round": lc_round,
                    "original_phrase": orig,
                    "reason": phrase_info.get("reason", ""),
                    "matched": replacement_status in ("replaced", "already_tagged"),
                    "replacement_status": replacement_status,
                })

            # 更新摘要供下一輪使用
            if round_phrases:
                success_phrases = [
                    p for p in round_phrases
                    if replacement_results.get(id(p)) == "replaced"
                ]
                already_tagged_phrases = [
                    p for p in round_phrases
                    if replacement_results.get(id(p)) == "already_tagged"
                ]
                failed_phrases = [
                    p for p in round_phrases
                    if replacement_results.get(id(p)) == "not_found"
                ]
                citation_tag_phrases = [
                    p for p in round_phrases
                    if replacement_results.get(id(p)) == "citation_tag"
                ]

                handled_phrases = success_phrases + already_tagged_phrases
                if handled_phrases:
                    already_tagged_summary += f"\n[第 {lc_round} 輪已處理]\n"
                    for i, p in enumerate(handled_phrases, 1):
                        status_text = "已新增標註" if replacement_results.get(id(p)) == "replaced" else "已在標註區內，未重複標註"
                        already_tagged_summary += (
                            f"  {i}. [{status_text}] \"{p.get('original_phrase', '')}\" "
                            f"→ {p.get('reason', '')}\n"
                        )

                if failed_phrases:
                    for p in failed_phrases:
                        failed_phrases_summary += (
                            f"\"{p.get('original_phrase', '')}\" "
                            f"→ {p.get('reason', '')}\n"
                        )
                if citation_tag_phrases:
                    for p in citation_tag_phrases:
                        citation_tag_summary += (
                            f"\"{p.get('original_phrase', '')}\" "
                            f"→ {p.get('reason', '')}\n"
                        )

            _log(
                f"[LC 第 {lc_round} 輪] 發現 {len(round_phrases)} 個片段，"
                f"標註 {round_annotated} 個，intercept={intercept}"
            )

            # 判斷是否繼續
            if intercept == "pass":
                lc_pass_count += 1
                _log(
                    f"[LC 第 {lc_round} 輪] pass "
                    f"(累積 {lc_pass_count}/{self.detection_strength})"
                )
                if lc_pass_count >= self.detection_strength:
                    _log(f"[LC] 已達檢測強度 {self.detection_strength}，結束掃描")
                    break

            if lc_round == self.max_scan_rounds:
                _log(f"[LC] 已達最大輪次 {self.max_scan_rounds}，結束掃描")

        # 彙總
        _log(f"\n{'='*60}")
        _log(
            f"[LC Subagent] 掃描完成：共 {lc_round} 輪，"
            f"發現 {len(all_phrases)} 個低信心片段，"
            f"成功標註 {total_annotated} 個"
        )
        _log(f"{'='*60}")

        if all_phrases:
            _log("\n--- 各輪低信心片段明細 ---")
            for item in all_phrases:
                matched_label = "✓" if item["matched"] else "✗"
                _log(
                    f"  [第{item['round']}輪] [{matched_label}] "
                    f"\"{item['original_phrase']}\" → {item['reason']}"
                )

        return {
            "success": True,
            "annotated_note": current_note,
            "total_rounds": lc_round,
            "total_annotated": total_annotated,
            "all_phrases": all_phrases,
            "error": None,
        }

    # ════════════════════════════════════════════════════════════
    # 內部方法
    # ════════════════════════════════════════════════════════════

    def _scan_round(
        self,
        current_note: str,
        interview_dialogue: str,
        conversation_history: str,
        system_prompt: str,
        round_num: int,
        already_tagged_summary: str,
        failed_phrases_summary: str,
        citation_tag_summary: str,
        record_diff_context: str = "",
        loaded_files_block: str = "",
        behavior_context: dict | None = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> dict[str, Any]:
        """
        執行一輪 LC 掃描。

        Returns:
            {"intercept": "detected" | "pass", "phrases": list}
        """
        def _log(msg: str):
            if log_callback:
                log_callback(msg)
            print(msg)

        # 組裝 user prompt
        record_diff_text = record_diff_context or "## 【病歷修改 diff 過程】\n（無病歷版本歷史）"
        user_prompt = f"""## 【最終正式病歷 NOTE 欄位】
{current_note}

{record_diff_text}

## 【人類醫師與 AI 主治醫師的互動過程】
{conversation_history or '（無互動過程）'}

## 【完整問診對話紀錄】
{interview_dialogue}
"""

        user_prompt += f"""
## 【當輪讀取檔案暫存區】
{loaded_files_block if loaded_files_block else '（空白）'}
"""

        # 非首輪時附加已標註與失敗摘要
        if round_num > 1:
            if already_tagged_summary:
                user_prompt += (
                    f"\n## 【先前各輪已標註的低信心片段（共 {round_num - 1} 輪，"
                    f"以下片段已用 **...** 標註於 NOTE 中，或已被判定落在既有標註區內，不需重複標註）】\n"
                    f"{already_tagged_summary}\n"
                )
            if failed_phrases_summary:
                user_prompt += (
                    f"\n## 【先前各輪曾識別但無法成功標註的低信心片段（"
                    f"這些片段未在未標註的 NOTE 原文中找到，請重新擷取能精確匹配病歷原文的 original_phrase）】\n"
                    f"{failed_phrases_summary}\n"
                )
            if citation_tag_summary:
                user_prompt += (
                    f"\n## 【先前各輪曾識別但因 original_phrase 包含、跨越或落在來源標籤 [ ... ] 內而被拒絕標註的片段】\n"
                    f"以下片段已被系統拒絕標註。請重新擷取不含來源標籤的病歷本文，"
                    f"來源標籤本身必須完整保留在低信心標註區塊之外。\n"
                    f"{citation_tag_summary}\n"
                )
            user_prompt += (
                "\n請檢查是否仍有遺漏的低信心片段。"
                "若所有低信心片段皆已標註完畢，請輸出 \"intercept\": \"pass\"。\n"
            )

        _log(f"\n[LC 第 {round_num} 輪 User Prompt]")
        _log(user_prompt[:3000] + ("..." if len(user_prompt) > 3000 else ""))
        if behavior_context:
            append_behavior_event(
                behavior_context.get("folder_path"),
                behavior_context.get("date_str"),
                agent="low_confidence_subagent",
                event_type="llm_input",
                label=f"輸入R{round_num}",
                title=f"低信心標註 Subagent 第 {round_num} 輪輸入",
                content=user_prompt,
            )

        # 呼叫 LLM（LC 不注入圖片：帶有圖片來源標籤的描述已視為高信心）
        _lc_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(self.MAX_JSON_RETRIES + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=_lc_messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                raw_output = (resp.choices[0].message.content or "").strip()
            except Exception as e:
                _log(f"[LC] LLM 呼叫失敗: {e}")
                return {"intercept": "pass", "phrases": []}

            _log(f"\n[LC 第 {round_num} 輪 LLM 輸出]")
            _log(raw_output[:5000] + ("..." if len(raw_output) > 5000 else ""))
            if behavior_context:
                append_behavior_event(
                    behavior_context.get("folder_path"),
                    behavior_context.get("date_str"),
                    agent="low_confidence_subagent",
                    event_type="llm_output",
                    label=f"輸出R{round_num}",
                    title=f"低信心標註 Subagent 第 {round_num} 輪輸出",
                    content=raw_output or "",
                    meta={"attempt": attempt},
                )

            # 解析 JSON
            json_match = re.search(r'\{[\s\S]*\}', raw_output)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    phrases = result.get("low_confidence_phrases", [])
                    intercept = str(
                        result.get("intercept", "pass")
                    ).strip().lower()

                    # 驗證格式
                    valid_phrases = [
                        p for p in phrases
                        if isinstance(p, dict)
                        and "original_phrase" in p
                        and "reason" in p
                    ]

                    # 自動修正 intercept 與 phrases 不一致
                    if valid_phrases and intercept == "pass":
                        intercept = "detected"
                    if not valid_phrases and intercept == "detected":
                        intercept = "pass"

                    return {
                        "intercept": intercept,
                        "phrases": valid_phrases,
                    }
                except json.JSONDecodeError:
                    pass

            # JSON 解析失敗
            if attempt < self.MAX_JSON_RETRIES:
                _log(
                    f"[LC] JSON 解析失敗，"
                    f"第 {attempt + 1} 次重試..."
                )
            else:
                _log("[LC] JSON 解析失敗，已達重試上限，回傳 pass")

        return {"intercept": "pass", "phrases": []}
