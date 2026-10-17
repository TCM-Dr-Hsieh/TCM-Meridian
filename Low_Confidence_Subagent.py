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
import model_json
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


def _plan_replacements(note: str, phrases: list[dict]) -> tuple[dict[int, str], list[tuple[int, int, str]]]:
    """Where each phrase of ONE scan round goes, resolved on the note exactly as the model saw it (not after the earlier phrases of the
    round have been applied: their added reasons would shift the count of a later `occurrence`).

    A phrase without `occurrence` takes the first place that is not in bold, not across a source tag and not already covered by a mark
    of this round; with `occurrence` it takes that place (1-based, top to bottom, places in bold counted). A place that only partly
    overlaps a mark of this round (or holds one) is not "already marked": the two are merged into one mark over both spans with both
    reasons, so the part the earlier mark did not cover is not left without a warning. Returns
    ({id(phrase): status}, [(start, end, reasons)]) with status replaced | already_tagged | citation_tag | not_found."""
    statuses: dict[int, str] = {}
    claims: list[list] = []         # [start, end, [reasons]]

    def overlapping(start: int, end: int) -> list[list]:
        return [claim for claim in claims if start < claim[1] and end > claim[0]]

    for info in phrases:
        original = info.get("original_phrase", "")
        if not original:
            statuses[id(info)] = "not_found"
            continue
        reason = info.get("reason", "")
        occurrence = info.get("occurrence")
        wanted = occurrence if isinstance(occurrence, int) and not isinstance(occurrence, bool) and occurrence >= 1 else None
        start = 0
        seen = 0
        found_bold = found_tag = False
        chosen: tuple[int, int, list[list]] | None = None
        while True:
            index = note.find(original, start)
            if index < 0:
                break
            end = index + len(original)
            start = end
            seen += 1
            if wanted is not None and seen != wanted:
                continue
            hits = overlapping(index, end)
            if _inside_bold_span(note, index) or any(claim[0] <= index and end <= claim[1] for claim in hits):
                found_bold = True               # in bold already, or wholly inside a mark of this round
            elif _overlaps_citation_tag(note, index, end):
                found_tag = True
            else:
                chosen = (index, end, hits)
                break
            if wanted is not None:
                break
        if chosen is not None:
            first, last, hits = chosen
            reasons: list[str] = []
            for claim in hits:
                first, last = min(first, claim[0]), max(last, claim[1])
                reasons += [text for text in claim[2] if text not in reasons]
                claims.remove(claim)
            if reason not in reasons:
                reasons.append(reason)
            claims.append([first, last, reasons])
            statuses[id(info)] = "replaced"
        else:
            statuses[id(info)] = "citation_tag" if found_tag else ("already_tagged" if found_bold else "not_found")
    return statuses, [(start, end, tuple(reasons)) for start, end, reasons in claims]


def _apply_replacements(note: str, claims: list[tuple[int, int, tuple[str, ...]]]) -> str:
    """Apply the planned marks from the end of the note backwards, so no position moves before it is used."""
    for start, end, reasons in sorted(claims, key=lambda claim: claim[0], reverse=True):
        note = note[:start] + f"**{note[start:end]}（{'；'.join(reasons)}）**" + note[end:]
    return note


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
        voice_transcript: str = "",
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
            voice_transcript: 診間對話逐字稿（語音辨識所得；接在完整問診對話紀錄之後）
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
        scan_errors = 0                  # rounds in which the model call or its output failed: not a "pass", the check did not happen
        consecutive_errors = 0
        skipped_items: list[dict] = []   # findings the model gave without a usable phrase/reason, and did not repair when asked

        _log(f"\n{'='*60}")
        _log(f"[LC Subagent] 開始迭代掃描（至多 {self.max_scan_rounds} 輪，檢測強度 {self.detection_strength}）")
        _log(f"{'='*60}")

        for lc_round in range(1, self.max_scan_rounds + 1):
            _log(f"\n--- Low Confidence Check 第 {lc_round}/{self.max_scan_rounds} 輪 ---")

            check_result = self._scan_round(
                current_note=current_note,
                interview_dialogue=interview_dialogue,
                voice_transcript=voice_transcript,
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

            if intercept == "error":
                scan_errors += 1
                consecutive_errors += 1
                _log(f"[LC 第 {lc_round} 輪] 掃描失敗（{check_result.get('error', '?')}），不算通過")
                if consecutive_errors >= 2:
                    _log("[LC] 連續 2 輪掃描失敗，停止掃描")
                    break
                continue
            consecutive_errors = 0
            if check_result.get("repair_error"):
                scan_errors += 1
                _log(f"[LC 第 {lc_round} 輪] {check_result['repair_error']}；已找到的合格片段保留，不合格項記為未標註")
            for item in check_result.get("invalid_items", []):
                skipped_items.append({"round": lc_round, "item": item})
                _log(f"[LC 第 {lc_round} 輪] 有一個發現格式不合、無法標註：{item}")

            # 標註替換：這一輪的所有位置先依「送給模型看的那份 NOTE」一次算好，再整批套用
            replacement_results, claims = _plan_replacements(current_note, round_phrases)
            current_note = _apply_replacements(current_note, claims)
            round_annotated = len(claims)

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

        # A round that failed is not a round that found nothing: if the scan never reached the passes it needs and some round failed, the
        # check is incomplete (what was marked before the failure is kept) and the caller must say so, not report "done".
        # An item that could not be placed is a finding that is not on the note: the scan is incomplete as well.
        incomplete = (scan_errors > 0 and lc_pass_count < self.detection_strength) or bool(skipped_items)
        return {
            "success": True,
            "annotated_note": current_note,
            "total_rounds": lc_round,
            "total_annotated": total_annotated,
            "all_phrases": all_phrases,
            "scan_errors": scan_errors,
            "skipped_items": skipped_items,
            "incomplete": incomplete,
            "error": None,
        }

    # ════════════════════════════════════════════════════════════
    # 內部方法
    # ════════════════════════════════════════════════════════════

    @staticmethod
    def _usable_phrase(item) -> bool:
        return (isinstance(item, dict)
                and isinstance(item.get("original_phrase"), str) and bool(item["original_phrase"].strip())
                and isinstance(item.get("reason"), str))

    @staticmethod
    def _describe_item(item) -> str:
        try:
            text = json.dumps(item, ensure_ascii=False)
        except (TypeError, ValueError):
            text = repr(item)
        return text if len(text) <= 160 else text[:160] + "…"

    @staticmethod
    def _structure_problem(result) -> str:
        """Why a parsed reply is not a usable scan answer ('' when it is): the keys and types the prompt asks for."""
        if not isinstance(result, dict):
            return "輸出不是 JSON 物件"
        if not isinstance(result.get("low_confidence_phrases"), list):
            return "缺少 low_confidence_phrases 陣列"
        if str(result.get("intercept", "")).strip().lower() not in ("pass", "detected"):
            return "intercept 不是 pass 或 detected"
        return ""

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
        voice_transcript: str = "",
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

## 【診間對話逐字稿】
{voice_transcript if voice_transcript else '（空白）'}
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

        best: dict | None = None        # the latest answer with the right structure that still holds items nobody can place
        for attempt in range(self.MAX_JSON_RETRIES + 1):
            repair = ""
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
                if best is not None:          # the first answer already held valid marks: keep them, and say the repair could not be asked
                    return {**best, "repair_error": f"請模型補齊時 LLM 呼叫失敗：{e}"}
                return {"intercept": "error", "phrases": [], "error": f"LLM 呼叫失敗：{e}"}

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

            # 解析 JSON，並驗證結構：缺必要欄位、型別不對、或給了片段卻沒有一個合格，都是「輸出不合規格」，不是「沒有問題」
            problem = "輸出無法解析為 JSON"
            json_match = re.search(r'\{[\s\S]*\}', raw_output)
            if json_match:
                try:
                    result = model_json.loads(json_match.group())
                except json.JSONDecodeError:
                    result = None
                if result is not None:
                    problem = self._structure_problem(result)
                    if not problem:
                        phrases = result["low_confidence_phrases"]
                        intercept = str(result["intercept"]).strip().lower()
                        valid_phrases = [p for p in phrases if self._usable_phrase(p)]
                        invalid_items = [self._describe_item(p) for p in phrases if not self._usable_phrase(p)]
                        if phrases and not valid_phrases:
                            problem = "low_confidence_phrases 裡沒有任何合格的片段（需要 original_phrase 與 reason）"
                        else:
                            # 自動修正 intercept 與 phrases 不一致
                            if valid_phrases and intercept == "pass":
                                intercept = "detected"
                            if not valid_phrases and intercept == "detected":
                                intercept = "pass"
                            answer = {"intercept": intercept, "phrases": valid_phrases, "invalid_items": invalid_items}
                            if not invalid_items:
                                return answer
                            # 合格的片段保留；不合格的是「發現了、卻沒有地方可標」，不能當作沒有：先請模型補齊重新輸出，
                            # 補不齊就連同不合格項一起回報（掃描結果標為未完成），不靜默丟棄
                            if best is None or len(invalid_items) <= len(best["invalid_items"]):
                                best = answer
                            problem = f"有 {len(invalid_items)} 個片段缺 original_phrase 或 reason"
                            repair = (
                                f"你上一個輸出有 {len(invalid_items)} 個片段不合格，系統無法標註：\n" + "\n".join(invalid_items)
                                + "\n請完整重新輸出整份 JSON：每個片段都要有 original_phrase（病歷原文的精確片段）與 reason；"
                                "已合格的片段請原樣保留；若確認某項其實不需要標註，直接不要列出。"
                            )

            # 輸出不合規格：重試，最後視為掃描失敗（有合格片段的答案則保留，並回報不合格項）
            if attempt < self.MAX_JSON_RETRIES:
                _log(
                    f"[LC] {problem}，"
                    f"第 {attempt + 1} 次重試..."
                )
                if repair:
                    _lc_messages = _lc_messages + [
                        {"role": "assistant", "content": raw_output},
                        {"role": "user", "content": repair},
                    ]
            else:
                _log(f"[LC] {problem}，已達重試上限，{'保留合格片段、回報不合格項' if best else '本輪視為掃描失敗'}")

        if best is not None:
            return best
        return {"intercept": "error", "phrases": [], "error": problem}
