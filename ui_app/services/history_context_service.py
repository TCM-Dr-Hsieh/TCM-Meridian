from __future__ import annotations

from deidentification_utils import format_patient_basic_info_for_llm
from ui_app.services.llm_config_resolver import resolve_main_child_llm_config


class HistoryContextService:
    """Build last-visit and history-summary context for MainAgent/IC Subagent."""

    def __init__(
        self,
        *,
        load_config,
        list_sessions,
        load_session_content,
        load_patient,
        save_history_summary,
    ):
        self.load_config = load_config
        self.list_sessions = list_sessions
        self.load_session_content = load_session_content
        self.load_patient = load_patient
        self.save_history_summary = save_history_summary

    def get_last_visit_content(self, folder_path: str, current_date: str) -> str:
        dates = self.list_sessions(folder_path)
        previous_dates = [d for d in dates if d < current_date]
        if not previous_dates:
            return ""

        last_date = previous_dates[0]
        content = self.load_session_content(folder_path, last_date)
        note = content.get("note", "")
        at = content.get("at", "")

        parts = [f"就診日期: {last_date}"]
        parts.append(f"\n--- NOTE ---\n{note if note else '（空白）'}")
        parts.append(f"\n--- ASSESSMENT & TREATMENT ---\n{at if at else '（空白）'}")
        return "\n".join(parts)

    def generate_history_summary(self, folder_path: str, current_date: str) -> str:
        cfg = self.load_config()
        main_cfg = cfg.get("main_agent", {})
        llm_cfg = resolve_main_child_llm_config(
            main_cfg,
            "history_summary",
            "history_summary_model_name",
            default_max_tokens=int(main_cfg.get("max_tokens", 4000) or 4000),
            default_temperature=0.5,
        )

        model_name = llm_cfg["model_name"]
        if not model_name:
            summary = "（未設定模型，無法產生歷史病歷摘要。請先在「模型設定」頁面設定模型。）"
            self.save_history_summary(folder_path, current_date, summary)
            return summary

        dates = self.list_sessions(folder_path)
        past_dates = [d for d in dates if d < current_date][:10]

        records_text_parts = []
        for d in reversed(past_dates):
            content = self.load_session_content(folder_path, d)
            note = content.get("note", "")
            at = content.get("at", "")
            records_text_parts.append(
                f"═══ {d} ═══\n"
                f"[NOTE]\n{note if note else '（空白）'}\n\n"
                f"[ASSESSMENT & TREATMENT]\n{at if at else '（空白）'}"
            )
        records_text = "\n\n".join(records_text_parts)
        if not records_text:
            records_text = (
                "（此為系統內首次就診，尚無先前病歷紀錄。請根據患者基本資料建立本次可用的"
                "首次歷史病歷摘要；若資料不足，請明確列出目前僅有資訊與待補資訊。）"
            )

        info = self.load_patient(folder_path)
        patient_text = "（無患者基本資料）"
        if info:
            bi = info.get("basic_info", {})
            patient_text = format_patient_basic_info_for_llm(bi, current_date)

        prompt = f"""你是一位經驗豐富的中醫師，請根據以下患者的基本資料與近期病歷紀錄，產生一份精煉的「歷史病歷摘要」。

要求：
1. 摘要應涵蓋：主要診斷、核心症狀演變、用藥/治療歷程、療效觀察、重要體質特徵
2. 按時間線整理，呈現病情的演變趨勢
3. 保留關鍵的量化資訊（如用藥劑量、檢查數值）
4. 標注仍在持續追蹤的問題
5. 語言精煉，避免重複，長度控制在 500-1000 字以內
6. 若沒有先前病歷紀錄，仍須根據患者基本資料產生首次摘要；不得只回答「無歷史病歷」。

【患者基本資料】
{patient_text}

【本次就診日期】
{current_date}

【近 {len(past_dates)} 次先前病歷紀錄】
{records_text}

請直接輸出摘要內容，不需要 JSON 格式："""

        try:
            from openai import OpenAI

            client = OpenAI(
                base_url=llm_cfg["api_url"],
                api_key=llm_cfg["api_key"],
            )

            print(f"\n{'='*60}")
            print(f"[History Summary] 正在為 {current_date} 產生歷史病歷摘要...")
            if past_dates:
                print(f"[History Summary] 涵蓋 {len(past_dates)} 次病歷: {', '.join(reversed(past_dates))}")
            else:
                print("[History Summary] 無先前病歷，改以患者基本資料產生首次摘要")
            print(f"{'='*60}")
            print("\n[History Summary] ══ 送入 LLM 的 Prompt ══")
            print(prompt)
            print(f"{'─'*40}\n")

            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=llm_cfg["max_tokens"],
                temperature=llm_cfg["temperature"],
            )
            summary = (resp.choices[0].message.content or "").strip()

            print("\n[History Summary] ══ LLM 輸出的摘要 ══")
            print(summary)
            print(f"{'─'*40}\n")

            summary = self._review_and_rewrite_history_summary(
                client=client,
                model_name=model_name,
                main_cfg=main_cfg,
                llm_cfg=llm_cfg,
                current_date=current_date,
                patient_text=patient_text,
                records_text=records_text,
                summary=summary,
            )

        except Exception as e:
            summary = f"（歷史病歷摘要產生失敗：{e}）"
            print(f"[History Summary] ❌ {summary}")

        self.save_history_summary(folder_path, current_date, summary)
        return summary

    def _review_and_rewrite_history_summary(
        self,
        *,
        client,
        model_name: str,
        main_cfg: dict,
        llm_cfg: dict,
        current_date: str,
        patient_text: str,
        records_text: str,
        summary: str,
    ) -> str:
        max_rounds = int(main_cfg.get("history_summary_review_rounds", 3) or 3)
        current_summary = summary

        for review_round in range(1, max_rounds + 1):
            review_prompt = f"""你是一位「歷史病歷摘要審查兼重寫員」。

請根據【患者基本資料】與【先前病歷紀錄】審查【待審查摘要】是否有明顯問題。

審查原則：
1. 不要過度嚴格，不要因為摘要不夠完美就退件。
2. 只有在摘要出現明顯錯誤、與資料矛盾、把未提供資訊寫成已確認、過度臆測、或有臨床上可能誤導的表述時，才需要改寫。
3. 若沒有先前病歷紀錄，摘要可只根據患者基本資料整理；不得因資料少就要求退件。
4. 改寫時只能使用提供的資料，不得新增未提供的診斷、症狀、檢驗值或治療內容。
5. 若只是措辭、格式或完整度的小問題，請直接同意。

你只能使用以下兩種輸出格式之一：
1. 同意
2. 不同意，改寫版本如下：
<完整改寫後摘要>

【患者基本資料】
{patient_text}

【本次就診日期】
{current_date}

【先前病歷紀錄】
{records_text}

【待審查摘要】
{current_summary}
"""

            print(f"\n[History Summary Review] 第 {review_round}/{max_rounds} 輪審查")
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": review_prompt}],
                max_completion_tokens=llm_cfg["max_tokens"],
                temperature=llm_cfg["temperature"],
            )
            review_text = (resp.choices[0].message.content or "").strip()
            print("[History Summary Review] 審查輸出：")
            print(review_text)
            print(f"{'─'*40}\n")

            if review_text.startswith("同意") and not review_text.startswith("不同意"):
                print(f"[History Summary Review] 第 {review_round} 輪同意，採用目前摘要")
                return current_summary

            rewritten = self._extract_history_summary_rewrite(review_text)
            if not rewritten:
                print("[History Summary Review] 審查輸出格式不明，為避免阻塞流程，採用目前摘要")
                return current_summary

            current_summary = rewritten
            print(f"[History Summary Review] 第 {review_round} 輪不同意，已套用改寫版本")

        print("[History Summary Review] 達最大審查輪數，採用最後一次改寫版本")
        return current_summary

    @staticmethod
    def _extract_history_summary_rewrite(review_text: str) -> str:
        marker = "改寫版本如下"
        if not review_text.startswith("不同意") or marker not in review_text:
            return ""
        rewritten = review_text.split(marker, 1)[1]
        rewritten = rewritten.lstrip("：:\n\r\t ")
        return rewritten.strip()
