from __future__ import annotations

from typing import Any, Callable

from ui_app.controllers.session_busy_guard import get_session_busy_guard


def build_model_settings_tab(
    ui: Any,
    app_state: dict[str, Any],
    load_config: Callable[[], dict],
    save_config: Callable[[dict], None],
    reset_agent_state: Callable[[], None],
):
    """Render model settings and persist agent configuration."""
    cfg = load_config()
    busy_guard = get_session_busy_guard(app_state)

    with ui.column().classes("w-full").style("max-width: 800px; margin: 0 auto;"):
        ui.label("⚙️ 模型設定").classes("section-title")
        ui.label("設定 AI 主治醫師、病歷登載助理、幻覺審查員、問診助理、低信心標註、病歷檢查員的 API 連線與模型參數。").style(
            "color: #666; font-size: 14px; margin-bottom: 16px;"
        )

        main_inputs = build_agent_section(
            ui,
            "🤖 AI 主治醫師 (Main Agent)",
            cfg.get("main_agent", {}),
            defaults={"max_tokens": 4000, "temperature": 0.7},
            extras=[
                {
                    "name": "max_sub_turns",
                    "label": "最大執行輪次",
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "tooltip": "單次對話中 Agent 可執行的最大子輪數 (ReAct 迴圈上限)",
                },
            ],
        )
        history_summary_inputs = build_agent_section(
            ui,
            "↳ 歷史病歷摘要/檢查（Main Agent 子模型）",
            main_child_section_config(
                cfg.get("main_agent", {}),
                "history_summary",
                "history_summary_model_name",
                default_max_tokens=int(cfg.get("main_agent", {}).get("max_tokens", 4000) or 4000),
                default_temperature=0.5,
            ),
            defaults={"max_tokens": 4000, "temperature": 0.5},
            note="用於新增就診日期時產生歷史病歷摘要，以及摘要檢查/重寫。Model Name 空白時沿用 Main Agent。",
        )
        summary_exit_inputs = build_agent_section(
            ui,
            "↳ 摘要並退出（Main Agent 子模型）",
            main_child_section_config(
                cfg.get("main_agent", {}),
                "summary_exit",
                "summary_exit_model_name",
                default_max_tokens=128,
                default_temperature=0.2,
            ),
            defaults={"max_tokens": 128, "temperature": 0.2},
            note="用於摘要並退出時，替 NOTE 與 ASSESSMENT & TREATMENT 產生索引短句。Model Name 空白時沿用 Main Agent。",
        )

        record_inputs = build_agent_section(
            ui,
            "📝 病歷登載助理 (Record Subagent)",
            cfg.get("record_subagent", {}),
            defaults={"max_tokens": 8000, "temperature": 0.7},
        )

        hallucination_inputs = build_agent_section(
            ui,
            "🔍 幻覺審查員 (Hallucination Reviewer)",
            cfg.get("hallucination_subagent", {}),
            defaults={"max_tokens": 8000, "temperature": 1.0},
            note="若 Model Name 留空，則病歷登載時跳過幻覺審查。檢測強度 0 = 不審查直接放行（研究對照組用）。",
            extras=[
                {
                    "name": "detection_strength",
                    "label": "檢測強度（累積 N 次 agree 通過；0=不審查/對照組）",
                    "default": 2,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                },
                {
                    "name": "max_review_rounds",
                    "label": "最大審查輪次",
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                },
            ],
        )

        ic_inputs = build_agent_section(
            ui,
            "🎯 問診助理 (Information Collection Subagent)",
            cfg.get("ic_subagent", {}),
            defaults={"max_tokens": 20000, "temperature": 0.7},
            note="若 Model Name 留空，則 Main Agent 無法啟動問診助理。",
            extras=[
                {
                    "name": "max_collection_rounds",
                    "label": "最大問診回合數",
                    "default": 10,
                    "min": 1,
                    "max": 30,
                    "step": 1,
                },
            ],
        )

        lc_inputs = build_agent_section(
            ui,
            "🏷️ 低信心標註 (Low Confidence Check Subagent)",
            cfg.get("lc_subagent", {}),
            defaults={"max_tokens": 20000, "temperature": 1.0},
            note="若 Model Name 留空，則 Main Agent 無法執行低信心標註。檢測強度 0 = 不掃描直接放行（研究對照組用）。",
            extras=[
                {
                    "name": "max_scan_rounds",
                    "label": "最大掃描輪次",
                    "default": 8,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                },
                {
                    "name": "detection_strength",
                    "label": "檢測強度（累積 N 次 pass 通過；0=不掃描/對照組）",
                    "default": 4,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                },
            ],
        )

        nr_inputs = build_agent_section(
            ui,
            "📋 病歷檢查員 (Note Review Subagent)",
            cfg.get("nr_subagent", {}),
            defaults={"max_tokens": 20000, "temperature": 1.0},
            note="若 Model Name 留空，則 Main Agent 無法執行病歷完整性掃描。",
        )

        with ui.row().classes("w-full gap-3"):
            btn_save_cfg = ui.button("💾 儲存設定", color="green").style("font-size: 15px;")
            cfg_status = ui.label("").style("font-size: 14px; line-height: 36px;")

        def patient_is_loaded() -> bool:
            return bool(app_state.get("selected_patient_folder"))

        def sync_save_state():
            if patient_is_loaded():
                btn_save_cfg.disable()
                cfg_status.text = "🔒 請先退出患者，才能儲存全域模型設定"
                cfg_status.style("color: #999;")
            else:
                btn_save_cfg.enable()
                if cfg_status.text.startswith("🔒"):
                    cfg_status.text = ""

        app_state["_sync_model_settings_save_state"] = sync_save_state
        sync_save_state()

        def on_save_config():
            if busy_guard.reject_if_busy(status_label=cfg_status):
                return
            if patient_is_loaded():
                sync_save_state()
                return
            new_cfg = load_config()
            main_cfg = dict(new_cfg.get("main_agent", {}))
            main_cfg.update(read_agent_section(main_inputs, {"max_tokens": 4000, "temperature": 0.7}, ["max_sub_turns"]))
            history_summary_cfg = read_agent_section(
                history_summary_inputs,
                {"max_tokens": 4000, "temperature": 0.5},
            )
            summary_exit_cfg = read_agent_section(
                summary_exit_inputs,
                {"max_tokens": 128, "temperature": 0.2},
            )
            main_cfg["history_summary"] = history_summary_cfg
            main_cfg["summary_exit"] = summary_exit_cfg
            main_cfg["history_summary_model_name"] = history_summary_cfg.get("model_name", "")
            main_cfg["summary_exit_model_name"] = summary_exit_cfg.get("model_name", "")
            new_cfg["main_agent"] = main_cfg
            new_cfg["record_subagent"] = read_agent_section(record_inputs, {"max_tokens": 8000, "temperature": 0.7})
            hallucination_cfg = read_agent_section(
                hallucination_inputs,
                {"max_tokens": 8000, "temperature": 1.0},
                ["detection_strength", "max_review_rounds"],
            )
            hallucination_control_group = False
            if hallucination_cfg.get("model_name"):
                # 直接讀原始輸入值：空欄位必須擋下，不可被共用的 `or 0` 默默當成對照組
                detection_strength = read_strength_or_none(hallucination_inputs["detection_strength"])
                max_review_rounds = int(hallucination_cfg.get("max_review_rounds", 0))
                if detection_strength is None:
                    cfg_status.text = "❌ 幻覺審查的檢測強度不可留空：請填 0（=不審查/對照組）或正整數"
                    cfg_status.style("color: var(--danger);")
                    return
                if detection_strength < 0 or max_review_rounds < 1:
                    cfg_status.text = "❌ 幻覺審查的最大審查輪次必須至少為 1，檢測強度不可為負（0=不審查/對照組）"
                    cfg_status.style("color: var(--danger);")
                    return
                if detection_strength > max_review_rounds:
                    cfg_status.text = "❌ 幻覺審查的檢測強度不可大於最大審查輪次"
                    cfg_status.style("color: var(--danger);")
                    return
                hallucination_cfg["detection_strength"] = detection_strength
                hallucination_control_group = detection_strength == 0
            new_cfg["hallucination_subagent"] = hallucination_cfg
            new_cfg["ic_subagent"] = read_agent_section(
                ic_inputs,
                {"max_tokens": 20000, "temperature": 0.7},
                ["max_collection_rounds"],
            )
            lc_cfg = read_agent_section(
                lc_inputs,
                {"max_tokens": 20000, "temperature": 1.0},
                ["max_scan_rounds", "detection_strength"],
            )
            lc_control_group = False
            if lc_cfg.get("model_name"):
                lc_strength = read_strength_or_none(lc_inputs["detection_strength"])
                lc_rounds = int(lc_cfg.get("max_scan_rounds", 0))
                if lc_strength is None:
                    cfg_status.text = "❌ 低信心標註的檢測強度不可留空：請填 0（=不掃描/對照組）或正整數"
                    cfg_status.style("color: var(--danger);")
                    return
                if lc_strength < 0 or lc_rounds < 1:
                    cfg_status.text = "❌ 低信心標註的最大掃描輪次必須至少為 1，檢測強度不可為負（0=不掃描/對照組）"
                    cfg_status.style("color: var(--danger);")
                    return
                if lc_strength > lc_rounds:
                    cfg_status.text = "❌ 低信心標註的檢測強度不可大於最大掃描輪次"
                    cfg_status.style("color: var(--danger);")
                    return
                lc_cfg["detection_strength"] = lc_strength
                lc_control_group = lc_strength == 0
            new_cfg["lc_subagent"] = lc_cfg
            new_cfg["nr_subagent"] = read_agent_section(nr_inputs, {"max_tokens": 20000, "temperature": 1.0})
            save_config(new_cfg)
            reset_agent_state()
            control_groups = []
            if hallucination_control_group:
                control_groups.append("幻覺審查")
            if lc_control_group:
                control_groups.append("低信心標註")
            if control_groups:
                cfg_status.text = (
                    f"⚠️ 設定已儲存：{'、'.join(control_groups)}檢測強度為 0，"
                    "將不檢查直接放行（對照組模式）"
                )
                cfg_status.style("color: var(--warning);")
            else:
                cfg_status.text = "✅ 設定已儲存（Agent 將在下次對話時重新載入）"
                cfg_status.style("color: var(--primary);")

        btn_save_cfg.on_click(on_save_config)


def build_agent_section(
    ui: Any,
    title: str,
    section_cfg: dict,
    *,
    defaults: dict[str, float | int],
    note: str = "",
    extras: list[dict] | None = None,
) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    with ui.card().classes("w-full q-pa-md q-mb-lg").style("border: 1px solid var(--border); border-radius: 12px;"):
        ui.label(title).style("font-weight: 700; font-size: 16px; color: var(--primary-dark);")
        if note:
            ui.label(note).style("color: #999; font-size: 12px; margin-bottom: 8px;")
        inputs["api_url"] = ui.input(label="API URL", value=section_cfg.get("api_url", "http://localhost:1234/v1")).classes("w-full")
        inputs["api_key"] = ui.input(label="API Key", value=section_cfg.get("api_key", "lm-studio")).classes("w-full")
        inputs["model_name"] = ui.input(label="Model Name", value=section_cfg.get("model_name", "")).classes("w-full")
        with ui.row().classes("w-full gap-4"):
            inputs["max_tokens"] = ui.number(
                label="Max Tokens",
                value=section_cfg.get("max_tokens", defaults["max_tokens"]),
                min=1,
                max=128000,
                step=100,
            ).classes("flex-1")
            inputs["temperature"] = ui.number(
                label="Temperature",
                value=section_cfg.get("temperature", defaults["temperature"]),
                min=0.0,
                max=2.0,
                step=0.05,
                format="%.2f",
            ).classes("flex-1")

        if extras:
            with ui.row().classes("w-full gap-4"):
                for extra in extras:
                    control = ui.number(
                        label=extra["label"],
                        value=section_cfg.get(extra["name"], extra["default"]),
                        min=extra.get("min"),
                        max=extra.get("max"),
                        step=extra.get("step", 1),
                    ).classes("flex-1")
                    if extra.get("tooltip"):
                        control.tooltip(extra["tooltip"])
                    inputs[extra["name"]] = control

    return inputs


def main_child_section_config(
    main_cfg: dict,
    child_key: str,
    legacy_model_key: str,
    *,
    default_max_tokens: int,
    default_temperature: float,
) -> dict:
    child_cfg = main_cfg.get(child_key, {})
    if not isinstance(child_cfg, dict):
        child_cfg = {}
    return {
        "api_url": child_cfg.get("api_url", main_cfg.get("api_url", "http://localhost:1234/v1")),
        "api_key": child_cfg.get("api_key", main_cfg.get("api_key", "lm-studio")),
        "model_name": child_cfg.get("model_name", main_cfg.get(legacy_model_key, "")),
        "max_tokens": child_cfg.get("max_tokens", default_max_tokens),
        "temperature": child_cfg.get("temperature", default_temperature),
    }


def read_strength_or_none(control: Any) -> int | None:
    """讀取檢測強度欄位的原始值。

    空欄位回傳 None，與明確填入的 0（對照組模式）區分開來——
    共用的 `int(value or 0)` 會把清空欄位默默變成 0，
    對「0 = 停用安全檢查」的欄位來說必須擋下而非靜默接受。
    """
    raw = control.value
    if raw is None:
        return None
    if isinstance(raw, str) and not raw.strip():
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def read_agent_section(
    inputs: dict[str, Any],
    defaults: dict[str, float | int],
    extra_keys: list[str] | None = None,
) -> dict:
    data = {
        "api_url": inputs["api_url"].value or "",
        "api_key": inputs["api_key"].value or "",
        "model_name": inputs["model_name"].value or "",
        "max_tokens": int(inputs["max_tokens"].value or defaults["max_tokens"]),
        "temperature": float(inputs["temperature"].value or defaults["temperature"]),
    }
    for key in extra_keys or []:
        data[key] = int(inputs[key].value or 0)
    return data
