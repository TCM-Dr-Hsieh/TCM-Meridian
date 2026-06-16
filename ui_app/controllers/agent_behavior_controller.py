from __future__ import annotations

import json
from functools import partial
from typing import Any

from agent_behavior_log import AGENT_COLUMNS, load_behavior_events


def _display_content(event: dict) -> str:
    content = event.get("content", "")
    if event.get("content_type") == "json":
        try:
            parsed = json.loads(content) if isinstance(content, str) else content
            return "```json\n" + json.dumps(parsed, ensure_ascii=False, indent=2) + "\n```"
        except Exception:
            return "```json\n" + str(content) + "\n```"

    stripped = content.strip() if isinstance(content, str) else str(content)
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
            return "```json\n" + json.dumps(parsed, ensure_ascii=False, indent=2) + "\n```"
        except Exception:
            pass
    return stripped or "（空白）"


def build_agent_behavior_tab(ui: Any, app_state: dict[str, Any]):
    """Render a compact multi-agent behavior timeline for debugging."""

    container = ui.column().classes("w-full gap-2").style("max-width: 1600px; margin: 0 auto;")
    app_state["agent_behavior_container"] = container

    dialog = ui.dialog()
    with dialog:
        with ui.card().classes("w-full").style("max-width: 1100px; max-height: 86vh; overflow: auto;"):
            modal_title = ui.label("").style("font-weight: 700; font-size: 18px; color: #1f2937;")
            modal_meta = ui.label("").style("font-size: 12px; color: #6b7280;")
            modal_body = ui.markdown("").style(
                "font-size: 14px; line-height: 1.55; white-space: normal;"
            )
            with ui.row().classes("w-full justify-end"):
                ui.button("關閉", on_click=dialog.close).props("outline")

    def open_event(event: dict):
        modal_title.text = event.get("title", event.get("label", "事件"))
        modal_meta.text = (
            f"{event.get('ts', '')}  |  {event.get('agent', '')}  |  "
            f"{event.get('event_type', '')}"
        )
        modal_body.content = _display_content(event)
        dialog.open()

    def button_style(event: dict) -> str:
        severity = event.get("severity", "normal")
        event_type = event.get("event_type", "")
        if severity == "error" or "error" in event_type:
            return "background:#fee2e2;color:#991b1b;border:1px solid #fecaca;"
        if severity == "warning" or event_type == "manual_stop":
            return "background:#fef3c7;color:#92400e;border:1px solid #fde68a;"
        if event_type == "llm_input":
            return "background:#dbeafe;color:#1e40af;border:1px solid #bfdbfe;"
        if event_type == "llm_output":
            return "background:#dcfce7;color:#166534;border:1px solid #bbf7d0;"
        if event_type == "rag_retrieval":
            return "background:#f3e8ff;color:#6b21a8;border:1px solid #e9d5ff;"
        return "background:#f3f4f6;color:#374151;border:1px solid #e5e7eb;"

    def render():
        fp = app_state.get("selected_patient_folder")
        dt = app_state.get("selected_session_date")
        events = load_behavior_events(fp, dt)

        container.clear()
        with container:
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("智能體互動行為").style("font-weight:700;font-size:20px;color:#1f2937;")
                ui.button("重新整理", on_click=render).props("outline")

            if not fp or not dt:
                ui.label("請先選取患者與就診日期。").style("color:#9ca3af;margin-top:32px;")
                return

            if not events:
                ui.label("尚無智能體行為紀錄。").style("color:#9ca3af;margin-top:32px;")
                return

            grid_style = (
                "display:grid;grid-template-columns:repeat(7,minmax(150px,1fr));"
                "gap:6px;align-items:stretch;width:100%;overflow-x:auto;"
            )
            with ui.element("div").style(grid_style):
                for _, label in AGENT_COLUMNS:
                    ui.label(label).style(
                        "font-weight:700;font-size:13px;text-align:center;"
                        "padding:8px;background:#f9fafb;border:1px solid #e5e7eb;"
                    )

            col_index = {agent: idx for idx, (agent, _) in enumerate(AGENT_COLUMNS)}
            for idx, event in enumerate(events, 1):
                with ui.element("div").style(grid_style):
                    event_col = col_index.get(event.get("agent"), 0)
                    for col in range(len(AGENT_COLUMNS)):
                        with ui.element("div").style(
                            "min-height:34px;border-left:1px solid #e5e7eb;"
                            "border-right:1px solid #f3f4f6;padding:3px;"
                        ):
                            if col == event_col:
                                label = event.get("label", event.get("event_type", "事件"))
                                prefix = "⚠ " if event.get("severity") in {"warning", "error"} else ""
                                btn = ui.button(
                                    f"{prefix}{idx}. {label}",
                                    on_click=partial(open_event, event),
                                ).props("flat dense no-caps")
                                btn.style(
                                    button_style(event)
                                    + "width:100%;min-height:28px;font-size:12px;"
                                    "white-space:normal;border-radius:6px;"
                                )

    app_state["render_agent_behavior"] = render
    render()
