from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class TabSpec:
    """Declarative definition for one top-level UI tab."""

    key: str
    label: str
    builder: Callable[[], None]
    panel_classes: str = "q-pa-md"
    tab_style: str = ""


CHAT_SCROLL_SELECTORS = {
    "main": ".main-chat-scroll",
    "auto": ".ic-chat-scroll",
}


def scroll_selector_to_bottom(ui: Any, selector: str):
    selector_js = json.dumps(selector)
    ui.run_javascript(
        f"""
        setTimeout(() => {{
          const el = document.querySelector({selector_js});
          if (el) el.scrollTop = el.scrollHeight;
        }}, 80);
        """
    )


def build_header(ui: Any, today: str):
    """Build the fixed app header."""
    with ui.header().classes("items-center justify-between q-px-lg").style(
        "background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 50%, #40916c 100%);"
    ):
        ui.label("🌿 杏林經緯  TCM-Meridian").style(
            "font-size: 22px; font-weight: 700; color: white; letter-spacing: 1px;"
        )
        ui.label(f"📅 {today}").style("color: rgba(255,255,255,0.85); font-size: 14px;")


def build_tab_shell(ui: Any, app_state: dict[str, Any], specs: list[TabSpec]):
    """Build tabs and panels from TabSpec entries."""
    tab_elements: dict[str, Any] = {}
    with ui.tabs().classes("w-full").style(
        "background: #fff; border-bottom: 2px solid var(--border);"
    ) as tabs:
        for spec in specs:
            tab = ui.tab(spec.label)
            if spec.tab_style:
                tab.style(spec.tab_style)
            scroll_selector = CHAT_SCROLL_SELECTORS.get(spec.key)
            if scroll_selector:
                tab.on("click", lambda *_, selector=scroll_selector: scroll_selector_to_bottom(ui, selector))
            tab_elements[spec.key] = tab

    app_state["tab_auto"] = tab_elements.get("auto")

    first_tab = tab_elements[specs[0].key]
    with ui.tab_panels(tabs, value=first_tab).classes("w-full").style("min-height: calc(100vh - 120px);"):
        for spec in specs:
            with ui.tab_panel(tab_elements[spec.key]).classes(spec.panel_classes):
                spec.builder()
