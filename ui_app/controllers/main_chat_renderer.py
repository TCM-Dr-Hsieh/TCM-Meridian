from __future__ import annotations

from typing import Any

from ui_app.rendering import simple_md_render


class MainChatRenderer:
    """Render the main doctor-agent chat transcript."""

    def __init__(self, ui: Any, container: Any):
        self.ui = ui
        self.container = container

    def render(self, messages: list[dict]):
        self.container.clear()
        with self.container:
            for msg in messages:
                if msg["role"] == "user":
                    self._render_user(msg)
                elif msg["role"] == "agent":
                    self._render_agent(msg)
        self.ui.run_javascript(
            """
            setTimeout(() => {
              const el = document.querySelector('.main-chat-scroll');
              if (el) el.scrollTop = el.scrollHeight;
            }, 0);
            """
        )

    def _render_user(self, msg: dict):
        with self.ui.column().classes("w-full"):
            self.ui.label("👨‍⚕️ 人類醫師:").style("font-weight: 700; font-size: 13px; color: #2d6a4f;")
            self.ui.html(f'<div class="chat-bubble-user">{simple_md_render(msg["content"])}</div>')

    def _render_agent(self, msg: dict):
        with self.ui.column().classes("w-full"):
            if msg.get("steps"):
                with self.ui.expansion("🔧 執行過程概述", icon="expand_more").classes("w-full").style(
                    "background: #fafafa; border-radius: 8px; margin: 4px 0;"
                ):
                    for step in msg["steps"]:
                        with self.ui.column().classes("step-detail-box"):
                            self.ui.label(f"🔹 {step['step_label']}").style("font-weight: 700; font-size: 12px;")
                            self.ui.label(f"💭 思考: {step['thinking'][:200]}...").style("font-size: 12px;")
                            self.ui.label(f"⚡ 動作: {step['action']}").style("font-size: 12px;")
                            if step.get("result"):
                                self.ui.label(f"📋 結果: {step['result']}").style("font-size: 12px;")
                            if step.get("next_step") and step["next_step"] != "無":
                                self.ui.label(f"➡️ 下一步: {step['next_step'][:150]}").style("font-size: 12px;")

            self.ui.label("🤖 醫療主Agent:").style("font-weight: 700; font-size: 13px; color: #555;")
            self.ui.html(f'<div class="chat-bubble-agent">{simple_md_render(msg["content"])}</div>')
