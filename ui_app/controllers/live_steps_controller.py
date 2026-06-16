from __future__ import annotations

from typing import Any

from ui_app.controllers.agent_run_state_controller import get_agent_run_state


class LiveStepsController:
    """Render incremental MainAgent live steps while a turn is running."""

    def __init__(self, ui: Any, app_state: dict[str, Any], container: Any, status_label: Any):
        self.ui = ui
        self.app_state = app_state
        self.agent_run_state = get_agent_run_state(app_state)
        self.container = container
        self.status_label = status_label
        self.rendered_count = 0
        self.timer = ui.timer(2.0, self.poll, active=False)

        app_state["_live_poll_timer"] = self.timer
        app_state["_live_steps_container"] = container
        app_state["_live_steps_controller"] = self

    def start(self):
        self.rendered_count = 0
        self.container.clear()
        self.container.set_visibility(True)
        with self.container:
            self.ui.label("⏳ Agent 執行中...").style(
                "color: #1565C0; font-weight: bold; font-size: 13px;"
            )
        self.timer.activate()

    def stop(self, *, clear: bool = True):
        self.timer.deactivate()
        if clear:
            self.container.clear()
        self.container.set_visibility(False)

    def poll(self):
        agent = self.agent_run_state.get_agent()
        if not agent or not hasattr(agent, "_live_steps"):
            return

        steps = agent._live_steps
        if len(steps) <= self.rendered_count:
            return

        with self.container:
            for i in range(self.rendered_count, len(steps)):
                s = steps[i]
                action = s.get("action", "?")
                thinking = s.get("thinking", "")[:120]
                label = s.get("step_label", f"Step {i+1}")

                with self.ui.row().classes("items-center gap-2").style(
                    "padding: 4px 8px; background: white; border-radius: 6px; "
                    "border-left: 3px solid #42A5F5;"
                ):
                    self.ui.label(f"✅ {label}").style(
                        "font-weight: 700; font-size: 12px; color: #1565C0; white-space: nowrap;"
                    )
                    self.ui.label(f"⚡ {action}").style(
                        "font-size: 12px; color: #333; white-space: nowrap;"
                    )
                    if thinking:
                        self.ui.label(f"💭 {thinking}...").style(
                            "font-size: 11px; color: #777; overflow: hidden; "
                            "text-overflow: ellipsis; white-space: nowrap; max-width: 220px;"
                        )

        self.rendered_count = len(steps)
        self.status_label.text = f"🤖 AI 執行中... (已完成 {len(steps)} 個子輪)"
