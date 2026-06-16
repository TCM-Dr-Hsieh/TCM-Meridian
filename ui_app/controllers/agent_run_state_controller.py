from __future__ import annotations

from typing import Any


class AgentRunStateController:
    """Own MainAgent instance, run flag, and session generation."""

    def __init__(self, *, app_state: dict[str, Any], ui_state: dict[str, Any] | None = None):
        self.app_state = app_state
        self.ui_state = ui_state
        self.app_state.setdefault("agent_instance", None)
        self.app_state.setdefault("session_generation", 0)
        if self.ui_state is not None:
            self.ui_state.setdefault("agent_running", False)

    def get_agent(self) -> Any:
        return self.app_state.get("agent_instance")

    def set_agent(self, agent: Any):
        self.app_state["agent_instance"] = agent

    def ensure_agent(self, factory: Any, cfg: dict) -> Any:
        agent = self.get_agent()
        if agent is None:
            agent = factory.create_main_agent(cfg)
            self.set_agent(agent)
        return agent

    def clear_agent(self):
        self.app_state["agent_instance"] = None

    def start_run(self):
        if self.ui_state is not None:
            self.ui_state["agent_running"] = True
        guard = self.app_state.get("_session_busy_guard")
        if guard:
            guard.sync_navigation()

    def finish_run(self):
        if self.ui_state is not None:
            self.ui_state["agent_running"] = False
        guard = self.app_state.get("_session_busy_guard")
        if guard:
            guard.sync_navigation()

    def bump_generation(self):
        self.app_state["session_generation"] = self.current_generation() + 1

    def current_generation(self) -> int:
        return int(self.app_state.get("session_generation", 0))

    def is_current_generation(self, generation: int) -> bool:
        return generation == self.current_generation()

    def reset_for_session_change(self):
        self.clear_agent()
        self.finish_run()


def get_agent_run_state(
    app_state: dict[str, Any],
    ui_state: dict[str, Any] | None = None,
) -> AgentRunStateController:
    controller = app_state.get("_agent_run_state")
    if controller is None:
        controller = AgentRunStateController(app_state=app_state, ui_state=ui_state)
        app_state["_agent_run_state"] = controller
    elif ui_state is not None and controller.ui_state is None:
        controller.ui_state = ui_state
        controller.ui_state.setdefault("agent_running", False)
    return controller
