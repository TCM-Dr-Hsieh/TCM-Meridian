from __future__ import annotations

from typing import Any


class InterviewStateController:
    """Own the core auto-interview state stored in app_state."""

    def __init__(self, app_state: dict[str, Any]):
        self.app_state = app_state
        self.app_state.setdefault("ic_subagent", None)
        self.app_state.setdefault("interview_dialogue", "")
        self.app_state.setdefault("interview_active", False)

    def get_subagent(self) -> Any:
        return self.app_state.get("ic_subagent")

    def set_subagent(self, subagent: Any):
        self.app_state["ic_subagent"] = subagent

    def is_active(self) -> bool:
        return bool(self.app_state.get("interview_active"))

    def get_dialogue(self) -> str:
        return self.app_state.get("interview_dialogue", "")

    def start(self, subagent: Any | None = None):
        if subagent is not None:
            self.set_subagent(subagent)
        self.app_state["interview_active"] = True
        guard = self.app_state.get("_session_busy_guard")
        if guard:
            guard.sync_navigation()

    def finish(self, dialogue: str = ""):
        self.app_state["interview_dialogue"] = dialogue or ""
        self.app_state["interview_active"] = False
        guard = self.app_state.get("_session_busy_guard")
        if guard:
            guard.sync_navigation()

    def stop(self, dialogue: str = ""):
        self.finish(dialogue)

    def clear_dialogue(self):
        self.app_state["interview_dialogue"] = ""

    def reset_for_session_change(self):
        self.app_state["ic_subagent"] = None
        self.app_state["interview_dialogue"] = ""
        self.app_state["interview_active"] = False
        guard = self.app_state.get("_session_busy_guard")
        if guard:
            guard.sync_navigation()


def get_interview_state(app_state: dict[str, Any]) -> InterviewStateController:
    controller = app_state.get("_interview_state")
    if controller is None:
        controller = InterviewStateController(app_state)
        app_state["_interview_state"] = controller
    return controller
