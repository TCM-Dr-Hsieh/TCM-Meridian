from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ui_app.controllers.agent_run_state_controller import get_agent_run_state


def _initial_state() -> dict[str, Any]:
    return {
        "selected_patient_folder": None,
        "selected_patient_info": None,
        "selected_session_date": None,
        "agent_instance": None,
        "ic_subagent": None,
        "interview_dialogue": "",
        "interview_active": False,
        "session_generation": 0,
    }


@dataclass
class AppContext:
    """Shared UI state facade.

    The legacy UI still reads and writes ``state`` as a dict. The facade gives
    new controllers a single place to reset cross-tab state without changing
    persisted data formats or Agent behavior.
    """

    state: dict[str, Any] = field(default_factory=_initial_state)

    def bump_session_generation(self):
        get_agent_run_state(self.state).bump_generation()

    def reset_agent_state(self):
        get_agent_run_state(self.state).reset_for_session_change()
        interview_state = self.state.get("_interview_state")
        if interview_state:
            interview_state.reset_for_session_change()
        else:
            self.state["ic_subagent"] = None
            self.state["interview_dialogue"] = ""
            self.state["interview_active"] = False

    def reset_patient_selection(self):
        self.state["selected_patient_folder"] = None
        self.state["selected_patient_info"] = None
        self.state["selected_session_date"] = None
        self.reset_agent_state()

    def register_callback(self, name: str, callback: Callable[..., Any] | None):
        if callback is None:
            self.state.pop(name, None)
        else:
            self.state[name] = callback
