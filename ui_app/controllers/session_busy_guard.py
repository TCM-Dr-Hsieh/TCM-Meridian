from __future__ import annotations

from typing import Any


class SessionBusyGuard:
    """Centralized guard for session-sensitive UI actions.

    The guard has two jobs:
    - reject actions that would conflict with editing, agent execution,
      interview flow, or async session transitions;
    - keep the patient/session navigation controls disabled while busy.
    """

    def __init__(self, *, app_state: dict[str, Any], ui_state: dict[str, Any] | None = None):
        self.app_state = app_state
        self.ui_state = ui_state
        self._navigation_controls: list[Any] = []
        self._status_label: Any | None = None
        self.app_state.setdefault("_session_transition_active", False)
        self.app_state["_session_busy_guard"] = self

    def bind_ui_state(self, ui_state: dict[str, Any]):
        self.ui_state = ui_state
        self.sync_navigation()

    def register_navigation_controls(self, controls: list[Any], status_label: Any | None = None):
        self._navigation_controls = [c for c in controls if c is not None]
        self._status_label = status_label
        self.sync_navigation()

    def begin_transition(self):
        self.app_state["_session_transition_active"] = True
        self.sync_navigation()
        # 右欄聊天送出鈕也視覺鎖定，與左欄導覽一致
        main_chat_input = self.app_state.get("_main_chat_input")
        if main_chat_input:
            main_chat_input.lock("session_transition")

    def end_transition(self):
        self.app_state["_session_transition_active"] = False
        self.sync_navigation()
        main_chat_input = self.app_state.get("_main_chat_input")
        if main_chat_input:
            main_chat_input.unlock("session_transition")

    def is_busy(self, *, block_edit: bool = True) -> bool:
        return self.reason(block_edit=block_edit) is not None

    def reason(self, *, block_edit: bool = True) -> str | None:
        if self.app_state.get("_session_transition_active"):
            return "session_transition"
        if block_edit and self.ui_state and self.ui_state.get("view_mode") == "edit":
            return "edit_mode"
        if self.ui_state and self.ui_state.get("agent_running"):
            return "agent_running"
        if self.app_state.get("interview_active"):
            return "interview_active"
        return None

    def message(self, reason: str | None = None) -> str:
        reason = reason or self.reason()
        if reason == "session_transition":
            return "⚠️ session 切換或儲存流程進行中，請稍候"
        if reason == "edit_mode":
            return "⚠️ 編輯模式中，請先按「修改完成」或切回瀏覽模式"
        if reason == "agent_running":
            return "⚠️ AI 執行中，請先中斷或等待完成"
        if reason == "interview_active":
            return "⚠️ 問診中，請先完成或中斷問診"
        return "⚠️ 目前狀態不允許執行此操作"

    def reject_if_busy(
        self,
        *,
        status_label: Any | None = None,
        block_edit: bool = True,
    ) -> bool:
        reason = self.reason(block_edit=block_edit)
        if reason is None:
            return False
        msg = self.message(reason)
        target = status_label or self._status_label
        if target is not None:
            try:
                target.text = msg
            except Exception:
                pass
        self.app_state["_session_busy_guard_last_message"] = msg
        self.sync_navigation()
        return True

    def sync_navigation(self):
        disabled = self.is_busy(block_edit=True)
        for control in self._navigation_controls:
            try:
                if disabled:
                    control.disable()
                else:
                    control.enable()
            except Exception:
                try:
                    control.set_enabled(not disabled)
                except Exception:
                    pass


def get_session_busy_guard(
    app_state: dict[str, Any],
    ui_state: dict[str, Any] | None = None,
) -> SessionBusyGuard:
    guard = app_state.get("_session_busy_guard")
    if guard is None:
        guard = SessionBusyGuard(app_state=app_state, ui_state=ui_state)
    elif ui_state is not None:
        guard.bind_ui_state(ui_state)
    return guard
