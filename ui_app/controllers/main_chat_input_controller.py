from __future__ import annotations

from typing import Any


class MainChatInputController:
    """Own the main chat input lock state.

    Multiple workflows can need the right-panel chat input disabled at the same
    time. A lock set prevents one workflow from accidentally re-enabling input
    while another workflow still needs it locked.
    """

    def __init__(self, *, app_state: dict[str, Any], btn_send: Any):
        self.app_state = app_state
        self.btn_send = btn_send
        self.locks: set[str] = set()
        self._sync()

    def lock(self, reason: str):
        if reason:
            self.locks.add(reason)
        self._sync()

    def unlock(self, reason: str):
        if reason:
            self.locks.discard(reason)
        self._sync()

    def reset(self):
        self.locks.clear()
        # session 切換/儲存流程進行中時，reset 不得解除 transition 鎖
        # （摘要並退出會在 transition 中途清空 session UI 並呼叫 reset）
        if self.app_state.get("_session_transition_active"):
            self.locks.add("session_transition")
        self._sync()

    def is_locked(self) -> bool:
        return bool(self.locks)

    def _sync(self):
        self.app_state["_main_chat_input_locks"] = sorted(self.locks)
        if self.is_locked():
            self.btn_send.disable()
        else:
            self.btn_send.enable()
