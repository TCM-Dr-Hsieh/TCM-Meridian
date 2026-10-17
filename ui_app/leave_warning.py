"""Ask before this page is closed or reloaded while that would lose work.

A reload starts a fresh page: an AI turn that is running, an interview, a record edit that is not finished, a visit being switched
and a remote-microphone capture that lives in this very tab are all tied to the page that is being left. (A recording from the
computer's own microphone belongs to the server and goes on, so it does not ask.) The patient and visit that were open come back on
their own (`LastSelectionStore`); the question only matters for what cannot.
"""
from __future__ import annotations

from typing import Any

JS = """
window.tcmLeaveBusy = false;
window.addEventListener('beforeunload', function (event) {
  var remote = window.tcmRemote && (window.tcmRemote.sessionId || window.tcmRemote.streaming);
  if (window.tcmLeaveBusy || remote) {
    event.preventDefault();
    event.returnValue = '';
  }
});
"""


def leaving_would_lose_work(app_state: dict[str, Any]) -> bool:
    """Something in this page's state is in progress (the lock that keeps the patient navigation shut, without the voice recording)."""
    guard = app_state.get("_session_busy_guard")
    return bool(guard is not None and guard.is_busy(block_edit=True, include_voice=False))


def register_leave_warning(ui, app_state: dict[str, Any]) -> None:
    """Put the warning in this page and keep its flag in step with the page's state."""
    ui.add_head_html(f"<script>{JS}</script>")
    last: dict[str, bool | None] = {"busy": None}

    def sync():
        busy = leaving_would_lose_work(app_state)
        if busy == last["busy"]:
            return
        last["busy"] = busy
        try:
            ui.run_javascript(f"window.tcmLeaveBusy = {'true' if busy else 'false'};")
        except Exception:                      # the page is going away: nothing to tell it
            last["busy"] = None

    ui.timer(1.0, sync)
