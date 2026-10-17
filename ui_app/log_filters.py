"""Keep one harmless NiceGUI message out of the console.

A page that was requested but never connected to the server (a health check, a crawler, a probe on the tunnel) is kept for the
reconnect timeout and then deleted. Its timers wait for a connection that never comes; when the page is deleted they wake up, find
their slot gone and each report "The parent slot of the element has been deleted." as a full traceback. Nothing is lost: there was
no page. With the long reconnect timeout (the page of a browser that lost its connection is kept for a minute) every such request
leaves three tracebacks a minute later, which hides the real errors.

The same message can come from a real UI problem (an element used after its parent was deleted), and that one must stay in the log.
So the filter looks at where the error came from, not at its text: only an error that is raised by a NiceGUI timer when it
enters its slot, and by nothing else (no frame of this program in it), is dropped. If a NiceGUI version changes that path the filter
stops matching and the error is logged again, which is the safe side.
"""
from __future__ import annotations

import logging
import traceback
from pathlib import Path

DROPPED_PAGE_MESSAGE = "The parent slot of the element has been deleted."

# (folder, file, function) of every frame of the traceback; the timer loop is what NiceGUI reports it from. The first frame is where
# the finished task is read (or, for a timer that catches its own error, the loop itself).
_READ_TASK = ("nicegui", "background_tasks.py", "_handle_exceptions")
_TIMER_LOOPS = {("nicegui", "timer.py", "_run_once"), ("nicegui", "timer.py", "_run_in_loop")}
_ENTER_SLOT = [("elements", "timer.py", "_get_context"), ("nicegui", "element.py", "parent_slot")]


def _frames(error: BaseException) -> list[tuple[str, str, str]]:
    return [(Path(frame.filename).parent.name, Path(frame.filename).name, frame.name) for frame in traceback.extract_tb(error.__traceback__)]


def is_timer_of_a_deleted_page(error: BaseException | None) -> bool:
    """The error is "parent slot deleted" raised by a NiceGUI timer entering its slot after the page was deleted, and by nothing else."""
    if not (isinstance(error, RuntimeError) and str(error) == DROPPED_PAGE_MESSAGE):
        return False
    frames = _frames(error)
    if frames and frames[0] == _READ_TASK:
        frames = frames[1:]
    return len(frames) == 3 and frames[0] in _TIMER_LOOPS and frames[1:] == _ENTER_SLOT


class DroppedPageFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not is_timer_of_a_deleted_page(record.exc_info[1] if record.exc_info else None)


def install_dropped_page_filter(logger: logging.Logger | None = None) -> None:
    """Add the filter to NiceGUI's logger once (the main script runs again for every browser connection)."""
    logger = logger or logging.getLogger("nicegui")
    if not any(isinstance(item, DroppedPageFilter) for item in logger.filters):
        logger.addFilter(DroppedPageFilter())
