"""Keep one harmless NiceGUI message out of the console.

A page that was requested but never connected to the server (a health check, a crawler, a probe on the tunnel) is kept for the
reconnect timeout and then deleted. Its timers wait for a connection that never comes; when the page is deleted they wake up, find
their slot gone and each report "The parent slot of the element has been deleted." as a full traceback. Nothing is lost: there was
no page. With the long reconnect timeout (the page of a browser that lost its connection is kept for a minute) every such request
leaves three tracebacks a minute later, which hides the real errors. Only this one error is dropped; everything else is logged as before.
"""
from __future__ import annotations

import logging

DROPPED_PAGE_MESSAGE = "The parent slot of the element has been deleted."


class DroppedPageFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        error = record.exc_info[1] if record.exc_info else None
        return not (isinstance(error, RuntimeError) and str(error) == DROPPED_PAGE_MESSAGE)


def install_dropped_page_filter(logger: logging.Logger | None = None) -> None:
    """Add the filter to NiceGUI's logger once (the main script runs again for every browser connection)."""
    logger = logger or logging.getLogger("nicegui")
    if not any(isinstance(item, DroppedPageFilter) for item in logger.filters):
        logger.addFilter(DroppedPageFilter())
