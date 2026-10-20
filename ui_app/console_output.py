"""The program's console output never stops an agent.

The agents print their prompts and replies (the session log is written separately). In a console window Python writes Unicode, so
every character gets through. When the output is sent to a file or a pipe (`start.cmd > log.txt`, a service wrapper), Python writes in
the Windows code page instead, cp950 on a Traditional Chinese system, which has no place for some characters of a record (「衂」, an
emoji such as ❌). The print then raises UnicodeEncodeError, inside the agent's call, and that call fails. Such a character is written
as an escape (`\\u8842`) instead: the log keeps it readable, the agent goes on.
"""
from __future__ import annotations

import sys


def tolerate_unencodable_output(*streams) -> None:
    """Write a character the stream's encoding has no place for as a backslash escape instead of raising. Defaults to stdout and stderr;
    a stream that is missing (no console at all) or cannot be reconfigured is left as it is."""
    for stream in streams or (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="backslashreplace")
        except (OSError, ValueError):
            pass
