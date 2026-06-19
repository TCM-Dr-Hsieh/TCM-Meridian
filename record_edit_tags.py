from __future__ import annotations

import difflib


HUMAN_EDIT_TAG = "[人類醫師_手動修改]"


def _append_human_edit_tag(line: str) -> str:
    if line.rstrip().endswith(HUMAN_EDIT_TAG):
        return line
    return line + HUMAN_EDIT_TAG


def tag_human_edits(old_text: str, new_text: str) -> str:
    old_lines = (old_text or "").split("\n")
    new_lines = (new_text or "").split("\n")
    result = []

    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    for op, _old_start, _old_end, new_start, new_end in matcher.get_opcodes():
        changed = op != "equal"
        for new_line in new_lines[new_start:new_end]:
            if changed and new_line.strip():
                result.append(_append_human_edit_tag(new_line))
            else:
                result.append(new_line)
    return "\n".join(result)
