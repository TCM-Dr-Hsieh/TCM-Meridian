from __future__ import annotations

import difflib
import re


def simple_md_render(text: str) -> str:
    """Small markdown subset used by the existing NiceGUI HTML labels."""
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    escaped = re.sub(
        r"^### (.+)$",
        r'<div style="font-size:15px; font-weight:700; color:#2d6a4f; margin:8px 0 4px;">\1</div>',
        escaped,
        flags=re.MULTILINE,
    )
    escaped = re.sub(
        r"^## (.+)$",
        r'<div style="font-size:16px; font-weight:700; color:#1b4332; margin:10px 0 4px;">\1</div>',
        escaped,
        flags=re.MULTILINE,
    )
    escaped = re.sub(
        r"^# (.+)$",
        r'<div style="font-size:18px; font-weight:700; color:#1b4332; margin:12px 0 6px;">\1</div>',
        escaped,
        flags=re.MULTILINE,
    )
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    return escaped.replace("\n", "<br>")


def strip_citations(text: str) -> str:
    """Hide square-bracket source tags in browse mode."""
    return re.sub(r"\s*\[[^\]]*\]", "", text)


def patient_info_html(basic_info: dict) -> str:
    return (
        f"<b>{basic_info.get('name', '')}</b><br>"
        f"ID: {basic_info.get('id', '')}<br>"
        f"性別: {basic_info.get('gender', '')} | 生日: {basic_info.get('birthday', '')}<br>"
        f"電話: {basic_info.get('phone', '')}"
    )


def tag_human_edits(old_text: str, new_text: str) -> str:
    tag = "[人類醫師_手動修改]"
    old_lines = (old_text or "").split("\n")
    new_lines = (new_text or "").split("\n")
    result = []

    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    for op, _old_start, _old_end, new_start, new_end in matcher.get_opcodes():
        changed = op != "equal"
        for new_line in new_lines[new_start:new_end]:
            stripped = new_line.strip()
            if not changed or not stripped or tag in new_line:
                result.append(new_line)
            else:
                result.append(new_line + tag)
    return "\n".join(result)


def build_conversation_text(messages: list[dict]) -> str:
    """Build the persisted plain-text main chat transcript."""
    parts: list[str] = []
    for msg in messages:
        if msg["role"] == "user":
            parts.append(f"【人類醫師】\n{msg['content']}")
        elif msg["role"] == "agent":
            parts.append(f"【主治醫師 Agent】\n{msg['content']}")
    return "\n\n".join(parts)


def format_interview_conversations(conversations: list[dict]) -> str:
    parts: list[str] = []
    for msg in conversations:
        round_num = msg.get("round", "?")
        if msg["role"] == "subagent":
            parts.append(f"＜第{round_num}回合(R{round_num})助理提問＞：{msg['content']}")
        elif msg["role"] == "patient":
            parts.append(f"＜第{round_num}回合(R{round_num})收到回應＞：{msg['content']}")
    return "\n".join(parts)


def generate_diff_html(old_text: str, new_text: str, title: str = "") -> str:
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    differ = difflib.unified_diff(old_lines, new_lines, lineterm="", n=3)

    html_parts = []
    if title:
        html_parts.append(f'<div style="font-weight:700; font-size:15px; margin-bottom:8px; color:#1b4332;">{title}</div>')

    html_parts.append(
        '<div style="font-family: monospace; font-size:13px; line-height:1.6; white-space:pre-wrap; '
        'background:#fafafa; border:1px solid #e0e0e0; border-radius:8px; padding:12px; overflow-x:auto;">'
    )

    has_diff = False
    for line in differ:
        escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if line.startswith("+++") or line.startswith("---"):
            continue
        elif line.startswith("@@"):
            html_parts.append(f'<span style="color:#888; font-style:italic;">{escaped}</span>\n')
            has_diff = True
        elif line.startswith("+"):
            html_parts.append(f'<span style="background:#d4edda; color:#155724;">{escaped}</span>\n')
            has_diff = True
        elif line.startswith("-"):
            html_parts.append(
                f'<span style="background:#f8d7da; color:#721c24; text-decoration:line-through;">{escaped}</span>\n'
            )
            has_diff = True
        else:
            html_parts.append(f"{escaped}\n")

    if not has_diff:
        html_parts.append('<span style="color:#888;">（無差異）</span>')

    html_parts.append("</div>")
    return "".join(html_parts)
