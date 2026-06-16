from __future__ import annotations

from datetime import date


def deidentify_name(name: str) -> str:
    """Keep only the first character of a patient name for LLM prompts."""
    cleaned = "".join((name or "").split())
    if not cleaned:
        return ""
    return cleaned[0] + ("某" * max(len(cleaned) - 1, 1))


def _age_years_months_on_date(birthday: str, reference_date: str) -> tuple[int, int] | None:
    try:
        birth_date = date.fromisoformat((birthday or "").strip())
        ref_date = date.fromisoformat((reference_date or "").strip())
    except Exception:
        return None

    if ref_date < birth_date:
        return None

    years = ref_date.year - birth_date.year
    months = ref_date.month - birth_date.month
    if ref_date.day < birth_date.day:
        months -= 1
    if months < 0:
        years -= 1
        months += 12
    return years, months


def deidentify_birthday_with_age(birthday: str, reference_date: str) -> str:
    """Keep birth year and computed age, but mask month/day."""
    birthday = (birthday or "").strip()
    if not birthday:
        return ""

    year = birthday[:4] if len(birthday) >= 4 and birthday[:4].isdigit() else "XXXX"
    masked = f"{year}-XX-XX"
    age = _age_years_months_on_date(birthday, reference_date)
    if age is None:
        return masked
    years, months = age
    return f"{masked} ({years}歲{months}月)"


def format_patient_basic_info_for_llm(
    basic_info: dict | None,
    reference_date: str,
    *,
    include_visit_date: bool = False,
) -> str:
    """Format de-identified patient basic info for LLM prompts."""
    bi = basic_info or {}
    lines = [
        f"姓名: {deidentify_name(bi.get('name', ''))}",
        f"性別: {bi.get('gender', '')}",
        f"生日: {deidentify_birthday_with_age(bi.get('birthday', ''), reference_date)}",
    ]
    if include_visit_date:
        lines.append(f"就診日期: {reference_date}")
    lines.append(f"備註: {bi.get('remark', '')}")
    return "\n".join(lines)
