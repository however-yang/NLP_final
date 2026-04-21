from __future__ import annotations

import re

from text_rich_mllm.utils.constants import AnswerType


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_text(text: str) -> str:
    text = normalize_whitespace(text).lower()
    text = text.replace("'", "'").replace("`", "'")
    text = re.sub(r"[^\w\s.%/-]", "", text)
    return normalize_whitespace(text)


def _parse_numeric_value(text: str) -> float | None:
    text = normalize_whitespace(text).lower().replace(",", "")
    text = text.replace("%", "")
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def normalize_numeric_text(text: str) -> str:
    value = _parse_numeric_value(text)
    if value is None:
        return normalize_text(text)
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def extract_choice_label(text: str) -> str:
    cleaned = normalize_whitespace(text).upper()
    if re.fullmatch(r"[A-Z][\s\.\)\]:-]*", cleaned):
        return cleaned[0]
    match = re.search(r"\b(?:ANSWER|OPTION|CHOICE)\s*(?:IS\s*)?[:\-]?\s*([A-Z])(?:\b|[\.\)])", cleaned)
    if match:
        return match.group(1)
    match = re.match(r"^\(?([A-Z])[\)\.\:\-]\s*", cleaned)
    if match:
        return match.group(1)
    return cleaned


def normalize_answer(text: str, answer_type: str) -> str:
    if answer_type == AnswerType.MULTIPLE_CHOICE.value:
        return extract_choice_label(text)
    if answer_type == AnswerType.NUMERIC.value:
        return normalize_numeric_text(text)
    return normalize_text(text)


def is_valid_choice_prediction(text: str) -> bool:
    label = extract_choice_label(text)
    return len(label) == 1 and "A" <= label <= "Z"


def numeric_equal(prediction: str, gold_answer: str, *, tolerance: float = 1e-3) -> bool:
    pred_value = _parse_numeric_value(prediction)
    gold_value = _parse_numeric_value(gold_answer)
    if pred_value is None or gold_value is None:
        return False
    scale = max(abs(gold_value), 1.0)
    return abs(pred_value - gold_value) <= max(tolerance, tolerance * scale)

