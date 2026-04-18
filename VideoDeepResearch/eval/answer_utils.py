import re
from typing import Dict, Tuple


_MCQ_PREFIX_RE = re.compile(r"^\(?([A-Za-z])\)?(?:\s*[\.\):])?\s*(.*)$")


def normalize_whitespace(text) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_options(options) -> list:
    if options is None:
        return []
    if isinstance(options, dict):
        keys = sorted(options.keys(), key=lambda value: str(value))
        return [normalize_whitespace(options[key]) for key in keys if normalize_whitespace(options[key])]
    if isinstance(options, (list, tuple)):
        return [normalize_whitespace(option) for option in options if normalize_whitespace(option)]
    value = normalize_whitespace(options)
    return [value] if value else []


def _option_letter_and_text(option: str, index: int):
    option_text = normalize_whitespace(option)
    if not option_text:
        return None, ""

    match = re.match(r"^\(?([A-Za-z])\)?(?:\s*[\.\):])\s*(.*)$", option_text)
    if match:
        return match.group(1).upper(), normalize_whitespace(match.group(2))
    return chr(ord("A") + index), option_text


def build_option_maps(options) -> Tuple[Dict[str, str], Dict[str, str]]:
    letter_to_text = {}
    normalized_option_to_letter = {}

    for index, option in enumerate(normalize_options(options)):
        letter, body = _option_letter_and_text(option, index)
        if not letter:
            continue
        letter_to_text.setdefault(letter, body)
        normalized_option_to_letter[normalize_whitespace(option).lower()] = letter
        if body:
            normalized_option_to_letter[body.lower()] = letter

    return letter_to_text, normalized_option_to_letter


def resolve_mcq_answer_text(answer, options=None) -> str:
    text = normalize_whitespace(answer)
    if not text:
        return ""

    letter_to_text, _ = build_option_maps(options)
    match = re.match(r"^\(?([A-Za-z])\)?\.?$", text)
    if not match:
        return text

    return letter_to_text.get(match.group(1).upper(), text)


def canonicalize_answer(answer, options=None) -> str:
    text = normalize_whitespace(answer)
    if not text:
        return ""

    letter_to_text, normalized_option_to_letter = build_option_maps(options)
    if letter_to_text:
        lowered = text.lower()
        if lowered in normalized_option_to_letter:
            return normalized_option_to_letter[lowered]

        prefix_match = _MCQ_PREFIX_RE.match(text)
        if prefix_match:
            letter = prefix_match.group(1).upper()
            suffix = normalize_whitespace(prefix_match.group(2))
            if letter in letter_to_text:
                option_body = normalize_whitespace(letter_to_text[letter]).lower()
                if not suffix or suffix.lower() == option_body:
                    return letter
    return text.lower()


def answers_match(predicted, gold, options=None):
    predicted_canonical = canonicalize_answer(predicted, options=options)
    gold_canonical = canonicalize_answer(gold, options=options)
    if not predicted_canonical or not gold_canonical:
        return None
    return predicted_canonical == gold_canonical
