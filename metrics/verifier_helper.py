import json
import os
import sys
from pathlib import Path


_EVAL_DIR = str(Path(__file__).resolve().parents[1] / "VideoDeepResearch" / "eval")
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from refine_prompt import verifier_prompt

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


TEXT_ONLY_SUFFIX = (
    "\n\nTEXT_ONLY_MODE:\n"
    "This verifier call has no access to video, audio, frames, OCR outputs, or hidden tool state. "
    "Use only the text in this prompt and the iteration summary above. "
    "There is intentionally no separate answer field in this verifier call; infer any final conclusion "
    "only from the trace itself. Treat unsupported sensory claims as unsupported rather than observed. "
    "However, if the TRACE explicitly attributes a claim to a named tool result and phrases it as a reported "
    "tool output, treat that attribution as textual evidence rather than as an unsupported direct observation. "
    "Tool names in the iteration summary alone do not supply missing numeric values. "
    "Set error_categories[].evidence to null or \"N/A (text-only pass)\".\n"
)

def _normalize_optional_number(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_optional_int(value):
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_trace_quality_scores(scores):
    if not isinstance(scores, dict):
        return None

    return {
        "logical_coherence": _normalize_optional_number(scores.get("logical_coherence")),
        "completeness": _normalize_optional_number(scores.get("completeness")),
        "factual_correctness": _normalize_optional_number(scores.get("factual_correctness")),
        "reasoning_order": _normalize_optional_number(scores.get("reasoning_order")),
    }


def _normalize_error_categories(value):
    if not isinstance(value, list):
        return []

    items = []
    for item in value:
        if not isinstance(item, dict):
            continue
        suggested_tools = item.get("suggested_tools")
        if not isinstance(suggested_tools, list):
            suggested_tools = []
        items.append(
            {
                "type": str(item.get("type") or "").strip(),
                "step_index": _normalize_optional_int(item.get("step_index")),
                "description": str(item.get("description") or "").strip(),
                "severity": str(item.get("severity") or "LOW").strip().upper() or "LOW",
                "suggested_tools": [str(tool).strip() for tool in suggested_tools if str(tool).strip()],
                "evidence": item.get("evidence"),
            }
        )
    return items


def _normalize_evidence_gaps(value):
    if not isinstance(value, list):
        return []

    items = []
    for item in value:
        if not isinstance(item, dict):
            continue
        items.append(
            {
                "step_index": _normalize_optional_int(item.get("step_index")),
                "summary": str(item.get("summary") or "").strip(),
                "scope": str(item.get("scope") or "").strip(),
            }
        )
    return items


def normalize_verifier_output(payload):
    if not isinstance(payload, dict):
        return None

    normalized = dict(payload)
    verdict = str(payload.get("verdict") or "FAIL").strip().upper()
    normalized["verdict"] = verdict if verdict in {"PASS", "FAIL"} else "FAIL"
    normalized["answer_correct"] = bool(payload.get("answer_correct"))
    normalized["trace_quality_scores"] = normalize_trace_quality_scores(payload.get("trace_quality_scores"))
    normalized["error_categories"] = _normalize_error_categories(payload.get("error_categories"))
    normalized["evidence_gaps"] = _normalize_evidence_gaps(payload.get("evidence_gaps"))
    normalized["confidence"] = _normalize_optional_number(payload.get("confidence"))
    normalized["summary"] = str(payload.get("summary") or "").strip()
    return normalized


def build_verifier_prompt(stage_record, iteration=0, max_iterations=1, history=None):
    question = str(stage_record.get("question") or "").strip()
    options = stage_record.get("options") or []
    if isinstance(options, dict):
        options = [str(options[key]) for key in sorted(options.keys(), key=lambda value: str(value))]

    question_block = question
    if options:
        question_block += "\nOptions:\n" + "\n".join(str(option) for option in options)

    trace_steps = list(stage_record.get("trace_steps") or [])
    trace_text = "\n".join(f"{index + 1}. {step}" for index, step in enumerate(trace_steps)) if trace_steps else "(empty trace)"
    ctx = (
        f"ITERATION: {iteration + 1}/{max_iterations}\n"
        f"PREVIOUS_ITERATIONS_SUMMARY:\n{json.dumps(history or [], ensure_ascii=False, indent=2)}\n"
    )
    return (
        ctx
        + "\n"
        + verifier_prompt.strip()
        + "\n\nQUESTION:\n"
        + question_block
        + "\n\nTRACE:\n"
        + trace_text
        + TEXT_ONLY_SUFFIX
    )


def create_verifier_client():
    if OpenAI is None:
        raise RuntimeError("openai package required: pip install openai")

    api_base = (
        os.environ.get("PLANNER_API_BASE")
        or os.environ.get("API_BASE_URL")
        or "https://api.openai.com/v1"
    ).strip()
    api_key = (
        os.environ.get("PLANNER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("API_KEY")
        or ""
    ).strip()
    if not api_key:
        raise RuntimeError("Set PLANNER_API_KEY, OPENAI_API_KEY, or API_KEY")

    base = api_base.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    client = OpenAI(base_url=base, api_key=api_key)
    model = (
        os.environ.get("VERIFIER_MODEL_NAME")
        or os.environ.get("PLANNER_MODEL_NAME")
        or os.environ.get("API_MODEL_NAME")
        or "gpt-4o-mini"
    )
    return client, model


def run_verifier(client, model, prompt):
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    content = completion.choices[0].message.content
    return content if isinstance(content, str) else (content or "")


def parse_verifier_output(raw_text):
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None
    try:
        parsed = json.loads(raw_text.strip())
    except json.JSONDecodeError:
        return None
    return normalize_verifier_output(parsed)


def resolve_verifier(stage_record, verifier_mode="hybrid", client=None, model=None):
    verifier_mode = str(verifier_mode or "hybrid").strip().lower()
    stored_output = normalize_verifier_output(stage_record.get("stored_verifier_output"))
    stored_raw = stage_record.get("stored_verifier_raw")
    if stored_output is None and stored_raw:
        stored_output = parse_verifier_output(stored_raw)

    if verifier_mode not in {"hybrid", "stored", "offline"}:
        raise ValueError("verifier_mode must be one of: hybrid, stored, offline")

    if verifier_mode in {"hybrid", "stored"} and isinstance(stored_output, dict):
        return {
            "verifier_mode_used": "stored",
            "verifier_raw": stored_raw or json.dumps(stored_output, ensure_ascii=False),
            "verifier_output": stored_output,
            "model": None,
        }

    if verifier_mode == "stored":
        return {
            "verifier_mode_used": "stored_missing",
            "verifier_raw": stored_raw,
            "verifier_output": None,
            "model": None,
        }

    if client is None or model is None:
        client, model = create_verifier_client()

    prompt = build_verifier_prompt(stage_record, iteration=0, max_iterations=1, history=[])
    raw_text = run_verifier(client, model, prompt)
    return {
        "verifier_mode_used": "offline",
        "verifier_raw": raw_text,
        "verifier_output": parse_verifier_output(raw_text),
        "model": model,
    }
