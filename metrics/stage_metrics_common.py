import json
import sys
from pathlib import Path


_EVAL_DIR = str(Path(__file__).resolve().parents[1] / "VideoDeepResearch" / "eval")
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from answer_utils import answers_match, normalize_options, normalize_whitespace


VALID_ANSWER_SOURCES = {"benchmark", "stage_local", "terminal_final"}


def stage_sort_key(stage_name: str):
    name = str(stage_name or "").strip()
    if name == "initial":
        return (0, 0, name)
    if name == "generated":
        return (1, 0, name)
    if name.startswith("ref") and name[3:].isdigit():
        return (2, int(name[3:]), name)
    return (3, 0, name)


def sorted_stage_names(stage_names):
    return sorted({str(name).strip() for name in stage_names if str(name).strip()}, key=stage_sort_key)


def first_nonempty(*values):
    for value in values:
        text = normalize_whitespace(value)
        if text:
            return text
    return None


def safe_json_obj(raw):
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def parse_csv_arg(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = str(value).split(",")
    return [str(item).strip() for item in items if str(item).strip()]


def parse_answer_source_overrides(value):
    overrides = {}
    for item in parse_csv_arg(value):
        if "=" not in item:
            raise ValueError(
                f"Invalid answer-source override {item!r}; expected STAGE=SOURCE."
            )
        stage_name, source_name = item.split("=", 1)
        stage_name = stage_name.strip()
        source_name = source_name.strip()
        if source_name not in VALID_ANSWER_SOURCES:
            raise ValueError(
                f"Invalid answer source {source_name!r}; "
                f"expected one of {sorted(VALID_ANSWER_SOURCES)}."
            )
        overrides[stage_name] = source_name
    return overrides


def default_answer_source(stage_name: str) -> str:
    return "benchmark" if str(stage_name or "").strip() == "initial" else "stage_local"


def get_terminal_stage(sample):
    terminal_stage = normalize_whitespace(sample.get("terminal_stage"))
    if terminal_stage:
        return terminal_stage

    available = [
        name for name in sample.get("stages", {})
        if sample["stages"].get(name, {}).get("available")
    ]
    ordered = sorted_stage_names(available)
    return ordered[-1] if ordered else None


def get_terminal_stage_answer(sample):
    terminal_stage = get_terminal_stage(sample)
    if not terminal_stage:
        return None
    stage = sample.get("stages", {}).get(terminal_stage, {})
    return first_nonempty(
        stage.get("stage_local_answer"),
        sample.get("final_answer"),
        sample.get("metadata", {}).get("final_answer"),
    )


def select_proposed_answer(sample, stage_name, answer_source_overrides=None):
    answer_source_overrides = answer_source_overrides or {}
    source_name = answer_source_overrides.get(stage_name, default_answer_source(stage_name))

    if source_name == "benchmark":
        proposed_answer = first_nonempty(sample.get("gold_answer"))
    elif source_name == "stage_local":
        proposed_answer = first_nonempty(
            sample.get("stages", {}).get(stage_name, {}).get("stage_local_answer")
        )
    elif source_name == "terminal_final":
        proposed_answer = get_terminal_stage_answer(sample)
    else:
        raise ValueError(
            f"Unsupported answer source {source_name!r}; expected one of "
            f"{sorted(VALID_ANSWER_SOURCES)}."
        )

    return source_name, proposed_answer


def build_stage_record(sample, stage_name, answer_source_overrides=None):
    stage = dict(sample.get("stages", {}).get(stage_name) or {})
    stage.setdefault("trace_steps", [])
    stage.setdefault("executed_tools", [])

    answer_source, proposed_answer = select_proposed_answer(
        sample,
        stage_name,
        answer_source_overrides=answer_source_overrides,
    )

    options = normalize_options(sample.get("options"))
    gold_answer = first_nonempty(sample.get("gold_answer"))
    is_correct = None
    if proposed_answer and gold_answer:
        is_correct = answers_match(proposed_answer, gold_answer, options=options)

    return {
        "sample_id": sample.get("sample_id"),
        "source_dir": sample.get("source_dir"),
        "video_path": sample.get("video_path", ""),
        "question": sample.get("question", ""),
        "question_id": sample.get("question_id"),
        "options": options,
        "gold_answer": gold_answer,
        "terminal_stage": get_terminal_stage(sample),
        "final_answer": first_nonempty(
            sample.get("final_answer"),
            sample.get("metadata", {}).get("final_answer"),
        ),
        "stage": stage_name,
        "available": bool(stage.get("available")),
        "is_terminal": stage_name == get_terminal_stage(sample),
        "trace_steps": list(stage.get("trace_steps") or []),
        "stage_local_answer": first_nonempty(stage.get("stage_local_answer")),
        "proposed_answer": proposed_answer,
        "proposed_answer_source": answer_source,
        "is_correct": is_correct,
        "stored_verifier_raw": stage.get("stored_verifier_raw"),
        "stored_verifier_output": stage.get("stored_verifier_output"),
        "planner_output": stage.get("planner_output"),
        "executed_tools": list(stage.get("executed_tools") or []),
        "metadata": dict(sample.get("metadata") or {}),
    }


def results_dir_default_output(input_path, filename="per_sample_metrics.jsonl"):
    input_path = Path(input_path)
    stem = input_path.name or "metrics"
    return Path(__file__).resolve().parent / "results" / stem / filename
