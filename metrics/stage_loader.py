import json
import re
from pathlib import Path

from stage_metrics_common import (
    first_nonempty,
    normalize_options,
    normalize_whitespace,
    safe_json_obj,
    sorted_stage_names,
)


def _load_json(path: Path):
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_json_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


def _load_jsonl_rows(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _step_to_text(step):
    if step is None:
        return ""
    if isinstance(step, str):
        return step.strip()
    if isinstance(step, dict):
        if isinstance(step.get("step"), str) and step.get("step").strip():
            return step["step"].strip()
        evidence = (
            step.get("evidence")
            or step.get("evidece")
            or step.get("evience")
            or step.get("evodence")
            or step.get("nevidence")
            or ""
        )
        inference = (
            step.get("inference")
            or step.get("Inference")
            or step.get("infefence")
            or ""
        )
        text = f"Evidence: {evidence}. Inference: {inference}".strip()
        return text if text != "Evidence: . Inference:" else ""
    return str(step).strip()


def normalize_trace_steps(trace_value):
    if isinstance(trace_value, dict):
        trace_value = trace_value.get("steps", trace_value)

    if isinstance(trace_value, list):
        texts = [_step_to_text(step) for step in trace_value]
        return [text for text in texts if text]

    text = normalize_whitespace(trace_value)
    return [text] if text else []


def _list_refinement_files(video_dir: Path):
    files = []
    for path in sorted(video_dir.glob("refinement_*.json")):
        match = re.match(r"refinement_(\d+)\.json$", path.name)
        if match:
            files.append((int(match.group(1)), path))
    return sorted(files, key=lambda item: item[0])


def _make_sample_id(video_path, question, question_id=None, source_dir=None):
    if source_dir:
        return str(Path(source_dir).resolve())
    question_key = normalize_whitespace(question) or "<missing-question>"
    if question_id is not None and str(question_id).strip():
        question_key = f"qid={question_id}"
    return f"{normalize_whitespace(video_path) or '<missing-video>'}::{question_key}"


def _extract_generated_artifacts(generated_file: dict):
    rounds = generated_file.get("generation_rounds")
    if not isinstance(rounds, list) or not rounds:
        return None, []

    for record in reversed(rounds):
        if not isinstance(record, dict):
            continue
        planner = safe_json_obj(record.get("planner_output")) or safe_json_obj(record.get("planner_raw"))
        executed_tools = record.get("executed_tools") if isinstance(record.get("executed_tools"), list) else []
        if planner is not None or executed_tools:
            return planner, executed_tools
    return None, []


def _coerce_stage(stage_name, payload, terminal_stage, sample_metadata):
    payload = dict(payload or {})
    trace_steps = normalize_trace_steps(payload.get("trace_steps"))
    if not trace_steps:
        trace_steps = normalize_trace_steps(payload.get("trace"))

    return {
        "name": stage_name,
        "available": bool(payload.get("available", bool(trace_steps))),
        "is_terminal": bool(payload.get("is_terminal", stage_name == terminal_stage)),
        "trace_steps": trace_steps,
        "stage_local_answer": first_nonempty(
            payload.get("stage_local_answer"),
            payload.get("answer"),
            payload.get("final_answer"),
            payload.get("refined_answer"),
        ),
        "stored_verifier_raw": payload.get("stored_verifier_raw") or payload.get("verifier_raw"),
        "stored_verifier_output": payload.get("stored_verifier_output") or payload.get("verifier_output"),
        "planner_output": safe_json_obj(payload.get("planner_output")) or safe_json_obj(payload.get("planner_raw")),
        "executed_tools": payload.get("executed_tools") if isinstance(payload.get("executed_tools"), list) else [],
        "metadata": sample_metadata,
    }


def _infer_terminal_stage_from_files(meta: dict, generated_file, refinement_files):
    terminal_stage = first_nonempty(meta.get("terminal_stage"))
    if terminal_stage:
        return terminal_stage

    refinement_numbers = [number for number, _ in refinement_files]
    if refinement_numbers:
        return f"ref{max(refinement_numbers)}"
    if isinstance(generated_file, dict) or meta.get("trace_generated"):
        return "generated"
    return "initial"


def _load_results_sample(video_dir: Path):
    video_dir = Path(video_dir)
    meta = _load_json(video_dir / "meta.json") or {}
    generated_file = _load_json(video_dir / "generated_trace.json")
    refinement_files = [(number, _load_json(path) or {}) for number, path in _list_refinement_files(video_dir)]
    refinement_payloads = {number: payload for number, payload in refinement_files}

    terminal_stage = _infer_terminal_stage_from_files(meta, generated_file, refinement_files)
    final_verifier_output = meta.get("final_verifier_output") or meta.get("verifier_output")
    final_verifier_raw = meta.get("final_verifier_raw") or meta.get("verifier_raw")

    stages = {
        "initial": {
            "name": "initial",
            "available": bool(normalize_trace_steps(meta.get("initial_trace_steps"))),
            "is_terminal": terminal_stage == "initial",
            "trace_steps": normalize_trace_steps(meta.get("initial_trace_steps")),
            "stage_local_answer": first_nonempty(meta.get("initial_answer")),
            "stored_verifier_raw": final_verifier_raw if terminal_stage == "initial" else None,
            "stored_verifier_output": final_verifier_output if terminal_stage == "initial" else None,
            "planner_output": None,
            "executed_tools": [],
        }
    }
    if isinstance(generated_file, dict):
        generated_planner, generated_tools = _extract_generated_artifacts(generated_file)
        generated_trace = normalize_trace_steps(generated_file.get("trace_steps"))
        first_refinement_payload = refinement_payloads.get(1) or {}
        stages["generated"] = {
            "name": "generated",
            "available": bool(generated_trace),
            "is_terminal": terminal_stage == "generated",
            "trace_steps": normalize_trace_steps(meta.get("final_trace"))
            if terminal_stage == "generated" and normalize_trace_steps(meta.get("final_trace"))
            else generated_trace,
            "stage_local_answer": first_nonempty(
                meta.get("final_answer") if terminal_stage == "generated" else None,
                generated_file.get("answer"),
            ),
            "stored_verifier_raw": (
                final_verifier_raw if terminal_stage == "generated" else first_refinement_payload.get("verifier_raw")
            ),
            "stored_verifier_output": (
                final_verifier_output
                if terminal_stage == "generated"
                else first_refinement_payload.get("verifier_output")
            ),
            "planner_output": generated_planner,
            "executed_tools": generated_tools,
        }

    refinement_numbers = sorted(refinement_payloads)

    for number in refinement_numbers:
        stage_name = f"ref{number}"
        current_payload = refinement_payloads[number]
        refiner_output = current_payload.get("refiner_output") or {}
        next_payload = refinement_payloads.get(number + 1) or {}
        trace_steps = normalize_trace_steps(refiner_output.get("refined_trace"))
        answer_text = first_nonempty(refiner_output.get("refined_answer"))

        if stage_name == terminal_stage:
            final_trace_steps = normalize_trace_steps(meta.get("final_trace"))
            if final_trace_steps:
                trace_steps = final_trace_steps
            answer_text = first_nonempty(meta.get("final_answer"), answer_text)

        stages[stage_name] = {
            "name": stage_name,
            "available": bool(trace_steps),
            "is_terminal": stage_name == terminal_stage,
            "trace_steps": trace_steps,
            "stage_local_answer": answer_text,
            "stored_verifier_raw": (
                final_verifier_raw if stage_name == terminal_stage else next_payload.get("verifier_raw")
            ),
            "stored_verifier_output": (
                final_verifier_output
                if stage_name == terminal_stage
                else next_payload.get("verifier_output")
            ),
            "planner_output": safe_json_obj(current_payload.get("planner_output"))
            or safe_json_obj(current_payload.get("planner_raw")),
            "executed_tools": current_payload.get("executed_tools")
            if isinstance(current_payload.get("executed_tools"), list)
            else [],
        }

    sample = {
        "sample_id": _make_sample_id(
            meta.get("video_path"),
            meta.get("question"),
            question_id=meta.get("question_id"),
            source_dir=str(video_dir.resolve()),
        ),
        "source_dir": str(video_dir.resolve()),
        "video_path": normalize_whitespace(meta.get("video_path")),
        "question": normalize_whitespace(meta.get("question")),
        "question_id": meta.get("question_id"),
        "options": normalize_options(meta.get("options")),
        "gold_answer": first_nonempty(meta.get("answer")),
        "final_answer": first_nonempty(meta.get("final_answer")),
        "terminal_stage": terminal_stage,
        "metadata": dict(meta),
        "stages": {},
    }

    for stage_name in sorted_stage_names(stages.keys()):
        sample["stages"][stage_name] = _coerce_stage(
            stage_name,
            stages[stage_name],
            terminal_stage=terminal_stage,
            sample_metadata=sample["metadata"],
        )

    return sample


def _coerce_normalized_sample(row):
    terminal_stage = first_nonempty(row.get("terminal_stage"))
    stages = {}
    for stage_name, payload in (row.get("stages") or {}).items():
        stages[stage_name] = _coerce_stage(
            stage_name,
            payload,
            terminal_stage=terminal_stage,
            sample_metadata=dict(row),
        )

    if not terminal_stage:
        terminal_stage = next(
            (stage_name for stage_name, payload in stages.items() if payload.get("is_terminal")),
            None,
        )
        if not terminal_stage and stages:
            terminal_stage = sorted_stage_names(stages.keys())[-1]

    sample = {
        "sample_id": first_nonempty(
            row.get("sample_id"),
            _make_sample_id(
                row.get("video_path"),
                row.get("question"),
                question_id=row.get("question_id"),
                source_dir=row.get("source_dir"),
            ),
        ),
        "source_dir": first_nonempty(row.get("source_dir")),
        "video_path": normalize_whitespace(row.get("video_path")),
        "question": normalize_whitespace(row.get("question")),
        "question_id": row.get("question_id"),
        "options": normalize_options(row.get("options")),
        "gold_answer": first_nonempty(row.get("gold_answer"), row.get("answer")),
        "final_answer": first_nonempty(row.get("final_answer")),
        "terminal_stage": terminal_stage,
        "metadata": dict(row),
        "stages": stages,
    }
    return sample


def _extract_generic_stage_local_answer(row):
    return first_nonempty(
        row.get("stage_local_answer"),
        row.get("proposed_answer"),
        row.get("final_answer"),
        row.get("predicted_answer"),
        row.get("initial_answer"),
    )


def _extract_generic_trace(row):
    for key in ("trace_steps", "trace", "final_trace", "reasoning_steps", "initial_trace_steps"):
        steps = normalize_trace_steps(row.get(key))
        if steps:
            return steps
    return []


def _coerce_generic_row(row, generic_stage="initial", source_file=None):
    if isinstance(row, dict) and isinstance(row.get("stages"), dict):
        sample = _coerce_normalized_sample(row)
        if source_file and not sample.get("source_dir"):
            sample["source_dir"] = str(source_file)
            sample["sample_id"] = _make_sample_id(
                sample.get("video_path"),
                sample.get("question"),
                question_id=sample.get("question_id"),
                source_dir=sample.get("source_dir"),
            )
        return sample

    stage_name = first_nonempty(row.get("stage"), generic_stage) or "initial"
    stage_payload = {
        "available": bool(_extract_generic_trace(row)),
        "trace_steps": _extract_generic_trace(row),
        "stage_local_answer": _extract_generic_stage_local_answer(row),
        "stored_verifier_raw": row.get("verifier_raw"),
        "stored_verifier_output": row.get("verifier_output"),
        "planner_output": safe_json_obj(row.get("planner_output")) or safe_json_obj(row.get("planner_raw")),
        "executed_tools": row.get("executed_tools") if isinstance(row.get("executed_tools"), list) else [],
    }
    sample = {
        "sample_id": first_nonempty(
            row.get("sample_id"),
            _make_sample_id(
                row.get("video_path"),
                row.get("question"),
                question_id=row.get("question_id"),
                source_dir=row.get("source_dir"),
            ),
        ),
        "source_dir": first_nonempty(row.get("source_dir"), str(source_file) if source_file else None),
        "video_path": normalize_whitespace(row.get("video_path") or row.get("video")),
        "question": normalize_whitespace(row.get("question")),
        "question_id": row.get("question_id"),
        "options": normalize_options(row.get("options")),
        "gold_answer": first_nonempty(
            row.get("gold_answer"),
            row.get("answer"),
            row.get("correct_answer"),
            row.get("ground_truth"),
        ),
        "final_answer": first_nonempty(row.get("final_answer")),
        "terminal_stage": first_nonempty(row.get("terminal_stage"), stage_name),
        "metadata": dict(row),
        "stages": {
            stage_name: _coerce_stage(
                stage_name,
                stage_payload,
                terminal_stage=first_nonempty(row.get("terminal_stage"), stage_name),
                sample_metadata=dict(row),
            )
        },
    }
    return sample


def _looks_like_results_root(path: Path):
    if (path / "meta.json").is_file():
        return True
    for child in path.iterdir():
        if child.is_dir() and (child / "meta.json").is_file():
            return True
    return False


def _iter_results_dirs(path: Path):
    if (path / "meta.json").is_file():
        return [path]
    return sorted(
        [
            child for child in path.iterdir()
            if child.is_dir() and (child / "meta.json").is_file()
        ],
        key=lambda item: item.name,
    )


def load_stage_samples(input_path, generic_stage="initial", max_samples=None):
    input_path = Path(input_path).expanduser().resolve()
    samples = []

    if input_path.is_dir() and _looks_like_results_root(input_path):
        for video_dir in _iter_results_dirs(input_path):
            samples.append(_load_results_sample(video_dir))
            if max_samples is not None and len(samples) >= max_samples:
                break
        return samples

    if input_path.suffix == ".jsonl":
        rows = _load_jsonl_rows(input_path)
        for row in rows[:max_samples] if max_samples is not None else rows:
            samples.append(_coerce_generic_row(row, generic_stage=generic_stage, source_file=input_path))
        return samples

    if input_path.suffix == ".json":
        rows = _load_json_rows(input_path)
        for row in rows[:max_samples] if max_samples is not None else rows:
            samples.append(_coerce_generic_row(row, generic_stage=generic_stage, source_file=input_path))
        return samples

    raise ValueError(
        f"Unsupported input path {input_path}. Expected a results directory, JSON file, or JSONL file."
    )
