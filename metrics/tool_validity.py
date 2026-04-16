import json
import os
from pathlib import Path

from experiment_utils import apply_experiment, get_experiment


ALLOWED_TOOLS = {
    "temporal_grounder",
    "frame_retriever",
    "asr",
    "audio_grounder",
    "ocr",
    "spatial_grounder",
    "counter",
    "dense_captioner",
    "action_recognizer",
    "chart_analyzer",
}


def load_data(path, max_samples=None):
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = []
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
                if max_samples is not None and len(rows) >= max_samples:
                    break
        return rows

    with path.open() as f:
        data = json.load(f)

    if isinstance(data, list):
        return data[:max_samples] if max_samples is not None else data
    return [data]


def _safe_json_obj(raw):
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        txt = raw.strip()
        if not txt:
            return None
        try:
            parsed = json.loads(txt)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def _extract_planner_candidates(sample):
    candidates = []

    for key in ("planner_output", "planner", "plan", "planner_raw", "plan_raw"):
        obj = _safe_json_obj(sample.get(key))
        if obj is not None:
            candidates.append((key, obj))

    for list_key in ("all_iterations", "iteration_history"):
        items = sample.get(list_key)
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            for key in ("planner_output", "planner_raw"):
                obj = _safe_json_obj(item.get(key))
                if obj is not None:
                    candidates.append((f"{list_key}[{idx}].{key}", obj))

    return candidates


def _choose_best_planner(candidates):
    best_source = None
    best_obj = None
    best_count = -1
    for source, obj in candidates:
        tool_calls = obj.get("tool_calls") if isinstance(obj, dict) else None
        count = len(tool_calls) if isinstance(tool_calls, list) else 0
        if count > best_count:
            best_count = count
            best_source = source
            best_obj = obj
    return best_source, best_obj


def _validate_tool_call(tool_call):
    reasons = []
    if not isinstance(tool_call, dict):
        return ["tool_call_not_object"]

    step = tool_call.get("step")
    if step is not None and not (isinstance(step, int) and step > 0):
        reasons.append("invalid_step")

    tool = tool_call.get("tool")
    if not isinstance(tool, str) or not tool.strip():
        reasons.append("missing_tool")
    elif tool not in ALLOWED_TOOLS:
        reasons.append("unknown_tool")

    arguments = tool_call.get("arguments")
    if not isinstance(arguments, dict):
        reasons.append("arguments_not_object")

    depends_on = tool_call.get("depends_on")
    if depends_on is not None:
        if not isinstance(depends_on, list) or any((not isinstance(x, int) or x < 0) for x in depends_on):
            reasons.append("invalid_depends_on")

    return reasons


def compute_metric(sample):
    candidates = _extract_planner_candidates(sample)
    planner_source, planner_obj = _choose_best_planner(candidates)

    if not isinstance(planner_obj, dict):
        return {
            "tool_validity_score": 0.0,
            "total_tool_calls": 0,
            "valid_tool_calls": 0,
            "invalid_tool_calls": 0,
            "planner_source": None,
            "invalid_details": [],
        }

    tool_calls = planner_obj.get("tool_calls")
    if not isinstance(tool_calls, list):
        tool_calls = []

    invalid_details = []
    valid_count = 0
    for i, call in enumerate(tool_calls):
        reasons = _validate_tool_call(call)
        if reasons:
            invalid_details.append(
                {
                    "index": i,
                    "tool": call.get("tool") if isinstance(call, dict) else None,
                    "reasons": reasons,
                }
            )
        else:
            valid_count += 1

    total_calls = len(tool_calls)
    invalid_count = total_calls - valid_count
    score = (valid_count / total_calls) if total_calls else 0.0

    return {
        "tool_validity_score": round(score, 4),
        "total_tool_calls": total_calls,
        "valid_tool_calls": valid_count,
        "invalid_tool_calls": invalid_count,
        "planner_source": planner_source,
        "invalid_details": invalid_details[:10],
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = "/fs/nexus-scratch/gnanesh/cot"

    input_file = os.environ.get("DATA_PATH") or os.path.join(project_root, "OmniVideoBench", "data_short_under1min.jsonl")
    experiment = get_experiment()
    output_file = os.path.join(script_dir, "results", experiment, "tool_validity.jsonl")

    max_samples = None
    if os.environ.get("MAX_SAMPLES"):
        try:
            max_samples = int(os.environ["MAX_SAMPLES"])
        except ValueError:
            pass

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    samples = load_data(input_file, max_samples=max_samples)

    rows = []
    for i, sample in enumerate(samples):
        sample = apply_experiment(sample, experiment)
        metric = compute_metric(sample)
        row = {
            "video": sample.get("video", ""),
            "question": sample.get("question", ""),
            **metric,
        }
        rows.append(row)

        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1} samples")

    n = len(rows)
    avg_score = sum(r.get("tool_validity_score", 0.0) for r in rows) / n if n else 0.0
    with_calls = sum(1 for r in rows if r.get("total_tool_calls", 0) > 0)
    fully_valid = sum(1 for r in rows if r.get("total_tool_calls", 0) > 0 and r.get("invalid_tool_calls", 0) == 0)

    summary = {
        "_summary": True,
        "metric": "Tool Validity",
        "total_samples": n,
        "samples_with_tool_calls": with_calls,
        "average_tool_validity_score": round(avg_score, 4),
        "fraction_fully_valid_when_present": round((fully_valid / with_calls), 4) if with_calls else 0.0,
    }

    with open(output_file, "w") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"Done. Output: {output_file}")
    print(
        "Overall: "
        f"avg_tool_validity_score={summary['average_tool_validity_score']:.4f}, "
        f"samples_with_tool_calls={with_calls}, n={n}"
    )


if __name__ == "__main__":
    main()
