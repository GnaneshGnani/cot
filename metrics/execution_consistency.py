import json
import os
from pathlib import Path

from experiment_utils import apply_experiment, get_experiment


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


def _normalize_step(value):
    if isinstance(value, int) and value > 0:
        return value
    return None


def _normalize_depends(depends):
    if depends is None:
        return []
    if not isinstance(depends, list):
        return None
    out = []
    for v in depends:
        if not isinstance(v, int) or v < 0:
            return None
        out.append(v)
    return sorted(set(out))


def _nonempty_output(value):
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) > 0
    return True


def _extract_candidates(sample):
    candidates = []

    direct_planner = None
    for key in ("planner_output", "planner", "plan", "planner_raw", "plan_raw"):
        direct_planner = _safe_json_obj(sample.get(key))
        if direct_planner is not None:
            break

    direct_executed = sample.get("executed_tools") if isinstance(sample.get("executed_tools"), list) else None
    if direct_planner is not None or direct_executed is not None:
        candidates.append(("sample", direct_planner, direct_executed or []))

    for list_key in ("all_iterations", "iteration_history"):
        items = sample.get(list_key)
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue

            planner = None
            for key in ("planner_output", "planner_raw"):
                planner = _safe_json_obj(item.get(key))
                if planner is not None:
                    break

            executed = item.get("executed_tools")
            executed = executed if isinstance(executed, list) else []

            if planner is not None or executed:
                candidates.append((f"{list_key}[{idx}]", planner, executed))

    return candidates


def _choose_best_candidate(candidates):
    if not candidates:
        return None, None, []

    best = None
    best_key = (-1, -1)
    for source, planner, executed in candidates:
        planned_len = len(planner.get("tool_calls") or []) if isinstance(planner, dict) else 0
        executed_len = len(executed) if isinstance(executed, list) else 0
        key = (planned_len, executed_len)
        if key > best_key:
            best_key = key
            best = (source, planner, executed)

    return best if best is not None else (None, None, [])


def _index_calls_by_step(calls):
    indexed = {}
    for call in calls:
        if not isinstance(call, dict):
            continue
        step = _normalize_step(call.get("step"))
        if step is None:
            continue
        if step not in indexed:
            indexed[step] = call
    return indexed


def _args_match(planned_call, executed_call):
    p_args = planned_call.get("arguments")
    e_args = executed_call.get("arguments")

    if not isinstance(p_args, dict) or not isinstance(e_args, dict):
        return False

    for key, val in p_args.items():
        if key not in e_args:
            return False
        if e_args[key] != val:
            return False
    return True


def _depends_equal(planned_call, executed_call):
    p = _normalize_depends(planned_call.get("depends_on"))
    e = _normalize_depends(executed_call.get("depends_on"))
    if p is None or e is None:
        return False
    return p == e


def _execution_dep_validity(executed_by_step):
    if not executed_by_step:
        return 0.0

    valid = 0
    total = 0
    steps = set(executed_by_step.keys())
    for step, call in executed_by_step.items():
        deps = _normalize_depends(call.get("depends_on"))
        total += 1
        if deps is None:
            continue
        ok = True
        for dep in deps:
            if dep not in steps:
                ok = False
                break
            if dep >= step:
                ok = False
                break
        if ok:
            valid += 1
    return valid / total if total else 0.0


def compute_metric(sample):
    source, planner, executed = _choose_best_candidate(_extract_candidates(sample))

    planned_calls = planner.get("tool_calls") if isinstance(planner, dict) else []
    planned_calls = planned_calls if isinstance(planned_calls, list) else []
    executed_calls = executed if isinstance(executed, list) else []

    planned_by_step = _index_calls_by_step(planned_calls)
    executed_by_step = _index_calls_by_step(executed_calls)

    planned_steps = set(planned_by_step.keys())
    executed_steps = set(executed_by_step.keys())
    common_steps = sorted(planned_steps & executed_steps)

    if not planned_steps and not executed_steps:
        return {
            "execution_consistency_score": 0.0,
            "rho_exec": 0.0,
            "artifact_source": source,
            "planned_tool_calls": 0,
            "executed_tool_calls": 0,
            "matched_steps": 0,
            "missing_execution_steps": [],
            "extra_execution_steps": [],
            "step_coverage": 0.0,
            "tool_name_match_rate": 0.0,
            "arguments_match_rate": 0.0,
            "depends_on_match_rate": 0.0,
            "execution_dependency_validity": 0.0,
            "output_presence_rate": 0.0,
        }

    coverage = len(common_steps) / len(planned_steps) if planned_steps else 0.0

    tool_match = 0
    args_match = 0
    dep_match = 0
    for step in common_steps:
        p = planned_by_step[step]
        e = executed_by_step[step]

        if str(p.get("tool") or "") == str(e.get("tool") or ""):
            tool_match += 1
        if _args_match(p, e):
            args_match += 1
        if _depends_equal(p, e):
            dep_match += 1

    common_n = len(common_steps)
    tool_rate = tool_match / common_n if common_n else 0.0
    args_rate = args_match / common_n if common_n else 0.0
    dep_rate = dep_match / common_n if common_n else 0.0

    dep_validity = _execution_dep_validity(executed_by_step)

    outputs_present = 0
    for call in executed_by_step.values():
        if _nonempty_output(call.get("output")):
            outputs_present += 1
    output_rate = outputs_present / len(executed_by_step) if executed_by_step else 0.0

    score = (
        0.45 * coverage
        + 0.20 * tool_rate
        + 0.15 * args_rate
        + 0.10 * dep_rate
        + 0.05 * dep_validity
        + 0.05 * output_rate
    )

    return {
        "execution_consistency_score": round(score, 4),
        "rho_exec": round(score, 4),
        "artifact_source": source,
        "planned_tool_calls": len(planned_steps),
        "executed_tool_calls": len(executed_steps),
        "matched_steps": common_n,
        "missing_execution_steps": sorted(planned_steps - executed_steps)[:20],
        "extra_execution_steps": sorted(executed_steps - planned_steps)[:20],
        "step_coverage": round(coverage, 4),
        "tool_name_match_rate": round(tool_rate, 4),
        "arguments_match_rate": round(args_rate, 4),
        "depends_on_match_rate": round(dep_rate, 4),
        "execution_dependency_validity": round(dep_validity, 4),
        "output_presence_rate": round(output_rate, 4),
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = "/fs/nexus-scratch/gnanesh/cot"

    input_file = os.environ.get("DATA_PATH") or os.path.join(project_root, "OmniVideoBench", "data_short_under1min.jsonl")
    experiment = get_experiment()
    output_file = os.path.join(script_dir, "results", experiment, "execution_consistency.jsonl")

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
    avg_score = sum(r.get("execution_consistency_score", 0.0) for r in rows) / n if n else 0.0
    with_exec = sum(1 for r in rows if r.get("executed_tool_calls", 0) > 0 or r.get("planned_tool_calls", 0) > 0)

    summary = {
        "_summary": True,
        "metric": "Execution Consistency",
        "total_samples": n,
        "samples_with_execution_artifacts": with_exec,
        "average_execution_consistency_score": round(avg_score, 4),
    }

    with open(output_file, "w") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"Done. Output: {output_file}")
    print(
        "Overall: "
        f"avg_execution_consistency_score={summary['average_execution_consistency_score']:.4f}, "
        f"samples_with_execution_artifacts={with_exec}, n={n}"
    )


if __name__ == "__main__":
    main()
