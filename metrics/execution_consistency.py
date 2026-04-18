import json
import os
from pathlib import Path

from stage_loader import load_stage_samples
from stage_metrics_common import (
    build_stage_record,
    parse_answer_source_overrides,
    parse_csv_arg,
    results_dir_default_output,
    safe_json_obj,
    sorted_stage_names,
)


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
    for value in depends:
        if not isinstance(value, int) or value < 0:
            return None
        out.append(value)
    return sorted(set(out))


def _nonempty_output(value):
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) > 0
    return True


def _index_calls_by_step(calls):
    indexed = {}
    for call in calls:
        if not isinstance(call, dict):
            continue
        step = _normalize_step(call.get("step"))
        if step is None:
            continue
        indexed.setdefault(step, call)
    return indexed


def _args_match(planned_call, executed_call):
    p_args = planned_call.get("arguments")
    e_args = executed_call.get("arguments")
    if not isinstance(p_args, dict) or not isinstance(e_args, dict):
        return False
    for key, value in p_args.items():
        if key not in e_args or e_args[key] != value:
            return False
    return True


def _depends_equal(planned_call, executed_call):
    planned_depends = _normalize_depends(planned_call.get("depends_on"))
    executed_depends = _normalize_depends(executed_call.get("depends_on"))
    if planned_depends is None or executed_depends is None:
        return False
    return planned_depends == executed_depends


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
            if dep not in steps or dep >= step:
                ok = False
                break
        if ok:
            valid += 1
    return valid / total if total else 0.0


def compute_stage_metric(stage_record):
    if not stage_record.get("available"):
        return {
            "applicable": False,
            "execution_consistency_score": None,
            "rho_exec": None,
            "artifact_source": None,
            "planned_tool_calls": 0,
            "executed_tool_calls": 0,
            "matched_steps": 0,
            "missing_execution_steps": [],
            "extra_execution_steps": [],
            "step_coverage": None,
            "tool_name_match_rate": None,
            "arguments_match_rate": None,
            "depends_on_match_rate": None,
            "execution_dependency_validity": None,
            "output_presence_rate": None,
            "skipped_reason": "stage_unavailable",
        }

    planner = safe_json_obj(stage_record.get("planner_output"))
    executed = stage_record.get("executed_tools") if isinstance(stage_record.get("executed_tools"), list) else []
    planned_calls = planner.get("tool_calls") if isinstance(planner, dict) else []
    planned_calls = planned_calls if isinstance(planned_calls, list) else []

    if not planned_calls and not executed:
        return {
            "applicable": False,
            "execution_consistency_score": None,
            "rho_exec": None,
            "artifact_source": None,
            "planned_tool_calls": 0,
            "executed_tool_calls": 0,
            "matched_steps": 0,
            "missing_execution_steps": [],
            "extra_execution_steps": [],
            "step_coverage": None,
            "tool_name_match_rate": None,
            "arguments_match_rate": None,
            "depends_on_match_rate": None,
            "execution_dependency_validity": None,
            "output_presence_rate": None,
            "skipped_reason": "missing_planner_and_execution_artifacts",
        }

    planned_by_step = _index_calls_by_step(planned_calls)
    executed_by_step = _index_calls_by_step(executed)

    planned_steps = set(planned_by_step.keys())
    executed_steps = set(executed_by_step.keys())
    common_steps = sorted(planned_steps & executed_steps)

    coverage = len(common_steps) / len(planned_steps) if planned_steps else 0.0

    tool_match = 0
    args_match = 0
    dep_match = 0
    for step in common_steps:
        planned_call = planned_by_step[step]
        executed_call = executed_by_step[step]
        if str(planned_call.get("tool") or "") == str(executed_call.get("tool") or ""):
            tool_match += 1
        if _args_match(planned_call, executed_call):
            args_match += 1
        if _depends_equal(planned_call, executed_call):
            dep_match += 1

    common_n = len(common_steps)
    tool_rate = tool_match / common_n if common_n else 0.0
    args_rate = args_match / common_n if common_n else 0.0
    dep_rate = dep_match / common_n if common_n else 0.0
    dep_validity = _execution_dep_validity(executed_by_step)

    outputs_present = sum(1 for call in executed_by_step.values() if _nonempty_output(call.get("output")))
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
        "applicable": True,
        "execution_consistency_score": round(score, 4),
        "rho_exec": round(score, 4),
        "artifact_source": "stage",
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
        "skipped_reason": None,
    }


def collect_metric_rows(samples, stage_filter=None, answer_source_overrides=None):
    stage_filter = set(parse_csv_arg(stage_filter)) if stage_filter else None
    rows = []

    for sample in samples:
        for stage_name in sorted_stage_names(sample.get("stages", {}).keys()):
            if stage_filter and stage_name not in stage_filter:
                continue
            stage_record = build_stage_record(
                sample,
                stage_name,
                answer_source_overrides=answer_source_overrides,
            )
            metric = compute_stage_metric(stage_record)
            rows.append(
                {
                    "sample_id": stage_record["sample_id"],
                    "source_dir": stage_record["source_dir"],
                    "video_path": stage_record["video_path"],
                    "question": stage_record["question"],
                    "question_id": stage_record["question_id"],
                    "stage": stage_name,
                    "is_terminal": stage_record["is_terminal"],
                    **metric,
                }
            )
    return rows


def main():
    default_input = os.environ.get("DATA_PATH")

    import argparse

    parser = argparse.ArgumentParser(description="Compute execution consistency over normalized stages.")
    parser.add_argument("input_path", nargs="?", default=default_input)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--stages", type=str, default=os.environ.get("STAGES"))
    parser.add_argument("--generic-stage", type=str, default=os.environ.get("GENERIC_STAGE", "initial"))
    parser.add_argument("--answer-sources", type=str, default=os.environ.get("ANSWER_SOURCE_OVERRIDES"))
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    if not args.input_path:
        raise SystemExit("Set DATA_PATH or pass an input path explicitly.")

    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output) if args.output else results_dir_default_output(input_path, "execution_consistency.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    answer_source_overrides = parse_answer_source_overrides(args.answer_sources)
    samples = load_stage_samples(input_path, generic_stage=args.generic_stage, max_samples=args.max_samples)
    rows = collect_metric_rows(
        samples,
        stage_filter=args.stages,
        answer_source_overrides=answer_source_overrides,
    )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    applicable_rows = [row for row in rows if row.get("applicable")]
    avg_score = (
        sum(row.get("execution_consistency_score", 0.0) for row in applicable_rows) / len(applicable_rows)
        if applicable_rows else 0.0
    )
    summary = {
        "_summary": True,
        "metric": "execution_consistency",
        "input_path": str(input_path),
        "total_stage_rows": len(rows),
        "applicable_stage_rows": len(applicable_rows),
        "average_execution_consistency_score": round(avg_score, 4),
    }
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"Done. Output: {output_path}")
    print(
        f"Overall: avg_execution_consistency_score={summary['average_execution_consistency_score']:.4f}, "
        f"applicable={summary['applicable_stage_rows']}, rows={summary['total_stage_rows']}"
    )


if __name__ == "__main__":
    main()
