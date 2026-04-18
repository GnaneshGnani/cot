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

_NUM = (int, float)
_STR = (str,)
_INT = (int,)

TOOL_ARG_SCHEMAS = {
    "temporal_grounder": {"required": ["query"], "optional_types": {"query": _STR, "start_time": _NUM, "end_time": _NUM, "top_k": _INT}},
    "frame_retriever": {"required": ["query"], "optional_types": {"query": _STR, "start_time": _NUM, "end_time": _NUM, "top_k": _INT}},
    "asr": {"required": [], "optional_types": {"start_time": _NUM, "end_time": _NUM, "language": _STR}},
    "audio_grounder": {"required": ["query"], "optional_types": {"query": _STR, "start_time": _NUM, "end_time": _NUM}},
    "ocr": {"required": [], "optional_types": {"start_time": _NUM, "end_time": _NUM, "query": _STR}},
    "spatial_grounder": {"required": ["query"], "optional_types": {"query": _STR, "start_time": _NUM, "end_time": _NUM}},
    "counter": {"required": ["query"], "optional_types": {"query": _STR, "start_time": _NUM, "end_time": _NUM}},
    "dense_captioner": {"required": [], "optional_types": {"start_time": _NUM, "end_time": _NUM, "query": _STR}},
    "action_recognizer": {"required": [], "optional_types": {"query": _STR, "start_time": _NUM, "end_time": _NUM}},
    "chart_analyzer": {"required": [], "optional_types": {"query": _STR, "start_time": _NUM, "end_time": _NUM}},
}


def _validate_arguments_for_tool(tool_name, arguments):
    reasons = []
    if not isinstance(arguments, dict):
        return ["arguments_not_object"]

    schema = TOOL_ARG_SCHEMAS.get(tool_name)
    if not schema:
        return reasons

    for key in schema.get("required", []):
        if key not in arguments:
            reasons.append(f"missing_required_arg:{key}")
        elif arguments[key] is None:
            reasons.append(f"null_required_arg:{key}")

    for key, type_tuple in (schema.get("optional_types") or {}).items():
        if key not in arguments or arguments[key] is None:
            continue
        if not isinstance(arguments[key], type_tuple):
            reasons.append(f"bad_type:{key}")
    return reasons


def _parse_tool_output(output_value):
    if output_value is None:
        return None
    if isinstance(output_value, dict):
        return output_value
    if not isinstance(output_value, str):
        return None
    text = output_value.strip()
    if not text:
        return None
    if "are:\n" in output_value:
        text = output_value.split("are:\n", 1)[-1].strip()
    if not (text.startswith("{") or text.startswith("[")):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _validate_parsed_output(tool_name, parsed):
    reasons = []
    if parsed is None:
        return ["unparseable_output"]

    if tool_name == "action_recognizer":
        if isinstance(parsed, list):
            return reasons if parsed else ["empty_action_list"]
        if isinstance(parsed, dict):
            actions = parsed.get("actions")
            if isinstance(actions, list) and actions:
                return reasons
            return ["missing_actions"]
        return ["bad_action_recognizer_shape"]

    if not isinstance(parsed, dict):
        return ["output_not_object"]

    if tool_name == "temporal_grounder":
        if not isinstance(parsed.get("segments"), list):
            reasons.append("missing_or_bad_segments")
    elif tool_name == "frame_retriever":
        if not isinstance(parsed.get("frames"), list):
            reasons.append("missing_or_bad_frames")
    elif tool_name == "asr":
        transcript = parsed.get("full_transcript") or parsed.get("transcript") or ""
        if not (isinstance(transcript, str) and transcript.strip()):
            reasons.append("missing_transcript")
    elif tool_name == "ocr":
        if not isinstance(parsed.get("detections"), list):
            reasons.append("missing_or_bad_detections")
    elif tool_name == "spatial_grounder":
        has_detections = isinstance(parsed.get("detections"), list) and parsed.get("detections")
        has_frames = isinstance(parsed.get("frames"), list) and parsed.get("frames")
        if not has_detections and not has_frames:
            reasons.append("missing_detections_and_frames")
    elif tool_name == "counter":
        has_count = parsed.get("count") is not None
        has_frames = isinstance(parsed.get("frames"), list) and parsed.get("frames")
        if not has_count and not has_frames:
            reasons.append("missing_count_and_frames")
    elif tool_name == "dense_captioner":
        if not isinstance(parsed.get("captions"), list):
            reasons.append("missing_or_bad_captions")
    elif tool_name == "audio_grounder":
        events = parsed.get("events") or []
        groups = parsed.get("distinct_event_groups") or []
        if not events and not groups:
            reasons.append("missing_events_and_groups")
    elif tool_name == "chart_analyzer":
        chart_type = str(parsed.get("chart_type", "") or "").strip().lower()
        if chart_type in ("", "unknown", "other") and not parsed.get("series"):
            reasons.append("weak_chart_structure")
    return reasons


def _validate_tool_call(tool_call):
    reasons = []
    if not isinstance(tool_call, dict):
        return ["tool_call_not_object"]

    step = tool_call.get("step")
    if step is not None and not (isinstance(step, int) and step > 0):
        reasons.append("invalid_step")

    tool_name = tool_call.get("tool")
    if not isinstance(tool_name, str) or not tool_name.strip():
        reasons.append("missing_tool")
    elif tool_name not in ALLOWED_TOOLS:
        reasons.append("unknown_tool")

    arguments = tool_call.get("arguments")
    if not isinstance(arguments, dict):
        reasons.append("arguments_not_object")
    elif isinstance(tool_name, str) and tool_name in ALLOWED_TOOLS:
        reasons.extend(_validate_arguments_for_tool(tool_name, arguments))

    depends_on = tool_call.get("depends_on")
    if depends_on is not None:
        if not isinstance(depends_on, list) or any((not isinstance(item, int) or item < 0) for item in depends_on):
            reasons.append("invalid_depends_on")
    return reasons


def compute_stage_metric(stage_record):
    if not stage_record.get("available"):
        return {
            "applicable": False,
            "tool_validity_score": None,
            "tool_combined_validity_score": None,
            "tool_input_validity_score": None,
            "total_tool_calls": 0,
            "valid_tool_calls": 0,
            "invalid_tool_calls": 0,
            "planner_source": None,
            "invalid_details": [],
            "tool_output_validity_score": None,
            "executed_source": None,
            "total_executed_calls": 0,
            "valid_output_calls": 0,
            "output_invalid_details": [],
            "skipped_reason": "stage_unavailable",
        }

    planner_obj = safe_json_obj(stage_record.get("planner_output"))
    executed_tools = stage_record.get("executed_tools") if isinstance(stage_record.get("executed_tools"), list) else []

    tool_calls = planner_obj.get("tool_calls") if isinstance(planner_obj, dict) else []
    tool_calls = tool_calls if isinstance(tool_calls, list) else []

    if not tool_calls and not executed_tools:
        return {
            "applicable": False,
            "tool_validity_score": None,
            "tool_combined_validity_score": None,
            "tool_input_validity_score": None,
            "total_tool_calls": 0,
            "valid_tool_calls": 0,
            "invalid_tool_calls": 0,
            "planner_source": None,
            "invalid_details": [],
            "tool_output_validity_score": None,
            "executed_source": None,
            "total_executed_calls": 0,
            "valid_output_calls": 0,
            "output_invalid_details": [],
            "skipped_reason": "missing_planner_and_execution_artifacts",
        }

    invalid_details = []
    valid_tool_calls = 0
    for index, call in enumerate(tool_calls):
        reasons = _validate_tool_call(call)
        if reasons:
            invalid_details.append(
                {
                    "index": index,
                    "tool": call.get("tool") if isinstance(call, dict) else None,
                    "reasons": reasons,
                }
            )
        else:
            valid_tool_calls += 1
    tool_input_validity_score = (valid_tool_calls / len(tool_calls)) if tool_calls else None

    output_invalid_details = []
    valid_output_calls = 0
    for index, call in enumerate(executed_tools):
        if not isinstance(call, dict):
            output_invalid_details.append({"index": index, "tool": None, "reasons": ["call_not_object"]})
            continue
        tool_name = call.get("tool")
        if not isinstance(tool_name, str) or tool_name not in ALLOWED_TOOLS:
            output_invalid_details.append({"index": index, "tool": tool_name, "reasons": ["unknown_or_missing_tool"]})
            continue
        parsed = _parse_tool_output(call.get("output"))
        reasons = _validate_parsed_output(tool_name, parsed)
        if reasons:
            output_invalid_details.append({"index": index, "tool": tool_name, "reasons": reasons})
        else:
            valid_output_calls += 1
    tool_output_validity_score = (valid_output_calls / len(executed_tools)) if executed_tools else None

    combined = tool_input_validity_score
    if tool_input_validity_score is not None and tool_output_validity_score is not None:
        combined = round((tool_input_validity_score + tool_output_validity_score) / 2.0, 4)
    elif combined is not None:
        combined = round(combined, 4)

    return {
        "applicable": True,
        "tool_validity_score": round(tool_input_validity_score, 4) if tool_input_validity_score is not None else None,
        "tool_combined_validity_score": combined,
        "tool_input_validity_score": round(tool_input_validity_score, 4) if tool_input_validity_score is not None else None,
        "total_tool_calls": len(tool_calls),
        "valid_tool_calls": valid_tool_calls,
        "invalid_tool_calls": len(tool_calls) - valid_tool_calls,
        "planner_source": "stage" if planner_obj is not None else None,
        "invalid_details": invalid_details[:10],
        "tool_output_validity_score": round(tool_output_validity_score, 4) if tool_output_validity_score is not None else None,
        "executed_source": "stage" if executed_tools else None,
        "total_executed_calls": len(executed_tools),
        "valid_output_calls": valid_output_calls,
        "output_invalid_details": output_invalid_details[:10],
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

    parser = argparse.ArgumentParser(description="Compute tool validity over normalized stages.")
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
    output_path = Path(args.output) if args.output else results_dir_default_output(input_path, "tool_validity.jsonl")
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
        sum(row.get("tool_combined_validity_score", 0.0) for row in applicable_rows if row.get("tool_combined_validity_score") is not None) / len(applicable_rows)
        if applicable_rows else 0.0
    )
    summary = {
        "_summary": True,
        "metric": "tool_validity",
        "input_path": str(input_path),
        "total_stage_rows": len(rows),
        "applicable_stage_rows": len(applicable_rows),
        "average_tool_combined_validity_score": round(avg_score, 4),
    }
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"Done. Output: {output_path}")
    print(
        f"Overall: avg_tool_combined_validity_score={summary['average_tool_combined_validity_score']:.4f}, "
        f"applicable={summary['applicable_stage_rows']}, rows={summary['total_stage_rows']}"
    )


if __name__ == "__main__":
    main()
