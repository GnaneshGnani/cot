import json
import os
from pathlib import Path

from stage_loader import load_stage_samples
from stage_metrics_common import (
    build_stage_record,
    parse_answer_source_overrides,
    parse_csv_arg,
    results_dir_default_output,
    sorted_stage_names,
)
from verifier_helper import normalize_trace_quality_scores, resolve_verifier


def trace_quality_to_scalar(trace_quality_scores):
    scores = normalize_trace_quality_scores(trace_quality_scores)
    if not isinstance(scores, dict):
        return None
    values = []
    for key in ("logical_coherence", "completeness", "factual_correctness", "reasoning_order"):
        value = scores.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value) / 10.0)
    if not values:
        return None
    return sum(values) / len(values)


def _quality_subscores(verifier_output):
    if not isinstance(verifier_output, dict):
        return None
    scores = normalize_trace_quality_scores(verifier_output.get("trace_quality_scores"))
    if not isinstance(scores, dict):
        return None
    return {
        "logical_coherence": scores.get("logical_coherence"),
        "completeness": scores.get("completeness"),
        "factual_correctness": scores.get("factual_correctness"),
        "reasoning_order": scores.get("reasoning_order"),
    }


def _evidence_gaps(verifier_output):
    if not isinstance(verifier_output, dict):
        return None
    gaps = verifier_output.get("evidence_gaps")
    return gaps if isinstance(gaps, list) else None


def compute_stage_metric(stage_record, verifier_mode="hybrid", client=None, model=None):
    if not stage_record.get("available"):
        return {
            "applicable": False,
            "verifier_quality_score": None,
            "verifier_mode_used": None,
            "verifier_raw": None,
            "verifier_output": None,
            "verdict": None,
            "confidence": None,
            "summary": None,
            "quality_scores": None,
            "answer_correct": None,
            "num_error_categories": None,
            "evidence_gaps": None,
            "num_evidence_gaps": None,
            "model": None,
            "skipped_reason": "stage_unavailable",
        }

    if not stage_record.get("trace_steps"):
        return {
            "applicable": False,
            "verifier_quality_score": None,
            "verifier_mode_used": None,
            "verifier_raw": None,
            "verifier_output": None,
            "verdict": None,
            "confidence": None,
            "summary": None,
            "quality_scores": None,
            "answer_correct": None,
            "num_error_categories": None,
            "evidence_gaps": None,
            "num_evidence_gaps": None,
            "model": None,
            "skipped_reason": "missing_trace",
        }

    verifier_result = resolve_verifier(
        stage_record,
        verifier_mode=verifier_mode,
        client=client,
        model=model,
    )
    verifier_output = verifier_result.get("verifier_output")
    quality_score = trace_quality_to_scalar(
        verifier_output.get("trace_quality_scores") if isinstance(verifier_output, dict) else None
    )

    applicable = isinstance(verifier_output, dict)
    return {
        "applicable": applicable,
        "verifier_quality_score": round(quality_score, 4) if quality_score is not None else None,
        "verifier_mode_used": verifier_result.get("verifier_mode_used"),
        "verifier_raw": verifier_result.get("verifier_raw"),
        "verifier_output": verifier_output,
        "verdict": verifier_output.get("verdict") if isinstance(verifier_output, dict) else None,
        "confidence": verifier_output.get("confidence") if isinstance(verifier_output, dict) else None,
        "summary": verifier_output.get("summary") if isinstance(verifier_output, dict) else None,
        "quality_scores": _quality_subscores(verifier_output),
        "answer_correct": verifier_output.get("answer_correct") if isinstance(verifier_output, dict) else None,
        "num_error_categories": len(verifier_output.get("error_categories") or []) if isinstance(verifier_output, dict) else None,
        "evidence_gaps": _evidence_gaps(verifier_output),
        "num_evidence_gaps": len(verifier_output.get("evidence_gaps") or []) if isinstance(verifier_output, dict) else None,
        "model": verifier_result.get("model"),
        "skipped_reason": None if applicable else "verifier_unavailable",
    }


def collect_metric_rows(samples, verifier_mode="hybrid", stage_filter=None, answer_source_overrides=None, client=None, model=None):
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
            metric = compute_stage_metric(
                stage_record,
                verifier_mode=verifier_mode,
                client=client,
                model=model,
            )
            rows.append(
                {
                    "sample_id": stage_record["sample_id"],
                    "source_dir": stage_record["source_dir"],
                    "video_path": stage_record["video_path"],
                    "question": stage_record["question"],
                    "question_id": stage_record["question_id"],
                    "stage": stage_name,
                    "is_terminal": stage_record["is_terminal"],
                    "proposed_answer_source": stage_record["proposed_answer_source"],
                    "proposed_answer": stage_record["proposed_answer"],
                    "gold_answer": stage_record["gold_answer"],
                    "is_correct": stage_record["is_correct"],
                    **metric,
                }
            )
    return rows


def main():
    default_input = os.environ.get("DATA_PATH")
    if not default_input:
        default_input = os.environ.get("RESULTS_ROOT")

    import argparse

    parser = argparse.ArgumentParser(description="Compute verifier metrics over normalized stages.")
    parser.add_argument("input_path", nargs="?", default=default_input)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--stages", type=str, default=os.environ.get("STAGES"))
    parser.add_argument("--generic-stage", type=str, default=os.environ.get("GENERIC_STAGE", "initial"))
    parser.add_argument("--answer-sources", type=str, default=os.environ.get("ANSWER_SOURCE_OVERRIDES"))
    parser.add_argument("--verifier-mode", type=str, default=os.environ.get("VERIFIER_MODE", "hybrid"))
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    if not args.input_path:
        raise SystemExit("Set DATA_PATH/RESULTS_ROOT or pass an input path explicitly.")

    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output) if args.output else results_dir_default_output(input_path, "verifier_metrics.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    answer_source_overrides = parse_answer_source_overrides(args.answer_sources)
    samples = load_stage_samples(input_path, generic_stage=args.generic_stage, max_samples=args.max_samples)
    rows = collect_metric_rows(
        samples,
        verifier_mode=args.verifier_mode,
        stage_filter=args.stages,
        answer_source_overrides=answer_source_overrides,
    )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    applicable_rows = [row for row in rows if row.get("applicable")]
    avg_score = (
        sum(row.get("verifier_quality_score", 0.0) for row in applicable_rows if row.get("verifier_quality_score") is not None) / len(applicable_rows)
        if applicable_rows else 0.0
    )
    summary = {
        "_summary": True,
        "metric": "verifier_metrics",
        "input_path": str(input_path),
        "verifier_mode": args.verifier_mode,
        "total_stage_rows": len(rows),
        "applicable_stage_rows": len(applicable_rows),
        "average_verifier_quality_score": round(avg_score, 4),
    }
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"Done. Output: {output_path}")
    print(
        f"Overall: avg_verifier_quality_score={summary['average_verifier_quality_score']:.4f}, "
        f"applicable={summary['applicable_stage_rows']}, rows={summary['total_stage_rows']}"
    )


if __name__ == "__main__":
    main()
