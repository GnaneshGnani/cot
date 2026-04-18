import argparse
import json
import os
from pathlib import Path

from answer_sufficiency import compute_stage_metric as compute_answer_sufficiency, load_qwen_judge
from execution_consistency import compute_stage_metric as compute_execution_consistency
from internal_coherence import compute_stage_metric as compute_internal_coherence
from stage_loader import load_stage_samples
from stage_metrics_common import (
    build_stage_record,
    parse_answer_source_overrides,
    parse_csv_arg,
    results_dir_default_output,
    sorted_stage_names,
)
from tool_validity import compute_stage_metric as compute_tool_validity
from verifier_helper import create_verifier_client
from verifier_metrics import compute_stage_metric as compute_verifier_metric


ALL_METRICS = (
    "answer_sufficiency",
    "internal_coherence",
    "execution_consistency",
    "tool_validity",
    "verifier_metrics",
)


def _load_answer_sufficiency_judge():
    model_name = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
    print(f"Loading answer sufficiency judge: {model_name}")
    return load_qwen_judge(model_name), model_name


def _load_internal_coherence_model():
    from sentence_transformers import SentenceTransformer

    model_name = os.environ.get("LANGUAGE_BERT_MODEL", "sentence-transformers/bert-base-nli-mean-tokens")
    print(f"Loading internal coherence model: {model_name}")
    return SentenceTransformer(model_name), model_name


def _maybe_create_verifier_client(verifier_mode):
    if verifier_mode == "stored":
        return None, None
    return create_verifier_client()


def run_metrics(
    samples,
    metrics_to_run=None,
    stage_filter=None,
    answer_source_overrides=None,
    verifier_mode="hybrid",
):
    metrics_to_run = tuple(metrics_to_run or ALL_METRICS)
    stage_filter = set(stage_filter or [])
    answer_source_overrides = answer_source_overrides or {}

    answer_judge = None
    answer_judge_model = None
    internal_model = None
    internal_model_name = None
    verifier_client = None
    verifier_model = None

    if "answer_sufficiency" in metrics_to_run:
        answer_judge, answer_judge_model = _load_answer_sufficiency_judge()
    if "internal_coherence" in metrics_to_run:
        internal_model, internal_model_name = _load_internal_coherence_model()
    if "verifier_metrics" in metrics_to_run:
        verifier_client, verifier_model = _maybe_create_verifier_client(verifier_mode)

    rows = []
    for sample in samples:
        out_row = {
            "sample_id": sample.get("sample_id"),
            "source_dir": sample.get("source_dir"),
            "video_path": sample.get("video_path"),
            "question": sample.get("question"),
            "question_id": sample.get("question_id"),
            "options": sample.get("options") or [],
            "gold_answer": sample.get("gold_answer"),
            "terminal_stage": sample.get("terminal_stage"),
            "stages": {},
        }

        for stage_name in sorted_stage_names(sample.get("stages", {}).keys()):
            if stage_filter and stage_name not in stage_filter:
                continue

            stage_record = build_stage_record(
                sample,
                stage_name,
                answer_source_overrides=answer_source_overrides,
            )
            stage_entry = {
                "available": stage_record["available"],
                "is_terminal": stage_record["is_terminal"],
                "trace_steps": stage_record["trace_steps"],
                "proposed_answer": stage_record["proposed_answer"],
                "proposed_answer_source": stage_record["proposed_answer_source"],
                "stage_local_answer": stage_record["stage_local_answer"],
                "is_correct": stage_record["is_correct"],
                "verifier_output": stage_record["stored_verifier_output"],
                "verifier_raw": stage_record["stored_verifier_raw"],
            }

            if "answer_sufficiency" in metrics_to_run:
                stage_entry["answer_sufficiency"] = compute_answer_sufficiency(stage_record, answer_judge)
                stage_entry["answer_sufficiency"]["model"] = answer_judge_model
            if "internal_coherence" in metrics_to_run:
                stage_entry["internal_coherence"] = compute_internal_coherence(stage_record, internal_model)
                stage_entry["internal_coherence"]["model"] = internal_model_name
            if "execution_consistency" in metrics_to_run:
                stage_entry["execution_consistency"] = compute_execution_consistency(stage_record)
            if "tool_validity" in metrics_to_run:
                stage_entry["tool_validity"] = compute_tool_validity(stage_record)
            if "verifier_metrics" in metrics_to_run:
                verifier_metric = compute_verifier_metric(
                    stage_record,
                    verifier_mode=verifier_mode,
                    client=verifier_client,
                    model=verifier_model,
                )
                stage_entry["verifier_metrics"] = verifier_metric
                stage_entry["verifier_output"] = verifier_metric.get("verifier_output")
                stage_entry["verifier_raw"] = verifier_metric.get("verifier_raw")

            out_row["stages"][stage_name] = stage_entry

        rows.append(out_row)

    return rows


def summarize_rows(rows):
    summary = {
        "total_samples": len(rows),
        "stage_counts": {},
    }
    for row in rows:
        for stage_name, stage_entry in (row.get("stages") or {}).items():
            summary["stage_counts"][stage_name] = summary["stage_counts"].get(stage_name, 0) + 1
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run the full stage-normalized metrics pipeline.")
    parser.add_argument("input_path", type=str, help="Results directory, JSON, or JSONL input.")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--stages", type=str, default=os.environ.get("STAGES"))
    parser.add_argument("--generic-stage", type=str, default=os.environ.get("GENERIC_STAGE", "initial"))
    parser.add_argument("--answer-sources", type=str, default=os.environ.get("ANSWER_SOURCE_OVERRIDES"))
    parser.add_argument("--verifier-mode", type=str, default=os.environ.get("VERIFIER_MODE", "hybrid"))
    parser.add_argument("--metrics", type=str, default=",".join(ALL_METRICS))
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output) if args.output else results_dir_default_output(input_path, "per_sample_metrics.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    answer_source_overrides = parse_answer_source_overrides(args.answer_sources)
    metrics_to_run = tuple(parse_csv_arg(args.metrics)) or ALL_METRICS
    stage_filter = parse_csv_arg(args.stages)

    print(f"Loading normalized samples from {input_path}...")
    samples = load_stage_samples(input_path, generic_stage=args.generic_stage, max_samples=args.max_samples)

    rows = run_metrics(
        samples,
        metrics_to_run=metrics_to_run,
        stage_filter=stage_filter,
        answer_source_overrides=answer_source_overrides,
        verifier_mode=args.verifier_mode,
    )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = output_path.with_name("summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summarize_rows(rows), handle, indent=2, ensure_ascii=False)

    print(f"Done. Per-sample output: {output_path}")
    print(f"Summary output: {summary_path}")


if __name__ == "__main__":
    main()
