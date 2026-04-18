import json
import os
from pathlib import Path

import numpy as np

from stage_loader import load_stage_samples
from stage_metrics_common import (
    build_stage_record,
    parse_answer_source_overrides,
    parse_csv_arg,
    results_dir_default_output,
    sorted_stage_names,
)


def _pairwise_cosine_matrix(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    normalized = embeddings / norms
    return np.dot(normalized, normalized.T)


def compute_stage_metric(stage_record, embedding_model, entailment_threshold=0.58):
    if not stage_record.get("available"):
        return {
            "applicable": False,
            "internal_coherence_score": None,
            "step_count": 0,
            "pair_count": 0,
            "entailed_pair_fraction": None,
            "average_pairwise_similarity": None,
            "skipped_reason": "stage_unavailable",
        }

    steps = list(stage_record.get("trace_steps") or [])
    n = len(steps)
    if n == 0:
        return {
            "applicable": False,
            "internal_coherence_score": None,
            "step_count": 0,
            "pair_count": 0,
            "entailed_pair_fraction": None,
            "average_pairwise_similarity": None,
            "skipped_reason": "missing_trace",
        }

    if n == 1:
        return {
            "applicable": True,
            "internal_coherence_score": 1.0,
            "step_count": 1,
            "pair_count": 0,
            "entailed_pair_fraction": 1.0,
            "average_pairwise_similarity": 1.0,
            "skipped_reason": None,
        }

    embeddings = embedding_model.encode(steps)
    sim = _pairwise_cosine_matrix(embeddings)

    upper = np.triu_indices(n, k=1)
    pair_sims = sim[upper]
    entailed = float(np.mean(pair_sims >= entailment_threshold)) if len(pair_sims) else 0.0
    avg_sim = float(np.mean(pair_sims)) if len(pair_sims) else 0.0

    return {
        "applicable": True,
        "internal_coherence_score": round(entailed, 4),
        "step_count": n,
        "pair_count": int(len(pair_sims)),
        "entailed_pair_fraction": round(entailed, 4),
        "average_pairwise_similarity": round(avg_sim, 4),
        "skipped_reason": None,
    }


def collect_metric_rows(samples, embedding_model, stage_filter=None, answer_source_overrides=None, entailment_threshold=0.58):
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
                embedding_model,
                entailment_threshold=entailment_threshold,
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
                    **metric,
                }
            )
    return rows


def main():
    default_input = os.environ.get("DATA_PATH")

    import argparse

    parser = argparse.ArgumentParser(description="Compute internal coherence over normalized stages.")
    parser.add_argument("input_path", nargs="?", default=default_input)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--stages", type=str, default=os.environ.get("STAGES"))
    parser.add_argument("--generic-stage", type=str, default=os.environ.get("GENERIC_STAGE", "initial"))
    parser.add_argument("--answer-sources", type=str, default=os.environ.get("ANSWER_SOURCE_OVERRIDES"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.environ.get("INTERNAL_COHERENCE_THRESHOLD", "0.58")),
    )
    args = parser.parse_args()
    if not args.input_path:
        raise SystemExit("Set DATA_PATH or pass an input path explicitly.")

    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output) if args.output else results_dir_default_output(input_path, "internal_coherence.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_name = os.environ.get("LANGUAGE_BERT_MODEL", "sentence-transformers/bert-base-nli-mean-tokens")
    answer_source_overrides = parse_answer_source_overrides(args.answer_sources)

    print(f"Loading samples from {input_path}...")
    samples = load_stage_samples(input_path, generic_stage=args.generic_stage, max_samples=args.max_samples)

    print(f"Loading internal coherence model: {model_name}")
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(model_name)

    rows = collect_metric_rows(
        samples,
        embedding_model,
        stage_filter=args.stages,
        answer_source_overrides=answer_source_overrides,
        entailment_threshold=args.threshold,
    )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    applicable_rows = [row for row in rows if row.get("applicable")]
    avg_score = (
        sum(row.get("internal_coherence_score", 0.0) for row in applicable_rows) / len(applicable_rows)
        if applicable_rows else 0.0
    )
    summary = {
        "_summary": True,
        "metric": "internal_coherence",
        "input_path": str(input_path),
        "total_stage_rows": len(rows),
        "applicable_stage_rows": len(applicable_rows),
        "average_internal_coherence_score": round(avg_score, 4),
        "entailment_threshold": args.threshold,
    }
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"Done. Output: {output_path}")
    print(
        f"Overall: avg_internal_coherence_score={summary['average_internal_coherence_score']:.4f}, "
        f"applicable={summary['applicable_stage_rows']}, rows={summary['total_stage_rows']}"
    )


if __name__ == "__main__":
    main()
