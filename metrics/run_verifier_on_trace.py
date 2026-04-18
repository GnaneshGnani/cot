import argparse
import json
from pathlib import Path

from stage_loader import load_stage_samples
from stage_metrics_common import build_stage_record, parse_answer_source_overrides, sorted_stage_names
from verifier_helper import create_verifier_client, resolve_verifier


def main():
    parser = argparse.ArgumentParser(description="Run the text-only verifier on normalized traces.")
    parser.add_argument("input_path", type=str)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--stages", type=str, default=None)
    parser.add_argument("--generic-stage", type=str, default="initial")
    parser.add_argument("--answer-sources", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else Path(__file__).resolve().parent / "results" / input_path.name / "verifier_on_trace.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    answer_source_overrides = parse_answer_source_overrides(args.answer_sources)
    samples = load_stage_samples(input_path, generic_stage=args.generic_stage, max_samples=args.max_samples)
    client, model = create_verifier_client()

    stage_filter = {item.strip() for item in (args.stages or "").split(",") if item.strip()} if args.stages else None

    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            for stage_name in sorted_stage_names(sample.get("stages", {}).keys()):
                if stage_filter and stage_name not in stage_filter:
                    continue
                stage_record = build_stage_record(
                    sample,
                    stage_name,
                    answer_source_overrides=answer_source_overrides,
                )
                verifier_result = resolve_verifier(
                    stage_record,
                    verifier_mode="offline",
                    client=client,
                    model=model,
                )
                row = {
                    "sample_id": stage_record["sample_id"],
                    "source_dir": stage_record["source_dir"],
                    "video_path": stage_record["video_path"],
                    "question": stage_record["question"],
                    "stage": stage_name,
                    "trace_step_count": len(stage_record["trace_steps"]),
                    "verifier_raw": verifier_result.get("verifier_raw"),
                    "verifier_output": verifier_result.get("verifier_output"),
                    "model": verifier_result.get("model"),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

        handle.write(
            json.dumps(
                {
                    "_summary": True,
                    "metric": "verifier_on_trace",
                    "model": model,
                    "total_stage_rows": written,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    print(f"Done. Wrote {written} rows to {output_path}")


if __name__ == "__main__":
    main()
