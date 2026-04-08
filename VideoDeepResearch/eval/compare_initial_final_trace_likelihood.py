import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List

from tqdm import tqdm

from score_trace_answer_likelihood import (
    build_answer_context,
    build_question_context,
    load_model_and_tokenizer,
    normalize_options,
    normalize_text,
    normalize_trace,
    normalize_trace_steps,
    resolve_device,
    score_continuation,
)


DEFAULT_INPUT = Path("/nfs-stor/ghazi.ahmad/videos/refiner_results.json")
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare score2 for initial vs final traces using gt_answer."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to refiner_results.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the comparison JSON. Defaults to <input_stem>_initial_final_compare_qwen3.json.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Qwen3 checkpoint to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Torch device, for example "auto", "cuda:0", or "cpu".',
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of samples to score.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the tokenizer/model from local cache only.",
    )
    return parser.parse_args()


def resolve_output_path(input_path: Path, output_path: Path) -> Path:
    if output_path is not None:
        return output_path
    return input_path.with_name(f"{input_path.stem}_initial_final_compare_qwen3.json")


def extract_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    refiner_result = item.get("refiner_result") or {}
    return {
        "video_path": normalize_text(refiner_result.get("video_path") or item.get("video_path")),
        "question": normalize_text(refiner_result.get("question") or item.get("question")),
        "options": normalize_options(refiner_result.get("options") or item.get("options")),
        "gt_answer": normalize_text(item.get("gt_answer")),
        "initial_trace": normalize_trace_steps(
            refiner_result.get("initial_trace")
            or item.get("initial_trace")
            or item.get("initial_trace_steps")
        ),
        "final_trace": normalize_trace_steps(
            refiner_result.get("final_trace") or item.get("final_trace")
        ),
        "initial_answer": normalize_text(refiner_result.get("initial_answer")),
        "final_answer": normalize_text(refiner_result.get("final_answer")),
    }


def score_trace(model, tokenizer, device: str, question: str, options: List[str], trace_steps: List[str], answer_text: str):
    trace_text = normalize_trace(trace_steps)
    if not question:
        raise ValueError("Missing question.")
    if not trace_text:
        raise ValueError("Missing trace.")
    if not answer_text:
        raise ValueError("Missing gt_answer.")

    question_context = build_question_context(question, options)
    answer_context = build_answer_context(question_context, trace_text)

    trace_score = score_continuation(
        model=model,
        tokenizer=tokenizer,
        user_content=question_context,
        target_text=trace_text,
        device=device,
    )
    answer_score = score_continuation(
        model=model,
        tokenizer=tokenizer,
        user_content=answer_context,
        target_text=answer_text,
        device=device,
    )

    total_tokens = trace_score["num_tokens"] + answer_score["num_tokens"]
    total_logprob = trace_score["logprob"] + answer_score["logprob"]

    return {
        "trace_logprob": trace_score["logprob"],
        "answer_logprob": answer_score["logprob"],
        "score2_logprob": total_logprob,
        "trace_num_tokens": trace_score["num_tokens"],
        "answer_num_tokens": answer_score["num_tokens"],
        "total_num_tokens": total_tokens,
        "trace_avg_logprob_per_token": trace_score["avg_logprob_per_token"],
        "answer_avg_logprob_per_token": answer_score["avg_logprob_per_token"],
        "score2_avg_logprob_per_token": (total_logprob / total_tokens) if total_tokens else None,
    }


def summarize_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [
        sample for sample in samples
        if "error" not in (sample.get("initial_trace_answer_likelihood") or {})
        and "error" not in (sample.get("final_trace_answer_likelihood") or {})
    ]

    def _metric_list(key: str):
        return [sample[key] for sample in valid]

    final_better_score2 = [s for s in valid if s["delta_score2_logprob"] > 0.0]
    final_better_score2_avg = [s for s in valid if s["delta_score2_avg_logprob_per_token"] > 0.0]
    final_better_trace_avg = [s for s in valid if s["delta_trace_avg_logprob_per_token"] > 0.0]

    summary = {
        "num_samples": len(samples),
        "num_valid": len(valid),
        "num_errors": len(samples) - len(valid),
        "final_better_raw_score2": len(final_better_score2),
        "initial_better_raw_score2": sum(1 for s in valid if s["delta_score2_logprob"] < 0.0),
        "ties_raw_score2": sum(1 for s in valid if s["delta_score2_logprob"] == 0.0),
        "final_better_score2_avg_per_token": len(final_better_score2_avg),
        "initial_better_score2_avg_per_token": sum(1 for s in valid if s["delta_score2_avg_logprob_per_token"] < 0.0),
        "ties_score2_avg_per_token": sum(1 for s in valid if s["delta_score2_avg_logprob_per_token"] == 0.0),
        "final_better_trace_avg_per_token": len(final_better_trace_avg),
        "initial_better_trace_avg_per_token": sum(1 for s in valid if s["delta_trace_avg_logprob_per_token"] < 0.0),
        "ties_trace_avg_per_token": sum(1 for s in valid if s["delta_trace_avg_logprob_per_token"] == 0.0),
    }

    if valid:
        for key in [
            "delta_score2_logprob",
            "delta_score2_avg_logprob_per_token",
            "delta_trace_avg_logprob_per_token",
        ]:
            values = _metric_list(key)
            summary[f"{key}_mean"] = mean(values)
            summary[f"{key}_median"] = median(values)
            summary[f"{key}_min"] = min(values)
            summary[f"{key}_max"] = max(values)

    return summary


def main():
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = resolve_output_path(input_path, args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {input_path}, got {type(data).__name__}.")

    if args.limit is not None:
        data = data[: args.limit]

    device = resolve_device(args.device)
    tokenizer, model = load_model_and_tokenizer(args, device)

    samples = []
    for item in tqdm(data, desc="Comparing initial vs final score2"):
        extracted = extract_fields(item)
        record = {
            "video_path": extracted["video_path"],
            "question": extracted["question"],
            "options": extracted["options"],
            "gt_answer": extracted["gt_answer"],
            "initial_trace": extracted["initial_trace"],
            "final_trace": extracted["final_trace"],
            "initial_trace_answer_likelihood": None,
            "final_trace_answer_likelihood": None,
        }

        try:
            initial_scores = score_trace(
                model=model,
                tokenizer=tokenizer,
                device=device,
                question=extracted["question"],
                options=extracted["options"],
                trace_steps=extracted["initial_trace"],
                answer_text=extracted["gt_answer"],
            )
            record["initial_trace_answer_likelihood"] = {
                "model": args.model,
                "trace_source": "refiner_result.initial_trace.steps",
                "answer_source": "gt_answer",
                "chat_template_enable_thinking": False,
                **initial_scores,
            }
        except Exception as exc:
            record["initial_trace_answer_likelihood"] = {
                "model": args.model,
                "error": str(exc),
            }

        try:
            final_scores = score_trace(
                model=model,
                tokenizer=tokenizer,
                device=device,
                question=extracted["question"],
                options=extracted["options"],
                trace_steps=extracted["final_trace"],
                answer_text=extracted["gt_answer"],
            )
            record["final_trace_answer_likelihood"] = {
                "model": args.model,
                "trace_source": "refiner_result.final_trace.steps",
                "answer_source": "gt_answer",
                "chat_template_enable_thinking": False,
                **final_scores,
            }
        except Exception as exc:
            record["final_trace_answer_likelihood"] = {
                "model": args.model,
                "error": str(exc),
            }

        initial_like = record["initial_trace_answer_likelihood"]
        final_like = record["final_trace_answer_likelihood"]
        if "error" not in initial_like and "error" not in final_like:
            record["delta_score2_logprob"] = (
                final_like["score2_logprob"] - initial_like["score2_logprob"]
            )
            record["delta_score2_avg_logprob_per_token"] = (
                final_like["score2_avg_logprob_per_token"]
                - initial_like["score2_avg_logprob_per_token"]
            )
            record["delta_trace_avg_logprob_per_token"] = (
                final_like["trace_avg_logprob_per_token"]
                - initial_like["trace_avg_logprob_per_token"]
            )
            record["final_stronger_raw_score2"] = record["delta_score2_logprob"] > 0.0
            record["final_stronger_score2_avg_per_token"] = (
                record["delta_score2_avg_logprob_per_token"] > 0.0
            )
            record["final_stronger_trace_avg_per_token"] = (
                record["delta_trace_avg_logprob_per_token"] > 0.0
            )

        samples.append(record)

    summary = summarize_samples(samples)
    payload = {
        "model": args.model,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "answer_source": "gt_answer",
        "summary": summary,
        "samples": samples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Saved comparison results to: {output_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
