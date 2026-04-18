import gc
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/fs/nexus-scratch/gnanesh/.cache/huggingface")
os.environ.setdefault("VLLM_PLUGINS", "")
os.environ.setdefault("VLLM_USE_MODELSCOPE", "false")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

try:
    import torch
except ImportError:
    class _TorchShim(object):
        @staticmethod
        def inference_mode():
            def _decorator(fn):
                return fn

            return _decorator

    torch = _TorchShim()

_EVAL_DIR = str(Path(__file__).resolve().parents[1] / "VideoDeepResearch" / "eval")
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)
from stage_loader import load_stage_samples
from stage_metrics_common import (
    build_stage_record,
    parse_answer_source_overrides,
    parse_csv_arg,
    results_dir_default_output,
    sorted_stage_names,
)


SYSTEM_PROMPT = """You are a strict evaluator for reasoning-trace answer sufficiency.

Your task is to judge whether a reasoning trace is sufficient to support a proposed answer.

A trace is sufficient if:
- a careful reader could derive the proposed answer from the trace alone
- the key evidence and inferences needed for the answer are present
- there are no major unresolved gaps or unsupported jumps

A trace is insufficient if:
- key evidence is missing
- the trace makes a crucial unsupported leap
- the trace stays ambiguous between multiple answers
- the trace gives observations but never actually supports the proposed answer

For multiple-choice questions, the trace must support the chosen option specifically.

Judge only whether the trace supports the proposed answer. Do not use outside knowledge.

Reply with exactly one word: Yes or No."""

USER_PROMPT = """## Question
{question}

## Options
{options_block}

## Proposed Answer
{proposed_answer}

## Reasoning Trace
{trace_text}

## Task
Is the reasoning trace sufficient to support the proposed answer?
Reply with exactly one word: Yes or No."""


def extract_assistant_response(full_output):
    text = str(full_output or "").strip()
    if "assistant" in text:
        text = text.split("assistant")[-1].strip()
    for marker in ("\nHuman:", "\nhuman:", "\nUser:", "\nuser:"):
        if marker in text:
            text = text.split(marker)[0].strip()
    return text


def parse_yes_no(text):
    raw = extract_assistant_response(text).strip().lower()
    tokens = re.findall(r"[a-z]+", raw)
    if not tokens:
        return None
    if tokens[0] == "yes":
        return True
    if tokens[0] == "no":
        return False
    if "yes" in tokens and "no" not in tokens[: tokens.index("yes")]:
        return True
    if "no" in tokens:
        return False
    return None


def build_prompt(stage_record):
    question = str(stage_record.get("question") or "").strip()
    options = stage_record.get("options") or []
    options_block = "\n".join(str(option) for option in options) if options else "(none)"
    trace_steps = list(stage_record.get("trace_steps") or [])
    trace_text = "\n".join(f"{index + 1}. {step}" for index, step in enumerate(trace_steps)) if trace_steps else "(missing trace)"
    proposed_answer = str(stage_record.get("proposed_answer") or "").strip()
    return {
        "question": question,
        "options": options,
        "trace_steps": trace_steps,
        "trace_text": trace_text,
        "proposed_answer": proposed_answer,
        "prompt": USER_PROMPT.format(
            question=question or "(missing question)",
            options_block=options_block,
            proposed_answer=proposed_answer or "(missing proposed answer)",
            trace_text=trace_text,
        ),
    }


def build_judge_text(prompt_data):
    return f"{SYSTEM_PROMPT}\n\n{prompt_data['prompt']}"


def load_qwen_judge(model_name):
    from refiner import VideoQADemo

    judge = VideoQADemo.__new__(VideoQADemo)
    judge.vlm_tensor_parallel_size = int(os.environ.get("VLM_TENSOR_PARALLEL_SIZE", "1"))
    judge.vlm_api_base = []
    judge.vlm_api_keys = []
    judge.vlm_model_name = model_name
    judge.local_vlm_model_name = model_name
    judge.vlm_server = None
    judge.processor = None
    judge._dense_captioner_vlm_server = None
    judge._dense_captioner_processor = None
    judge._setup_environment()

    server, processor, resolved_local_model = judge._load_local_vlm_runtime(model_name)
    judge.vlm_server = server
    judge.processor = processor
    judge._resolved_local_model = resolved_local_model
    return judge


@torch.inference_mode()
def compute_stage_metric(stage_record, judge):
    if not stage_record.get("available"):
        return {
            "applicable": False,
            "sufficiency_score": None,
            "is_sufficient": None,
            "skipped_reason": "stage_unavailable",
            "model_response": None,
        }

    prompt_data = build_prompt(stage_record)
    if not prompt_data["trace_steps"]:
        return {
            "applicable": False,
            "sufficiency_score": None,
            "is_sufficient": None,
            "skipped_reason": "missing_trace",
            "model_response": None,
        }
    if not prompt_data["proposed_answer"]:
        return {
            "applicable": False,
            "sufficiency_score": None,
            "is_sufficient": None,
            "skipped_reason": "missing_proposed_answer",
            "model_response": None,
        }
    if not prompt_data["question"]:
        return {
            "applicable": False,
            "sufficiency_score": None,
            "is_sufficient": None,
            "skipped_reason": "missing_question",
            "model_response": None,
        }

    model_response = extract_assistant_response(judge._vlm_summarize_text(build_judge_text(prompt_data)))
    verdict = parse_yes_no(model_response)
    return {
        "applicable": True,
        "sufficiency_score": 1.0 if verdict else 0.0,
        "is_sufficient": bool(verdict),
        "skipped_reason": None,
        "model_response": model_response,
    }


def collect_metric_rows(samples, judge, stage_filter=None, answer_source_overrides=None):
    stage_filter = set(parse_csv_arg(stage_filter)) if stage_filter else None
    rows = []

    for sample in samples:
        stage_names = sorted_stage_names(sample.get("stages", {}).keys())
        for stage_name in stage_names:
            if stage_filter and stage_name not in stage_filter:
                continue
            stage_record = build_stage_record(
                sample,
                stage_name,
                answer_source_overrides=answer_source_overrides,
            )
            metric = compute_stage_metric(stage_record, judge)
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

    import argparse

    parser = argparse.ArgumentParser(description="Compute answer sufficiency over normalized stages.")
    parser.add_argument("input_path", nargs="?", default=default_input)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--stages", type=str, default=os.environ.get("STAGES"))
    parser.add_argument("--generic-stage", type=str, default=os.environ.get("GENERIC_STAGE", "initial"))
    parser.add_argument(
        "--answer-sources",
        type=str,
        default=os.environ.get("ANSWER_SOURCE_OVERRIDES"),
        help="Comma-separated overrides like initial=benchmark,generated=stage_local",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    if not args.input_path:
        raise SystemExit("Set DATA_PATH or pass an input path explicitly.")

    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output) if args.output else results_dir_default_output(input_path, "answer_sufficiency.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_name = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
    answer_source_overrides = parse_answer_source_overrides(args.answer_sources)

    print(f"Loading samples from {input_path}...")
    samples = load_stage_samples(input_path, generic_stage=args.generic_stage, max_samples=args.max_samples)

    print(f"Loading sufficiency judge: {model_name}")
    judge = load_qwen_judge(model_name)

    rows = collect_metric_rows(
        samples,
        judge,
        stage_filter=args.stages,
        answer_source_overrides=answer_source_overrides,
    )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    applicable_rows = [row for row in rows if row.get("applicable")]
    avg_score = (
        sum(row.get("sufficiency_score", 0.0) for row in applicable_rows) / len(applicable_rows)
        if applicable_rows else 0.0
    )

    summary = {
        "_summary": True,
        "metric": "answer_sufficiency",
        "model": model_name,
        "input_path": str(input_path),
        "total_stage_rows": len(rows),
        "applicable_stage_rows": len(applicable_rows),
        "average_sufficiency_score": round(avg_score, 4),
    }
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"Done. Output: {output_path}")
    print(
        f"Overall: avg_sufficiency_score={summary['average_sufficiency_score']:.4f}, "
        f"applicable={summary['applicable_stage_rows']}, rows={summary['total_stage_rows']}"
    )
    gc.collect()


if __name__ == "__main__":
    main()
