import gc
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/nfs-stor/ghazi.ahmad/HF_HOME")
os.environ.setdefault("VLLM_PLUGINS", "")

import torch

EVAL_DIR = "/nfs-stor/ghazi.ahmad/cot/VideoDeepResearch/eval"
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

from refiner import VideoQADemo


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

TRACE_FIELDS = (
    "final_trace",
    "trace",
    "reasoning_trace",
    "initial_trace_steps",
    "reasoning_steps",
)

ANSWER_FIELDS = (
    "final_answer",
    "predicted_answer",
    "initial_answer",
    "answer",
)


def load_data(path, max_samples=None):
    path = Path(path)
    if path.is_dir():
        return load_meta_dir(path, max_samples=max_samples)

    if path.suffix == ".jsonl":
        rows = []
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_source_file"] = str(path)
                rows.append(row)
                if max_samples is not None and len(rows) >= max_samples:
                    break
        return rows

    with path.open() as f:
        data = json.load(f)

    if isinstance(data, list):
        rows = data[:max_samples] if max_samples is not None else data
        for row in rows:
            if isinstance(row, dict):
                row.setdefault("_source_file", str(path))
        return rows

    if isinstance(data, dict):
        data["_source_file"] = str(path)
        return [data]

    return []


def load_meta_dir(root, max_samples=None):
    rows = []
    for meta_path in sorted(Path(root).rglob("meta.json")):
        with meta_path.open() as f:
            row = json.load(f)
        row["_source_file"] = str(meta_path)
        rows.append(row)
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows


def extract_assistant_response(full_output):
    text = str(full_output or "").strip()
    if "assistant" in text:
        text = text.split("assistant")[-1].strip()
    for marker in ("\nHuman:", "\nhuman:", "\nUser:", "\nuser:"):
        if marker in text:
            text = text.split(marker)[0].strip()
    return text


def normalize_options(options):
    if options is None:
        return []
    if isinstance(options, dict):
        return [str(options[k]).strip() for k in sorted(options.keys(), key=lambda x: str(x)) if str(options[k]).strip()]
    if isinstance(options, (list, tuple)):
        return [str(option).strip() for option in options if str(option).strip()]
    value = str(options).strip()
    return [value] if value else []


def extract_choice_letter(text):
    match = re.search(r"\b([A-Z])\b", str(text or "").upper())
    return match.group(1) if match else None


def normalize_answer_text(text, options=None):
    text = str(text or "").strip()
    if not text:
        return ""

    letter = extract_choice_letter(text)
    if letter:
        return letter

    lowered = text.lower()
    for idx, option in enumerate(options or []):
        option_text = str(option).strip()
        if not option_text:
            continue
        if "." in option_text:
            option_text = option_text.split(".", 1)[1].strip()
        if option_text and option_text.lower() == lowered:
            return chr(ord("A") + idx)
    return lowered


def answers_match(proposed_answer, reference_answer, options=None):
    if not proposed_answer or not reference_answer:
        return None
    return normalize_answer_text(proposed_answer, options) == normalize_answer_text(reference_answer, options)


def step_to_text(step):
    if step is None:
        return ""
    if isinstance(step, str):
        return step.strip()
    if isinstance(step, dict):
        if isinstance(step.get("step"), str) and step.get("step").strip():
            return step["step"].strip()

        evidence = (
            step.get("evidence")
            or step.get("evidece")
            or step.get("evience")
            or step.get("evodence")
            or step.get("nevidence")
            or ""
        )
        inference = (
            step.get("inference")
            or step.get("Inference")
            or step.get("infefence")
            or ""
        )
        text = f"Evidence: {evidence}. Inference: {inference}".strip()
        return text if text != "Evidence: . Inference:" else ""
    return str(step).strip()


def normalize_trace_steps(trace_value):
    if isinstance(trace_value, dict):
        trace_value = trace_value.get("steps", trace_value)

    if isinstance(trace_value, list):
        return [step_to_text(step) for step in trace_value if step_to_text(step)]

    if trace_value is None:
        return []

    text = str(trace_value).strip()
    if not text:
        return []
    return [text]


def format_trace(steps):
    if not steps:
        return "(missing trace)"
    return "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(steps))


def get_trace_steps(sample):
    for field in TRACE_FIELDS:
        if field not in sample:
            continue
        steps = normalize_trace_steps(sample.get(field))
        if steps:
            return steps, field
    return [], None


def get_proposed_answer(sample):
    for field in ANSWER_FIELDS:
        value = sample.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text, field
    return "", None


def build_prompt(sample):
    question = str(sample.get("question") or "").strip()
    options = normalize_options(sample.get("options"))
    options_block = "\n".join(options) if options else "(none)"
    trace_steps, trace_field = get_trace_steps(sample)
    proposed_answer, answer_field = get_proposed_answer(sample)

    return {
        "question": question,
        "options": options,
        "trace_steps": trace_steps,
        "trace_field": trace_field,
        "trace_text": format_trace(trace_steps),
        "proposed_answer": proposed_answer,
        "answer_field": answer_field,
        "reference_answer": str(sample.get("answer") or "").strip(),
        "prompt": USER_PROMPT.format(
            question=question or "(missing question)",
            options_block=options_block,
            proposed_answer=proposed_answer or "(missing proposed answer)",
            trace_text=format_trace(trace_steps),
        ),
    }


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


def build_judge_text(prompt_data):
    return f"{SYSTEM_PROMPT}\n\n{prompt_data['prompt']}"


def load_qwen_judge(model_name):
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
def compute_metric(sample, judge):
    prompt_data = build_prompt(sample)
    if not prompt_data["question"] or not prompt_data["proposed_answer"] or not prompt_data["trace_steps"]:
        return {
            **prompt_data,
            "model_response": "(missing question, proposed answer, or trace)",
            "is_sufficient": False,
            "sufficiency_score": 0.0,
            "answer_matches_reference": answers_match(
                prompt_data["proposed_answer"],
                prompt_data["reference_answer"],
                prompt_data["options"],
            ),
        }

    model_response = extract_assistant_response(judge._vlm_summarize_text(build_judge_text(prompt_data)))
    verdict = parse_yes_no(model_response)

    return {
        **prompt_data,
        "model_response": model_response,
        "is_sufficient": bool(verdict),
        "sufficiency_score": 1.0 if verdict else 0.0,
        "answer_matches_reference": answers_match(
            prompt_data["proposed_answer"],
            prompt_data["reference_answer"],
            prompt_data["options"],
        ),
    }


def default_output_path(input_path):
    input_path = Path(input_path)
    if input_path.is_dir():
        return input_path / "answer_sufficiency.jsonl"
    if input_path.suffix == ".jsonl":
        return input_path.with_name(f"{input_path.stem}_answer_sufficiency.jsonl")
    return input_path.with_name("answer_sufficiency.jsonl")


def main():
    default_input = "/nfs-stor/ghazi.ahmad/cot/VideoDeepResearch/eval/results_generated_full_context"
    input_path = Path(os.environ.get("DATA_PATH") or default_input)
    output_path = Path(os.environ.get("OUTPUT_PATH") or default_output_path(input_path))

    model_name = os.environ.get("QWEN_MODEL", "Qwen/Qwen3.5-9B")

    max_samples = None
    if os.environ.get("MAX_SAMPLES"):
        try:
            max_samples = int(os.environ["MAX_SAMPLES"])
        except ValueError:
            pass

    print(f"Loading data from {input_path}...")
    samples = load_data(input_path, max_samples=max_samples)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_name}...")
    judge = load_qwen_judge(model_name)

    results = []
    for idx, sample in enumerate(samples):
        result = compute_metric(sample, judge)
        row = {
            "source_file": sample.get("_source_file", ""),
            "video_path": sample.get("video_path", ""),
            "question": result["question"],
            "options": result["options"],
            "trace_field": result["trace_field"],
            "answer_field": result["answer_field"],
            "proposed_answer": result["proposed_answer"],
            "reference_answer": result["reference_answer"],
            "answer_matches_reference": result["answer_matches_reference"],
            "model_response": result["model_response"],
            "is_sufficient": result["is_sufficient"],
            "sufficiency_score": result["sufficiency_score"],
        }
        results.append(row)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1} samples")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        gc.collect()

    n = len(results)
    avg_score = sum(row["sufficiency_score"] for row in results) / n if n else 0.0
    sufficient_rate = sum(1 for row in results if row["is_sufficient"]) / n if n else 0.0
    comparable = [row for row in results if row["answer_matches_reference"] is not None]
    agreement_rate = (
        sum(1 for row in comparable if row["answer_matches_reference"]) / len(comparable)
        if comparable else 0.0
    )

    summary = {
        "_summary": True,
        "metric": "answer_sufficiency",
        "model": model_name,
        "input_path": str(input_path),
        "total_samples": n,
        "average_sufficiency_score": avg_score,
        "sufficient_rate": sufficient_rate,
        "answer_reference_agreement_rate": agreement_rate,
        "comparable_answer_count": len(comparable),
    }

    with output_path.open("w") as out:
        for row in results:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"Done. Output: {output_path}")
    print(
        f"Overall: avg_sufficiency_score={avg_score:.4f}, "
        f"sufficient_rate={sufficient_rate:.4f}, n={n}"
    )


if __name__ == "__main__":
    main()
