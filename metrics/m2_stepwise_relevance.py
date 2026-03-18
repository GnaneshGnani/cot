import gc
import json
import os
import re
import sys

import torch
from experiment_utils import apply_experiment, get_experiment
from prompts import M2_STEPWISE_RELEVANCE_SYSTEM, M2_STEPWISE_RELEVANCE_USER
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

def load_data(path, max_samples = None):
    if path.suffix == ".jsonl":
        samples = []
        with path.open() as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
                if max_samples is not None and len(samples) >= max_samples:
                    break
        return samples

    with path.open() as f:
        data = json.load(f)
    return data[:max_samples] if max_samples is not None else data

def extract_assistant_response(full_output):
    if "assistant" in full_output:
        full_output = full_output.split("assistant")[-1].strip()
    # Truncate at next-turn markers (model sometimes continues with "Human:", "user:", etc.)
    for marker in ("\nHuman:", "\nhuman:", "\nUser:", "\nuser:"):
        if marker in full_output:
            full_output = full_output.split(marker)[0].strip()
    return full_output

def _step_text(step):
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
    return f"Evidence: {evidence}. Inference: {inference}"


def format_steps(reasoning_steps):
    if not reasoning_steps:
        return ""
    lines = []
    for i, step in enumerate(reasoning_steps, 1):
        lines.append(f"Step {i}: {_step_text(step)}")
    return "\n".join(lines)


def parse_relevance_scores(text, n_steps):
    text = extract_assistant_response(text)
    scores = []
    tokens = re.split(r"[,;\s]+", text.lower().strip())
    for tok in tokens:
        if "yes" in tok or tok == "y":
            scores.append(1.0)
        elif "no" in tok or tok == "n":
            scores.append(0.0)
        if len(scores) >= n_steps:
            break
    while len(scores) < n_steps:
        scores.append(0.5)
    return scores[:n_steps]


def build_prompt(sample):
    question = sample.get("question", "")
    reasoning_steps = sample.get("reasoning_steps") or []
    steps_text = format_steps(reasoning_steps)
    n = len(reasoning_steps)
    return M2_STEPWISE_RELEVANCE_USER.format(
        question=question, steps_text=steps_text, n=n
    )


def compute_metric(sample, model, processor, device):
    reasoning_steps = sample.get("reasoning_steps") or []
    if not reasoning_steps:
        return {"step_scores": [], "srs_score": 0.0, "total_steps": 0}

    prompt = build_prompt(sample)
    conversations = [
        {"role": "system", "content": [{"type": "text", "text": M2_STEPWISE_RELEVANCE_SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]

    inputs = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        )

    output_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    output_text = output_texts[0] if output_texts else ""

    step_scores = parse_relevance_scores(output_text, len(reasoning_steps))
    srs = sum(step_scores) / len(step_scores) if step_scores else 0.0

    return {
        "step_scores": step_scores,
        "srs_score": srs,
        "total_steps": len(reasoning_steps),
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = "/fs/nexus-scratch/gnanesh/cot"
    input_file = os.environ.get("DATA_PATH") or os.path.join(project_root, "OmniVideoBench", "data_short_under1min.jsonl")
    experiment = get_experiment()
    output_file = os.path.join(script_dir, "results", experiment, "m2_stepwise_relevance.jsonl")
    model_name = "Qwen/Qwen2.5-Omni-7B"
    cache_dir = "/fs/nexus-scratch/gnanesh/.cache/huggingface"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {model_name} on {device}...")
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto" if device == "cuda" else None,
        cache_dir=cache_dir,
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
    )

    from pathlib import Path
    max_samples = None
    if os.environ.get("MAX_SAMPLES"):
        try:
            max_samples = int(os.environ["MAX_SAMPLES"])
        except ValueError:
            pass
    samples = load_data(Path(input_file), max_samples=max_samples)

    results = []
    for i, sample in enumerate(samples):
        sample = apply_experiment(sample, experiment)
        result = compute_metric(sample, model, processor, model.device)
        output = {
            "video": sample.get("video", ""),
            "question": sample.get("question", ""),
            **result,
        }
        results.append(output)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} samples")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        gc.collect()

    # Compute overall performance
    n = len(results)
    avg_srs_score = sum(r["srs_score"] for r in results) / n if n else 0.0
    summary = {
        "_summary": True,
        "metric": "m2_stepwise_relevance",
        "total_samples": n,
        "average_srs_score": avg_srs_score,
    }

    with open(output_file, "w") as out:
        for result in results:
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
        out.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"Done. Output: {output_file}")
    print(f"Overall: avg_srs_score={avg_srs_score:.4f}, n={n}")


if __name__ == "__main__":
    main()
