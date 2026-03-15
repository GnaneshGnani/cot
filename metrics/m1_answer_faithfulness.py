import gc
import json
import os
import sys

import torch
from experiment_utils import apply_experiment, get_experiment
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

def extract_choice_letter(text):
    text = text.strip()
    for ch in text:
        if "A" <= ch <= "Z":
            return ch
    return None

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

    return evidence, inference


def format_reasoning_trace(reasoning_steps):
    if not reasoning_steps:
        return ""

    lines = []
    for i, step in enumerate(reasoning_steps, 1):
        evidence, inference = _step_text(step)
        lines.append(f"Step {i}: Evidence: {evidence}, Inference: {inference}")
    
    return "\n".join(lines)

def build_prompt(sample):
    question = sample.get("question", "")
    options = sample.get("options") or []
    reasoning_steps = sample.get("reasoning_steps") or []

    trace_text = format_reasoning_trace(reasoning_steps)
    if options:
        option_text = "\n".join(options)
        prompt = (
            f"Given the following reasoning trace, predict the answer to the question. "
            f"Do not include the answer in your reasoning.\n\n"
            f"Reasoning trace:\n{trace_text}\n\n"
            f"Question: {question}\n"
            f"Options:\n{option_text}\n\n"
            f"Answer with only the option letter (A, B, C, or D)."
        )
    else:
        prompt = (
            f"Given the following reasoning trace, predict the answer to the question.\n\n"
            f"Reasoning trace:\n{trace_text}\n\n"
            f"Question: {question}\n\n"
            f"Answer concisely."
        )
    return prompt


def compute_af_score(predicted, ground_truth, correct_option, options):
    if options and correct_option:
        pred_letter = extract_choice_letter(predicted)
        gt_letter = str(correct_option)[0] if correct_option else None

        if pred_letter is not None and gt_letter is not None:
            is_correct = pred_letter.upper() == gt_letter.upper()
            return 1.0 if is_correct else 0.0, is_correct

    gt_norm = str(ground_truth or "").strip().lower()
    pred_norm = str(predicted or "").strip().lower()

    if not gt_norm:
        return 0.0, False

    is_correct = gt_norm in pred_norm or pred_norm in gt_norm
    return 1.0 if is_correct else 0.0, is_correct


def compute_metric(sample, model, processor, device):
    prompt = build_prompt(sample)
    conversations = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant. Given a reasoning trace and question, predict the answer based only on the trace.",
                }
            ],
        },

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

    input_length = inputs["input_ids"].shape[1]
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
            stop_strings=["\nHuman:", "\nUser:", "\nhuman:", "\nuser:"],
            tokenizer=processor.tokenizer,
        )

    # Decode only the newly generated tokens (exclude the input prompt)
    generated_ids_only = generated_ids[:, input_length:]
    output_texts = processor.batch_decode(
        generated_ids_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    output_text = output_texts[0] if output_texts else ""
    predicted = extract_assistant_response(output_text)

    ground_truth = sample.get("answer")
    correct_option = sample.get("correct_option")
    options = sample.get("options") or []

    af_score, is_correct = compute_af_score(
        predicted, ground_truth, correct_option, options
    )

    return {
        "predicted_answer": predicted,
        "ground_truth": ground_truth,
        "options": options,
        "correct_option": correct_option,
        "af_score": af_score,
        "is_correct": is_correct,
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = "/fs/nexus-scratch/gnanesh/cot"
    input_file = os.path.join(project_root, "OmniVideoBench", "data_short_under1min.jsonl")
    experiment = get_experiment()
    output_file = os.path.join(script_dir, "results", experiment, "m1_answer_faithfulness.jsonl")
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
    samples = load_data(Path(input_file))

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

        print("="*100)
        print(results)
        print("="*100)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} samples")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        gc.collect()

    # Compute overall performance
    n = len(results)
    avg_af_score = sum(r["af_score"] for r in results) / n if n else 0.0
    accuracy = sum(1 for r in results if r["is_correct"]) / n if n else 0.0
    summary = {
        "_summary": True,
        "metric": "m1_answer_faithfulness",
        "total_samples": n,
        "average_af_score": avg_af_score,
        "accuracy": accuracy,
    }

    with open(output_file, "w") as out:
        for result in results:
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
        out.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"Done. Output: {output_file}")
    print(f"Overall: avg_af_score={avg_af_score:.4f}, accuracy={accuracy:.4f}, n={n}")

if __name__ == "__main__":
    main()
