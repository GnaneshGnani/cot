import gc
import json
import os
import re

import numpy as np
import torch
from experiment_utils import apply_experiment, get_experiment
from prompts import M4_COMPLETENESS_SYSTEM, M4_COMPLETENESS_USER
from sentence_transformers import SentenceTransformer
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor


def load_data(path, max_samples=None):
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
    for marker in ("\nHuman:", "\nhuman:", "\nUser:", "\nuser:"):
        if marker in full_output:
            full_output = full_output.split(marker)[0].strip()
    return full_output


def parse_subgoals(text):
    text = extract_assistant_response(text)
    subgoals = []
    for line in text.strip().split("\n"):
        line = line.strip()
        m = re.match(r"^[\d\.\-\*]+\s*(.+)", line)
        if m:
            subgoals.append(m.group(1).strip())
        elif line and len(subgoals) > 0 and not line[0].isdigit():
            subgoals[-1] += " " + line
        elif line and len(line) > 5:
            subgoals.append(line)
    return [s for s in subgoals if len(s) > 10][:5]


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
    return f"{evidence} {inference}".strip() or "(empty)"


def build_skeleton_prompt(sample):
    question = sample.get("question", "")
    answer = sample.get("answer", "")
    options = sample.get("options") or []
    if options and sample.get("correct_option"):
        answer = str(sample.get("correct_option", ""))
    return M4_COMPLETENESS_USER.format(question=question, answer=answer)


def generate_reference_skeleton(sample, model, processor, device):
    prompt = build_skeleton_prompt(sample)
    conversations = [
        {"role": "system", "content": [{"type": "text", "text": M4_COMPLETENESS_SYSTEM}]},
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

    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        )
    gen_text = processor.tokenizer.decode(
        generated_ids[0][input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return parse_subgoals(gen_text)


def compute_coverage(subgoals, step_texts, embedding_model, threshold=0.8):
    if not subgoals:
        return 1.0
    if not step_texts:
        return 0.0
    sub_emb = embedding_model.encode(subgoals)
    step_emb = embedding_model.encode(step_texts)
    covered = 0
    for i in range(len(sub_emb)):
        sims = np.dot(step_emb, sub_emb[i]) / (
            np.linalg.norm(step_emb, axis=1) * np.linalg.norm(sub_emb[i]) + 1e-9
        )
        if np.max(sims) > threshold:
            covered += 1
    return covered / len(subgoals)


def compute_metric(sample, llm_model, processor, embedding_model, device):
    reasoning_steps = sample.get("reasoning_steps") or []
    if not reasoning_steps:
        return {"subgoals": [], "cs_score": 0.0, "covered_count": 0, "total_subgoals": 0}

    subgoals = generate_reference_skeleton(sample, llm_model, processor, device)
    if not subgoals:
        return {"subgoals": [], "cs_score": 1.0, "covered_count": 0, "total_subgoals": 0}

    step_texts = [_step_text(s) for s in reasoning_steps]
    cs_score = compute_coverage(subgoals, step_texts, embedding_model)
    s_emb = embedding_model.encode(step_texts)
    g_emb = embedding_model.encode(subgoals)
    covered_count = 0
    for i in range(len(g_emb)):
        sims = np.dot(s_emb, g_emb[i]) / (np.linalg.norm(s_emb, axis=1) * np.linalg.norm(g_emb[i]) + 1e-9)
        if np.max(sims) > 0.8:
            covered_count += 1

    return {
        "subgoals": subgoals,
        "cs_score": cs_score,
        "covered_count": covered_count,
        "total_subgoals": len(subgoals),
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = "/fs/nexus-scratch/gnanesh/cot"
    input_file = os.environ.get("DATA_PATH") or os.path.join(project_root, "OmniVideoBench", "data_short_under1min.jsonl")
    experiment = get_experiment()
    output_file = os.path.join(script_dir, "results", experiment, "m4_completeness.jsonl")
    model_name = "Qwen/Qwen2.5-Omni-7B"
    cache_dir = "/fs/nexus-scratch/gnanesh/.cache/huggingface"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {model_name} on {device}...")
    llm_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto" if device == "cuda" else None,
        cache_dir=cache_dir,
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    print("Loading embedding model...")
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    max_samples = None
    if os.environ.get("MAX_SAMPLES"):
        try:
            max_samples = int(os.environ["MAX_SAMPLES"])
        except ValueError:
            pass
    samples = load_data(__import__("pathlib").Path(input_file), max_samples=max_samples)

    results = []
    for i, sample in enumerate(samples):
        sample = apply_experiment(sample, experiment)
        result = compute_metric(sample, llm_model, processor, embedding_model, llm_model.device)
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

    n = len(results)
    avg_cs = sum(r["cs_score"] for r in results) / n if n else 0.0
    summary = {
        "_summary": True,
        "metric": "m4_completeness",
        "total_samples": n,
        "average_cs_score": avg_cs,
    }

    with open(output_file, "w") as out:
        for r in results:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
        out.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"Done. Output: {output_file}")
    print(f"Overall: avg_cs_score={avg_cs:.4f}, n={n}")


if __name__ == "__main__":
    main()
