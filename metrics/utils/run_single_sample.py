import json
import os
import sys

import torch
from sentence_transformers import SentenceTransformer
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

from m1_answer_faithfulness import compute_metric as compute_m1, load_data as load_data_m1
from m2_stepwise_relevance import compute_metric as compute_m2
from m3_causal_coherence import compute_metric as compute_m3
from m4_completeness import compute_metric as compute_m4
from m6_efficiency import compute_metric as compute_m6
from experiment_utils import apply_experiment

OQS_WEIGHTS = {
    "af_score": 0.25,
    "srs_score": 0.15,
    "ccc_score": 0.20,
    "cs_score": 0.20,
    "fas_score": 0.10,
    "efficiency_score": 0.10,
}


def compute_oqs(scores):
    total = 0.0
    for key, w in OQS_WEIGHTS.items():
        val = scores.get(key)
        if val is None:
            val = 0.5
        elif isinstance(val, bool):
            val = 1.0 if val else 0.0
        total += w * float(val)
    return total


def assign_tier(oqs):
    if oqs >= 0.85:
        return "A", "Accept"
    if oqs >= 0.60:
        return "B", "Needs Completion"
    return "C", "Regenerate"


def run_all_metrics(sample, llm_model, processor, emb_model, device, videos_dir=None):
    sample = apply_experiment(sample, os.environ.get("EXPERIMENT", "full_cot"))
    scores = {}
    debug = {}

    r1 = compute_m1(sample, llm_model, processor, device)
    scores["af_score"] = r1.get("af_score", 0)
    scores["is_correct"] = r1.get("is_correct", False)
    debug["m1_predicted"] = r1.get("predicted_answer", "")
    debug["m1_ground_truth"] = r1.get("ground_truth", "")

    r2 = compute_m2(sample, llm_model, processor, device)
    scores["srs_score"] = r2.get("srs_score", 0)

    r3 = compute_m3(sample, llm_model, processor, device)
    scores["ccc_score"] = r3.get("ccc_score", 0)
    debug["m3_step_scores"] = r3.get("step_coherence_scores", [])

    r4 = compute_m4(sample, llm_model, processor, emb_model, device)
    scores["cs_score"] = r4.get("cs_score", 0)
    debug["m4_subgoals"] = r4.get("subgoals", [])
    debug["m4_covered"] = r4.get("covered_count", 0)
    debug["m4_total"] = r4.get("total_subgoals", 0)

    try:
        from m5_factual_accuracy import compute_metric as compute_m5
        r5 = compute_m5(sample, llm_model, processor, videos_dir or "", device)
        scores["fas_score"] = r5.get("fas_score", 0.5)
        scores["tgs_score"] = r5.get("tgs_score", 1.0)
        debug["m5_step_details"] = r5.get("step_details", [])
        debug["m5_total_ts"] = r5.get("total_timestamps", 0)
        debug["m5_valid_ts"] = r5.get("valid_timestamps", 0)
        debug["m5_invalid_ts"] = r5.get("invalid_timestamps", [])
    except ImportError:
        scores["fas_score"] = 0.5
        scores["tgs_score"] = 1.0
        debug["m5_step_details"] = []
        debug["m5_total_ts"] = 0
        debug["m5_valid_ts"] = 0
        debug["m5_invalid_ts"] = []

    r6 = compute_m6(sample, emb_model)
    scores["efficiency_score"] = r6.get("efficiency_score", 0)
    debug["m6_redundant_pairs"] = r6.get("redundant_pairs", 0)
    debug["m6_total_pairs"] = r6.get("total_pairs", 0)

    return scores, debug


def _step_text(step):
    e = step.get("evidence") or step.get("evidece") or step.get("evience") or step.get("evodence") or step.get("nevidence") or ""
    i = step.get("inference") or step.get("Inference") or step.get("infefence") or ""
    return f"Evidence: {e}. Inference: {i}"


def generate_model_trace(sample, model, processor, videos_dir, device):
    video_rel = sample.get("video") or ""
    video_path = os.path.join(videos_dir, video_rel) if video_rel else ""
    if not video_path or not os.path.exists(video_path):
        return None
    question = sample.get("question", "")
    options = sample.get("options") or []
    opt_text = "\n".join(options) if options else ""
    prompt = (
        "Watch this video and answer the question. First, produce a reasoning trace with numbered steps. "
        "Each step should have: evidence (what you see/hear) and inference (what you conclude). "
        "End with your final answer.\n\n"
        f"Question: {question}\n"
    )
    if opt_text:
        prompt += f"Options:\n{opt_text}\n\n"
    prompt += "Output format:\nStep 1: Evidence: ... Inference: ...\nStep 2: Evidence: ... Inference: ...\n...\nAnswer: (letter or text)"

    conversations = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful video QA assistant. Output a structured reasoning trace, then the answer."}]},
        {"role": "user", "content": [{"type": "video", "video": video_path}, {"type": "text", "text": prompt}]},
    ]
    try:
        inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True,
        ).to(device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
            )
        text = processor.tokenizer.decode(gen[0][input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return text
    except Exception as e:
        return f"(error: {e})"


def print_report(sample, scores, debug=None):
    vid = sample.get("video", "")
    q = sample.get("question", "")
    if len(q) > 80:
        q = q[:80] + "..."

    oqs = compute_oqs(scores)
    tier, tier_label = assign_tier(oqs)
    process_ok = oqs >= 0.75
    answer_ok = scores.get("is_correct", False)
    if process_ok and answer_ok:
        classification = "Genuine Reasoning (GR)"
    elif process_ok and not answer_ok:
        classification = "Reasoning-Execution Gap"
    elif not process_ok and answer_ok:
        classification = "Answer Hacking"
    else:
        classification = "Full Failure"

    print("Sample:", vid)
    print("Question:", q)
    print()
    print("=== Question & Answer ===")
    options = sample.get("options") or []
    for o in options:
        print(" ", o)
    correct_opt = sample.get("correct_option", "")
    gt_answer = sample.get("answer", "")
    print("Correct option:", correct_opt)
    print("Ground-truth answer:", gt_answer)
    if debug:
        print("M1 predicted (from trace only):", debug.get("m1_predicted", ""))
    print()
    print("=== Individual Metrics ===")
    print(f"M1 (AF):   {scores.get('af_score', 0):.2f}  {'✓' if scores.get('af_score', 0) >= 0.9 else ''}")
    print(f"M2 (SRS):  {scores.get('srs_score', 0):.2f}")
    print(f"M3 (CCC):  {scores.get('ccc_score', 0):.2f}")
    if debug and debug.get("m3_step_scores"):
        print("  M3 per-step:", debug["m3_step_scores"])
    print(f"M4 (CS):   {scores.get('cs_score', 0):.2f}")
    if debug and debug.get("m4_subgoals"):
        print("  M4 rubrics (subgoals):")
        for i, sg in enumerate(debug["m4_subgoals"], 1):
            print(f"    {i}. {sg}")
        print(f"  M4 covered: {debug.get('m4_covered', 0)}/{debug.get('m4_total', 0)}")
    print(f"M5 (FAS):  {scores.get('fas_score', 0):.2f}  TGS={scores.get('tgs_score', 1.0):.2f}")
    if debug and debug.get("m5_step_details"):
        print("  M5 per-step verification:")
        for i, d in enumerate(debug["m5_step_details"], 1):
            cl = d.get("claim", "") or ""
            mr = d.get("model_response", "")
            print(f"    Step {i} (seg {d.get('segment_sec', 0)}s): score={d.get('score', 0):.2f}")
            print(f"      claim: {cl}")
            print(f"      model: {mr}")
    if debug and debug.get("m5_total_ts", 0) > 0:
        print(f"  M5 TGS timestamps: {debug.get('m5_valid_ts', 0)} valid, {debug.get('m5_total_ts', 0)} total")
        inv = debug.get("m5_invalid_ts", [])
        if inv:
            print(f"  M5 invalid timestamps: {inv}")
    print(f"M6 (ERS):  {scores.get('efficiency_score', 0):.2f}")
    if debug:
        rp, tp = debug.get("m6_redundant_pairs", 0), debug.get("m6_total_pairs", 0)
        print(f"  M6 redundant_pairs: {rp}/{tp}")
    print()
    print("=== Overall Quality Score ===")
    print(f"OQS: {oqs:.2f}")
    print(f"Tier: {tier} ({tier_label})")
    print()
    print(f"Process: {'✓' if process_ok else '✗'} {'Correct' if process_ok else 'Incorrect'}")
    print(f"Answer: {'✓' if answer_ok else '✗'} {'Correct' if answer_ok else 'Incorrect'}")
    print(f"Classification: {classification}")
    print()
    print("=== Original Trace (from dataset) ===")
    steps = sample.get("reasoning_steps") or []
    for i, step in enumerate(steps, 1):
        print(f"Step {i}: {_step_text(step)}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = "/fs/nexus-scratch/gnanesh/cot"
    input_file = os.path.join(project_root, "OmniVideoBench", "data_short_under1min.jsonl")
    videos_dir = os.path.join(project_root, "OmniVideoBench")
    model_name = "Qwen/Qwen2.5-Omni-7B"
    cache_dir = "/fs/nexus-scratch/gnanesh/.cache/huggingface"

    idx = 0
    if len(sys.argv) > 1:
        try:
            idx = int(sys.argv[1])
        except ValueError:
            pass

    from pathlib import Path
    samples = load_data_m1(Path(input_file))
    if idx >= len(samples):
        print(f"Sample index {idx} out of range (max {len(samples)-1})")
        sys.exit(1)
    sample = samples[idx]

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
    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    scores, debug = run_all_metrics(sample, llm_model, processor, emb_model, llm_model.device, videos_dir)
    print_report(sample, scores, debug)

    if "--generate-trace" in sys.argv or "-g" in sys.argv:
        print()
        print("=== Model-Generated Trace (from video+question) ===")
        gen_trace = generate_model_trace(sample, llm_model, processor, videos_dir, llm_model.device)
        if gen_trace:
            print(gen_trace)
        else:
            print("(video not found or error)")


if __name__ == "__main__":
    main()
