import gc
import json
import os
import re
import subprocess
import tempfile

import torch
from experiment_utils import apply_experiment, get_experiment
from prompts import M5_FACTUAL_ACCURACY_SYSTEM, M5_FACTUAL_ACCURACY_USER
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


def parse_duration(duration_str):
    if not duration_str:
        return None
    m = re.match(r"(\d{1,2}):(\d{2})", str(duration_str).strip())
    if not m:
        return None
    minutes, seconds = int(m.group(1)), int(m.group(2))
    return minutes * 60 + seconds


_TS_MMSS = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})")
_TS_SEC = re.compile(r"(\d+)\s*seconds?", re.I)


def _parse_mmss(m):
    h, m_, s = m.groups()
    return (int(h) if h else 0) * 3600 + int(m_) * 60 + int(s)


def extract_timestamps(text):
    out = []
    for m in _TS_MMSS.finditer(text):
        out.append(_parse_mmss(m))
    for m in _TS_SEC.finditer(text):
        out.append(float(m.group(1)))
    return out


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


def extract_clip(video_path, start_sec, duration_sec=10):
    if not os.path.exists(video_path):
        return None
    fd, tmp = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(max(0, start_sec)),
                "-t",
                str(duration_sec),
                "-i",
                str(video_path),
                "-c",
                "copy",
                tmp,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=True,
        )
        return tmp
    except Exception:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except Exception:
                pass
        return None


def verify_claim(claim, clip_path, model, processor, device):
    if not claim or not clip_path or not os.path.exists(clip_path):
        return 0.5, "(no clip)"
    prompt = M5_FACTUAL_ACCURACY_USER.format(claim=claim)
    conversations = [
        {"role": "system", "content": [{"type": "text", "text": M5_FACTUAL_ACCURACY_SYSTEM}]},
        {"role": "user", "content": [{"type": "video", "video": clip_path}, {"type": "text", "text": prompt}]},
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
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
            )
        gen_text = processor.tokenizer.decode(
            generated_ids[0][input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        raw = extract_assistant_response(gen_text).strip()
        gen_lower = raw.lower()
        if "yes" in gen_lower and "no" not in gen_lower[:gen_lower.find("yes")]:
            return 1.0, raw
        if "no" in gen_lower:
            return 0.0, raw
        return 0.5, raw
    except Exception as e:
        return 0.5, f"(error: {e})"


def compute_metric(sample, model, processor, videos_dir, device, segment_length=10):
    reasoning_steps = sample.get("reasoning_steps") or []
    if not reasoning_steps:
        return {
            "step_scores": [], "fas_score": 1.0, "total_steps": 0,
            "tgs_score": 1.0, "total_timestamps": 0, "valid_timestamps": 0, "invalid_timestamps": [],
        }

    video_rel = sample.get("video") or ""
    duration_str = sample.get("duration")
    duration_sec = parse_duration(duration_str)
    if duration_sec is None:
        duration_sec = 60

    video_path = os.path.join(videos_dir, video_rel) if video_rel else ""
    if not os.path.exists(video_path):
        details = [{"claim": _step_text(s), "score": 0.5, "model_response": "(video not found)", "segment_sec": 0} for s in reasoning_steps]
        return {
            "step_scores": [0.5] * len(reasoning_steps), "step_details": details, "fas_score": 0.5, "total_steps": len(reasoning_steps),
            "tgs_score": 1.0, "total_timestamps": 0, "valid_timestamps": 0, "invalid_timestamps": [],
        }

    step_scores = []
    step_details = []
    temp_clips = []
    all_timestamps = []

    for step in reasoning_steps:
        claim = _step_text(step)
        timestamps = extract_timestamps(claim)
        all_timestamps.extend(timestamps)
        start_sec = 0
        if timestamps:
            start_sec = min(timestamps[0], max(0, duration_sec - segment_length))
        clip_path = extract_clip(video_path, start_sec, segment_length)
        if clip_path:
            temp_clips.append(clip_path)
        score, model_response = verify_claim(claim, clip_path, model, processor, device)
        step_scores.append(score)
        step_details.append({"claim": claim, "score": score, "model_response": model_response, "segment_sec": start_sec})

    # TGS: fraction of timestamps within video bounds (TGS is sub-component of FAS per EvaluationMetrics.md)
    if not all_timestamps:
        tgs_score = 1.0
        valid_timestamps = 0
        invalid_timestamps = []
    else:
        valid = []
        invalid = []
        for t in all_timestamps:
            if 0 <= t <= duration_sec + 1e-3:
                valid.append(t)
            else:
                invalid.append(t)
        tgs_score = len(valid) / len(all_timestamps)
        valid_timestamps = len(valid)
        invalid_timestamps = invalid

    for p in temp_clips:
        try:
            if os.path.exists(p):
                os.unlink(p)
        except Exception:
            pass

    fas = sum(step_scores) / len(step_scores) if step_scores else 0.5
    return {
        "step_scores": step_scores,
        "step_details": step_details,
        "fas_score": fas,
        "total_steps": len(reasoning_steps),
        "tgs_score": tgs_score,
        "total_timestamps": len(all_timestamps),
        "valid_timestamps": valid_timestamps,
        "invalid_timestamps": invalid_timestamps,
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = "/fs/nexus-scratch/gnanesh/cot"
    input_file = os.environ.get("DATA_PATH") or os.path.join(project_root, "OmniVideoBench", "data_short_under1min.jsonl")
    videos_dir = os.environ.get("VIDEOS_DIR") or os.path.join(project_root, "OmniVideoBench")
    experiment = get_experiment()
    output_file = os.path.join(script_dir, "results", experiment, "m5_factual_accuracy.jsonl")
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
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name, cache_dir=cache_dir)

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
        result = compute_metric(sample, model, processor, videos_dir, model.device)
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
    avg_fas = sum(r["fas_score"] for r in results) / n if n else 0.0
    tgs_vals = [r["tgs_score"] for r in results if "tgs_score" in r]
    avg_tgs = sum(tgs_vals) / len(tgs_vals) if tgs_vals else 0.0
    summary = {
        "_summary": True,
        "metric": "m5_factual_accuracy",
        "total_samples": n,
        "average_fas_score": avg_fas,
        "average_tgs_score": round(avg_tgs, 4),
    }

    with open(output_file, "w") as out:
        for r in results:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
        out.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"Done. Output: {output_file}")
    print(f"Overall: avg_fas_score={avg_fas:.4f}, avg_tgs_score={avg_tgs:.4f}, n={n}")


if __name__ == "__main__":
    main()
