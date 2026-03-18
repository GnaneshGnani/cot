"""
Run Qwen 2.5 Omni on VideoMathQA benchmark with 4 configurations:
1. no_options_no_reasoning - Without options, no reasoning (direct answer)
2. no_options_reasoning    - Without options, with reasoning (think then answer)
3. with_options_no_reasoning - With options, no reasoning (direct letter)
4. with_options_reasoning    - With options, with reasoning (think then letter)

Handles OOM (GPU/system) and other errors gracefully - logs, continues.
Uses video truncation (--max-duration-seconds) to avoid OOM on long videos.
"""
import argparse
import gc
import json
import os
import subprocess
import tempfile
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

PROJECT_ROOT = "/fs/nexus-scratch/gnanesh/cot"
VMQA_DIR = os.path.join(PROJECT_ROOT, "VideoMathQA")
CACHE_DIR = "/fs/nexus-scratch/gnanesh/.cache/huggingface"

MODES = [
    "no_options_no_reasoning",
    "no_options_reasoning",
    "with_options_no_reasoning",
    "with_options_reasoning",
]


def extract_choice_letter(text):
    text = str(text).strip()
    for ch in text:
        if "A" <= ch <= "Z":
            return ch
    return None


def extract_assistant_response(full_output):
    if "assistant" in full_output.lower():
        full_output = full_output.split("assistant")[-1].strip()
    for marker in ("\nHuman:", "\nhuman:", "\nUser:", "\nuser:"):
        if marker in full_output:
            full_output = full_output.split(marker)[0].strip()
    return full_output


def load_videomathqa(data_path=None, max_samples=None):
    path_to_try = data_path
    if not path_to_try:
        for name in ("videomathqa_mcq_test.json", "annotations.json"):
            p = os.path.join(VMQA_DIR, name)
            if os.path.exists(p):
                path_to_try = p
                break
    if not path_to_try or not os.path.exists(path_to_try):
        raise FileNotFoundError(f"VideoMathQA data not found. Set --data-path or ensure {VMQA_DIR} has videomathqa_mcq_test.json")

    path = Path(path_to_try)
    if path.suffix == ".jsonl":
        samples = []
        with path.open() as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
                if max_samples and len(samples) >= max_samples:
                    break
    else:
        with path.open() as f:
            data = json.load(f)
        data = data if isinstance(data, list) else [data]
        raw = data[:max_samples] if max_samples else data
        samples = []
        for obj in raw:
            vid = obj.get("videoID", "")
            samples.append({
                "video": os.path.join(VMQA_DIR, "videos", f"{vid}.mp4") if vid else "",
                "videoID": vid,
                "question": obj.get("question", ""),
                "options": obj.get("options") or [],
                "answer": obj.get("answer", ""),
                "correct_option": obj.get("answer", ""),
            })

    for s in samples:
        if s.get("video") and not os.path.isabs(s["video"]):
            s["video"] = os.path.join(VMQA_DIR, s["video"])
    return samples


def build_prompt(sample, with_options, with_reasoning):
    question = sample.get("question", "")
    options = sample.get("options") or []

    if with_reasoning:
        if with_options and options:
            opt_text = "\n".join(options)
            prompt = (
                "Watch this video and answer the multiple-choice question.\n"
                "Think step by step, explain your reasoning, then give the final answer as the option letter only.\n\n"
                f"Question: {question}\n"
                f"Options:\n{opt_text}\n\n"
                "Answer with only the option letter (e.g., A, B, C, D, or E) at the end."
            )
        else:
            prompt = (
                "Watch this video and answer the question.\n"
                "Think step by step, explain your reasoning, then give your final answer concisely.\n\n"
                f"Question: {question}"
            )
    else:
        if with_options and options:
            opt_text = "\n".join(options)
            prompt = (
                "You are a helpful assistant for video question answering.\n"
                "Watch the video and answer using the option letter only. Do not output any explanation.\n\n"
                f"Question: {question}\n"
                f"Options:\n{opt_text}\n\n"
                "Answer with only the option letter (e.g., A, B, C, D, or E)."
            )
        else:
            prompt = (
                "You are a helpful assistant for video question answering.\n"
                "Watch the video and answer concisely. Do not output any explanation.\n\n"
                f"Question: {question}"
            )
    return prompt


def _get_video_for_processor(video_path, max_duration_seconds, ffmpeg_dir):
    """Return video path, optionally truncated to max_duration_seconds to reduce memory."""
    if not max_duration_seconds or not os.path.exists(video_path):
        return video_path
    ffprobe = os.path.join(ffmpeg_dir, "ffprobe") if ffmpeg_dir else "ffprobe"
    ffmpeg_exe = os.path.join(ffmpeg_dir, "ffmpeg") if ffmpeg_dir else "ffmpeg"
    tmp = None
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=10,
        )
        if out.returncode != 0 or not out.stdout or not out.stdout.strip():
            return video_path
        duration_sec = float(out.stdout.strip())
        if duration_sec <= max_duration_seconds:
            return video_path
        fd, tmp = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        subprocess.run(
            [ffmpeg_exe, "-y", "-i", str(video_path), "-t", str(max_duration_seconds), "-c", "copy", tmp],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120, check=True,
        )
        return tmp
    except Exception:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except Exception:
                pass
        return video_path


def run_inference(
    model,
    processor,
    sample,
    mode: str,
    max_new_tokens: int,
    max_duration_seconds: int = None,
    ffmpeg_dir: str = None,
):
    video_path = sample.get("video", "")
    if not video_path or not os.path.exists(video_path):
        return {"error": "video_not_found", "predicted_letter": None, "model_output": ""}

    path_for_processor = _get_video_for_processor(video_path, max_duration_seconds, ffmpeg_dir)
    cleanup_temp = path_for_processor != video_path

    with_options = "with_options" in mode
    with_reasoning = "reasoning" in mode
    prompt = build_prompt(sample, with_options, with_reasoning)

    conversations = [
        {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human capable of perceiving auditory and visual inputs and answering questions about videos."}]},
        {"role": "user", "content": [{"type": "video", "video": str(path_for_processor)}, {"type": "text", "text": prompt}]},
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
        ).to(model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
            )
        text = processor.tokenizer.decode(gen[0][input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        assistant_response = extract_assistant_response(text)
        pred_letter = extract_choice_letter(assistant_response)
        return {"predicted_letter": pred_letter, "model_output": assistant_response}
    except (torch.cuda.OutOfMemoryError, MemoryError, RuntimeError, OSError) as e:
        return {"error": str(e), "predicted_letter": None, "model_output": ""}
    except Exception as e:
        return {"error": str(e), "predicted_letter": None, "model_output": ""}
    finally:
        if cleanup_temp and path_for_processor and os.path.exists(str(path_for_processor)):
            try:
                os.unlink(path_for_processor)
            except Exception:
                pass


def run_mode(
    model,
    processor,
    samples,
    mode: str,
    output_path: Path,
    max_new_tokens: int = 512,
    max_duration_seconds: int = None,
    ffmpeg_dir: str = None,
):
    correct = 0
    total = 0
    errors = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as fout:
        for idx, sample in enumerate(tqdm(samples, desc=mode, unit="sample")):
            result = run_inference(
                model, processor, sample, mode, max_new_tokens,
                max_duration_seconds=max_duration_seconds, ffmpeg_dir=ffmpeg_dir,
            )
            gt = sample.get("correct_option") or sample.get("answer", "")
            gt_letter = str(gt)[0] if gt else None
            pred_letter = result.get("predicted_letter")
            is_correct = None
            if "error" in result:
                errors += 1
            elif gt_letter and pred_letter:
                total += 1
                is_correct = pred_letter.upper() == gt_letter.upper()
                if is_correct:
                    correct += 1

            record = {
                "index": idx,
                "videoID": sample.get("videoID", ""),
                "question": sample.get("question", ""),
                "answer": sample.get("answer", ""),
                "correct_option": gt_letter,
                "predicted_letter": pred_letter,
                "is_correct": is_correct,
                "model_output": result.get("model_output", "")[:500],
            }
            if "error" in result:
                record["error"] = result["error"]
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    acc = correct / total if total else 0
    print(f"  {mode}: accuracy={acc:.3f} ({correct}/{total}), errors={errors}")
    return {"mode": mode, "accuracy": acc, "correct": correct, "total": total, "errors": errors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None, help="Path to VideoMathQA json/jsonl (default: load from VMQA dir)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: cot/metrics/results/videomathqa_inference)")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-duration-seconds", type=int, default=90, help="Truncate videos longer than this (reduces OOM). Default 90.")
    parser.add_argument("--ffmpeg-dir", type=str, default=None, help="Dir containing ffmpeg/ffprobe (default: from PATH)")
    parser.add_argument("--modes", type=str, nargs="+", default=MODES, help="Which modes to run")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir or os.path.join(script_dir, "results", "videomathqa_inference"))

    print("Loading VideoMathQA...")
    samples = load_videomathqa(args.data_path, args.max_samples)
    print(f"Loaded {len(samples)} samples")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {args.model_name} on {device}...")
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto" if device == "cuda" else None,
        cache_dir=CACHE_DIR,
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_name, cache_dir=CACHE_DIR)

    ffmpeg_dir = args.ffmpeg_dir
    if not ffmpeg_dir and os.environ.get("PATH"):
        for p in os.environ["PATH"].split(":"):
            if os.path.exists(os.path.join(p, "ffmpeg")) and os.path.exists(os.path.join(p, "ffprobe")):
                ffmpeg_dir = p
                break

    summary = []
    for mode in args.modes:
        if mode not in MODES:
            print(f"Unknown mode {mode}, skipping")
            continue
        out_path = output_dir / f"{mode}.jsonl"
        s = run_mode(
            model, processor, samples, mode, out_path, args.max_new_tokens,
            max_duration_seconds=args.max_duration_seconds, ffmpeg_dir=ffmpeg_dir,
        )
        summary.append(s)

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
