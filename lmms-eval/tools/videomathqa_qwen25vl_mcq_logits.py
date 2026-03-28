#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import datasets
import numpy as np
import torch
from loguru import logger
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from lmms_eval.tasks.videomathqa.utils import (
    extract_characters_regex,
    videomathqa_doc_to_text,
    videomathqa_mcq_aggregate_results,
    videomathqa_process_results,
    videomathqa_doc_to_visual,
)

try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:  # pragma: no cover
    raise SystemExit("qwen_vl_utils is required. Install it with `pip install qwen-vl-utils`.") from exc


DEFAULT_POST_PROMPT = "\nAnswer with the option's letter (A, B, C, D or E) from the given choices directly."
DATASET_PATH = "MBZUAI/VideoMathQA"
DATASET_NAME = "mcq"
CANDIDATE_LETTERS = tuple("ABCDE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VideoMathQA MCQ with Qwen2.5-VL and dump per-option logits/logprobs."
    )
    parser.add_argument("--pretrained", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True, help="Path to the output JSONL file.")
    parser.add_argument("--summary-output", default=None, help="Optional path for a JSON summary file.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--question-id", default=None, help="Score only a single question_id if provided.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--max-pixels", type=int, default=12845056)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-num-frames", type=int, default=32)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def resolve_torch_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    return getattr(torch, name)


def get_candidate_letters(num_options: int) -> List[str]:
    if num_options < 2 or num_options > len(CANDIDATE_LETTERS):
        raise ValueError(f"Expected between 2 and 5 options, got {num_options}")
    return list(CANDIDATE_LETTERS[:num_options])


def build_prompt(doc: Dict[str, Any]) -> str:
    return videomathqa_doc_to_text(
        doc,
        lmms_eval_specific_kwargs={"post_prompt": DEFAULT_POST_PROMPT},
    )


def build_messages(
    doc: Dict[str, Any],
    prompt: str,
    system_prompt: str,
    max_pixels: int,
    min_pixels: int,
) -> List[Dict[str, Any]]:
    visuals = videomathqa_doc_to_visual(doc)
    content: List[Dict[str, Any]] = []
    for visual in visuals:
        lower = str(visual).lower()
        if lower.endswith((".mp4", ".avi", ".mov", ".mkv")):
            content.append(
                {
                    "type": "video",
                    "video": visual,
                    "max_pixels": max_pixels,
                    "min_pixels": min_pixels,
                }
            )
        else:
            raise ValueError(f"Unsupported visual input for VideoMathQA: {visual}")
    content.append({"type": "text", "text": prompt})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def downsample_video_inputs(video_inputs: Any, max_num_frames: int) -> Any:
    if video_inputs is None:
        return None

    downsampled = deepcopy(video_inputs)
    for idx, video in enumerate(downsampled):
        total_frames = int(video.shape[0])
        if total_frames <= max_num_frames:
            continue
        indices = np.linspace(0, total_frames - 1, max_num_frames, dtype=int)
        indices = np.unique(indices)
        if (total_frames - 1) not in indices:
            indices = np.unique(np.append(indices, total_frames - 1))
        downsampled[idx] = video[indices]
    return downsampled


def build_processor_inputs(
    processor: AutoProcessor,
    prompt_text: str,
    image_inputs: Any,
    video_inputs: Any,
) -> Dict[str, torch.Tensor]:
    return processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        return_tensors="pt",
    )


def move_inputs(inputs: Dict[str, torch.Tensor], device: str, device_map: str) -> Dict[str, torch.Tensor]:
    target = "cuda" if device_map == "auto" else device
    return inputs.to(target)


def tokenize_suffix(tokenizer: AutoTokenizer, prompt_text: str, continuation: str) -> List[int]:
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt_text + continuation, add_special_tokens=False)["input_ids"]
    suffix_ids = full_ids[len(prompt_ids) :]
    if not suffix_ids:
        raise ValueError(f"No continuation token ids found for continuation={continuation!r}")
    return suffix_ids


def score_continuation(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    prompt_text: str,
    continuation: str,
    image_inputs: Any,
    video_inputs: Any,
    device: str,
    device_map: str,
) -> Dict[str, Any]:
    base_inputs = build_processor_inputs(processor, prompt_text, image_inputs, video_inputs)
    base_len = int(base_inputs["input_ids"].shape[1])
    full_inputs = build_processor_inputs(processor, prompt_text + continuation, image_inputs, video_inputs)
    full_input_ids = full_inputs["input_ids"][0]
    continuation_ids = full_input_ids[base_len:].tolist()
    if not continuation_ids:
        raise ValueError(f"Failed to isolate continuation ids for continuation={continuation!r}")

    full_inputs = move_inputs(full_inputs, device=device, device_map=device_map)
    with torch.inference_mode():
        outputs = model(**full_inputs, use_cache=False)
    logits = outputs.logits[0].float()
    log_probs = torch.log_softmax(logits, dim=-1)

    token_logprobs: List[float] = []
    token_logits: List[float] = []
    for offset, token_id in enumerate(continuation_ids):
        position = base_len - 1 + offset
        token_logprobs.append(float(log_probs[position, token_id].item()))
        token_logits.append(float(logits[position, token_id].item()))

    return {
        "continuation_ids": continuation_ids,
        "token_logprobs": token_logprobs,
        "token_logits": token_logits,
        "sum_logprob": float(sum(token_logprobs)),
        "avg_logprob": float(sum(token_logprobs) / len(token_logprobs)),
    }


def score_doc(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    tokenizer: AutoTokenizer,
    doc: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    prompt = build_prompt(doc)
    messages = build_messages(
        doc=doc,
        prompt=prompt,
        system_prompt=args.system_prompt,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
    )
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    video_inputs = downsample_video_inputs(video_inputs, args.max_num_frames)

    base_inputs = build_processor_inputs(processor, prompt_text, image_inputs, video_inputs)
    base_inputs = move_inputs(base_inputs, device=args.device, device_map=args.device_map)

    with torch.inference_mode():
        outputs = model(**base_inputs, use_cache=False)

    next_token_logits = outputs.logits[0, -1, :].float().cpu()
    next_token_logprobs = torch.log_softmax(next_token_logits, dim=-1)

    option_scores: Dict[str, Dict[str, Any]] = {}
    available_letters = get_candidate_letters(len(doc["options"]))

    for letter in available_letters:
        continuation_ids = tokenize_suffix(tokenizer, prompt_text, letter)
        first_token_id = int(continuation_ids[0])
        entry: Dict[str, Any] = {
            "candidate": letter,
            "continuation_ids": continuation_ids,
            "first_token_id": first_token_id,
            "first_token_text": tokenizer.decode([first_token_id]),
            "first_token_logit": float(next_token_logits[first_token_id].item()),
            "first_token_logprob": float(next_token_logprobs[first_token_id].item()),
        }
        if len(continuation_ids) == 1:
            entry["sum_logprob"] = entry["first_token_logprob"]
            entry["avg_logprob"] = entry["first_token_logprob"]
            entry["token_logits"] = [entry["first_token_logit"]]
            entry["token_logprobs"] = [entry["first_token_logprob"]]
        else:
            entry.update(
                score_continuation(
                    model=model,
                    processor=processor,
                    prompt_text=prompt_text,
                    continuation=letter,
                    image_inputs=image_inputs,
                    video_inputs=video_inputs,
                    device=args.device,
                    device_map=args.device_map,
                )
            )
        option_scores[letter] = entry

    restricted_logits = torch.tensor(
        [option_scores[letter]["first_token_logit"] for letter in available_letters],
        dtype=torch.float32,
    )
    restricted_probs = torch.softmax(restricted_logits, dim=0)
    restricted_logprobs = torch.log_softmax(restricted_logits, dim=0)
    for idx, letter in enumerate(available_letters):
        option_scores[letter]["restricted_prob"] = float(restricted_probs[idx].item())
        option_scores[letter]["restricted_logprob"] = float(restricted_logprobs[idx].item())

    predicted_letter = max(available_letters, key=lambda letter: option_scores[letter]["sum_logprob"])
    parsed_answer = extract_characters_regex(predicted_letter)
    processed = videomathqa_process_results(doc, [predicted_letter])["videomathqa_perception_score"]

    return {
        "question_id": doc["question_id"],
        "videoID": doc["videoID"],
        "category": doc["category"],
        "length": doc["length"],
        "answer": doc["answer"],
        "available_options": available_letters,
        "prompt": prompt,
        "predicted_option": predicted_letter,
        "parsed_predicted_option": parsed_answer,
        "correct": int(predicted_letter == doc["answer"]),
        "option_scores": option_scores,
        "processed_result": processed,
    }


def iter_docs(dataset: Iterable[Dict[str, Any]], args: argparse.Namespace) -> Iterable[tuple[int, Dict[str, Any]]]:
    yielded = 0
    for idx in range(args.start_index, len(dataset)):
        doc = dataset[idx]
        if args.question_id is not None and str(doc["question_id"]) != str(args.question_id):
            continue
        yield idx, doc
        yielded += 1
        if args.max_samples is not None and yielded >= args.max_samples:
            break


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": resolve_torch_dtype(args.torch_dtype),
        "device_map": args.device_map,
    }
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation

    logger.info("Loading dataset {} / {} split={}", DATASET_PATH, DATASET_NAME, args.split)
    dataset = datasets.load_dataset(
        DATASET_PATH,
        DATASET_NAME,
        split=args.split,
        token=False,
        cache_dir="videomathqa",
        video=True,
    )

    logger.info("Loading model {}", args.pretrained)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.pretrained, **model_kwargs).eval()
    processor = AutoProcessor.from_pretrained(
        args.pretrained,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)

    all_processed_results: List[Dict[str, Any]] = []
    num_examples = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for dataset_index, doc in iter_docs(dataset, args):
            result = score_doc(model=model, processor=processor, tokenizer=tokenizer, doc=doc, args=args)
            result["dataset_index"] = dataset_index
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            all_processed_results.append(result["processed_result"])
            num_examples += 1

            if args.verbose:
                logger.info(
                    "[{}] qid={} pred={} gold={} correct={}",
                    dataset_index,
                    result["question_id"],
                    result["predicted_option"],
                    result["answer"],
                    result["correct"],
                )

    overall = videomathqa_mcq_aggregate_results(all_processed_results) if all_processed_results else 0.0
    accuracy = (
        sum(item["pred_answer"] == item["answer"] for item in all_processed_results) / len(all_processed_results) * 100.0
        if all_processed_results
        else 0.0
    )

    summary = {
        "model": args.pretrained,
        "split": args.split,
        "num_examples": num_examples,
        "overall_score": overall,
        "accuracy": accuracy,
        "output_path": str(output_path),
    }
    logger.info("Finished {} examples. Overall score: {:.2f}", num_examples, overall)

    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
