import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


DEFAULT_INPUT = Path("/nfs-stor/ghazi.ahmad/videos/refiner_results.json")
DEFAULT_MODEL = "Qwen/Qwen3-8B"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute log P(trace|q) + log P(answer|q,trace) for refiner_results.json."
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
        help="Where to write the scored JSON. Defaults to <input_stem>_score2_qwen3.json.",
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


def normalize_options(options: Any) -> List[str]:
    if options is None:
        return []
    if isinstance(options, dict):
        def _sort_key(key: Any):
            key_str = str(key)
            return (0, int(key_str)) if key_str.isdigit() else (1, key_str)

        ordered = []
        for key in sorted(options.keys(), key=_sort_key):
            value = str(options[key]).strip()
            if value:
                ordered.append(value)
        return ordered
    if isinstance(options, (list, tuple)):
        return [str(option).strip() for option in options if str(option).strip()]
    value = str(options).strip()
    return [value] if value else []


def normalize_trace_steps(trace_value: Any) -> List[str]:
    if isinstance(trace_value, dict):
        trace_value = trace_value.get("steps", trace_value)

    if isinstance(trace_value, list):
        return [str(step).strip() for step in trace_value if str(step).strip()]

    if trace_value is None:
        return []

    text = str(trace_value).strip()
    return [text] if text else []


def normalize_trace(trace_value: Any) -> str:
    steps = normalize_trace_steps(trace_value)
    return "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(steps))


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_question_context(question: str, options: List[str]) -> str:
    parts = [f"Question:\n{question.strip()}"]
    if options:
        parts.append("Options:\n" + "\n".join(options))
    return "\n\n".join(parts).strip()


def build_answer_context(question_context: str, trace_text: str) -> str:
    return f"{question_context}\n\nTrace:\n{trace_text}".strip()


def resolve_output_path(input_path: Path, output_path: Path) -> Path:
    if output_path is not None:
        return output_path
    return input_path.with_name(f"{input_path.stem}_score2_qwen3.json")


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def resolve_dtype(dtype_arg: str, device: str):
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if device.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


def apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


@torch.inference_mode()
def score_continuation(model, tokenizer, user_content: str, target_text: str, device: str) -> Dict[str, float]:
    if not target_text:
        raise ValueError("Target text is empty.")

    messages = [{"role": "user", "content": user_content}]
    prompt_text = apply_chat_template(tokenizer, messages)
    full_text = prompt_text + target_text

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Prompt tokens are not a prefix of full tokens; refusing to score incorrectly.")

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    labels = input_ids.clone()
    labels[:, : len(prompt_ids)] = -100

    max_positions = getattr(model.config, "max_position_embeddings", None)
    if max_positions is not None and input_ids.shape[1] > max_positions:
        raise ValueError(
            f"Sequence length {input_ids.shape[1]} exceeds model limit {max_positions}."
        )

    logits = model(input_ids=input_ids).logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    valid_mask = shift_labels != -100

    if not bool(valid_mask.any()):
        raise ValueError("No continuation tokens were selected for scoring.")

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    gather_ids = shift_labels.masked_fill(~valid_mask, 0).unsqueeze(-1)
    token_log_probs = log_probs.gather(dim=-1, index=gather_ids).squeeze(-1)
    selected = token_log_probs.masked_select(valid_mask)

    total_logprob = float(selected.sum().item())
    num_tokens = int(selected.numel())
    avg_logprob = total_logprob / num_tokens

    return {
        "logprob": total_logprob,
        "num_tokens": num_tokens,
        "avg_logprob_per_token": avg_logprob,
    }


def load_model_and_tokenizer(args, device: str):
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    load_kwargs = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }

    def _from_pretrained(model_cls):
        try:
            return model_cls.from_pretrained(
                args.model,
                dtype=dtype,
                **load_kwargs,
            )
        except TypeError:
            # Older transformers still expect `torch_dtype`.
            return model_cls.from_pretrained(
                args.model,
                torch_dtype=dtype,
                **load_kwargs,
            )

    config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model_type = getattr(config, "model_type", "") or ""

    if model_type == "qwen3_vl":
        try:
            from transformers import Qwen3VLForConditionalGeneration

            model = _from_pretrained(Qwen3VLForConditionalGeneration)
        except (ImportError, AttributeError):
            try:
                from transformers import AutoModelForImageTextToText

                model = _from_pretrained(AutoModelForImageTextToText)
            except (ImportError, AttributeError):
                from transformers import AutoModelForVision2Seq

                model = _from_pretrained(AutoModelForVision2Seq)
    else:
        model = _from_pretrained(AutoModelForCausalLM)

    model.to(device)
    model.eval()
    return tokenizer, model


def extract_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    refiner_result = item.get("refiner_result") or {}
    video_path = normalize_text(refiner_result.get("video_path") or item.get("video_path"))
    question = normalize_text(refiner_result.get("question") or item.get("question"))
    options = normalize_options(refiner_result.get("options") or item.get("options"))
    initial_trace_steps = normalize_trace_steps(
        refiner_result.get("initial_trace")
        or item.get("initial_trace")
        or item.get("initial_trace_steps")
    )
    final_trace_steps = normalize_trace_steps(
        refiner_result.get("final_trace") or item.get("final_trace")
    )
    trace_text = normalize_trace(final_trace_steps)
    answer_text = normalize_text(
        refiner_result.get("final_answer")
        or item.get("final_answer")
        or refiner_result.get("initial_answer")
        or item.get("answer")
    )
    return {
        "video_path": video_path,
        "question": question,
        "options": options,
        "initial_trace": initial_trace_steps,
        "final_trace": final_trace_steps,
        "trace_text": trace_text,
        "answer_text": answer_text,
    }


def main():
    args = parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = resolve_output_path(input_path, args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {input_path}, got {type(data).__name__}.")

    if args.limit is not None:
        data = data[: args.limit]

    device = resolve_device(args.device)
    tokenizer, model = load_model_and_tokenizer(args, device)

    scored_items = []
    for item in tqdm(data, desc="Scoring score2"):
        extracted = extract_fields(item)
        record = {
            "video_path": extracted["video_path"],
            "question": extracted["question"],
            "options": extracted["options"],
            "initial_trace": extracted["initial_trace"],
            "final_trace": extracted["final_trace"],
            "trace_answer_likelihood": None,
        }

        try:
            question = extracted["question"]
            trace_text = extracted["trace_text"]
            answer_text = extracted["answer_text"]

            if not question:
                raise ValueError("Missing question.")
            if not trace_text:
                raise ValueError("Missing refiner_result.final_trace.")
            if not answer_text:
                raise ValueError("Missing refiner_result.final_answer.")

            question_context = build_question_context(question, extracted["options"])
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

            record["trace_answer_likelihood"] = {
                "model": args.model,
                "trace_source": "refiner_result.final_trace.steps",
                "trace_serialization": "numbered_steps_joined_with_newlines",
                "answer_source": "refiner_result.final_answer",
                "chat_template_enable_thinking": False,
                "trace_logprob": trace_score["logprob"],
                "answer_logprob": answer_score["logprob"],
                "score2_logprob": trace_score["logprob"] + answer_score["logprob"],
                "trace_num_tokens": trace_score["num_tokens"],
                "answer_num_tokens": answer_score["num_tokens"],
                "trace_avg_logprob_per_token": trace_score["avg_logprob_per_token"],
                "answer_avg_logprob_per_token": answer_score["avg_logprob_per_token"],
            }
        except Exception as exc:
            record["trace_answer_likelihood"] = {
                "model": args.model,
                "error": str(exc),
            }

        scored_items.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(scored_items, f, ensure_ascii=False, indent=2)

    print(f"Saved scored results to: {output_path}")


if __name__ == "__main__":
    main()
