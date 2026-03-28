import argparse
import csv
import json
import math
import os
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


TRACE_PATH = Path(__file__).resolve().parent / "traces" / "videomathqa_mcq_w_options.json"
CHOICES = ["A", "B", "C", "D", "E"]
PREDICT_NOW = "Predict the answer now. Reply with only the option letter."
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "confidence_report"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score MCQ answer confidence and generate diagnostic reports."
    )
    parser.add_argument(
        "--trace-path",
        type=Path,
        default=TRACE_PATH,
        help="Path to the trace JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV, JSON summary, and plots.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of items to score.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional torch device override.",
    )
    return parser.parse_args()


def load_data(trace_path):
    with trace_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_answer_letter(value):
    if value is None:
        return None
    value = str(value).strip().upper()
    if not value:
        return None
    letter = value[0]
    return letter if letter in CHOICES else None


def normalize_options(options):
    if options is None:
        return []
    if isinstance(options, dict):
        ordered = []
        for key in sorted(options):
            option = str(options[key]).strip()
            if option:
                ordered.append(option)
        return ordered
    if isinstance(options, (list, tuple)):
        return [str(option).strip() for option in options if str(option).strip()]
    option = str(options).strip()
    return [option] if option else []


def messages_at_decision_point(messages):
    trimmed = [dict(message) for message in messages[:-1]]
    trimmed.append({"role": "assistant", "content": PREDICT_NOW})
    return trimmed


def build_model(model_name, device=None):
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    model_kwargs = {
        "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto" if device is None else None,
    }
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        **model_kwargs,
    )
    if device is not None:
        model = model.to(device)

    choice_token_ids = {
        c: processor.tokenizer.encode(c, add_special_tokens=False)[0] for c in CHOICES
    }
    return processor, model, choice_token_ids


def score_item(item, processor, model, choice_token_ids):
    messages = messages_at_decision_point(item["messages"])
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1]

    choice_logits = torch.tensor(
        [logits[choice_token_ids[c]].item() for c in CHOICES],
        dtype=torch.float64,
    )
    probs = torch.softmax(choice_logits, dim=0)
    pred_index = int(probs.argmax())
    pred = CHOICES[pred_index]
    return {
        "pred": pred,
        "probs": {c: float(probs[i]) for i, c in enumerate(CHOICES)},
        "logits": {c: float(choice_logits[i]) for i, c in enumerate(CHOICES)},
    }


def entropy(prob_values):
    total = 0.0
    for p in prob_values:
        if p > 0.0:
            total -= p * math.log(p)
    return total


def rank_of_choice(probs, gt):
    ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    for idx, (choice, _) in enumerate(ordered, start=1):
        if choice == gt:
            return idx
    return None


def multiclass_brier(probs, gt):
    score = 0.0
    for choice, prob in probs.items():
        target = 1.0 if choice == gt else 0.0
        score += (prob - target) ** 2
    return score


def compute_ece(rows, n_bins=10):
    if not rows:
        return 0.0, []

    bins = []
    total = len(rows)
    for bin_idx in range(n_bins):
        lo = bin_idx / n_bins
        hi = (bin_idx + 1) / n_bins
        bucket = [
            row for row in rows
            if (row["p_pred"] >= lo and row["p_pred"] < hi) or (bin_idx == n_bins - 1 and row["p_pred"] == 1.0)
        ]
        if not bucket:
            bins.append(
                {
                    "bin_start": lo,
                    "bin_end": hi,
                    "count": 0,
                    "avg_confidence": None,
                    "accuracy": None,
                }
            )
            continue

        avg_conf = sum(row["p_pred"] for row in bucket) / len(bucket)
        accuracy = sum(1 for row in bucket if row["correct"]) / len(bucket)
        bins.append(
            {
                "bin_start": lo,
                "bin_end": hi,
                "count": len(bucket),
                "avg_confidence": avg_conf,
                "accuracy": accuracy,
            }
        )

    ece = 0.0
    for bucket in bins:
        if bucket["count"] == 0:
            continue
        weight = bucket["count"] / total
        ece += weight * abs(bucket["accuracy"] - bucket["avg_confidence"])
    return ece, bins


def selective_accuracy(rows, thresholds):
    output = []
    total = len(rows)
    for threshold in thresholds:
        kept = [row for row in rows if row["p_pred"] >= threshold]
        coverage = len(kept) / total if total else 0.0
        accuracy = sum(1 for row in kept if row["correct"]) / len(kept) if kept else None
        output.append(
            {
                "threshold": threshold,
                "coverage": coverage,
                "accuracy": accuracy,
                "count": len(kept),
            }
        )
    return output


def summarize(rows):
    valid_gt_rows = [row for row in rows if row["gt"] is not None]
    accuracy = (
        sum(1 for row in valid_gt_rows if row["correct"]) / len(valid_gt_rows)
        if valid_gt_rows else None
    )
    avg_confidence = sum(row["p_pred"] for row in rows) / len(rows) if rows else None
    avg_gt_prob = (
        sum(row["p_gt"] for row in valid_gt_rows) / len(valid_gt_rows)
        if valid_gt_rows else None
    )
    avg_margin = sum(row["margin"] for row in rows) / len(rows) if rows else None
    avg_entropy = sum(row["entropy"] for row in rows) / len(rows) if rows else None
    avg_nll = (
        sum(row["nll"] for row in valid_gt_rows) / len(valid_gt_rows)
        if valid_gt_rows else None
    )
    avg_brier = (
        sum(row["brier"] for row in valid_gt_rows) / len(valid_gt_rows)
        if valid_gt_rows else None
    )
    ece, ece_bins = compute_ece(valid_gt_rows)

    gt_rank_counts = {str(rank): 0 for rank in range(1, len(CHOICES) + 1)}
    for row in valid_gt_rows:
        if row["gt_rank"] is not None:
            gt_rank_counts[str(row["gt_rank"])] += 1

    return {
        "num_items": len(rows),
        "num_items_with_gt": len(valid_gt_rows),
        "accuracy": accuracy,
        "avg_confidence": avg_confidence,
        "avg_gt_prob": avg_gt_prob,
        "avg_margin": avg_margin,
        "avg_entropy": avg_entropy,
        "avg_nll": avg_nll,
        "avg_brier": avg_brier,
        "ece_10_bins": ece,
        "ece_bins": ece_bins,
        "gt_rank_counts": gt_rank_counts,
        "selective_accuracy": selective_accuracy(valid_gt_rows, [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    }


def write_csv(rows, path):
    fieldnames = [
        "index",
        "question",
        "pred",
        "gt",
        "correct",
        "p_pred",
        "p_gt",
        "margin",
        "entropy",
        "nll",
        "brier",
        "gt_rank",
        "A",
        "B",
        "C",
        "D",
        "E",
        "options_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "index": row["index"],
                "question": row["question"],
                "pred": row["pred"],
                "gt": row["gt"],
                "correct": row["correct"],
                "p_pred": row["p_pred"],
                "p_gt": row["p_gt"],
                "margin": row["margin"],
                "entropy": row["entropy"],
                "nll": row["nll"],
                "brier": row["brier"],
                "gt_rank": row["gt_rank"],
                "A": row["probs"]["A"],
                "B": row["probs"]["B"],
                "C": row["probs"]["C"],
                "D": row["probs"]["D"],
                "E": row["probs"]["E"],
                "options_json": json.dumps(row["options"], ensure_ascii=False),
            })


def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def save_plots(rows, summary, output_dir):
    plt = try_import_matplotlib()
    if plt is None:
        print("matplotlib is not available; skipping plots.")
        return False

    valid_gt_rows = [row for row in rows if row["gt"] is not None]
    correct_rows = [row for row in valid_gt_rows if row["correct"]]
    incorrect_rows = [row for row in valid_gt_rows if not row["correct"]]

    if valid_gt_rows:
        fig, ax = plt.subplots(figsize=(6, 6))
        bin_centers = []
        accuracy = []
        for bucket in summary["ece_bins"]:
            if bucket["count"] == 0:
                continue
            bin_centers.append(bucket["avg_confidence"])
            accuracy.append(bucket["accuracy"])
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
        ax.scatter(bin_centers, accuracy, color="tab:blue", s=50)
        ax.set_title("Reliability Diagram")
        ax.set_xlabel("Average confidence")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(output_dir / "reliability.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        bins = 15
        ax.hist([row["p_pred"] for row in correct_rows], bins=bins, alpha=0.6, label="Correct")
        ax.hist([row["p_pred"] for row in incorrect_rows], bins=bins, alpha=0.6, label="Incorrect")
        ax.set_title("Predicted Confidence Histogram")
        ax.set_xlabel("Top-1 confidence")
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "confidence_hist.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist([row["margin"] for row in correct_rows], bins=15, alpha=0.6, label="Correct")
        ax.hist([row["margin"] for row in incorrect_rows], bins=15, alpha=0.6, label="Incorrect")
        ax.set_title("Top-1 Minus Top-2 Margin")
        ax.set_xlabel("Margin")
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "margin_hist.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist([row["entropy"] for row in correct_rows], bins=15, alpha=0.6, label="Correct")
        ax.hist([row["entropy"] for row in incorrect_rows], bins=15, alpha=0.6, label="Incorrect")
        ax.set_title("Entropy Histogram")
        ax.set_xlabel("Entropy")
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "entropy_hist.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ranks = sorted(int(rank) for rank in summary["gt_rank_counts"].keys())
        counts = [summary["gt_rank_counts"][str(rank)] for rank in ranks]
        ax.bar(ranks, counts)
        ax.set_title("Ground-Truth Rank")
        ax.set_xlabel("Rank of ground-truth option")
        ax.set_ylabel("Count")
        ax.set_xticks(ranks)
        fig.tight_layout()
        fig.savefig(output_dir / "gt_rank_hist.png", dpi=160)
        plt.close(fig)

    return True


def main():
    args = parse_args()
    model_name = os.getenv("API_MODEL_NAME_VLM", "Qwen/Qwen2.5-VL-7B-Instruct")
    data = load_data(args.trace_path)
    if args.limit is not None:
        data = data[:args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor, model, choice_token_ids = build_model(model_name, device=args.device)

    rows = []
    for idx, item in enumerate(data):
        scored = score_item(item, processor, model, choice_token_ids)
        probs = scored["probs"]
        ordered_probs = sorted(probs.values(), reverse=True)
        gt = normalize_answer_letter(item.get("gt") or item.get("others", {}).get("answer"))
        p_pred = probs[scored["pred"]]
        p_gt = probs[gt] if gt in probs else None
        row = {
            "index": idx,
            "question": item.get("question", ""),
            "pred": scored["pred"],
            "gt": gt,
            "correct": gt == scored["pred"] if gt is not None else None,
            "p_pred": p_pred,
            "p_gt": p_gt,
            "margin": ordered_probs[0] - ordered_probs[1],
            "entropy": entropy(probs.values()),
            "nll": -math.log(max(p_gt, 1e-12)) if p_gt is not None else None,
            "brier": multiclass_brier(probs, gt) if gt is not None else None,
            "gt_rank": rank_of_choice(probs, gt) if gt is not None else None,
            "probs": probs,
            "logits": scored["logits"],
            "options": normalize_options(item.get("others", {}).get("options")),
        }
        rows.append(row)

    summary = summarize(rows)
    csv_path = args.output_dir / "per_item_metrics.csv"
    summary_path = args.output_dir / "summary.json"
    write_csv(rows, csv_path)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    plots_written = save_plots(rows, summary, args.output_dir)
    print(json.dumps({
        "model_name": model_name,
        "trace_path": str(args.trace_path),
        "output_dir": str(args.output_dir),
        "num_items": summary["num_items"],
        "accuracy": summary["accuracy"],
        "avg_confidence": summary["avg_confidence"],
        "avg_gt_prob": summary["avg_gt_prob"],
        "avg_margin": summary["avg_margin"],
        "avg_entropy": summary["avg_entropy"],
        "avg_nll": summary["avg_nll"],
        "avg_brier": summary["avg_brier"],
        "ece_10_bins": summary["ece_10_bins"],
        "plots_written": plots_written,
    }, indent=2))


if __name__ == "__main__":
    main()
