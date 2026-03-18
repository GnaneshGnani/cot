"""
perturbation_generator.py

Generates controlled perturbations of combined_20_samples.json for metric
robustness testing. Each perturbation targets one or more specific metrics
to verify they respond in the expected direction.

Usage:
    python perturbation_generator.py                     # generate all 10
    python perturbation_generator.py --perturb shuffled  # generate one
"""

import argparse
import copy
import json
import os
import random
import re
import sys

PROJECT_ROOT = "/fs/nexus-scratch/gnanesh/cot"
METRICS_DIR = os.path.join(PROJECT_ROOT, "cot", "metrics")
INPUT_FILE = os.path.join(METRICS_DIR, "results", "combined_20", "combined_20_samples.json")
OUTPUT_DIR = os.path.join(METRICS_DIR, "results", "robustness_perturbs")

EVIDENCE_KEYS = ("evidence", "evidece", "evience", "evodence", "nevidence")
INFERENCE_KEYS = ("inference", "Inference", "infefence")

PERTURBATIONS = [
    "shuffled",
    "duplicate_steps",
    "corrupt_final",
    "invalid_timestamps",
    "inject_irrelevant",
    "single_step",
    "wrong_answer",
    "generic_evidence",
    "contradiction",
    "keyword_stuffing",
]

# Two fixed irrelevant steps used for P5
_IRRELEVANT_STEPS = [
    {
        "evidence": "The capital of France is Paris.",
        "inference": "This is a well-known geographical fact unrelated to the video.",
        "modality": "text",
        "evidece": None, "evience": None, "evodence": None, "nevidence": None,
        "Inference": None, "infefence": None, "nModality": None,
    },
    {
        "evidence": "The boiling point of water is 100 degrees Celsius at sea level.",
        "inference": "This is a basic chemistry fact unrelated to the video.",
        "modality": "text",
        "evidece": None, "evience": None, "evodence": None, "nevidence": None,
        "Inference": None, "infefence": None, "nModality": None,
    },
]

_OPTION_CYCLE = {"A": "B", "B": "C", "C": "D", "D": "A"}


def _get_evidence(step):
    for k in EVIDENCE_KEYS:
        v = step.get(k)
        if v:
            return v
    return ""


def _set_all_evidence(step, value):
    for k in EVIDENCE_KEYS:
        if k in step:
            step[k] = value if step[k] else step[k]
    # Always set the canonical key so metrics that look for "evidence" find it
    step["evidence"] = value


def _get_inference(step):
    for k in INFERENCE_KEYS:
        v = step.get(k)
        if v:
            return v
    return ""


def _set_all_inference(step, value):
    for k in INFERENCE_KEYS:
        if k in step:
            step[k] = value if step[k] else step[k]
    step["inference"] = value


def _timestamp_to_seconds(h, m, s):
    return int(h or 0) * 3600 + int(m) * 60 + int(s)


def _inflate_timestamps(text):
    """Replace all MM:SS or H:MM:SS timestamps with values 10× larger."""
    def replacer(match):
        groups = match.groups()
        if len(groups) == 3:
            total = _timestamp_to_seconds(*groups)
        else:
            total = int(groups[0]) * 60 + int(groups[1])
        total_inflated = total * 10
        mins, secs = divmod(total_inflated, 60)
        return f"{mins}:{secs:02d}"

    # Match optional H:MM:SS or MM:SS
    pattern = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})")
    return pattern.sub(replacer, text)


# ---------------------------------------------------------------------------
# Perturbation functions — each takes a list of samples, returns a new list
# ---------------------------------------------------------------------------

def p1_shuffled(samples):
    """Randomly reorder reasoning_steps within each sample (seed=42)."""
    out = []
    rng = random.Random(42)
    for s in samples:
        s = copy.deepcopy(s)
        steps = s.get("reasoning_steps") or []
        if len(steps) > 1:
            rng.shuffle(steps)
            s["reasoning_steps"] = steps
        out.append(s)
    return out


def p2_duplicate_steps(samples):
    """Copy each step and insert it immediately after the original."""
    out = []
    for s in samples:
        s = copy.deepcopy(s)
        steps = s.get("reasoning_steps") or []
        doubled = []
        for step in steps:
            doubled.append(step)
            doubled.append(copy.deepcopy(step))
        s["reasoning_steps"] = doubled
        out.append(s)
    return out


def p3_corrupt_final(samples):
    """Replace the inference of the last step with an indeterminate statement."""
    out = []
    for s in samples:
        s = copy.deepcopy(s)
        steps = s.get("reasoning_steps") or []
        if steps:
            last = steps[-1]
            _set_all_inference(last, "Therefore, the answer cannot be determined from the available evidence.")
        out.append(s)
    return out


def p4_invalid_timestamps(samples):
    """Multiply all MM:SS timestamps in evidence fields by 10, exceeding video duration."""
    out = []
    for s in samples:
        s = copy.deepcopy(s)
        steps = s.get("reasoning_steps") or []
        for step in steps:
            for k in EVIDENCE_KEYS:
                v = step.get(k)
                if v and isinstance(v, str):
                    step[k] = _inflate_timestamps(v)
        out.append(s)
    return out


def p5_inject_irrelevant(samples):
    """Insert 2 off-topic steps at fixed positions in the step list."""
    out = []
    rng = random.Random(42)
    for s in samples:
        s = copy.deepcopy(s)
        steps = list(s.get("reasoning_steps") or [])
        irr = [copy.deepcopy(x) for x in _IRRELEVANT_STEPS]
        # Insert at index 1 (after first step) and near the end
        ins1 = min(1, len(steps))
        ins2 = max(ins1 + 1, len(steps) - 1)
        steps.insert(ins1, irr[0])
        steps.insert(ins2, irr[1])
        s["reasoning_steps"] = steps
        out.append(s)
    return out


def p6_single_step(samples):
    """Keep only the first reasoning step."""
    out = []
    for s in samples:
        s = copy.deepcopy(s)
        steps = s.get("reasoning_steps") or []
        s["reasoning_steps"] = steps[:1]
        out.append(s)
    return out


def p7_wrong_answer(samples):
    """Cycle correct_option A→B→C→D→A; also update answer text to match."""
    out = []
    for s in samples:
        s = copy.deepcopy(s)
        opt = str(s.get("correct_option", "")).strip().upper()
        new_opt = _OPTION_CYCLE.get(opt, opt)
        s["correct_option"] = new_opt
        # Try to update the answer text to match the new option label
        options = s.get("options") or []
        for o in options:
            o_str = str(o)
            if o_str.startswith(new_opt + ".") or o_str.startswith(new_opt + " "):
                s["answer"] = o_str[len(new_opt):].lstrip(". ").strip()
                break
        out.append(s)
    return out


def p8_generic_evidence(samples):
    """Replace all evidence fields with a generic placeholder, keep inferences."""
    generic = "Some visual content was observed in the video."
    out = []
    for s in samples:
        s = copy.deepcopy(s)
        steps = s.get("reasoning_steps") or []
        for step in steps:
            _set_all_evidence(step, generic)
        out.append(s)
    return out


def p9_contradiction(samples):
    """Append a step that contradicts the final answer."""
    out = []
    for s in samples:
        s = copy.deepcopy(s)
        opt = str(s.get("correct_option", "")).strip().upper() or "the stated option"
        contradiction_step = {
            "evidence": "",
            "inference": (
                f"Upon further review, the answer is definitely not {opt} "
                "based on the available evidence."
            ),
            "modality": "text",
            "evidece": None, "evience": None, "evodence": None, "nevidence": None,
            "Inference": None, "infefence": None, "nModality": None,
        }
        steps = list(s.get("reasoning_steps") or [])
        steps.append(contradiction_step)
        s["reasoning_steps"] = steps
        out.append(s)
    return out


def p10_keyword_stuffing(samples):
    """Prepend the question text to every evidence field."""
    out = []
    for s in samples:
        s = copy.deepcopy(s)
        question = s.get("question", "")
        steps = s.get("reasoning_steps") or []
        for step in steps:
            for k in EVIDENCE_KEYS:
                v = step.get(k)
                if v and isinstance(v, str):
                    step[k] = f"{question} {v}"
            # Ensure canonical key is also stuffed
            canon = step.get("evidence") or ""
            if canon and not canon.startswith(question):
                step["evidence"] = f"{question} {canon}"
        out.append(s)
    return out


PERTURBATION_FNS = {
    "shuffled": p1_shuffled,
    "duplicate_steps": p2_duplicate_steps,
    "corrupt_final": p3_corrupt_final,
    "invalid_timestamps": p4_invalid_timestamps,
    "inject_irrelevant": p5_inject_irrelevant,
    "single_step": p6_single_step,
    "wrong_answer": p7_wrong_answer,
    "generic_evidence": p8_generic_evidence,
    "contradiction": p9_contradiction,
    "keyword_stuffing": p10_keyword_stuffing,
}


def load_samples(path):
    with open(path) as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    # JSONL fallback
    samples = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            samples.append(json.loads(line))
    return samples


def write_jsonl(samples, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Written {len(samples)} samples → {path}")


def generate_one(name, samples, output_dir):
    fn = PERTURBATION_FNS[name]
    perturbed = fn(samples)
    out_path = os.path.join(output_dir, f"{name}.jsonl")
    write_jsonl(perturbed, out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate perturbation datasets for metric robustness testing.")
    parser.add_argument("--perturb", choices=PERTURBATIONS, default=None,
                        help="Name of a single perturbation to generate. Omit to generate all.")
    parser.add_argument("--input", default=INPUT_FILE,
                        help="Path to combined_20_samples.json (JSON array or JSONL).")
    parser.add_argument("--output_dir", default=OUTPUT_DIR,
                        help="Directory to write perturbed JSONL files.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    samples = load_samples(args.input)
    print(f"Loaded {len(samples)} samples from {args.input}")

    targets = [args.perturb] if args.perturb else PERTURBATIONS
    for name in targets:
        print(f"Generating perturbation: {name}")
        generate_one(name, samples, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
