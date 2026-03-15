import json
import os
import re

from experiment_utils import apply_experiment, get_experiment

_TIMESTAMP_MMSS = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})")
_TIMESTAMP_SECONDS = re.compile(r"(\d+)\s*seconds?")

def _parse_mmss_match(match):
    h_str, m_str, s_str = match.groups()
    hours = int(h_str) if h_str else 0
    minutes = int(m_str)
    seconds = int(s_str)
    return hours * 3600 + minutes * 60 + seconds


def extract_timestamps(text):
    timestamps = []
    for m in _TIMESTAMP_MMSS.finditer(text):
        timestamps.append(_parse_mmss_match(m))
    for m in _TIMESTAMP_SECONDS.finditer(text, re.IGNORECASE):
        timestamps.append(float(m.group(1)))
    return timestamps

def parse_duration(duration_str):
    if not duration_str:
        return None
    m = re.match(r"(\d{1,2}):(\d{2})", str(duration_str).strip())
    if not m:
        return None
    minutes, seconds = int(m.group(1)), int(m.group(2))
    return minutes * 60 + seconds

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
    return f"{evidence} {inference}"


def compute_metric(sample):
    reasoning_steps = sample.get("reasoning_steps") or []
    duration_sec = parse_duration(sample.get("duration"))

    all_timestamps = []
    for step in reasoning_steps:
        all_timestamps.extend(extract_timestamps(_step_text(step)))

    if not all_timestamps:
        return {
            "total_timestamps": 0,
            "valid_timestamps": 0,
            "tgs_score": 1.0,
            "invalid_timestamps": [],
        }

    if duration_sec is None:
        duration_sec = max(all_timestamps) + 1

    valid = []
    invalid = []
    for t in all_timestamps:
        if 0 <= t <= duration_sec + 1e-3:
            valid.append(t)
        else:
            invalid.append(t)

    tgs = len(valid) / len(all_timestamps) if all_timestamps else 1.0

    return {
        "total_timestamps": len(all_timestamps),
        "valid_timestamps": len(valid),
        "tgs_score": tgs,
        "invalid_timestamps": invalid,
    }

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = "/fs/nexus-scratch/gnanesh/cot"
    input_file = os.path.join(project_root, "OmniVideoBench", "data_short_under1min.jsonl")
    experiment = get_experiment()
    output_file = os.path.join(script_dir, "results", experiment, "m9_temporal_grounding.jsonl")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    results = []
    with open(input_file) as f:
        for i, line in enumerate(f):
            sample = json.loads(line)
            sample = apply_experiment(sample, experiment)
            result = compute_metric(sample)
            output = {
                "video": sample.get("video", ""),
                "question": sample.get("question", ""),
                **result,
            }
            results.append(output)

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1} samples")

    # Compute overall performance
    n = len(results)
    avg_tgs_score = sum(r["tgs_score"] for r in results) / n if n else 0.0
    summary = {
        "_summary": True,
        "metric": "m9_temporal_grounding",
        "total_samples": n,
        "average_tgs_score": avg_tgs_score,
    }

    with open(output_file, "w") as out:
        for result in results:
            out.write(json.dumps(result) + "\n")
        out.write(json.dumps(summary) + "\n")

    print(f"Done. Output: {output_file}")
    print(f"Overall: avg_tgs_score={avg_tgs_score:.4f}, n={n}")


if __name__ == "__main__":
    main()
