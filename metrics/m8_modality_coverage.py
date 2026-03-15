import json
import os

from experiment_utils import apply_experiment, get_experiment


def _get_modality(step):
    return str(step.get("modality") or "").lower()


def compute_metric(sample):
    reasoning_steps = sample.get("reasoning_steps") or []
    if not reasoning_steps:
        return {"mc_visual": 0.0, "mc_audio": 0.0, "total_steps": 0}

    total = len(reasoning_steps)
    visual_count = sum(1 for s in reasoning_steps if _get_modality(s) == "vision")
    audio_count = sum(1 for s in reasoning_steps if _get_modality(s) == "audio")

    return {
        "mc_visual": visual_count / total,
        "mc_audio": audio_count / total,
        "total_steps": total,
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = "/fs/nexus-scratch/gnanesh/cot"
    input_file = os.path.join(project_root, "OmniVideoBench", "data_short_under1min.jsonl")
    experiment = get_experiment()
    output_file = os.path.join(script_dir, "results", experiment, "m8_modality_coverage.jsonl")

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
    avg_mc_visual = sum(r["mc_visual"] for r in results) / n if n else 0.0
    avg_mc_audio = sum(r["mc_audio"] for r in results) / n if n else 0.0
    summary = {
        "_summary": True,
        "metric": "m8_modality_coverage",
        "total_samples": n,
        "average_mc_visual": avg_mc_visual,
        "average_mc_audio": avg_mc_audio,
    }

    with open(output_file, "w") as out:
        for result in results:
            out.write(json.dumps(result) + "\n")
        out.write(json.dumps(summary) + "\n")

    print(f"Done. Output: {output_file}")
    print(f"Overall: avg_mc_visual={avg_mc_visual:.4f}, avg_mc_audio={avg_mc_audio:.4f}, n={n}")


if __name__ == "__main__":
    main()
