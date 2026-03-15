import json
import os

from experiment_utils import apply_experiment, get_experiment

PLAN_KEYWORDS = [
    "plan", "first", "then", "next", "finally",
    "step 1", "step 2", "step 3", "strategy",
]

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
    return f"{evidence} {inference}".lower()


def compute_metric(sample):
    reasoning_steps = sample.get("reasoning_steps") or []
    keywords_found = []

    for step in reasoning_steps:
        text = _step_text(step)
        for kw in PLAN_KEYWORDS:
            if kw in text and kw not in keywords_found:
                keywords_found.append(kw)

    has_plan = len(keywords_found) > 0

    return {
        "has_plan": has_plan,
        "plan_keywords_found": keywords_found,
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = "/fs/nexus-scratch/gnanesh/cot"
    input_file = os.path.join(project_root, "OmniVideoBench", "data_short_under1min.jsonl")
    experiment = get_experiment()
    output_file = os.path.join(script_dir, "results", experiment, "m7_planning_presence.jsonl")

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
    fraction_with_plan = sum(1 for r in results if r["has_plan"]) / n if n else 0.0
    summary = {
        "_summary": True,
        "metric": "m7_planning_presence",
        "total_samples": n,
        "fraction_with_plan": fraction_with_plan,
    }

    with open(output_file, "w") as out:
        for result in results:
            out.write(json.dumps(result) + "\n")
        out.write(json.dumps(summary) + "\n")

    print(f"Done. Output: {output_file}")
    print(f"Overall: fraction_with_plan={fraction_with_plan:.4f}, n={n}")


if __name__ == "__main__":
    main()
