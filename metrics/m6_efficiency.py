import json
import os

from sentence_transformers import SentenceTransformer
from experiment_utils import apply_experiment, get_experiment
import numpy as np


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


def compute_metric(sample, model):
    reasoning_steps = sample.get("reasoning_steps") or []
    if len(reasoning_steps) < 2:
        total_pairs = 0
        redundant_pairs = 0
        score = 1.0
    else:
        texts = [_step_text(s) for s in reasoning_steps]
        embeddings = model.encode(texts)
        n = len(embeddings)
        total_pairs = n * (n - 1) // 2
        redundant_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-9
                )
                if sim > 0.92:
                    redundant_pairs += 1
        score = 1.0 - (redundant_pairs / total_pairs) if total_pairs > 0 else 1.0

    return {
        "redundant_pairs": redundant_pairs,
        "total_pairs": total_pairs,
        "efficiency_score": score,
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = "/fs/nexus-scratch/gnanesh/cot"
    input_file = os.environ.get("DATA_PATH") or os.path.join(project_root, "OmniVideoBench", "data_short_under1min.jsonl")
    experiment = get_experiment()
    output_file = os.path.join(script_dir, "results", experiment, "m6_efficiency.jsonl")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print("Loading sentence-transformer model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    results = []
    with open(input_file) as f:
        for i, line in enumerate(f):
            sample = json.loads(line)
            sample = apply_experiment(sample, experiment)
            result = compute_metric(sample, model)
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
    avg_efficiency_score = sum(r["efficiency_score"] for r in results) / n if n else 0.0
    summary = {
        "_summary": True,
        "metric": "m6_efficiency",
        "total_samples": n,
        "average_efficiency_score": avg_efficiency_score,
    }

    with open(output_file, "w") as out:
        for result in results:
            out.write(json.dumps(result) + "\n")
        out.write(json.dumps(summary) + "\n")

    print(f"Done. Output: {output_file}")
    print(f"Overall: avg_efficiency_score={avg_efficiency_score:.4f}, n={n}")


if __name__ == "__main__":
    main()
