import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

from experiment_utils import apply_experiment, get_experiment


def load_data(path, max_samples=None):
    from pathlib import Path

    path = Path(path)
    if path.suffix == ".jsonl":
        rows = []
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
                if max_samples is not None and len(rows) >= max_samples:
                    break
        return rows

    with path.open() as f:
        data = json.load(f)

    if isinstance(data, list):
        return data[:max_samples] if max_samples is not None else data
    return [data]


def _step_text(step):
    if isinstance(step, str):
        return step.strip()
    if not isinstance(step, dict):
        return ""

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

    text = f"{evidence} {inference}".strip()
    return text


def _extract_steps(sample):
    steps = sample.get("reasoning_steps")
    if isinstance(steps, list):
        texts = [_step_text(s) for s in steps]
        return [t for t in texts if t]

    trace = sample.get("trace")
    if isinstance(trace, list):
        texts = [_step_text(s) for s in trace]
        return [t for t in texts if t]

    if isinstance(trace, dict) and isinstance(trace.get("steps"), list):
        texts = [_step_text(s) for s in trace.get("steps")]
        return [t for t in texts if t]

    return []


def _pairwise_cosine_matrix(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    normalized = embeddings / norms
    return np.dot(normalized, normalized.T)


def compute_metric(sample, embedding_model, entailment_threshold=0.58):
    steps = _extract_steps(sample)
    n = len(steps)

    if n <= 1:
        return {
            "internal_coherence_score": 1.0 if n == 1 else 0.0,
            "step_count": n,
            "pair_count": 0,
            "entailed_pair_fraction": 1.0 if n == 1 else 0.0,
            "average_pairwise_similarity": 1.0 if n == 1 else 0.0,
        }

    embeddings = embedding_model.encode(steps)
    sim = _pairwise_cosine_matrix(embeddings)

    upper = np.triu_indices(n, k=1)
    pair_sims = sim[upper]
    pair_count = len(pair_sims)

    entailed = float(np.mean(pair_sims >= entailment_threshold)) if pair_count else 0.0
    avg_sim = float(np.mean(pair_sims)) if pair_count else 0.0

    return {
        "internal_coherence_score": round(entailed, 4),
        "step_count": n,
        "pair_count": pair_count,
        "entailed_pair_fraction": round(entailed, 4),
        "average_pairwise_similarity": round(avg_sim, 4),
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = "/fs/nexus-scratch/gnanesh/cot"

    input_file = os.environ.get("DATA_PATH") or os.path.join(project_root, "OmniVideoBench", "data_short_under1min.jsonl")
    experiment = get_experiment()
    output_file = os.path.join(script_dir, "results", experiment, "internal_coherence.jsonl")

    model_name = os.environ.get("LANGUAGE_BERT_MODEL", "sentence-transformers/bert-base-nli-mean-tokens")

    max_samples = None
    if os.environ.get("MAX_SAMPLES"):
        try:
            max_samples = int(os.environ["MAX_SAMPLES"])
        except ValueError:
            pass

    threshold = 0.58
    if os.environ.get("INTERNAL_COHERENCE_THRESHOLD"):
        try:
            threshold = float(os.environ["INTERNAL_COHERENCE_THRESHOLD"])
        except ValueError:
            pass

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Loading Language-BERT model: {model_name}")
    embedding_model = SentenceTransformer(model_name)

    samples = load_data(input_file, max_samples=max_samples)

    rows = []
    for i, sample in enumerate(samples):
        sample = apply_experiment(sample, experiment)
        metric = compute_metric(sample, embedding_model, entailment_threshold=threshold)
        row = {
            "video": sample.get("video", ""),
            "question": sample.get("question", ""),
            **metric,
        }
        rows.append(row)

        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1} samples")

    n = len(rows)
    avg_score = sum(r.get("internal_coherence_score", 0.0) for r in rows) / n if n else 0.0
    avg_pairs = sum(r.get("pair_count", 0) for r in rows) / n if n else 0.0

    summary = {
        "_summary": True,
        "metric": "Internal Coherence",
        "total_samples": n,
        "average_internal_coherence_score": round(avg_score, 4),
        "average_pair_count": round(avg_pairs, 2),
        "entailment_threshold": threshold,
    }

    with open(output_file, "w") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"Done. Output: {output_file}")
    print(
        "Overall: "
        f"avg_internal_coherence_score={summary['average_internal_coherence_score']:.4f}, "
        f"avg_pair_count={summary['average_pair_count']}, n={n}"
    )


if __name__ == "__main__":
    main()
