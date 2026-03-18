import json
import os
import sys

OQS_WEIGHTS = {
    "af_score": 0.25,
    "srs_score": 0.15,
    "ccc_score": 0.20,
    "cs_score": 0.20,
    "fas_score": 0.10,
    "efficiency_score": 0.10,
}

METRIC_FILES = [
    "m1_answer_faithfulness.jsonl",
    "m2_stepwise_relevance.jsonl",
    "m3_causal_coherence.jsonl",
    "m4_completeness.jsonl",
    "m5_factual_accuracy.jsonl",
    "m6_efficiency.jsonl",
]


def load_all_metric_results(results_dir):
    """Load all metric files and merge by (video, question), keeping every field."""
    data = {}
    for filename in METRIC_FILES:
        path = os.path.join(results_dir, filename)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("_summary"):
                    continue
                vid = row.get("video", "")
                q = row.get("question", "")
                k = (vid, q)
                if k not in data:
                    data[k] = {"video": vid, "question": q}
                for key, val in row.items():
                    if key not in ("video", "question", "_summary"):
                        data[k][key] = val
    return list(data.values())


def compute_oqs(row):
    weighted_sum = 0.0
    for key, w in OQS_WEIGHTS.items():
        val = row.get(key)
        if val is None:
            val = 0.5
        elif isinstance(val, bool):
            val = 1.0 if val else 0.0
        weighted_sum += w * float(val)
    return weighted_sum


def assign_tier(oqs):
    if oqs >= 0.85:
        return "A"
    if oqs >= 0.60:
        return "B"
    return "C"


def build_aggregated(rows):
    """Build aggregated summary from per-sample rows."""
    n = len(rows)
    if n == 0:
        return {}
    agg = {
        "total_samples": n,
        "average_oqs": round(sum(r["oqs_score"] for r in rows) / n, 4),
        "tier_counts": {},
        "accuracy": round(sum(1 for r in rows if r.get("is_correct")) / n, 4),
    }
    for r in rows:
        t = r.get("tier", "")
        agg["tier_counts"][t] = agg["tier_counts"].get(t, 0) + 1
    for key in ("af_score", "srs_score", "ccc_score", "cs_score", "fas_score", "efficiency_score"):
        vals = [r[key] for r in rows if key in r and r[key] is not None]
        if vals:
            agg[f"average_{key}"] = round(sum(vals) / len(vals), 4)
    if any("tgs_score" in r for r in rows):
        tgs_vals = [r["tgs_score"] for r in rows if "tgs_score" in r]
        if tgs_vals:
            agg["average_tgs_score"] = round(sum(tgs_vals) / len(tgs_vals), 4)
    return agg


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiment = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("EXPERIMENT", "full_cot")
    results_dir = os.path.join(script_dir, "results", experiment)
    per_sample_file = os.path.join(results_dir, "per_sample_detailed.jsonl")
    aggregated_file = os.path.join(results_dir, "aggregated.json")

    if not os.path.exists(results_dir):
        print(f"Results dir not found: {results_dir}")
        sys.exit(1)

    rows = load_all_metric_results(results_dir)
    if not rows:
        print(f"No metric results found in {results_dir}")
        sys.exit(1)

    for row in rows:
        oqs = compute_oqs(row)
        row["oqs_score"] = round(oqs, 4)
        row["tier"] = assign_tier(oqs)

    with open(per_sample_file, "w") as out:
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    agg = build_aggregated(rows)
    with open(aggregated_file, "w") as out:
        json.dump(agg, out, indent=2, ensure_ascii=False)

    n = len(rows)
    tier_counts = agg.get("tier_counts", {})
    print(f"Done. Per-sample: {per_sample_file}, Aggregated: {aggregated_file}")
    print(f"Overall: avg_oqs={agg.get('average_oqs', 0):.4f}, n={n}, tiers={tier_counts}")


if __name__ == "__main__":
    main()
