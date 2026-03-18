"""
compare_robustness.py

Compares per-metric scores across all robustness perturbation experiments
against the combined_20 baseline. Prints a delta table and marks each cell
PASS or FAIL based on the expected direction of change.

Output: results/robustness_summary.txt  (also printed to stdout)

Usage:
    python compare_robustness.py
"""

import json
import os

METRICS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(METRICS_DIR, "results")
BASELINE_EXPERIMENT = "combined_20"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "robustness_summary.txt")

# Ordered list of metric keys as they appear in aggregated.json
METRIC_KEYS = [
    "average_af_score",
    "average_srs_score",
    "average_ccc_score",
    "average_cs_score",
    "average_fas_score",
    "average_tgs_score",
    "average_efficiency_score",
    "average_oqs",
]

METRIC_LABELS = {
    "average_af_score":       "M1_AF",
    "average_srs_score":      "M2_SRS",
    "average_ccc_score":      "M3_CCC",
    "average_cs_score":       "M4_CS",
    "average_fas_score":      "M5_FAS",
    "average_tgs_score":      "M5_TGS",
    "average_efficiency_score": "M6_EFF",
    "average_oqs":            "OQS",
}

# Expected direction per perturbation per metric.
# "down"  → score should decrease from baseline
# "up"    → score should increase (adversarial test)
# "flat"  → no meaningful change expected
# "edge"  → known edge-case default (e.g. M3/M6 = 1.0 for single step); treated as PASS
EXPECTED = {
    #                          M1_AF    M2_SRS   M3_CCC   M4_CS    M5_FAS   M5_TGS   M6_EFF   OQS
    "perturb_shuffled":         ["down",  "flat",  "down",  "flat",  "flat",  "flat",  "flat",  "down"],
    "perturb_duplicate_steps":  ["flat",  "flat",  "flat",  "flat",  "flat",  "flat",  "down",  "down"],
    "perturb_corrupt_final":    ["down",  "flat",  "down",  "flat",  "flat",  "flat",  "flat",  "down"],
    "perturb_invalid_timestamps":["flat", "flat",  "flat",  "flat",  "down",  "down",  "flat",  "down"],
    "perturb_inject_irrelevant":["down",  "down",  "down",  "flat",  "flat",  "flat",  "flat",  "down"],
    "perturb_single_step":      ["down",  "flat",  "edge",  "down",  "flat",  "flat",  "edge",  "down"],
    "perturb_wrong_answer":     ["down",  "flat",  "flat",  "down",  "flat",  "flat",  "flat",  "down"],
    "perturb_generic_evidence": ["down",  "down",  "down",  "down",  "down",  "flat",  "flat",  "down"],
    "perturb_contradiction":    ["down",  "flat",  "down",  "flat",  "flat",  "flat",  "flat",  "down"],
    "perturb_keyword_stuffing": ["flat",  "up",    "flat",  "flat",  "flat",  "flat",  "flat",  "flat"],
}

PERTURBATION_ORDER = [
    "perturb_shuffled",
    "perturb_duplicate_steps",
    "perturb_corrupt_final",
    "perturb_invalid_timestamps",
    "perturb_inject_irrelevant",
    "perturb_single_step",
    "perturb_wrong_answer",
    "perturb_generic_evidence",
    "perturb_contradiction",
    "perturb_keyword_stuffing",
]

# Threshold below which a delta is considered meaningful
DELTA_THRESHOLD = 0.02


def load_aggregated(experiment):
    path = os.path.join(RESULTS_DIR, experiment, "aggregated.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def evaluate_direction(delta, expected):
    """Return 'PASS', 'FAIL', or 'EDGE' for one cell."""
    if expected == "edge":
        return "EDGE"
    if expected == "flat":
        # Allow small movements in either direction
        return "PASS" if abs(delta) <= 0.10 else "FAIL"
    if expected == "down":
        return "PASS" if delta <= DELTA_THRESHOLD else "FAIL"
    if expected == "up":
        return "PASS" if delta >= -DELTA_THRESHOLD else "FAIL"
    return "????"


def fmt_delta(delta):
    if delta is None:
        return "  N/A  "
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:+.3f}"


def fmt_verdict(verdict):
    return f"[{verdict}]"


def build_table(baseline, experiments):
    col_labels = [METRIC_LABELS[k] for k in METRIC_KEYS]
    col_w = 14
    name_w = 30

    header = f"{'Perturbation':<{name_w}}" + "".join(f"{lbl:^{col_w}}" for lbl in col_labels)
    sep = "-" * len(header)

    lines = []
    lines.append(sep)
    lines.append("ROBUSTNESS SUMMARY: delta = (perturb - baseline)")
    lines.append(f"Baseline experiment : {BASELINE_EXPERIMENT}")
    lines.append(f"Delta threshold     : ±{DELTA_THRESHOLD}")
    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    # Baseline row
    baseline_row = f"{'BASELINE':<{name_w}}"
    for k in METRIC_KEYS:
        v = baseline.get(k)
        cell = f"{v:.3f}" if v is not None else "N/A"
        baseline_row += f"{cell:^{col_w}}"
    lines.append(baseline_row)
    lines.append(sep)

    total_cells = 0
    pass_cells = 0
    fail_cells = 0
    edge_cells = 0

    for exp_name in PERTURBATION_ORDER:
        data = experiments.get(exp_name)
        short_name = exp_name.replace("perturb_", "")
        if data is None:
            lines.append(f"{short_name:<{name_w}}{'(results not found)':^{col_w * len(METRIC_KEYS)}}")
            continue

        expected_row = EXPECTED.get(exp_name, ["flat"] * len(METRIC_KEYS))
        row = f"{short_name:<{name_w}}"
        for i, k in enumerate(METRIC_KEYS):
            b_val = baseline.get(k)
            p_val = data.get(k)
            if b_val is None or p_val is None:
                cell = f"{'N/A':^{col_w}}"
            else:
                delta = p_val - b_val
                exp_dir = expected_row[i] if i < len(expected_row) else "flat"
                verdict = evaluate_direction(delta, exp_dir)
                total_cells += 1
                if verdict == "PASS":
                    pass_cells += 1
                elif verdict == "FAIL":
                    fail_cells += 1
                else:
                    edge_cells += 1
                cell_str = f"{fmt_delta(delta)} {fmt_verdict(verdict)}"
                cell = f"{cell_str:^{col_w}}"
            row += cell
        lines.append(row)

    lines.append(sep)

    # Summary counts
    total_scored = total_cells
    lines.append(f"PASS: {pass_cells}/{total_scored}  |  FAIL: {fail_cells}/{total_scored}  |  EDGE (expected default): {edge_cells}")
    lines.append("")
    lines.append("Legend:")
    lines.append("  PASS  — delta direction matches expectation")
    lines.append("  FAIL  — delta direction is opposite to expectation (metric robustness issue)")
    lines.append("  EDGE  — known edge-case default value (not counted in PASS/FAIL)")
    lines.append("  flat  — expected delta within ±0.10")
    lines.append(sep)

    return "\n".join(lines)


def print_metric_notes(baseline, experiments):
    """Print targeted observations for known adversarial tests."""
    lines = []
    lines.append("\nTARGETED OBSERVATIONS")
    lines.append("-" * 60)

    # P10: keyword stuffing — check if M2 was gamed
    ks_data = experiments.get("perturb_keyword_stuffing")
    if ks_data and baseline.get("average_srs_score") is not None:
        delta = ks_data["average_srs_score"] - baseline["average_srs_score"]
        if delta > 0.10:
            lines.append(f"[WARNING] P10 keyword_stuffing raised M2_SRS by {delta:+.3f}.")
            lines.append("          M2 relevance prompt may be gameable by keyword repetition.")
            lines.append("          Consider adding adversarial negative examples to the system prompt.")
        else:
            lines.append(f"[OK] P10 keyword_stuffing M2_SRS delta = {delta:+.3f} (not significantly gamed).")

    # P2: duplicate steps — M6 should be near 0
    dup_data = experiments.get("perturb_duplicate_steps")
    if dup_data and dup_data.get("average_efficiency_score") is not None:
        eff = dup_data["average_efficiency_score"]
        if eff > 0.15:
            lines.append(f"[WARNING] P2 duplicate_steps M6_EFF = {eff:.3f} (expected near 0).")
            lines.append("          Cosine similarity threshold (0.92) may be too strict.")
        else:
            lines.append(f"[OK] P2 duplicate_steps M6_EFF = {eff:.3f} (correctly near 0).")

    # P6: single_step — M3 and M6 should default to 1.0
    ss_data = experiments.get("perturb_single_step")
    if ss_data:
        ccc = ss_data.get("average_ccc_score")
        eff = ss_data.get("average_efficiency_score")
        if ccc is not None:
            note = "PASS (expected default=1.0)" if abs(ccc - 1.0) < 0.05 else f"UNEXPECTED value={ccc:.3f}"
            lines.append(f"[P6] single_step M3_CCC = {ccc:.3f}  → {note}")
        if eff is not None:
            note = "PASS (expected default=1.0)" if abs(eff - 1.0) < 0.05 else f"UNEXPECTED value={eff:.3f}"
            lines.append(f"[P6] single_step M6_EFF = {eff:.3f}  → {note}")

    return "\n".join(lines)


def main():
    baseline = load_aggregated(BASELINE_EXPERIMENT)
    if baseline is None:
        print(f"ERROR: Baseline aggregated.json not found at results/{BASELINE_EXPERIMENT}/aggregated.json")
        print("Run compute_oqs.py combined_20 first.")
        return

    experiments = {}
    missing = []
    for exp_name in PERTURBATION_ORDER:
        data = load_aggregated(exp_name)
        experiments[exp_name] = data
        if data is None:
            missing.append(exp_name)

    if missing:
        print(f"WARNING: Results not found for: {', '.join(missing)}")
        print("These experiments will show as '(results not found)' in the table.\n")

    table = build_table(baseline, experiments)
    notes = print_metric_notes(baseline, experiments)
    full_output = table + notes + "\n"

    print(full_output)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write(full_output)
    print(f"\nSummary saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
