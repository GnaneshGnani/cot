# Robustness Perturbations

This document describes the 10 perturbation types applied to reasoning traces for metric robustness testing. Each perturbation targets one or more metrics to verify they respond in the expected direction when quality is degraded.

**Source:** `combined_20_samples.json` (20 samples from OmniVideoBench + VideoMathQA)  
**Output:** JSONL/JSON files in `data/` (e.g., `data/shuffled.jsonl`, `data/shuffled.json`)

---

## 1. Shuffled

**What it does:** Randomly reorders all `reasoning_steps` within each sample. Uses a fixed random seed (42) for reproducibility.

**Implementation:** For each sample, if there are ≥ 2 steps, `random.shuffle(steps)` is applied. The logical/causal order is destroyed.

**Primary target:** M3 Causal Chain Coherence (CCC) — the chain should break because later steps no longer follow from earlier ones.

**Secondary target:** M1 Answer Faithfulness (AF) — the model may still derive the answer from scrambled steps, but derivation order is wrong.

**Expected direction:** M3 CCC ↓↓, M1 AF ↓, others ≈

---

## 2. Duplicate Steps

**What it does:** Replaces each reasoning step with the step followed by an identical copy of itself. Every step appears twice in a row.

**Implementation:** For each step `S`, the sequence becomes `[S, copy(S), S', copy(S'), ...]`. Number of steps doubles.

**Primary target:** M6 Efficiency (EFF) — duplicate pairs have cosine similarity = 1.0 (exceeds 0.92 threshold), so the score should drop sharply.

**Expected direction:** M6 EFF ↓↓, OQS ↓, others ≈

---

## 3. Corrupt Final

**What it does:** Replaces the `inference` field of the **last** reasoning step with a generic non-committal string: *"Therefore, the answer cannot be determined from the available evidence."*

**Implementation:** Only the final step’s inference is modified. All other steps and evidence remain unchanged.

**Primary target:** M1 Answer Faithfulness (AF) — the trace no longer supports deriving the correct answer because the conclusion is removed.

**Secondary target:** M3 CCC — the last causal link is broken.

**Expected direction:** M1 AF ↓↓, M3 CCC ↓, others ≈

---

## 4. Invalid Timestamps

**What it does:** Multiplies all timestamps in `evidence` fields by 10. Timestamps in the form `M:SS` or `H:MM:SS` are inflated so they exceed typical video durations.

**Implementation:** Regex matches `(\d+):(\d{1,2}):(\d{2})` or `(\d{1,2}):(\d{2})`, converts to total seconds, multiplies by 10, and reformats. Example: `0:05` → `0:50`, `1:30` → `15:00`.

**Primary target:** M5 Temporal Grounding Score (TGS) — timestamps fall outside video bounds.

**Secondary target:** M5 Factual Accuracy (FAS) — clips extracted at wrong offsets may not support the claims.

**Expected direction:** M5 TGS ↓↓, M5 FAS ↓, others ≈

---

## 5. Inject Irrelevant

**What it does:** Inserts two off-topic steps into each sample’s reasoning trace. The inserted steps are unrelated geography/chemistry facts: *"The capital of France is Paris"* and *"The boiling point of water is 100°C at sea level."*

**Implementation:** Two fixed irrelevant steps are inserted at positions 1 (after the first step) and near the end of the step list. Each has `modality: "text"`.

**Primary target:** M2 Stepwise Relevance Score (SRS) — irrelevant steps should be marked as not relevant to the question.

**Secondary target:** M3 CCC — logical flow is broken where irrelevant steps are inserted.

**Expected direction:** M2 SRS ↓↓, M3 CCC ↓, M1 AF ↓, others ≈

---

## 6. Single Step

**What it does:** Keeps only the **first** reasoning step for each sample. All other steps are discarded.

**Implementation:** `reasoning_steps = reasoning_steps[:1]`.

**Primary target:** M4 Completeness Score (CS) — one step cannot cover all required subgoals for the question.

**Note:** M3 CCC and M6 EFF default to 1.0 when there are &lt; 2 steps (edge case).

**Expected direction:** M4 CS ↓↓, M1 AF ↓, M3/M6 = 1.0 (edge), others ≈

---

## 7. Wrong Answer

**What it does:** Cycles the `correct_option` to the next option: A→B, B→C, C→D, D→A. The `answer` text is updated to match the new option. **All reasoning steps are unchanged** — they still support the original correct answer.

**Implementation:** For each sample, `correct_option` is cycled using the mapping. The matching option text from `options` is used to set `answer`.

**Primary target:** M1 AF — the model re-predicts from steps; steps point to the original answer, not the flipped one.

**Secondary target:** M4 CS — subgoals are generated for the wrong answer, so coverage of the trace may drop.

**Expected direction:** M1 AF ↓↓, M4 CS ↓, others ≈

---

## 8. Generic Evidence

**What it does:** Replaces every `evidence` field in every step with the same placeholder: *"Some visual content was observed in the video."* All `inference` fields are left intact.

**Implementation:** All keys `evidence`, `evidece`, `evience`, `evodence`, `nevidence` are set to the generic string.

**Primary target:** M2 SRS (vague evidence), M4 CS (low semantic content), M5 FAS (no concrete claims to verify against the clip).

**Expected direction:** M2 SRS ↓, M3 CCC ↓, M4 CS ↓↓, M5 FAS ↓↓, others ≈

---

## 9. Contradiction

**What it does:** Appends one extra step to each sample that explicitly contradicts the final answer. The appended step says, e.g., *"Upon further review, the answer is definitely not C based on the available evidence."* (where C is the correct option).

**Implementation:** A new step with empty evidence and a contradiction inference is appended. The step uses `modality: "text"`.

**Primary target:** M3 CCC — the last pair (real conclusion → contradiction) violates logical coherence.

**Secondary target:** M1 AF — the contradictory final step may confuse the model’s answer prediction.

**Expected direction:** M3 CCC ↓↓, M1 AF ↓, others ≈

---

## 10. Keyword Stuffing

**What it does:** Prepends the full question text to every `evidence` field. Example: if the question is *"What was the mood of the boy on the left?"*, each evidence becomes *"What was the mood of the boy on the left? [original evidence]"*.

**Implementation:** For each step, `evidence = question + " " + original_evidence`. This is an **adversarial** test to see if the relevance metric can be gamed by repeating the question.

**Primary target:** M2 SRS (adversarial) — ideally the score should stay flat; a large increase would indicate the metric is gameable.

**Expected direction:** M2 SRS ↑? (concerning if large), others ≈

---

## Summary Table

| Perturbation       | Primary Metric(s) | Expected Change                         |
|--------------------|-------------------|-----------------------------------------|
| shuffled           | M3 CCC            | M3 ↓↓, M1 ↓                             |
| duplicate_steps    | M6 EFF            | M6 ↓↓, OQS ↓                            |
| corrupt_final      | M1 AF             | M1 ↓↓, M3 ↓                             |
| invalid_timestamps | M5 TGS, M5 FAS    | M5 TGS ↓↓, M5 FAS ↓                      |
| inject_irrelevant  | M2 SRS            | M2 ↓↓, M3 ↓, M1 ↓                       |
| single_step        | M4 CS             | M4 ↓↓, M1 ↓, M3/M6 = 1.0 (edge)         |
| wrong_answer       | M1 AF, M4 CS      | M1 ↓↓, M4 ↓                             |
| generic_evidence   | M2, M4, M5 FAS    | M2 ↓, M3 ↓, M4 ↓↓, M5 FAS ↓↓            |
| contradiction     | M3 CCC            | M3 ↓↓, M1 ↓                             |
| keyword_stuffing   | M2 (adversarial)  | M2 flat or ↑ (adversarial check)        |

---

## Running Metrics on Each Perturbation

To run all metrics (M1–M6) and compute OQS for a perturbation:

```bash
cd /fs/nexus-scratch/gnanesh/cot/cot/metrics
export DATA_PATH="$PWD/data/<perturbation>.jsonl"   # e.g., shuffled.jsonl
export EXPERIMENT="perturb_<perturbation>"          # e.g., perturb_shuffled
export VIDEOS_DIR=""  # or path to OmniVideoBench videos

python m6_efficiency.py
python m1_answer_faithfulness.py
python m2_stepwise_relevance.py
python m3_causal_coherence.py
python m4_completeness.py
python m5_factual_accuracy.py
python compute_oqs.py perturb_<perturbation>
```

Results are written to `results/perturb_<perturbation>/`.

Use the SLURM array job for batch execution:

```bash
sbatch run_robustness.slurm
```

Then run the comparison:

```bash
python compare_robustness.py
```
