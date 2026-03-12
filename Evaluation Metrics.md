# Process-Aware Evaluation Metrics for Multimodal Reasoning

## Overview

Traditional video reasoning benchmarks evaluate models solely on answer correctness. This creates a critical problem: **answer hacking** — models can guess the right answer without genuine reasoning, or produce sloppy reasoning that accidentally reaches the correct conclusion.

This evaluation framework introduces **process metrics** that measure the *quality of the reasoning pathway* independently from the final answer. Combined with answer correctness, these metrics reveal whether a model truly understands or is merely shortcutting.

---

## Part 1: Individual Step-Level Metrics

These metrics evaluate a reasoning trace **step-by-step**, before considering the final answer.

### Metric 1: Answer Faithfulness

**Purpose**: Can the reasoning trace alone determine the correct answer?

**Procedure**:
1. Hide the correct answer from view
2. Give a language model only: [Question] + [Reasoning Trace]
3. Ask the model to predict the answer (without seeing the ground truth)
4. Compare predicted answer to ground truth
5. Compute similarity using BERTScore (continuous, 0-1) or exact match (for multiple choice)

**Formula**:
```
Answer Faithfulness = Similarity(Predicted_Answer, Ground_Truth_Answer)
```

**What it catches**: 
- Incomplete traces that omit critical information
- Traces that contradict themselves (model cannot extract consistent answer)
- Vague or ambiguous reasoning that leads to incorrect predictions

**Example**:
- **Good trace**: "The person picks up a red ball at 0:05s. Then throws it at 0:15s. Therefore, the answer is: thrown."
  - Answer Faithfulness = 1.0 (model correctly predicts "thrown" from the trace alone)
- **Bad trace**: "At some point, there is a ball and a person."
  - Answer Faithfulness = 0.2 (model cannot reliably infer what happened)

---

### Metric 2: Stepwise Relevance Score

**Purpose**: Does each reasoning step actually contribute to answering the question?

**Procedure**:
1. For each step in the trace, ask: "Is this step relevant to answering the question?"
2. Use a binary or continuous (0-1) judgment
3. Average scores across all steps

**Formula**:
```
Stepwise Relevance = (1 / NumSteps) × Σ Relevance(step_i)
```

**What it catches**:
- Irrelevant tangents or unnecessary details
- Verbose but non-contributory observations
- Circular or repetitive reasoning

**Example**:
- Trace: "The video is 2 minutes long. It has background music. At 0:30s, a person walks. They pick up a ball at 0:45s."
- Step relevance scores: [0.1 (irrelevant), 0.2 (weakly relevant), 0.9 (relevant), 1.0 (relevant)]
- Stepwise Relevance = (0.1 + 0.2 + 0.9 + 1.0) / 4 = 0.55

---

### Metric 3: Causal Chain Coherence

**Purpose**: Does each step logically follow from the previous ones?

**Procedure**:
1. For each step (starting from step 2), ask: "Given {step_1, ..., step_{i-1}} and the question, is step_i a valid logical consequence?"
2. Judge as: 0 (invalid), 0.5 (partially valid), 1.0 (fully valid)
3. Average across all steps

**Formula**:
```
Causal Chain Coherence = (1 / (NumSteps - 1)) × Σ_(i=2)^n Coherence(step_i | steps_1...i-1)
```

**What it catches**:
- Logical gaps (non-sequiturs)
- Jumps in reasoning without justification
- Contradictions (step_i contradicts earlier steps)

**Critical insight**: Low coherence at position _i_ identifies where the reasoning breaks down — that's a _gap location_ for trace completion.

**Example**:
- Frame 1 (step_1): "The video shows a kitchen."
- Frame 2 (step_2): "There is a person in the kitchen." → Coherence = 1.0 (valid from step_1)
- Frame 3 (step_3): "The robot is performing surgery." → Coherence = 0.0 (invalid jump from kitchen to surgery)
- Frame 4 (step_4): "Therefore, the kitchen needs a new stove." → Coherence = 0.3 (partially valid, assumes step_3 was correct, but step_3 is incoherent)

Causal Chain Coherence = (1.0 + 0.0 + 0.3) / 3 = 0.43

---

### Metric 4: Completeness Score

**Purpose**: Does the trace cover all necessary reasoning steps?

**Procedure**:
1. Generate a "reference skeleton" using the strongest available model (e.g., GPT-4o) given only [Question, Video, Ground-Truth Answer]. This skeleton is a list of 3-5 sub-goals that *must* be addressed to reach the answer.
2. For each sub-goal in the skeleton, check if the trace semantically covers it:
   - Embed both the sub-goal and each step in the trace
   - Compute cosine similarity
   - Mark sub-goal as "covered" if max similarity > 0.8
3. Completeness = (Number of covered sub-goals) / (Total sub-goals)

**Formula**:
```
Completeness = |{covered_subgoals}| / |{required_subgoals}|
```

**What it catches**:
- Traces that skip critical reasoning steps
- "Cheating" answers that happen to be correct but skip intermediate reasoning

**Example**:
- Question: "Why does the person look angry at the end of the video?"
- Reference skeleton (from GPT-4o):
  1. Identify what happens to the person's belongings
  2. Recognize the person's facial expression changes
  3. Link the event to the facial expression
  4. Infer the emotional cause
- Submitted trace only covers: [step 1, step 2, step 4]
- Completeness = 3/4 = 0.75 (skipped the linking step)

---

### Metric 5: Factual Accuracy per Step

**Purpose**: Are the visual/audio claims in each step actually true?

**Procedure**:
1. For each step, extract factual claims (e.g., "Person is wearing red shirt", "Music changes at 1:23s")
2. For visual claims: show relevant video frames to a vision model or human; ask: "Is this claim true?" → Boolean judgment
3. For audio claims: transcribe audio or use audio classification; check claim accuracy
4. For temporal claims: verify timestamps against actual video events
5. Average accuracy across all steps

**Formula**:
```
Factual Accuracy = (1 / NumSteps) × Σ Accuracy(step_i)
```

**What it catches**:
- Hallucinations (claims about things that don't exist in the video/audio)
- Misidentifications (wrong objects, wrong people, wrong sounds)
- Temporal mistakes (claiming an event at 0:10s when it's actually at 0:20s)

**Example**:
- Step 1: "At 0:05s, the person picks up a red ball." → Check video frame at 0:05s → Actually a blue ball → Factual Accuracy of step 1 = 0
- Step 2: "At 0:15s, the person throws the ball." → Check video frame at 0:15s → Correct → Factual Accuracy of step 2 = 1.0
- Step 3: "There is background music playing." → Check audio → Yes, there is → Factual Accuracy of step 3 = 1.0
- Factual Accuracy = (0 + 1.0 + 1.0) / 3 = 0.67

---

### Metric 6: Efficiency / Redundancy Score

**Purpose**: Does the trace avoid unnecessary repetition or redundancy?

**Procedure**:
1. Embed all steps in the trace into a vector space
2. Compute pairwise cosine similarity between steps
3. Count "redundant pairs": pairs where similarity > 0.92 (near-identical content)
4. Compute redundancy ratio = (redundant pairs) / (total possible pairs)
5. Efficiency score = 1 - redundancy ratio

**Formula**:
```
RedundantPairs = |{(i,j) : i < j, cosine_similarity(step_i, step_j) > 0.92}|
Efficiency Score = 1 - (RedundantPairs / C(NumSteps, 2))
```

**What it catches**:
- Circular reasoning (same point stated multiple times)
- Inefficient traces (could be shorter without losing information)
- Redundant evidence gathering

**Example**:
- Trace with 4 steps: [A, B, C, A_again] (A is repeated)
- Similarity pairs:
  - (A, B) = 0.3, (A, C) = 0.4, (A, A_again) = 0.98 ← redundant!
  - (B, C) = 0.5, (B, A_again) = 0.2
  - (C, A_again) = 0.4
- 1 redundant pair out of 6 total pairs
- Efficiency Score = 1 - (1/6) = 0.83

---

### Metric 7: Planning Presence Score

**Purpose**: Does the trace begin with an explicit plan?

**Procedure**:
1. Ask a binary classifier: "Does the trace start with an explicit multi-step plan before executing reasoning?"
2. Example of "yes": "First, I will identify what the person is doing. Then, I will look at their facial expression. Finally, I will infer their emotion."
3. Example of "no": "The person is walking. They look sad. So the answer is that they are unhappy."
4. Score = 1 (has plan) or 0 (no plan)

**Formula**:
```
Planning Presence = 1 if {explicit plan present at start} else 0
```

**What it catches**:
- Models that reason in an ad-hoc manner vs. with structured intent
- Systems that lack meta-cognitive planning

**Note**: Planning Presence is *tracked separately* and not folded into the composite score. It's a binary property.

---

## Part 2: Overall Quality Score

**Purpose**: Combine all step-level metrics into a single overall quality score.

**Formula**:
```
Overall Quality Score =
    0.25 × Answer Faithfulness
  + 0.15 × Stepwise Relevance
  + 0.20 × Causal Chain Coherence
  + 0.20 × Completeness
  + 0.10 × Factual Accuracy
  + 0.10 × Efficiency Score
```

**Weights breakdown** (chosen to reflect importance for video reasoning):
- Answer Faithfulness (25%): More weight because if the trace doesn't determine the answer, it fails the core purpose
- Causal Chain Coherence (20%): Logical coherence is critical for genuine reasoning
- Completeness (20%): Completeness prevents shortcuts
- Stepwise Relevance (15%): Relevance filters noise
- Factual Accuracy (10%): Factual accuracy is necessary but can be partially masked by strong faithfulness
- Efficiency Score (10%): Efficiency is nice-to-have, not critical

**All metrics are on 0-1 scale**, so Overall Quality Score ∈ [0, 1].

**Interpretation**:
- Overall Quality Score ≥ 0.85 → **Tier A (Accept)**: Near-perfect reasoning trace
- 0.60 ≤ Overall Quality Score < 0.85 → **Tier B (Needs Completion)**: Good structure but missing a detail
- Overall Quality Score < 0.60 → **Tier C (Regenerate)**: Fundamentally flawed reasoning

---

## Part 3: 2×2 Evaluation Matrix

**Purpose**: Separate genuine reasoning from shortcuts by comparing reasoning quality with answer correctness.

### Setup

**Binary Signal 1: Correct Reasoning Process?**
```
Reasoning_Correct = True if Overall_Quality_Score(predicted_trace) ≥ 0.75, else False
```

**Binary Signal 2: Correct Answer?**
```
Answer_Correct = True if Similarity(predicted_answer, ground_truth) ≥ threshold, else False
(For MCQ: exact match. For open-ended: semantic similarity ≥ 0.8)
```

### The 2×2 Matrix

```
                          │ Correct Answer           │ Incorrect Answer
──────────────────────────┼──────────────────────────┼───────────────────────────────
Correct Reasoning         │ ★ Genuine Reasoning      │ Reasoning-Execution Gap
(Quality Score ≥ 0.75)    │                          │
──────────────────────────┼──────────────────────────┼───────────────────────────────
Incorrect Reasoning       │ Answer Hacking           │ Full Failure
(Quality Score < 0.75)    │                          │
```

### Four Outcomes Explained

#### Genuine Reasoning ★

- **Definition**: Model reasons correctly AND reaches the right answer
- **Interpretation**: This is the *gold standard*. Model demonstrates real understanding.
- **Example**: Model correctly identifies objects, places, temporal relationships, and concludes with the right answer

#### Answer Hacking

- **Definition**: Model reaches the right answer BUT the reasoning process is flawed
- **Interpretation**: Model got lucky, memorized, or used a shortcut. Unreliable on distribution shift.
- **Example**: Model says "I don't see clear evidence, but the answer is B" and happens to be right (maybe B is always common in this dataset)

#### Reasoning-Execution Gap

- **Definition**: Model reasons correctly BUT reaches the wrong answer
- **Interpretation**: Strong reasoning process but fails at the final step (calculation error, retrieval failure, integration error)
- **Example**: Model correctly identifies all evidence but misinterprets the question or makes an arithmetic mistake

#### Full Failure

- **Definition**: Model's reasoning is flawed AND the answer is wrong
- **Interpretation**: No understanding demonstrated. Model should not be trusted.
- **Example**: Model hallucinates objects, produces incoherent reasoning, and gets the answer wrong

### Model-Level Summary Metrics

```
Genuine Reasoning Rate = Count(Genuine Reasoning cases) / Total_Samples
→ Primary signal of true reasoning ability

Answer Hacking Rate = Count(Answer Hacking cases) / Total_Samples  
→ Should be minimized. High rate = model is gaming the benchmark

Reasoning-Execution Gap Rate = Count(Gap cases) / Total_Samples
→ Indicates weak link in final execution (calculation? retrieval?)

Process Consistency = (Genuine Reasoning + Full Failure) / Total_Samples
→ How often does process match outcome? (both are "consistent")
→ High consistency = model either reliably reasons or reliably fails
→ Low consistency = model is unreliable/random
```

**Interpretation at Model Level**:
- High accuracy, high Answer Hacking Rate → **Model is shortcutting**. Not generalizable.
- High Genuine Reasoning Rate, moderate accuracy → **Model reasons well but hits execution limits**. More valuable than high-accuracy shortcutters.
- High Answer Hacking Rate on audio-involved questions, Low on visual-only → **Model doesn't understand audio grounding**. Gap identified.

---

## Part 4: Trajectory Similarity Score

**Purpose**: Measure how similar the *predicted reasoning path* is to a *reference reasoning path*.

### Setup

Given:
- **Reference trace**: T_ref = {step_1^ref, step_2^ref, ..., step_n^ref}
- **Predicted trace**: T_pred = {step_1^pred, step_2^pred, ..., step_m^pred}

### Procedure

1. **Embed all steps**:
   ```
   E_ref = {embed(step) for step in T_ref}
   E_pred = {embed(step) for step in T_pred}
   ```

2. **Compute recall-oriented similarity**: For each reference step, find the best match in predicted trace:
   ```
   For each step_ref in T_ref:
       best_match = max_{step_pred in T_pred} cosine_sim(embed(step_ref), embed(step_pred))
       match_scores.append(best_match)
   
   Recall = mean(match_scores)  # Did model predict all required steps?
   ```

3. **Compute precision-oriented similarity**: For each predicted step, find best match in reference:
   ```
   For each step_pred in T_pred:
       best_match = max_{step_ref in T_ref} cosine_sim(embed(step_pred), embed(step_ref))
       match_scores.append(best_match)
   
   Precision = mean(match_scores)  # Are model's steps on-topic?
   ```

4. **Harmonic mean (F1-style)**:
   ```
   Trajectory Similarity (F1) = 2 × (Recall × Precision) / (Recall + Precision)
   ```

### Interpretation

- **Score = 1.0**: Perfect match between predicted and reference reasoning paths
- **Score = 0.8**: Model covers most required steps, minimal hallucination
- **Score = 0.5**: Significant divergence; model misses required steps or adds irrelevant ones
- **Score < 0.3**: Reasoning paths bear little resemblance

### Example

Reference trace:
1. "Person walks to the shelf"
2. "Person picks up a blue box"
3. "Person opens the box"
4. "Box contains a letter"

Predicted trace:
1. "A person is moving through the room"
2. "They pick up a blue object"
3. "The object is opened"
4. "Inside is a piece of paper"
5. "The person looks happy"

Embedding similarities:
- Ref step 1 → best match in Pred: "A person is moving" (sim=0.85)
- Ref step 2 → best match in Pred: "They pick up a blue object" (sim=0.92)
- Ref step 3 → best match in Pred: "The object is opened" (sim=0.88)
- Ref step 4 → best match in Pred: "Inside is a piece of paper" (sim=0.78)
- Recall = (0.85 + 0.92 + 0.88 + 0.78) / 4 = 0.86

- Pred step 1 → best match in Ref: step 1 (sim=0.85)
- Pred step 2 → best match in Ref: step 2 (sim=0.92)
- Pred step 3 → best match in Ref: step 3 (sim=0.88)
- Pred step 4 → best match in Ref: step 4 (sim=0.78)
- Pred step 5 → best match in Ref: step 4 (sim=0.25, low because "happy" doesn't match any ref step)
- Precision = (0.85 + 0.92 + 0.88 + 0.78 + 0.25) / 5 = 0.74

Trajectory Similarity (F1) = 2 × (0.86 × 0.74) / (0.86 + 0.74) = 2 × 0.6364 / 1.6 = 0.796 ≈ 0.80

---

## Part 5: Process-Outcome Score

**Purpose**: Combine reasoning quality and answer correctness into a single metric that values good process.

**Formula**:
```
Process-Outcome Score = α × Trajectory Similarity + (1 - α) × Answer Correctness
```

Where:
- **Trajectory Similarity**: The F1-style trajectory similarity score (0-1)
- **Answer Correctness**: 1 if answer is correct, 0 if incorrect
- **α**: Weighting parameter (default = 0.5)

**Interpretation**:

By default (α=0.5):
- **Process-Outcome Score = 1.0**: Perfect reasoning path AND correct answer
- **Process-Outcome Score = 0.75**: Either perfect reasoning with wrong answer, OR poor reasoning with correct answer
- **Process-Outcome Score = 0.50**: Significant issues in either process or outcome
- **Process-Outcome Score = 0.0**: Completely failed reasoning AND wrong answer

**Flexibility**: Adjust α based on use case:
- **α = 0.7** (process > outcome): For interpretability-focused tasks (medical imaging, safety-critical)
- **α = 0.3** (outcome > process): For pure performance tasks where only answer matters
- **α = 0.5** (balanced): For research on reasoning quality

---

## Part 6: End-to-End Evaluation Pipeline

### Stage 1: Fast Pre-filter
```
For each sample:
    Run Answer Faithfulness metric
    If Answer Faithfulness = 0 (trace cannot derive answer):
        → REJECT sample (reasoning is fundamentally broken)
```

### Stage 2: Step-Level Scoring
```
For each sample:
    For each step in trace:
        Compute: Stepwise Relevance, Causal Chain Coherence,
                 Factual Accuracy (via batched LLM call)
        Store step-level scores
```

### Stage 3: Global Completeness
```
For each sample:
    Generate reference skeleton (GPT-4o: Q + Video → required steps)
    Compute Completeness (coverage of required steps)
```

### Stage 4: Compute Composite Scores
```
For each sample:
    Overall Quality Score =
        0.25 × Answer Faithfulness
      + 0.15 × Stepwise Relevance
      + 0.20 × Causal Chain Coherence
      + 0.20 × Completeness
      + 0.10 × Factual Accuracy
      + 0.10 × Efficiency Score
    
    Assign to Tier:
        Tier A: Overall Quality Score ≥ 0.85 (accept as-is)
        Tier B: 0.60 ≤ Overall Quality Score < 0.85 (needs trace completion)
        Tier C: Overall Quality Score < 0.60 (needs full regeneration)
```

### Stage 5: Human Spot-Check
```
Randomly sample 5% from each tier
Have humans rate same samples using same metrics
Compute Cohen's Kappa (inter-rater agreement)
Target: κ > 0.70
```

### Stage 6: 2×2 Matrix Assignment
```
For each sample:
    Reasoning Correct = (Overall Quality Score ≥ 0.75)?
    Answer Correct   = (Answer Similarity ≥ threshold)?
    
    Assign to: Genuine Reasoning, Answer Hacking,
               Reasoning-Execution Gap, or Full Failure
    
Aggregate per model:
    Genuine Reasoning Rate, Answer Hacking Rate,
    Reasoning-Execution Gap Rate, Full Failure Rate
    Process Consistency
```

### Stage 7: Cross-Modal Analysis (for audio-visual tasks)
```
Of samples with audio-critical reasoning:
    Answer Hacking Rate (with audio) vs. Answer Hacking Rate (without audio)
    → If 2x worse with audio, model doesn't understand multimodal grounding
```

---

## Summary Table: What Each Metric Measures

| Metric | Measures | Good Range | Catches |
|--------|----------|------------|---------|
| **Answer Faithfulness** | Can the trace determine the answer? | ≥ 0.9 | Incomplete traces, contradictions |
| **Stepwise Relevance** | Are all steps relevant? | ≥ 0.8 | Irrelevant tangents, verbosity |
| **Causal Chain Coherence** | Do steps follow logically? | ≥ 0.75 | Logical gaps, non-sequiturs |
| **Completeness** | Does trace cover all sub-goals? | ≥ 0.85 | Skipped reasoning steps |
| **Factual Accuracy** | Are factual claims true? | ≥ 0.9 | Hallucinations, misidentifications |
| **Efficiency Score** | Is the trace efficient? | ≥ 0.85 | Redundancy, circular reasoning |
| **Planning Presence** | Does trace begin with a plan? | Binary | Lack of meta-cognitive structure |
| **Overall Quality Score** | Overall trace quality | ≥ 0.85 (Tier A) | Combined reasoning quality |
| **Trajectory Similarity** | Reasoning path similarity | ≥ 0.8 | Deviations from expected logic |
| **Process-Outcome Score** | Process + outcome balance | Custom by α | Task-specific quality |
| **2×2 Matrix** | Shortcutting detection | High Genuine %, Low Hacking % | Answer hacking, unreliable models |

---

## Implementation Notes

### Cost Considerations
- **Per sample**: ~5-10 API calls to GPT-4o (for Completeness, Causal Chain Coherence, Trajectory Similarity, and other human-level judgments)
- **For 1000-sample benchmark**: ~7,000 API calls ≈ $100-200 (depending on model pricing)
- **Optimization**: Use open-source embedding models (e.g., sentence-transformers) for embedding; reserve GPT-4o for semantic judgments

### Quality Assurance
- Always compute inter-rater agreement on a holdout set (human raters vs. LLM-as-judge)
- For video frames: use multiple vision models (CLIP + LLaVA + GPT-4V) to verify factual claims
- For audio: use multiple ASR systems (Whisper + Google Speech-to-Text) to catch errors

### Expected Distributions
On a well-curated benchmark:
- **Genuine Reasoning Rate**: 50-70% (depends on model strength)
- **Answer Hacking Rate**: 5-15% (indicates dataset is hard enough)
- **Reasoning-Execution Gap Rate**: 5-10% (execution failures, not reasoning)
- **Full Failure Rate**: 10-20% (models out of their depth)

If Answer Hacking Rate > 20%, the benchmark may be too easy (shortcuts are too rewarding).
If Reasoning-Execution Gap Rate > 15%, the benchmark may have perception bottlenecks (correct reasoning but failing due to visual parsing).

---

## References to Foundational Work

- Inspired by Process Reward Models in reasoning literature
- Builds on step-wise evaluation from math reasoning (e.g., Ape, MetaMath)
- Extends to multimodal domain (video + audio)
- Draws from Causal Chain Coherence literature in NLG evaluation
- 2×2 matrix concept from classification metrics (precision/recall, but applied to reasoning)
