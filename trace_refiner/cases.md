### Case 1: Trace correct, Answer correct

- **Verifier verdict:** PASS
- **Action:** Output as-is. No refinement needed.

### Case 2: Trace correct, Answer wrong

- **Verifier detection:** Logical consistency check finds trace supports a DIFFERENT answer than what's stated
- **Action:** Planner calls Video QA Re-answerer to independently answer. Refiner corrects the answer to match the trace's logical conclusion. Re-verify.

### Case 3: Trace wrong, Answer correct

This has several sub-cases:

#### 3a: Timestamps wrong, inference correct

- **Verifier detection:** Factual grounding check finds events at different times than claimed, but the logical reasoning is sound
- **Action:** Planner calls Temporal Grounder to re-localize events mentioned in each step. Frame Retriever extracts frames at corrected timestamps to confirm. Refiner patches timestamps in the trace.

#### 3b: Timestamps wrong, inference wrong

- **Verifier detection:** Both temporal and logical errors
- **Action:** Full re-grounding -- Temporal Grounder + Dense Captioner + relevant modality tools. Refiner rewrites affected steps.

#### 3c: Perception errors (wrong object identity, wrong count, wrong text)

- **Verifier detection:** Factual grounding check finds claimed observations don't match video
- **Action:** Planner dispatches Spatial Grounder / Counter / OCR as needed. Refiner corrects perceptual claims.

#### 3d: Incomplete trace (missing steps, insufficient evidence)

- **Verifier detection:** Completeness check; trace doesn't cover all reasoning needed for the answer
- **Action:** Planner identifies gaps. Calls Dense Captioner on full video or relevant segments. May also call ASR / Audio Grounder if audio evidence is missing. Refiner inserts new steps.

#### 3e: Wrong modality attribution

- **Verifier detection:** A step claims visual evidence for something that's actually from audio, or vice versa
- **Action:** ASR + Audio Grounder to check if claim is audio-sourced; Frame Retriever + Dense Captioner to check visual source. Refiner corrects modality tags.

### Case 4: Trace wrong, Answer wrong

- **Verifier detection:** Multiple error categories flagged
- **Action:** Full pipeline activation. Planner creates a comprehensive re-grounding plan. After tool execution, Refiner produces a substantially revised trace. Video QA Re-answerer independently derives the answer. Re-verify (potentially multiple iterations).

### Case 5: Ambiguous / Edge cases

#### 5a: Answer is among multiple valid interpretations

- **Action:** Verifier flags low confidence. Planner gathers exhaustive evidence. If multiple answers are defensible, flag for human review.

#### 5b: Question itself is flawed or ambiguous

- **Action:** Flag for human review with explanation.

#### 5c: Video quality issues (corrupted, missing audio, etc.)

- **Action:** Dense Captioner and ASR will return low-confidence or empty results. Pipeline flags as "unverifiable" with explanation.