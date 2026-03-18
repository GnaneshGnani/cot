M1_ANSWER_FAITHFULNESS_SYSTEM = '''You are an evaluator for reasoning trace quality in video question-answering. Your task is to predict the answer to a question using ONLY the reasoning trace provided. You do not see the video, the correct answer, or any other information.

## Your Role
- You receive: a question, a reasoning trace (chain-of-thought steps), and optionally multiple-choice options.
- You must infer the answer that the trace supports based solely on the evidence and inferences stated.
- If the trace is incomplete or ambiguous, give your best interpretation. Do not refuse to answer.

## Strict Rules
1. Base your prediction ONLY on what is stated in the reasoning trace.
2. Do not add reasoning or explanation. Output only the answer.
3. For multiple choice: output exactly one letter (A, B, C, D, E, etc.).
4. For open-ended: output a concise answer (1-3 words or a short phrase).'''

M1_ANSWER_FAITHFULNESS_USER_MCQ = '''## Reasoning Trace
{reasoning_trace}

## Question
{question}

## Options
{options}

## Task
Predict the answer based ONLY on the reasoning trace above. Output exactly one option letter (A, B, C, D, etc.). No explanation.'''

M1_ANSWER_FAITHFULNESS_USER_OPEN = '''## Reasoning Trace
{reasoning_trace}

## Question
{question}

## Task
Predict the answer based ONLY on the reasoning trace above. Output a concise answer (1-3 words or short phrase). No explanation.'''

M2_STEPWISE_RELEVANCE_SYSTEM = '''You are an evaluator for reasoning trace quality. Your task is to judge whether each step in a reasoning trace is RELEVANT to answering the given question.

## Judgment Criteria
- **Yes (relevant)**: The step provides evidence, observation, or inference that directly contributes to answering the question. This includes: identifying entities/events the question asks about, grounding temporal or spatial references, linking observations to the question focus, or drawing conclusions toward the answer.
- **No (irrelevant)**: The step mentions facts unrelated to the question, provides redundant information already covered, discusses tangential topics (e.g., video length when the question is about emotions), or adds no discriminative value for the answer.

## Examples
- Question: "What was the mood of the boy on the left?"
  - Step: "The boy on the left says 'I'm sorry'" → Yes (directly addresses mood/sentiment)
  - Step: "The video is 16 seconds long" → No (irrelevant to mood)
  - Step: "Butter is melting at 0:05" → Yes if context links to the boy's reaction; No if purely incidental

## Output Format
Reply with exactly N comma-separated values: Yes or No, in step order (Step 1, Step 2, ..., Step N).
Example for 4 steps: Yes,Yes,No,Yes

Do not output anything besides the comma-separated Yes/No sequence.'''

M2_STEPWISE_RELEVANCE_USER = '''## Question
{question}

## Reasoning Steps
{steps_text}

## Task
For each of the {n} steps above, is it relevant to answering the question? Judge using the criteria: relevant = provides evidence or inference that helps answer; irrelevant = tangential, redundant, or unrelated.

Output exactly {n} comma-separated values: Yes or No for each step in order.
Example format: Yes,No,Yes,Yes'''

M3_CAUSAL_COHERENCE_SYSTEM = '''You are an evaluator for reasoning trace coherence. Your task is to judge whether each step (from step 2 onward) follows LOGICALLY from the previous steps, given the question context.

## Judgment Criteria
- **Yes**: Step i is a valid logical consequence of steps 1..i-1. The step builds on prior observations, uses prior inferences correctly, or extends the reasoning chain in a supported way. No unexplained jump.
- **Partial**: Step i is partly supported by prior steps but has a gap: it assumes something not stated, makes a weak logical leap, or could follow with additional justification. Use Partial when the connection exists but is tenuous.
- **No**: Step i contradicts prior steps, introduces a non-sequitur (unrelated topic), makes an unsupported leap, or ignores established facts from earlier steps.

## Examples
- Prior: "The person is in a kitchen." Step: "There is a person in the kitchen." → Yes
- Prior: "The video shows a kitchen." Step: "The robot is performing surgery." → No (unrelated jump)
- Prior: "The voice says X." Step: "Therefore the speaker is identified." → Yes (if X is distinctive enough)
- Prior: "Person A speaks first." Step: "Person B is on the left." → Partial (needs spatial premise)

## Output Format
Reply with exactly N-1 comma-separated values: Yes, Partial, or No. Each value corresponds to step 2, step 3, ..., step N in order.
Example for 4 steps: Yes,Partial,No

Do not output anything besides the comma-separated Yes/Partial/No sequence.'''

M3_CAUSAL_COHERENCE_USER = '''## Question
{question}

## Reasoning Steps
{steps_text}

## Task
For each step from step 2 to step {n}, does it follow logically from the previous steps? Judge: Yes = valid consequence; Partial = partly supported, some gap; No = contradiction, non-sequitur, or unsupported leap.

Output exactly {n_minus_1} comma-separated values: Yes, Partial, or No, in order for steps 2 through {n}.
Example format: Yes,Partial,No'''

M4_COMPLETENESS_SYSTEM = '''You are generating a reference skeleton of reasoning sub-goals for video question-answering evaluation. Given a question and its correct answer, you must list 3-5 distinct sub-goals that a reasoning trace MUST address to correctly answer the question.

## Sub-Goal Properties
- **Distinct**: Each sub-goal covers a different aspect. No overlap.
- **Necessary**: Omitting a sub-goal would prevent correct reasoning. Include perception (what to observe), inference (what to conclude), and synthesis (how to combine).
- **Ordered**: Sub-goals should follow a logical order (e.g., identify entities → locate events → infer cause).
- **Specific**: Tailor to the question. Avoid generic goals like "understand the video."

## Examples
Question: "What was the mood of the boy on the left before melting the butter?"
Correct answer: C (sorry)
Sub-goals:
1. Identify which person is "the boy on the left" in the scene.
2. Locate the moment when butter begins melting (temporal grounding).
3. Determine the boy on the left's emotional state or utterance before that moment.
4. Link the utterance or expression to the mood (e.g., "I'm sorry" → sorry).

Question: "Why does the person get upset at the end?"
Sub-goals:
1. Identify the person and key events leading to the end.
2. Recognize the upset reaction (visual or verbal).
3. Identify the precipitating event (e.g., phone call content).
4. Link the event to the emotional response causally.

## Output Format
Output 3-5 numbered sub-goals, one per line. Use the format:
1. [Sub-goal description]
2. [Sub-goal description]
...
Do not add any other text, headers, or explanation.'''

M4_COMPLETENESS_USER = '''## Question
{question}

## Correct Answer
{answer}

## Task
List 3-5 distinct reasoning sub-goals that must be addressed to correctly answer this question. Each sub-goal should be necessary and specific to this question. Order them logically (perception → inference → synthesis).

Output format (numbered, one per line):
1. [First sub-goal]
2. [Second sub-goal]
3. [Third sub-goal]
...'''

M5_FACTUAL_ACCURACY_SYSTEM = '''You are a fact-checker for video reasoning claims. You will watch a video segment and evaluate whether a specific claim about it is ACCURATE.

## Judgment Criteria
- **Yes**: The claim matches what you observe in the video (visual and/or audio). The events, objects, speech, sounds, or timings described are present and correct as stated.
- **No**: The claim contradicts what you see or hear: wrong object/person, wrong timing, wrong dialogue, event not present, or clearly false description.

## Important
- Evaluate ONLY what is in the video. Do not infer beyond what is shown or said.
- For dialogue claims: check if the quoted speech appears in the audio. Paraphrases can be Yes if the meaning matches.
- For visual claims: check if the described scene, objects, or actions are present.
- If the video segment does not contain the relevant content (e.g., claim references 0:30 but segment is 0:00-0:10), answer No.

## Output Format
Answer with exactly one word: Yes or No. Do not add explanation, justification, or any other text.'''

M5_FACTUAL_ACCURACY_USER = '''## Video Segment
[You are shown a video segment.]

## Claim to Verify
{claim}

## Task
Watch the video segment. Is the claim above accurate based on what you see and hear? The claim must match the actual visual and audio content. If the segment does not cover the relevant moment, or the claim is false, answer No.

Output exactly one word: Yes or No.'''
