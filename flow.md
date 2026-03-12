# OmniModal Agentic Reasoning Framework

## ASCII Flowchart

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                   OMNIMODAL AGENTIC REASONING FRAMEWORK                       │
│                  (Iterative Thought → Action → Observation)                   │
└───────────────────────────────────────────────────────────────────────────────┘


                      INPUT: Question + Video + Audio
                                    │
                                    ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        GLOBAL CONTEXT MEMORY                                 ║
║                                                                              ║
║  Persists across ALL iterations. Updated after every cycle.                  ║
║                                                                              ║
║  Contains:                                                                   ║
║   • Original question + modality inputs                                      ║
║   • Global plan (high-level sub-goals, updated as needed)                    ║
║   • Evidence collected so far (visual, audio, knowledge)                     ║
║   • Reasoning trace built so far (step-by-step)                              ║
║   • Confidence scores per sub-goal                                           ║
║   • Unresolved questions / conflicts                                         ║
║   • Iteration count                                                          ║
╚══════════════════════════════╤════════════════════════════════════════════════╝
                               │
                               ▼
              ┌────────────────────────────────────┐
              │   META-ORCHESTRATOR (Planner)       │
              │                                    │
              │  Reads global context.              │
              │  Decides:                           │
              │   1. What sub-goal to tackle next?  │
              │   2. Which sub-agent to call?       │
              │   3. Is the answer ready?           │
              │      → If YES: exit loop            │
              │      → If NO: enter loop ───┐       │
              └─────────────────────────────┼───────┘
                                            │
         ┌──────────────────────────────────┘
         │
         │   ╔══════════════════════════════════════════════════════════╗
         │   ║           ITERATIVE REASONING LOOP                       ║
         │   ║           (repeats until answer is ready                 ║
         │   ║            or max iterations reached)                    ║
         ▼   ║                                                          ║
   ┌─────────╨──────────────────────────────────────────────────┐       ║
   │                                                            │       ║
   │  ┌──────────────────────────────────────────────────────┐  │       ║
   │  │  STEP 1: THOUGHT                                     │  │       ║
   │  │                                                       │  │       ║
   │  │  The orchestrator generates a reasoning intent:       │  │       ║
   │  │   • "I need to find what happens at 0:30s"            │  │       ║
   │  │   • "The audio might contain a clue about the mood"   │  │       ║
   │  │   • "I should verify if the visual and audio align"   │  │       ║
   │  │                                                       │  │       ║
   │  │  Based on: global context + current sub-goal          │  │       ║
   │  └────────────────────────┬─────────────────────────────┘  │       ║
   │                           │                                │       ║
   │                           ▼                                │       ║
   │  ┌──────────────────────────────────────────────────────┐  │       ║
   │  │  STEP 2: ACTION                                      │  │       ║
   │  │                                                       │  │       ║
   │  │  The orchestrator dispatches to a sub-agent:          │  │       ║
   │  │                                                       │  │       ║
   │  │   ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │  │       ║
   │  │   │   VISUAL     │  │    AUDIO     │  │ KNOWLEDGE │  │  │       ║
   │  │   │   GROUNDER   │  │   GROUNDER   │  │ RETRIEVAL │  │  │       ║
   │  │   │              │  │              │  │           │  │  │       ║
   │  │   │ • Localize   │  │ • Transcribe │  │ • Entity  │  │  │       ║
   │  │   │   frames     │  │   speech     │  │   lookup  │  │  │       ║
   │  │   │ • Detect     │  │ • Detect     │  │ • World   │  │  │       ║
   │  │   │   objects    │  │   sounds     │  │   facts   │  │  │       ║
   │  │   │ • Describe   │  │ • Identify   │  │           │  │  │       ║
   │  │   │   actions    │  │   music/tone │  │           │  │  │       ║
   │  │   └──────┬───────┘  └──────┬───────┘  └─────┬─────┘  │  │       ║
   │  │          └─────────────────┴─────────────────┘        │  │       ║
   │  │                            │                           │  │       ║
   │  │   ┌────────────────────────┘                           │  │       ║
   │  │   │  ┌──────────────┐                                  │  │       ║
   │  │   │  │   VERIFIER   │  (can also be called as action)  │  │       ║
   │  │   │  │              │                                  │  │       ║
   │  │   │  │ • Cross-check│                                  │  │       ║
   │  │   │  │   audio vs   │                                  │  │       ║
   │  │   │  │   visual     │                                  │  │       ║
   │  │   │  │ • Flag       │                                  │  │       ║
   │  │   │  │   conflicts  │                                  │  │       ║
   │  │   │  └──────┬───────┘                                  │  │       ║
   │  │   │         │                                          │  │       ║
   │  └───┼─────────┼─────────────────────────────────────────┘  │       ║
   │      └─────────┤                                            │       ║
   │                ▼                                            │       ║
   │  ┌──────────────────────────────────────────────────────┐  │       ║
   │  │  STEP 3: OBSERVATION                                  │  │       ║
   │  │                                                       │  │       ║
   │  │  The sub-agent returns its result:                    │  │       ║
   │  │   • "At 0:30s, a person slams the door"               │  │       ║
   │  │   • "Audio: loud bang at 0:30s, matches visual"        │  │       ║
   │  │   • "Conflict: visual=smile, audio=anger"             │  │       ║
   │  │                                                       │  │       ║
   │  │  The orchestrator:                                    │  │       ║
   │  │   1. Reads the observation                            │  │       ║
   │  │   2. Updates GLOBAL CONTEXT MEMORY                    │  │       ║
   │  │      (adds evidence, updates trace, marks sub-goal)   │  │       ║
   │  │   3. Updates confidence scores                        │  │       ║
   │  └────────────────────────┬─────────────────────────────┘  │       ║
   │                           │                                │       ║
   │                           ▼                                │       ║
   │  ┌──────────────────────────────────────────────────────┐  │       ║
   │  │  DECISION: Continue or Stop?                          │  │       ║
   │  │                                                       │  │       ║
   │  │  Orchestrator checks:                                 │  │       ║
   │  │   • All sub-goals resolved?    ── YES ──→ EXIT LOOP   │  │       ║
   │  │   • Confidence high enough?    ── YES ──→ EXIT LOOP   │  │       ║
   │  │   • Max iterations reached?    ── YES ──→ EXIT LOOP   │  │       ║
   │  │   • Otherwise                  ── NO  ──→ LOOP BACK   │  │       ║
   │  │                                    │         │         │  │       ║
   │  └────────────────────────────────────┼─────────┼─────────┘  │       ║
   │                                       │         │            │       ║
   └───────────────────────────────────────┼─────────┘            ║
                                           │  ▲                   ║
                                           │  │ (next iteration)  ║
                                           │  │ THOUGHT → ACTION  ║
                                           │  │ → OBSERVATION     ║
                                           │  └───────────────────╝
                                           │
                                           ▼
              ┌────────────────────────────────────────┐
              │         SYNTHESIZER                     │
              │                                        │
              │  Reads final GLOBAL CONTEXT MEMORY.     │
              │  Builds structured output:              │
              │                                        │
              │   [PLAN]      High-level sub-goals     │
              │   [EXECUTION] Step-by-step reasoning   │
              │   [SYNTHESIS] Final answer + summary   │
              └────────────────────┬───────────────────┘
                                   │
                                   ▼
              OUTPUT: Answer + Full Reasoning Trace + Confidence
                                   │
                                   ▼
              ┌────────────────────────────────────────┐
              │         EVALUATION                      │
              │                                        │
              │  Process-Aware Metrics:                 │
              │   • Answer Faithfulness                 │
              │   • Stepwise Relevance                  │
              │   • Causal Chain Coherence              │
              │   • Completeness                        │
              │   • Factual Accuracy                    │
              │   • Efficiency                          │
              │   • 2×2 Matrix Classification           │
              └────────────────────────────────────────┘
```


## How the Loop Works (Walkthrough Example)

**Question**: "Why does the person get upset at the end of the video?"

### Iteration 1
- **Thought**: "I should first understand what happens throughout the video visually."
- **Action**: Call Visual Grounder → scan full video, extract key events
- **Observation**: "0:00-0:20 person cooking, 0:25 phone rings, 0:30 person answers, 0:45 person slams phone, 0:50 person sits down looking upset"
- **Global Context Update**: Visual timeline added. Sub-goal "identify key events" marked done. Next: understand audio.

### Iteration 2
- **Thought**: "The phone call seems important. I need to hear what was said."
- **Action**: Call Audio Grounder → transcribe audio from 0:25-0:45
- **Observation**: "Phone rings at 0:25. Person says 'Hello?'... inaudible caller... Person says 'What do you mean it's cancelled?!' at 0:40"
- **Global Context Update**: Audio evidence added. Sub-goal "understand phone call" marked done. Conflict: need to verify cancellation context.

### Iteration 3
- **Thought**: "I should verify: does the visual reaction match the audio? And what was cancelled?"
- **Action**: Call Verifier → cross-check visual (upset face at 0:45) with audio ("cancelled" at 0:40)
- **Observation**: "Visual and audio align: upset expression follows hearing about cancellation. High confidence."
- **Global Context Update**: Evidence verified. All sub-goals resolved. Confidence = 0.92. → EXIT LOOP

### Synthesizer Output
- **[PLAN]**: 1) Identify visual events, 2) Understand audio context, 3) Verify cross-modal alignment
- **[EXECUTION]**: Step-by-step trace from iterations 1-3
- **[SYNTHESIS]**: "The person gets upset because they receive a phone call informing them something was cancelled, as evidenced by their verbal reaction at 0:40 and slammed phone at 0:45."
