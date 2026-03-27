## Component Design

### Verifier Agent

**Role:** Decide if a trace is correct and sufficient; if not, produce a structured diagnosis.

**Input:** Question, trace (with timestamps), answer, and optionally the video.

**Should the Verifier see the video?** **Yes, strongly recommended.** Without the video, the Verifier can only check logical consistency (does the trace support the answer?). With the video, it can also check **factual grounding** (does the trace match what actually happens?). We recommend a two-level verification:


| Level | Name                          | Input                                 | What it checks                                                                                     |
| ----- | ----------------------------- | ------------------------------------- | -------------------------------------------------------------------------------------------------- |
| L1    | **Logical Consistency Check** | Question + Trace + Answer (text only) | Does the reasoning chain logically lead to the answer? Are steps coherent?                         |
| L2    | **Factual Grounding Check**   | Question + Trace + Answer + Video     | Are claimed observations actually in the video? Are timestamps correct? Are audio claims accurate? |


**Output schema (structured JSON):**

```json
{
  "verdict": "PASS | FAIL",
  "answer_correct": true/false,
  "error_categories": [
    {
      "type": "TIMESTAMP_ERROR | INFERENCE_ERROR | PERCEPTION_ERROR | INCOMPLETE_TRACE | ANSWER_ERROR | MODALITY_ERROR",
      "step_index": 2,
      "description": "Step 2 claims event at 01:23-01:30 but the event occurs at 02:10-02:18",
      "severity": "HIGH | MEDIUM | LOW",
      "suggested_tools": ["temporal_grounder", "frame_retriever"]
    }
  ],
  "confidence": 0.85
}
```

**Candidate models for the Verifier:**

- **Gemini 2.5 Pro / 3.1 Pro** -- best native video understanding, can process hours of video, strong reasoning. **Top recommendation.**
- **GPT-4o / GPT-4.1** -- strong multimodal reasoning, good at structured output
- **Qwen2.5-VL-72B** -- open-source alternative with hour-level video support and temporal encoding
- For L1 (text-only): any strong reasoning LLM (Claude, GPT-4o, Qwen2.5-72B) works

---

### Planner / Orchestrator Agent

**Role:** Given the Verifier's diagnosis, decide which tools to call, in what order, and with what parameters. This is the "brain" of the pipeline.

**Design:** An LLM agent with tool-use capability. It receives the diagnosis and generates a **plan** -- a sequence of tool calls with arguments.

**Example plan for a timestamp error:**

1. Call `temporal_grounder(query="when does the player score the goal", video=...)` to get correct timestamp
2. Call `frame_retriever(video=..., timestamps=[corrected_range])` to extract evidence frames
3. Pass results to Trace Refinement Agent

**Candidate models:**

- **Gemini 2.5 Pro** -- native tool-use, strong video context
- **GPT-4o / GPT-4.1** -- excellent function-calling / tool-use
- **Claude 4 Opus/Sonnet** -- strong reasoning and planning

**Key design decision:** The Planner should have access to the **video** so it can make informed decisions about which tools to dispatch. If cost is a concern, it can operate on a summarized representation (e.g., dense captions + ASR transcript generated once upfront).

---

### Tool Suite

#### Tool 1: Temporal Grounder

**Purpose:** Given a natural-language query, localize the relevant time segment(s) in a video.


| Candidate                | Notes                                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------------------- |
| **TimeLens** (Dec 2025)  | SOTA among open-source; beats GPT-5 and Gemini-2.5-Flash on VTG benchmarks. RLVR-trained. **Top pick.** |
| **TRACE** (ICLR 2025)    | Causal event modeling; outputs (timestamp, saliency, caption) tuples                                    |
| **TimeChat** (CVPR 2024) | Sliding Q-Former for long videos; strong zero-shot                                                      |
| **Qwen2.5-VL**           | Built-in temporal encoding with second-level grounding                                                  |
| **Gemini 2.5 Pro**       | Can be prompted for temporal grounding directly                                                         |


#### Tool 2: Frame Retriever

**Purpose:** Extract the most relevant frames for a query or timestamp range.


| Candidate                    | Notes                                                             |
| ---------------------------- | ----------------------------------------------------------------- |
| **K-frames** (2025)          | Scene-driven any-k selection with RL curriculum                   |
| **KeyScore + STACFP** (2025) | Caption-aware scoring; 99% frame reduction                        |
| **FOCUS** (2025)             | Multi-armed bandit; processes less than 2% of frames              |
| **CLIP/SigLIP similarity**   | Simple baseline: encode query + frames, rank by cosine similarity |


#### Tool 3: ASR (Automatic Speech Recognition)

**Purpose:** Transcribe speech with word-level timestamps.


| Candidate                        | Notes                                                       |
| -------------------------------- | ----------------------------------------------------------- |
| **Whisper large-v3** (OpenAI)    | Robust multilingual; word-level timestamps; widely deployed |
| **Cohere Transcribe** (Mar 2026) | 2B params, 14 languages, 5.42 WER, 525 min/min throughput   |
| **WhisperX**                     | Whisper + forced alignment for precise word timestamps      |


#### Tool 4: Audio Grounder

**Purpose:** Localize non-speech audio events (music, sound effects, environmental sounds) in time.


| Candidate                       | Notes                                                    |
| ------------------------------- | -------------------------------------------------------- |
| **FLAM** (2025)                 | Frame-level open-vocabulary audio grounding; 50x speedup |
| **LongAudio-RAG** (2026)        | Structured event detection for multi-hour audio          |
| **LAION-CLAP**                  | Audio-text contrastive model; 40ms resolution            |
| **Qwen2-Audio / Qwen2.5-Audio** | Audio LLM with grounding capability                      |


#### Tool 5: OCR

**Purpose:** Extract text visible in video frames (scoreboards, signs, subtitles, equations).


| Candidate                          | Notes                                                    |
| ---------------------------------- | -------------------------------------------------------- |
| **PaddleOCR v5 / PP-Structure v3** | Real-time video OCR, multi-language, handles distortions |
| **PaddleOCR-VL-1.5**               | 0.9B VLM; 94.5% accuracy on document parsing             |
| **EasyOCR**                        | Lightweight alternative; good for quick integration      |
| **TrOCR** (Microsoft)              | Transformer-based; good for handwriting/printed text     |


#### Tool 6: Spatial / Object Grounder

**Purpose:** Detect objects, produce bounding boxes or segmentation masks given a text query.


| Candidate                      | Notes                                                                                  |
| ------------------------------ | -------------------------------------------------------------------------------------- |
| **Grounded SAM 2**             | GroundingDINO + SAM2; detection + segmentation + tracking across frames. **Top pick.** |
| **GroundingDINO 1.6 / DINO-X** | Open-vocab detection; 52.5 AP zero-shot on COCO                                        |
| **Florence-2**                 | Microsoft; unified vision model for detection, captioning, grounding                   |
| **OWLv2** (Google)             | Open-world detection                                                                   |


#### Tool 7: Counter

**Purpose:** Count specific objects in a frame or across frames.


| Candidate                  | Notes                                                                                                 |
| -------------------------- | ----------------------------------------------------------------------------------------------------- |
| **CountGD++** (Dec 2025)   | Open-world counting with text + visual exemplars; can specify what NOT to count; LLM-agent compatible |
| **CountGD** (NeurIPS 2024) | Multi-modal open-world counting; repurposes GroundingDINO                                             |


#### Tool 8: Dense Captioner (NEW -- recommended addition)

**Purpose:** Generate detailed frame-by-frame or segment-level descriptions of video content. Essential for the Planner to understand what's happening without re-watching the full video.


| Candidate          | Notes                                                            |
| ------------------ | ---------------------------------------------------------------- |
| **Qwen2.5-VL-72B** | Hour-level video understanding; dense temporal captions          |
| **PLLaVA**         | Parameter-free LLaVA extension; SOTA on Video ChatGPT benchmarks |
| **LLaVA-Video**    | Strong video captioning                                          |
| **Gemini 2.5 Pro** | Can caption video segments natively                              |


#### Tool 9: Action Recognizer (NEW -- recommended addition)

**Purpose:** Classify or detect human actions/activities in video segments. Useful when traces describe actions incorrectly.


| Candidate        | Notes                                              |
| ---------------- | -------------------------------------------------- |
| **InternVideo2** | Strong video action recognition + retrieval        |
| **VideoMAE v2**  | Self-supervised; fine-tuned for action recognition |
| **Qwen2.5-VL**   | Can be prompted for action description             |


#### Tool 10: Video QA Re-answerer (NEW -- recommended addition)

**Purpose:** Given the video and question, independently derive an answer to compare against the trace's answer. Acts as a cross-check.


| Candidate          | Notes                               |
| ------------------ | ----------------------------------- |
| **Gemini 2.5 Pro** | Best video QA performance currently |
| **GPT-4o**         | Strong multimodal QA                |
| **Qwen2.5-VL-72B** | Best open-source video QA           |


---

### Trace Refinement Agent

**Role:** Given the original trace, the Verifier's diagnosis, and the tool outputs, produce a corrected trace.

**Design:** An LLM that performs **targeted edits** to the trace rather than rewriting from scratch. This preserves correct parts while fixing identified errors.

**Input:**

- Original trace (with step indices)
- Verifier diagnosis (which steps are wrong and why)
- Tool outputs (corrected timestamps, transcripts, frame descriptions, etc.)
- Question and answer

**Output:** Refined trace in the same format as the original. If the answer is also wrong, propose a corrected answer.

**Candidate models:** Same as Planner (Gemini 2.5 Pro, GPT-4o, Claude)
