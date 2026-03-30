# Refiner (`VideoQADemo`): model backends

This document explains how to configure [`refiner.py`](refiner.py) `VideoQADemo` for:

- **Local models** (vLLM in-process for video tools, Hugging Face models for chart analysis, etc.)
- **Remote APIs** (e.g. OpenAI **gpt-5** or other OpenAI-compatible chat/vision endpoints)
- **Planner, verifier, and refiner** via **vLLM** using the existing **pickle file-queue** worker (not HTTP), or via **HTTP** to a vLLM/OpenAI-compatible server

All primary settings are **constructor arguments** on `VideoQADemo`. Edit the code where you construct the demo (or wrap defaults in your own helper).

---

## Roles of each model

| Role | What calls it | Local option | Remote API option |
|------|----------------|--------------|-------------------|
| **VLM** (frame/video tools: OCR fallback, spatial, counter, dense caption, action, etc.) | In-process `vLLM.LLM` | `vlm_model_name`, leave `vlm_api_base` unset | Set `vlm_api_base` + `vlm_model_name` (vision-capable model id on that API) |
| **Planner / verifier / refiner** (text JSON) | `_text2text` | vLLM **pickle queue** when `planner_api_base` is localhost (see below) | OpenAI-compatible **HTTP** when base URL is **not** localhost (or force with env) |
| **Chart analyzer** | `chart_mode` selects backend | **`api`:** `chart_model_name` + HTTP vision (defaults: `gpt-5`, inherits `planner_api_*` if `chart_api_base` unset) · **`vlm`:** same path as other VLM tools (`vlm_model_name`) · **`internvl`:** HF InternVL on `chart_device` |

---

## Tools and models (reference)

### Configurable on `VideoQADemo` (defaults in [`refiner.py`](refiner.py))

These are the **string ids** passed into the pipeline; change them in code when you construct `VideoQADemo`.

| Parameter | Default | Used for |
|-----------|---------|----------|
| `planner_model_name` | `deepseek-ai/DeepSeek-V3` | **Verifier**, **planner**, and **refiner** (all call `_text2text` with the same model and `planner_api_*`) |
| `planner_api_base` / `planner_api_keys` | `["http://localhost:8000/v1"]` / `["EMPTY"]` | Routing for that text LLM (pickle queue vs HTTP; see below) |
| `vlm_model_name` | `Qwen/Qwen2.5-VL-7B-Instruct` | **VLM-backed tools** (see table below) when `vlm_api_base` is unset (in-process vLLM) or when set (remote vision chat) |
| `vlm_api_base` / `vlm_api_keys` | unset / `["EMPTY"]` | If unset, load local VLM; if set (non-empty first URL), **no** in-process VLM—vision tools use HTTP + `vlm_model_name` |
| `vlm_tensor_parallel_size` | `1` | Tensor parallel size for in-process VLM only |
| `chart_mode` | `"api"` | `"api"` = OpenAI-compatible vision HTTP · `"vlm"` = reuse **`vlm_model_name`** (local vLLM or `vlm_api_*`) · `"internvl"` = separate HF **`chart_model_name`** (InternVL) on **`chart_device`** |
| `chart_model_name` | `"gpt-5"` | **`api`:** vision chat model id · **`internvl`:** e.g. `OpenGVLab/InternVL2_5-8B` (ignored for **`vlm`** mode) |
| `chart_api_base` / `chart_api_keys` | `None` / `None` | **`api` only:** `None` = inherit **`planner_api_base`** / **`planner_api_keys`**; set explicitly to override |
| `chart_device` | `"cuda:0"` | GPU for **`internvl`** mode only |

### Refinement agents (LLM, not executor tools)

| Agent | Role | Model |
|-------|------|--------|
| Verifier | Diagnoses trace quality (JSON) | `planner_model_name` |
| Planner | Produces tool plan (JSON) | `planner_model_name` |
| Refiner | Rewrites trace / answer (JSON) | `planner_model_name` |

### Executor tools (`refiner_tools.py`)

The planner’s plan is executed by name. Registered tools and what actually runs:

| Tool | What it does | Model / backend |
|------|----------------|-----------------|
| `temporal_grounder` | Text-to-video segment retrieval | **LanguageBind** + **BGE-M3** (text query) via [`retriever_languagebind.py`](../retriever_languagebind.py) (`LanguageBind_Video_FT` / `LanguageBind_Image`, optional local `BGE_M3_MODEL_PATH`); **no** `vlm_model_name` |
| `frame_retriever` | Frames at timestamps or from retrieval | Same retriever as above + on-disk frames |
| `asr` | Speech-to-text | **WhisperX** (default weights name `large-v3`, overridable with `WHISPERX_MODEL`); fallback: subtitles / `extract_subtitles` |
| `audio_grounder` | Audio event search in a window | **LAION CLAP** (`CLAP_Module.load_ckpt()`); fallback: subtitle stub |
| `ocr` | Text in a frame | **PaddleOCR** → **pytesseract** → **`vlm_model_name`** (VLM JSON) |
| `spatial_grounder` | Objects / regions from a frame | **`vlm_model_name`** (vLLM or remote vision API) |
| `counter` | Count objects in a frame | **`vlm_model_name`** |
| `dense_captioner` | Captions over a time range | **`vlm_model_name`** |
| `action_recognizer` | Human actions in a range | **`vlm_model_name`** |
| `chart_analyzer` | Chart / plot reading | Set **`chart_mode`**: **`api`** (default `gpt-5`, inherits planner API if chart URL unset), **`vlm`** (same stack as spatial/counter), or **`internvl`** (dedicated HF load) |

`video_qa_reanswerer` exists in prompts but is **not** registered in the handler map (commented out).

### Environment toggles for optional backends (tools)

Whisper, CLAP, and PaddleOCR can be disabled or tuned via env vars in `refiner_tools.py` (e.g. `REFINER_DISABLE_WHISPERX`, `REFINER_DISABLE_CLAP`, `REFINER_DISABLE_PADDLEOCR`, `WHISPERX_MODEL`, `WHISPERX_BATCH`, CLAP window/hop/threshold). These do not replace `VideoQADemo` constructor model ids; they only switch or configure fixed pipelines.

---

## Planner, verifier, and refiner: vLLM (pickle queue)

These agents share **`planner_model_name`**, **`planner_api_base`**, and **`planner_api_keys`**.

`_text2text` chooses the backend using `_use_openai_http(planner_api_base)`:

- If the first URL in `planner_api_base` contains **`localhost`** or **`127.0.0.1`**, the code does **not** use the HTTP client. It writes requests under **`./vllm_io_files/vllm_input_planner/`** and waits for a matching file in **`./vllm_io_files/vllm_output_planner/`** (pickled string response). Your separate **vLLM worker process** must read those inputs and write outputs—this is the legacy “vLLM for planner/verifier” path.

**Typical setup (unchanged defaults):**

```python
demo = VideoQADemo(
    ...,
    planner_model_name="deepseek-ai/DeepSeek-V3",  # must match what the worker serves
    planner_api_base=["http://localhost:8000/v1"],
    planner_api_keys=["EMPTY"],
)
```

Run your vLLM-sidecar worker from the **same working directory** so `./vllm_io_files/...` paths resolve, or adjust paths in your worker to match.

**Force pickle queue even if you use a non-localhost URL in the list:** set:

```bash
export REFINER_USE_VLLM_PICKLE=1
```

**Force HTTP** (OpenAI-compatible `chat.completions`) even for localhost—set:

```bash
export REFINER_FORCE_HTTP_LLM=1
```

That is useful when vLLM exposes an OpenAI-compatible server on `http://localhost:8000/v1` and you want the client to call it over HTTP instead of the pickle files.

---

## Planner / verifier / refiner: remote GPT (e.g. gpt-5)

Point `planner_api_base` at your provider’s OpenAI-compatible base URL and pass a real API key. Set **`planner_model_name`** to the model id your provider returns (e.g. `gpt-5` if that is the id on your account).

```python
demo = VideoQADemo(
    ...,
    planner_model_name="gpt-5",
    planner_api_base=["https://api.openai.com/v1"],
    planner_api_keys=[os.environ["OPENAI_API_KEY"]],  # or paste key in code for local tests only
)
```

Because the URL is not localhost, `_use_openai_http` is true and requests use the **HTTP** path (no `vllm_io_files`).

You can pass multiple bases/keys as comma-separated strings or parallel lists for failover (see `_norm_api_list` in `refiner.py`).

---

## VLM tools: local vLLM (default)

Leave **`vlm_api_base` as default (`None`)**. The process loads **`vlm_model_name`** with in-process vLLM (`LLM(...)`) and uses **`vlm_tensor_parallel_size`**.

```python
demo = VideoQADemo(
    ...,
    vlm_model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    vlm_tensor_parallel_size=1,
)
```

---

## VLM tools: remote vision API (e.g. GPT vision)

Set a non-empty **`vlm_api_base`** so the demo skips loading local vLLM and sends frames as base64 JPEGs to the chat vision API.

```python
demo = VideoQADemo(
    ...,
    vlm_model_name="gpt-5",  # or gpt-4o, etc.—must match provider
    vlm_api_base=["https://api.openai.com/v1"],
    vlm_api_keys=[os.environ["OPENAI_API_KEY"]],
)
```

---

## Chart analyzer modes (`chart_mode`)

### `api` (default) — e.g. gpt-5

Uses OpenAI-style **vision** `chat.completions` with **`chart_model_name`** (default **`gpt-5`**). If **`chart_api_base`** and **`chart_api_keys`** are omitted, the chart tool **reuses** **`planner_api_base`** / **`planner_api_keys`** (so one OpenAI key is enough when planner is already on `https://api.openai.com/v1`).

```python
demo = VideoQADemo(
    planner_model_name="gpt-5",
    planner_api_base=["https://api.openai.com/v1"],
    planner_api_keys=[os.environ["OPENAI_API_KEY"]],
    chart_mode="api",
    chart_model_name="gpt-5",
)
```

### `vlm` — same Qwen (vLLM or remote) as other frame tools

Chart analysis goes through the same **`vlm_model_name`** path as **`spatial_grounder`** / **`counter`** (in-process vLLM or **`vlm_api_*`**). **`chart_model_name`** is not used for inference in this mode.

```python
demo = VideoQADemo(
    chart_mode="vlm",
    vlm_model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    vlm_api_base=None,
)
```

### `internvl` — dedicated local InternVL

Loads **`chart_model_name`** with Hugging Face on **`chart_device`** (use an InternVL repo id).

```python
demo = VideoQADemo(
    chart_mode="internvl",
    chart_model_name="OpenGVLab/InternVL2_5-8B",
    chart_device="cuda:0",
)
```

---

## Mixing backends (example)

- **Planner/verifier/refiner:** GPT over HTTP  
- **VLM tools:** local Qwen vLLM  
- **Chart:** default **`api`** with **`gpt-5`** (inherits planner API)  

```python
demo = VideoQADemo(
    planner_model_name="gpt-5",
    planner_api_base=["https://api.openai.com/v1"],
    planner_api_keys=[os.environ["OPENAI_API_KEY"]],
    vlm_model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    vlm_api_base=None,
    chart_mode="api",
    chart_model_name="gpt-5",
)
```

---

## Environment variables (reference)

| Variable | Effect |
|----------|--------|
| `REFINER_FORCE_HTTP_LLM=1` | Planner `_text2text` always uses HTTP OpenAI client |
| `REFINER_USE_VLLM_PICKLE=1` | Planner `_text2text` always uses pickle file queue |
| `REFINER_DEBUG_*`, Whisper/CLAP/OCR toggles | Unrelated to model routing; see code and slurm scripts |

Model names and API URLs for the refiner pipeline are intended to be set in **`VideoQADemo(...)`** as shown above.
