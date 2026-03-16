# cot

Core repository for long-video reasoning/evaluation workflows.

## Structure
- `VideoDeepResearch/`: main codebase (demo, evaluation, training).
- `metrics/`: metric implementations and utilities.
- `Evaluation Metrics.md`: metric notes.
- `flow.md`: pipeline/process notes.

## Quick Links
- Main project README: `VideoDeepResearch/readme.md`
- Evaluation guide: `VideoDeepResearch/eval/readme.md`
- VideoMathQA trace generation: `VideoDeepResearch/readme.md` ("Generate VideoMathQA Traces")
- Trace output (default): `VideoDeepResearch/eval/videomathqa_traces.json`

## Trace Model Configs
- `VideoDeepResearch/eval/videomathqa_traces.json`
  - Script: `VideoDeepResearch/eval/generate_traces.py` (via `VideoDeepResearch/eval/gen.sh`)
  - Planner: `avery00/VideoExplorer-Planner-7B` (local vLLM server, `API_MODEL_NAME`)
  - Temporal grounder agent: `avery00/VideoExplorer-TemporalGrounder` (local vLLM server, `API_MODEL_NAME_TEMPORAL_GROUNDING`)
- `VideoDeepResearch/eval/gpt_videomathqa_traces.json`
  - Script: `VideoDeepResearch/eval/deepseek_gen_traces.py`
  - Planner: `gpt-5` (`API_MODEL_NAME`, OpenAI-compatible API)
  - Temporal grounder agent: `gpt-5` (`API_MODEL_NAME_TEMPORAL_GROUNDING`, OpenAI-compatible API)

Model selection is controlled by env vars in the corresponding run scripts (`gen.sh`, `deepseek_gen.sh`).
