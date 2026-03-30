#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

set +u
source /home/ghazi/miniconda3/etc/profile.d/conda.sh
conda activate cot
set -u

mkdir -p ./logs ./traces

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/share/data/drive_1/huggingface_cache

export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script}"
export API_BASE_URL=https://api.openai.com/v1
export API_BASE_URL_TEMPORAL_GROUNDING=https://api.openai.com/v1
export API_KEY="$OPENAI_API_KEY"
export API_KEY_TEMPORAL_GROUNDING="$OPENAI_API_KEY"
export API_MODEL_NAME=gpt-5
export API_MODEL_NAME_TEMPORAL_GROUNDING=gpt-5

export VLLM_WORKER_MULTIPROC_METHOD=spawn

export TOPK=10
threads=6

CUDA_VISIBLE_DEVICES=7 python3 refiner.py\
    --benchmark_dir /share/data/drive_1/ghazi/VideoMathQA \
    --annotation_file mcq.json \
    --output ./traces/gpt_videomathqa_mcq_traces.json \
    | tee -a ./logs/generate_traces_gpt_videomathqa_mcq.log
