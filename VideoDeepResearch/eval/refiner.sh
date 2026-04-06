#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

set +u
source /apps/local/anaconda3/etc/profile.d/conda.sh # Adjust the path to your conda.sh if necessary
conda activate cot
export CUDA_HOME=/nfs-stor/ghazi.ahmad/miniconda3
export CUDA_PATH="$CUDA_HOME"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$CUDA_HOME/include${CPATH:+:$CPATH}"
set -u


mkdir -p ./logs ./traces

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/nfs-stor/ghazi.ahmad/HF_HOME
export CLAP_CKPT_PATH="${CLAP_CKPT_PATH:-$HF_HOME/assets/laion_clap/630k-audioset-best.pt}"
export RETRIEVER_ASSETS_ROOT="${RETRIEVER_ASSETS_ROOT:-/home/ghazi.ahmad/model_assets}"
export LANGUAGEBIND_VIDEO_MODEL_PATH="${LANGUAGEBIND_VIDEO_MODEL_PATH:-$RETRIEVER_ASSETS_ROOT/LanguageBind_Video_FT}"
export LANGUAGEBIND_IMAGE_MODEL_PATH="${LANGUAGEBIND_IMAGE_MODEL_PATH:-$RETRIEVER_ASSETS_ROOT/LanguageBind_Image}"
export LANGUAGEBIND_VIDEO_TOKENIZER_PATH="${LANGUAGEBIND_VIDEO_TOKENIZER_PATH:-$LANGUAGEBIND_VIDEO_MODEL_PATH}"
export BGE_M3_MODEL_PATH="${BGE_M3_MODEL_PATH:-$RETRIEVER_ASSETS_ROOT/bge-m3}"

export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script}"
export API_BASE_URL=https://api.openai.com/v1
export API_BASE_URL_TEMPORAL_GROUNDING=https://api.openai.com/v1
export API_KEY="$OPENAI_API_KEY"
export API_KEY_TEMPORAL_GROUNDING="$OPENAI_API_KEY"
export API_MODEL_NAME=gpt-5.4
export API_MODEL_NAME_TEMPORAL_GROUNDING=gpt-5.4
export PLANNER_API_BASE="${PLANNER_API_BASE:-$API_BASE_URL}"
export PLANNER_API_KEY="${PLANNER_API_KEY:-$API_KEY}"
export PLANNER_MODEL_NAME="${PLANNER_MODEL_NAME:-$API_MODEL_NAME}"

# Default VLM tools to local Qwen3.
export LOCAL_VLM_MODEL_NAME="${LOCAL_VLM_MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
export VLM_API_BASE="${VLM_API_BASE:-}"
export VLM_API_KEY="${VLM_API_KEY:-$OPENAI_API_KEY}"
export VLM_MODEL_NAME="${VLM_MODEL_NAME:-$LOCAL_VLM_MODEL_NAME}"
export CHART_MODE="${CHART_MODE:-vlm}"
export CHART_MODEL_NAME="${CHART_MODEL_NAME:-$VLM_MODEL_NAME}"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WHISPERX_CONDA_ENV="${WHISPERX_CONDA_ENV:-${CONDA_DEFAULT_ENV:-cot}}"
export WHISPERX_MODEL="${WHISPERX_MODEL:-small}"
export WHISPERX_DOWNLOAD_ROOT="${WHISPERX_DOWNLOAD_ROOT:-$HF_HOME/hub}"
export WHISPERX_LOCAL_FILES_ONLY="${WHISPERX_LOCAL_FILES_ONLY:-0}"
export WHISPERX_FFMPEG_PATH="${WHISPERX_FFMPEG_PATH:-$(command -v ffmpeg || true)}"
if [ -n "${WHISPERX_FFMPEG_PATH}" ]; then
  export FFMPEG_BINARY="${FFMPEG_BINARY:-$WHISPERX_FFMPEG_PATH}"
  export IMAGEIO_FFMPEG_EXE="${IMAGEIO_FFMPEG_EXE:-$WHISPERX_FFMPEG_PATH}"
  export PATH="$(dirname "$WHISPERX_FFMPEG_PATH"):${PATH}"
fi
export WHISPERX_DEVICE="${WHISPERX_DEVICE:-cuda:0}"
export WHISPERX_AUX_DEVICE="${WHISPERX_AUX_DEVICE:-cpu}"
export WHISPERX_COMPUTE_TYPE="${WHISPERX_COMPUTE_TYPE:-float16}"

export TOPK=10
threads=6

CUDA_VISIBLE_DEVICES=2 python3 refiner.py "$@"
