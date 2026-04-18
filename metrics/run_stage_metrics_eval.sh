#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PARENT_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

ensure_module_cmd() {
  if type module >/dev/null 2>&1; then
    return 0
  fi

  for init_script in \
    /etc/profile.d/modules.sh \
    /usr/share/lmod/lmod/init/bash \
    /usr/share/Modules/init/bash; do
    if [[ -f "$init_script" ]]; then
      # shellcheck disable=SC1090
      source "$init_script"
      break
    fi
  done

  type module >/dev/null 2>&1
}

if [[ "${SKIP_MODULE_LOAD:-0}" != "1" ]]; then
  if ensure_module_cmd; then
    module purge
    module load gcc/11.2.0
    module load cuda/12.8.1
    module load cudnn/v9.10.2
    module load Python3/3.10.14
  else
    echo "WARNING: Could not initialize environment modules; continuing with existing shell environment." >&2
  fi
fi

if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(realpath "$(which nvcc)")")")"
fi
if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/lib64" ]]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

DEFAULT_VENV="/fs/nexus-scratch/gnanesh/venv_vdr3/bin/activate"
if [[ -z "${VIRTUAL_ENV:-}" && -f "$DEFAULT_VENV" ]]; then
  # shellcheck disable=SC1090
  source "$DEFAULT_VENV"
fi

export PATH="/fs/nexus-scratch/gnanesh/ffmpeg-7.0.2-amd64-static:$PATH"

RUNTIME_CACHE="${VDR_RUNTIME_ROOT:-${NEXUS_RUNTIME_ROOT:-/fs/nexus-scratch/gnanesh/.cache}}"
export HF_HOME="${HF_HOME:-$RUNTIME_CACHE/huggingface}"
export PADDLE_PDX_CACHE_HOME="${PADDLE_PDX_CACHE_HOME:-$RUNTIME_CACHE/paddlex}"
export PADDLE_HOME="${PADDLE_HOME:-$RUNTIME_CACHE/paddle}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$RUNTIME_CACHE/triton}"
export TMPDIR="${TMPDIR:-$RUNTIME_CACHE/tmp}"
export TORCH_HOME="${TORCH_HOME:-$RUNTIME_CACHE/torch}"
mkdir -p "$HF_HOME/hub" "$PADDLE_PDX_CACHE_HOME" "$PADDLE_HOME" "$TRITON_CACHE_DIR" "$TMPDIR" "$TORCH_HOME"

export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-false}"
export VLLM_PLUGINS="${VLLM_PLUGINS:-}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FLAGS_enable_pir_api="${FLAGS_enable_pir_api:-0}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::UserWarning:pyannote.audio.core.io,ignore::UserWarning:pyannote.audio.utils.reproducibility}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VERIFIER_MODE="${VERIFIER_MODE:-hybrid}"
METRICS_LIST="${METRICS_LIST:-answer_sufficiency,internal_coherence,execution_consistency,tool_validity,verifier_metrics}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
OPENAI_KEY_FILE="${OPENAI_KEY_FILE:-$PARENT_ROOT/OPENAI_API_KEY.txt}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$PARENT_ROOT/HF_TOKEN.txt}"

OVB_RESULTS="${OVB_RESULTS:-$PROJECT_ROOT/VideoDeepResearch/eval/results_omnivideobench}"
VMQA_RESULTS="${VMQA_RESULTS:-$PROJECT_ROOT/VideoDeepResearch/eval/results_generated_full_context_videomathqa_10pct}"
OUTPUT_BASE="${OUTPUT_BASE:-$SCRIPT_DIR/results}"

if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
  HF_TOKEN_VAL="$(grep -E -v '^\s*(#|$)' "$HF_TOKEN_FILE" | head -1 | xargs || true)"
  if [[ -n "$HF_TOKEN_VAL" ]]; then
    export HF_TOKEN="$HF_TOKEN_VAL"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN_VAL"
  fi
fi

if [[ -z "${OPENAI_API_KEY:-${PLANNER_API_KEY:-${API_KEY:-}}}" && -f "$OPENAI_KEY_FILE" ]]; then
  export OPENAI_API_KEY="$(tr -d '\r\n' < "$OPENAI_KEY_FILE")"
fi

if [[ "$VERIFIER_MODE" != "stored" && -z "${OPENAI_API_KEY:-${PLANNER_API_KEY:-${API_KEY:-}}}" ]]; then
  echo "VERIFIER_MODE=$VERIFIER_MODE requires OPENAI_API_KEY or PLANNER_API_KEY (or set VERIFIER_MODE=stored)." >&2
  exit 1
fi

run_one() {
  local input_path="$1"
  local name="$2"

  local out_dir="$OUTPUT_BASE/$name"
  mkdir -p "$out_dir"

  echo "=== Running metrics for $name ==="
  echo "Input: $input_path"
  echo "Output: $out_dir/per_sample_metrics.jsonl"

  "$PYTHON_BIN" "$SCRIPT_DIR/run_stage_metrics.py" \
    "$input_path" \
    --verifier-mode "$VERIFIER_MODE" \
    --metrics "$METRICS_LIST" \
    --output "$out_dir/per_sample_metrics.jsonl" \
    ${EXTRA_ARGS}
}

run_one "$OVB_RESULTS" "results_omnivideobench"
run_one "$VMQA_RESULTS" "results_generated_full_context_videomathqa_10pct"

echo "=== Done ==="
echo "OmniVideoBench: $OUTPUT_BASE/results_omnivideobench/per_sample_metrics.jsonl"
echo "VideoMathQA:    $OUTPUT_BASE/results_generated_full_context_videomathqa_10pct/per_sample_metrics.jsonl"
