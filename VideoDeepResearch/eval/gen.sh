#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

set +u
source /home/ghazi/miniconda3/etc/profile.d/conda.sh
conda activate cot
set -u

mkdir -p ./vllm_log ./logs


export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/share/data/drive_1/huggingface_cache

export API_MODEL_NAME_TEMPORAL_GROUNDING=/share/data/drive_1/huggingface_cache/hub/models--avery00--VideoExplorer-TemporalGrounder/snapshots/af140b39ac8e57d09d31f6b6fbf2491942ad19bd
export API_BASE_URL_TEMPORAL_GROUNDING=http://localhost:22345/v1
export API_KEY_TEMPORAL_GROUNDING=EMPTY

export API_MODEL_NAME=/share/data/drive_1/huggingface_cache/hub/models--avery00--VideoExplorer-Planner-7B/snapshots/60a51925ca5b91851e6a96b4ef4c85c4fea99476
export API_BASE_URL=http://localhost:22346/v1

export API_KEY=EMPTY

export API_MODEL_NAME_VLM=/share/data/drive_1/huggingface_cache/hub/models--Qwen--Qwen2.5-Omni-7B/snapshots/ae9e1690543ffd5c0221dc27f79834d0294cba00



#（planner）
CUDA_VISIBLE_DEVICES=5 python servers_io/vllm_server_planner.py > ./vllm_log/planner.log 2>&1 &
planner_pid=$!

# （temporal grounder）
CUDA_VISIBLE_DEVICES=6 python servers_io/vllm_server_temporal_grounder.py > ./vllm_log/temporal_grounder.log 2>&1 &
grounder_pid=$!

echo "Waiting for servers to start..."
sleep 10




bash ./clear.sh
export TOPK=10
threads=6

#（VideoMathQA）
CUDA_VISIBLE_DEVICES=7 python3 generate_traces.py \
    --benchmark_dir /share/data/drive_1/ghazi/VideoMathQA \
    --output ./videomathqa_traces.json \
    | tee -a ./logs/generate_traces_videomathqa.log


cleanup() {
    pkill -9 -P "${planner_pid:-0}" 2>/dev/null || true
    pkill -9 -P "${grounder_pid:-0}" 2>/dev/null || true
    kill -9 "${planner_pid:-0}" "${grounder_pid:-0}" 2>/dev/null || true
    pkill -9 -f vllm_server_planner.py 2>/dev/null || true
    pkill -9 -f vllm_server_temporal_grounder.py 2>/dev/null || true
}
trap cleanup EXIT
