#!/bin/bash
cd ./eval


export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/share/data/drive_1/huggingface_cache
DATASET_DIR=/share/users/ghazi/VideoDeepResearch/videos

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
mkdir -p ./logs


# 运行demo脚本
CUDA_VISIBLE_DEVICES=7 python demo.py \
    --video_path /share/users/ghazi/VideoDeepResearch/videos/875b24c9-a2ab-4965-8186-76495a5b553d.mp4 \
    --question "Among Walmart, Target, Whole Foods, and Albertsons, which store shows the highest discrepancy between customer-rated Store Cleanliness and Value for Dollar, and what is the approximate magnitude of that difference in percentage points?" \
    --topk $TOPK | tee -a ./logs/demo.log

# # 运行评估脚本
# for dataset in mlvu; do
#     for i in $(seq 0 $(($threads-1))); do
#         log_file=./logs/eval_${dataset}_thread${i}.log
#         CUDA_VISIBLE_DEVICES=$((i+2)) python eval.py \
#             --dataset $dataset \
#             --dataset_folder $DATASET_DIR \
#             --thread_num $threads \
#             --thread_idx $i \
#             --clip_duration 10 \
#             --end_sample_number 100000 | tee -a $log_file &
#     done
# done

# wait
# echo "✅ All evaluation threads finished."



pkill -9 -P $planner_pid 2>/dev/null || true
pkill -9 -P $grounder_pid 2>/dev/null || true
kill -9 $planner_pid $grounder_pid 2>/dev/null || true
pkill -9 -f vllm_server_planner.py 2>/dev/null || true
pkill -9 -f vllm_server_temporal_grounder.py 2>/dev/null || true
