#!/bin/bash
#SBATCH --job-name=vmqa_test
#SBATCH --output=logs/eval_videomathqa_test_%j.out
#SBATCH --error=logs/eval_videomathqa_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=high
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --mail-user=gnanesh@umd.edu
#SBATCH --mail-type=END,FAIL

module purge
module load gcc/11.2.0
module load cuda/12.4.1
module load Python3/3.10.14

export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.4

source /fs/nexus-scratch/gnanesh/venv_ada/bin/activate
export PATH="/fs/nexus-scratch/gnanesh/ffmpeg-7.0.2-amd64-static:$PATH"
export HF_HOME=/fs/nexus-scratch/gnanesh/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MAX_SAMPLES=1
export EXPERIMENT=videomathqa_test

PROJECT_ROOT=/fs/nexus-scratch/gnanesh/cot
VMQA_DIR="$PROJECT_ROOT/VideoMathQA"
METRICS_DIR="$PROJECT_ROOT/cot/metrics"
RESULTS_DIR="$METRICS_DIR/results/videomathqa_test"
CONVERTED_DATA="$RESULTS_DIR/videomathqa_converted.jsonl"

cd "$METRICS_DIR"
mkdir -p results logs

echo "=== Converting 1 VideoMathQA sample ==="
export VIDEOMATHQA_DIR="$VMQA_DIR"
python convert_videomathqa.py

export DATA_PATH="$CONVERTED_DATA"
export VIDEOS_DIR="$VMQA_DIR"

echo "=== Running all metrics on 1 sample ==="
python m6_efficiency.py
python m1_answer_faithfulness.py
python m2_stepwise_relevance.py
python m3_causal_coherence.py
python m4_completeness.py
python m5_factual_accuracy.py

echo "=== Computing OQS ==="
python compute_oqs.py videomathqa_test

echo "=== Done. Results in $RESULTS_DIR ==="
