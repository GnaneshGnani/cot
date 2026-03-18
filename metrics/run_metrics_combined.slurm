#!/bin/bash
#SBATCH --job-name=metrics_combined
#SBATCH --output=logs/eval_combined_%j.out
#SBATCH --error=logs/eval_combined_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=high
#SBATCH --gres=gpu:l40s:2
#SBATCH --mem=64G
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

PROJECT_ROOT=/fs/nexus-scratch/gnanesh/cot
METRICS_DIR="$PROJECT_ROOT/cot/metrics"
RESULTS_DIR="$METRICS_DIR/results/combined_20"
DATA_FILE="$RESULTS_DIR/combined_20_samples.jsonl"

cd "$METRICS_DIR"
mkdir -p results logs

echo "=== Creating combined dataset (10 OVB + 10 VideoMathQA) ==="
export OUTPUT_PATH="$DATA_FILE"
python create_combined_dataset.py
if [ $? -ne 0 ]; then
    echo "Dataset creation failed"
    exit 1
fi

export DATA_PATH="$DATA_FILE"
export VIDEOS_DIR=""
export EXPERIMENT=combined_20

echo "=== Running all metrics on 20 samples ==="
python m6_efficiency.py
python m1_answer_faithfulness.py
python m2_stepwise_relevance.py
python m3_causal_coherence.py
python m4_completeness.py
python m5_factual_accuracy.py

echo "=== Computing OQS ==="
python compute_oqs.py combined_20

echo "=== Done. Results in $RESULTS_DIR ==="
