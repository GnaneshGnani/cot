#!/bin/bash
#SBATCH --job-name=vmqa_metrics
#SBATCH --output=logs/eval_videomathqa_%j.out
#SBATCH --error=logs/eval_videomathqa_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=high
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=120G
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
VMQA_DIR="$PROJECT_ROOT/VideoMathQA"
METRICS_DIR="$PROJECT_ROOT/cot/metrics"
RESULTS_DIR="$METRICS_DIR/results/videomathqa"
CONVERTED_DATA="$RESULTS_DIR/videomathqa_converted.jsonl"

cd "$METRICS_DIR"
mkdir -p results logs

echo "=== Converting VideoMathQA to metrics format ==="
export VIDEOMATHQA_DIR="$VMQA_DIR"
python convert_videomathqa.py
if [ $? -ne 0 ]; then
    echo "Conversion failed"
    exit 1
fi

export DATA_PATH="$CONVERTED_DATA"
export VIDEOS_DIR="$VMQA_DIR"
export EXPERIMENT=videomathqa

echo "=== Running all metrics on VideoMathQA ==="
python m6_efficiency.py
python m1_answer_faithfulness.py
python m2_stepwise_relevance.py
python m3_causal_coherence.py
python m4_completeness.py
python m5_factual_accuracy.py

echo "=== Computing OQS ==="
python compute_oqs.py videomathqa

echo "=== Done. Results in $RESULTS_DIR ==="
echo "  - per_sample_detailed.jsonl: scores + debug per sample"
echo "  - aggregated.json: summary statistics"
