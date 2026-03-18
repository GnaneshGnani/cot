#!/bin/bash
#SBATCH --job-name=robustness
#SBATCH --array=0-9%5
#SBATCH --output=logs/robustness_%A_%a.out
#SBATCH --error=logs/robustness_%A_%a.err
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
PERTURBS_DIR="$METRICS_DIR/results/robustness_perturbs"

cd "$METRICS_DIR"
mkdir -p logs "$PERTURBS_DIR"

# Map array task ID to perturbation name
PERTURBATION_NAMES=(
    shuffled
    duplicate_steps
    corrupt_final
    invalid_timestamps
    inject_irrelevant
    single_step
    wrong_answer
    generic_evidence
    contradiction
    keyword_stuffing
)

PERTURB="${PERTURBATION_NAMES[$SLURM_ARRAY_TASK_ID]}"
EXPERIMENT="perturb_${PERTURB}"
DATA_PATH="${PERTURBS_DIR}/${PERTURB}.jsonl"

echo "=== Task ${SLURM_ARRAY_TASK_ID}: perturbation=${PERTURB}, experiment=${EXPERIMENT} ==="

# Step 1: generate this perturbation's data file (20 samples for faster run)
echo "--- Generating perturbed data ---"
python perturbation_generator.py --perturb "$PERTURB" --max_samples 20
if [ $? -ne 0 ]; then
    echo "ERROR: perturbation_generator.py failed for ${PERTURB}"
    exit 1
fi

# Step 2: run all 6 metrics (m6 first as it's GPU-free and fast)
export DATA_PATH
export EXPERIMENT
export VIDEOS_DIR=""

echo "--- Running M6 Efficiency ---"
python m6_efficiency.py

echo "--- Running M1 Answer Faithfulness ---"
python m1_answer_faithfulness.py

echo "--- Running M2 Stepwise Relevance ---"
python m2_stepwise_relevance.py

echo "--- Running M3 Causal Coherence ---"
python m3_causal_coherence.py

# M4, M5, compute_oqs skipped (not present in this setup)
if [ -f m4_completeness.py ]; then
    echo "--- Running M4 Completeness ---"
    python m4_completeness.py
fi
if [ -f m5_factual_accuracy.py ]; then
    echo "--- Running M5 Factual Accuracy ---"
    python m5_factual_accuracy.py
fi
if [ -f compute_oqs.py ]; then
    echo "--- Computing OQS for ${EXPERIMENT} ---"
    python compute_oqs.py "$EXPERIMENT"
fi

echo "=== Done: ${EXPERIMENT} ==="
