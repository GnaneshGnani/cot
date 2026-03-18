#!/bin/bash
#SBATCH --job-name=eval_metrics
#SBATCH --output=logs/eval_metrics_%j.out
#SBATCH --error=logs/eval_metrics_%j.err
#SBATCH --time=12:00:00
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

cd /fs/nexus-scratch/gnanesh/cot/cot/metrics
mkdir -p results

for exp in skip_last_step truncate_50 truncate_25 truncate_75 first_step_only evidence_only inferences_only vision_steps_only audio_steps_only; do
    echo "=== Experiment: $exp ==="
    python m6_efficiency.py $exp
    python m1_answer_faithfulness.py $exp
    python m2_stepwise_relevance.py $exp
    python m3_causal_coherence.py $exp
    python m4_completeness.py $exp
    python m5_factual_accuracy.py $exp
done
