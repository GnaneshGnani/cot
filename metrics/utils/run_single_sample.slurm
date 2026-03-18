#!/bin/bash
#SBATCH --job-name=run_single_sample
#SBATCH --output=logs/run_single_sample_%j.out
#SBATCH --error=logs/run_single_sample_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=high
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48G
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
mkdir -p logs
python run_single_sample.py 0
