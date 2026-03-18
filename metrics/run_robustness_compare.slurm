#!/bin/bash
#SBATCH --job-name=robustness_compare
#SBATCH --output=logs/robustness_compare_%j.out
#SBATCH --error=logs/robustness_compare_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=high
#SBATCH --mem=8G
#SBATCH --mail-user=gnanesh@umd.edu
#SBATCH --mail-type=END,FAIL

module purge
module load gcc/11.2.0
module load Python3/3.10.14

source /fs/nexus-scratch/gnanesh/venv_ada/bin/activate

PROJECT_ROOT=/fs/nexus-scratch/gnanesh/cot
METRICS_DIR="$PROJECT_ROOT/cot/metrics"

cd "$METRICS_DIR"
mkdir -p logs

echo "=== Running robustness comparison ==="
python compare_robustness.py

echo "=== Done. See results/robustness_summary.txt ==="
