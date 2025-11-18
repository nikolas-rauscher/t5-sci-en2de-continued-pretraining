#!/bin/bash
#SBATCH --job-name=flex_eval_24h_analyze
#SBATCH --partition=A100-80GB,H200,H100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=175G
#SBATCH --time=24:00:00
#SBATCH --output=/netscratch/nrauscher/projects/BA-hydra/evaluation/logs/flexible_eval_24h_analyze_%j.out
#SBATCH --error=/netscratch/nrauscher/projects/BA-hydra/evaluation/logs/flexible_eval_24h_analyze_%j.err

# Usage: sbatch jobs/flexible_eval_24h_with_analysis.sh <experiment_name>

EXPERIMENT=${1:-"quick_test"}

echo "=========================================="
echo "SLURM Job: Flexible Evaluation + Analysis (24h)"
echo "Experiment: $EXPERIMENT"
echo "=========================================="

PROJECT_ROOT="/netscratch/nrauscher/projects/BA-hydra"
export PROJECT_ROOT
cd "$PROJECT_ROOT" || exit 1

mkdir -p evaluation/logs

echo "Activating eval env..."
source .venv_eval/bin/activate

echo "Running eval_pipeline..."
export PYTHONUNBUFFERED=1
if ! python -u src/eval_pipeline.py experiment=$EXPERIMENT; then
  echo " Evaluation failed"
  exit 1
fi

# Determine latest Hydra run dir for eval_pipeline
RUN_DIR=$(ls -td logs/eval_pipeline/runs/* | head -n 1)
echo "Latest run dir: $RUN_DIR"

echo "Running analysis to produce heatmaps..."
python evaluation/analysis_scripts/analyze_evaluation_results.py "$RUN_DIR" || true

echo "Done. Generated analysis under: $RUN_DIR"

