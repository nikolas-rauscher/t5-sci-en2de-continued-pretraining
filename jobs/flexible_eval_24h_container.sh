#!/bin/bash
#SBATCH --job-name=flex_eval_24h_ctn
#SBATCH --partition=H100,H200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=75G
#SBATCH --time=24:00:00
#SBATCH --output=/netscratch/nrauscher/projects/BA-hydra/evaluation/logs/flexible_eval_24h_container_%j.out
#SBATCH --error=/netscratch/nrauscher/projects/BA-hydra/evaluation/logs/A_flexible_eval_24h_container_%j.err

# Usage: sbatch jobs/flexible_eval_24h_container.sh [experiment_name]

# Keep this script intentionally simple and robust
set -e

EXPERIMENT=${1:-"quick_test"}
export EXPERIMENT

echo "=========================================="
echo "SLURM Job: Flexible Evaluation (container)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Experiment: $EXPERIMENT"
echo "=========================================="

PROJECT_ROOT="/netscratch/nrauscher/projects/BA-hydra"
IMG="/netscratch/nrauscher/containers/hydra_pytorch_25.05.25-final.sqsh"

mkdir -p "$PROJECT_ROOT/evaluation/logs"

echo "Launching container: $IMG"

# Run everything inside the container with minimal quoting; assume venv exists
srun -K \
  --container-image="$IMG" \
  --container-mounts=/netscratch:/netscratch,/fscratch:/fscratch \
  --container-workdir="$PROJECT_ROOT" \
  /bin/bash -lc 'set -e; export PYTHONUNBUFFERED=1; export HYDRA_FULL_ERROR=1; echo Inside container; echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"; echo "Working dir: $(pwd)"; mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 || echo 0); if [ "$mem" -ge 80000 ]; then export EVAL_MAX_PARALLEL=4; elif [ "$mem" -ge 60000 ]; then export EVAL_MAX_PARALLEL=2; else export EVAL_MAX_PARALLEL=1; fi; source .venv_eval/bin/activate; which python; python --version; echo "Starting evaluation ($EXPERIMENT) with EVAL_MAX_PARALLEL=$EVAL_MAX_PARALLEL"; python src/eval_pipeline.py experiment="$EXPERIMENT"'

STATUS=$?

if [ $STATUS -eq 0 ]; then
  echo "\n=========================================="
  echo "✅ Evaluation completed successfully!"
  echo "Experiment: $EXPERIMENT"
  echo "End time: $(date)"
  echo "=========================================="
else
  echo "\n=========================================="
  echo "❌ Evaluation failed!"
  echo "Experiment: $EXPERIMENT"
  echo "End time: $(date)"
  echo "Check logs for details."
  echo "=========================================="
  exit 1
fi

echo "\nFinal GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader,nounits || true

echo "\nJob completed successfully! 🎉"
