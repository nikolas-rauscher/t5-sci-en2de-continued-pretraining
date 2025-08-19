#!/bin/bash
#SBATCH --job-name=flexible_eval_24h
#SBATCH --partition=A100-80GB,V100-32GB,A100-40GB,H100,RTXA6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=75G
#SBATCH --time=24:00:00
#SBATCH --output=/netscratch/nrauscher/projects/BA-hydra/evaluation/logs/flexible_eval_24h_%j.out
#SBATCH --error=/netscratch/nrauscher/projects/BA-hydra/evaluation/logs/flexible_eval_24h_%j.err

# Flexible Evaluation Pipeline Job - 24 Hour Version
# For long-running tasks like summarization with multiple checkpoints
# Usage: sbatch jobs/flexible_eval_24h.sh [experiment_name]
# Examples:
#   sbatch jobs/flexible_eval_24h.sh run1_eval_summarization
#   sbatch jobs/flexible_eval_24h.sh run1_eval_qasper  
#   sbatch jobs/flexible_eval_24h.sh checkpoint_progression

# Get experiment name from command line argument or default
EXPERIMENT=${1:-"quick_test"}

echo "=========================================="
echo "SLURM Job: Flexible Evaluation Pipeline (24h)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Experiment: $EXPERIMENT"
echo "Time limit: 24 hours"
echo "=========================================="

# Project paths
PROJECT_ROOT="/netscratch/nrauscher/projects/BA-hydra"
cd "$PROJECT_ROOT"

# Create log directories
mkdir -p evaluation/logs

# Activate evaluation environment
echo "Activating evaluation environment..."
source .venv_eval/bin/activate

# Show environment info
echo ""
echo "Environment setup:"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "HF_DATASETS_CACHE: $PWD/data/mmlu_datasets"
echo "HF_HOME: $PWD/data/huggingface_cache"

# Show GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits

# Check cache directories
echo ""
echo "Cache Status:"
if [ -d "data/mmlu_datasets" ]; then
    echo "✓ MMLU datasets cache found ($(du -sh data/mmlu_datasets | cut -f1))"
else
    echo "✗ MMLU datasets cache missing"
fi

if [ -d "data/huggingface_cache" ]; then
    echo "✓ HuggingFace cache found ($(du -sh data/huggingface_cache | cut -f1))"
else
    echo "✗ HuggingFace cache missing"
fi

echo ""
echo "=========================================="
echo "Starting Evaluation: $EXPERIMENT"
echo "=========================================="

# Set Python to unbuffered mode for real-time logging
export PYTHONUNBUFFERED=1

# Run evaluation with error handling
if python src/eval_pipeline.py experiment=$EXPERIMENT; then
    echo ""
    echo "=========================================="
    echo "✅ Evaluation completed successfully!"
    echo "Experiment: $EXPERIMENT"
    echo "End time: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "❌ Evaluation failed!"
    echo "Experiment: $EXPERIMENT"
    echo "End time: $(date)"
    echo "Check logs for details."
    echo "=========================================="
    exit 1
fi

# Show final GPU memory
echo ""
echo "Final GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader,nounits

echo ""
echo "Job completed successfully! 🎉"