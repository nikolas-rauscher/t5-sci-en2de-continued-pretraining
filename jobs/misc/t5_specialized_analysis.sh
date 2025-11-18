#!/bin/bash
#SBATCH --job-name=t5-specialized-analysis
#SBATCH --partition=A100-80GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --time=11:00:00
#SBATCH --output=logs/t5_analysis_%j.out
#SBATCH --error=logs/t5_analysis_%j.err

# Set up environment
cd /netscratch/nrauscher/projects/BA-hydra
source .venv_eval/bin/activate

# GPU optimization
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE=/netscratch/nrauscher/projects/BA-hydra/.cache

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting T5-specialized analysis on GPU"
echo "Date: $(date)"
echo "GPU Info:"
nvidia-smi

# Run the analysis
python t5_specialized_analysis.py

echo "Analysis completed at: $(date)"