#!/bin/bash
#SBATCH --job-name=perplexity_eval
#SBATCH --output=/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/logs/perplexity_eval_%j.out
#SBATCH --error=/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/logs/perplexity_eval_%j.err
#SBATCH --partition=A100-80GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00

# Load necessary modules
module load cuda/12.1

# Activate environment
source /netscratch/nrauscher/projects/BA-hydra/.venv/bin/activate

# Change to project directory
cd /netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer

# Create output directories
mkdir -p evaluation/perplexity_comparison
mkdir -p logs

# Run the perplexity evaluation
echo "Starting perplexity evaluation at $(date)"
echo "Comparing T5-base vs German Gold Model on German scientific text"
echo "=================================================="

python scripts/evaluate_perplexity_comparison.py

echo "=================================================="
echo "Evaluation completed at $(date)"

# Show results location
echo "Results saved to:"
echo "  - evaluation/perplexity_comparison/perplexity_comparison.png"
echo "  - evaluation/perplexity_comparison/perplexity_comparison.csv"
echo "  - evaluation/perplexity_comparison/perplexity_results.json"