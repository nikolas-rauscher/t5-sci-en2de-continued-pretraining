#!/bin/bash
#SBATCH --job-name=prepare_german_sliding_windows
#SBATCH --partition=A100-40GB,V100-32GB,RTXA6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/logs/slurm_%j_prepare_german_data.out
#SBATCH --error=/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/logs/slurm_%j_prepare_german_data.err

echo "=============================================="
echo "German Sliding Window Data Preparation"
echo "Creating sliding windows for German continued pretraining"
echo "Window size: 512 tokens, Overlap: 50%"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="

cd /netscratch/nrauscher/projects/BA-hydra

# Create log directory
mkdir -p cross_lingual_transfer/logs

# Check if Wechsel model exists (needed for tokenizer)
if [ ! -d "cross_lingual_transfer/models/german_T5_Optimized_50Olap_clean_restart_487k/tokenizer" ]; then
    echo "ERROR: German tokenizer not found!"
    echo "Please run Wechsel transfer first:"
    echo "  sbatch cross_lingual_transfer/jobs/run_german_transfer.sh"
    exit 1
fi

echo "✅ German tokenizer found"

# Activate environment
echo "Activating .venv_crosslingual_transfer environment..."
source .venv_crosslingual_transfer/bin/activate

# Verify environment
echo "Python: $(which python)"
echo "Environment: $VIRTUAL_ENV"

echo "=============================================="
echo "Starting German sliding window preparation..."
echo "This will download German data and create sliding windows"
echo "=============================================="

python cross_lingual_transfer/scripts/prepare_german_sliding_windows.py

if [ $? -eq 0 ]; then
    echo "=============================================="
    echo "✅ German data preparation completed successfully!"
    echo "✅ Data saved to: cross_lingual_transfer/data/german/sliding_windows/"
    echo "Next step: Run German continued pretraining:"
    echo "  sbatch cross_lingual_transfer/jobs/run_german_continued_pretraining.sh"
    echo "Time: $(date)"
    echo "=============================================="
else
    echo "❌ German data preparation failed!"
    exit 1
fi