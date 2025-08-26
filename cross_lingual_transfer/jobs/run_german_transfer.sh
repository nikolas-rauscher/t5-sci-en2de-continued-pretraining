#!/bin/bash
#SBATCH --job-name=german_cross_lingual_transfer
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=cross_lingual_transfer/logs/slurm_%j_german_transfer.out
#SBATCH --error=cross_lingual_transfer/logs/slurm_%j_german_transfer.err

# German cross-lingual transfer pipeline
# 1. Install requirements
# 2. Prepare German data  
# 3. Apply Wechsel transfer
# 4. Run German continued pretraining
# 5. Evaluate cross-lingual transfer

echo "Starting German cross-lingual transfer pipeline..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Setup environment
source ~/.bashrc
cd /netscratch/nrauscher/projects/BA-hydra

# Create log directory
mkdir -p cross_lingual_transfer/logs

# Activate conda environment (adjust as needed)
# conda activate your_env

echo "=== Step 1: Installing Requirements ==="
python cross_lingual_transfer/scripts/install_requirements.py

echo "=== Step 2: Preparing German Data ==="
python cross_lingual_transfer/scripts/german_data_preparation.py

echo "=== Step 3: Applying Wechsel Transfer ==="
python cross_lingual_transfer/scripts/wechsel_transfer.py

echo "=== Step 4: German Continued Pretraining ==="
python src/train.py experiment=german_continued_pretraining

echo "=== Step 5: Evaluation ==="
python cross_lingual_transfer/scripts/run_german_evaluation.py

echo "German cross-lingual transfer pipeline completed!"
echo "Results saved in cross_lingual_transfer/"