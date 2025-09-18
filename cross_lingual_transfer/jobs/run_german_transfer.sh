#!/bin/bash
#SBATCH --job-name=german_cross_lingual_transfer
#SBATCH --partition=A100-80GB,V100-32GB,A100-40GB,H100,RTXA6000,V100-32GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --mem=100G
#SBATCH --time=6:00:00
#SBATCH --output=/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/logs/slurm_%j_german_transfer.out
#SBATCH --error=/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/logs/slurm_%j_german_transfer.err

# German cross-lingual transfer pipeline
# Apply Wechsel transfer from best English checkpoint to German

echo "=============================================="
echo "Starting German Cross-Lingual Transfer"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo "=============================================="

# Setup environment
cd /netscratch/nrauscher/projects/BA-hydra

# Create log directory
mkdir -p cross_lingual_transfer/logs

# Activate crosslingual transfer environment
echo "Activating .venv_crosslingual_transfer environment..."
source .venv_crosslingual_transfer/bin/activate

# Verify environment
echo "Python: $(which python)"
echo "Environment: $VIRTUAL_ENV"

echo "=== Applying ROBUST Wechsel Transfer ==="
echo "Source: Clean Restart Checkpoint (487k steps, val_ppl 3.72168)"
echo "Target: German T5 (Optimized_50Olap_clean_restart_487k)"
echo "Method: Wechsel embedding transfer with robustness fixes"
echo "=============================================="

python cross_lingual_transfer/scripts/wechsel_transfer_robust.py

if [ $? -eq 0 ]; then
    echo "=============================================="
    echo "✅ German Cross-Lingual Transfer completed successfully!"
    echo "✅ Results saved in: cross_lingual_transfer/models/german_T5_Optimized_50Olap_clean_restart_487k/"
    echo "Time: $(date)"
    echo "=============================================="
else
    echo "❌ German Cross-Lingual Transfer failed!"
    exit 1
fi