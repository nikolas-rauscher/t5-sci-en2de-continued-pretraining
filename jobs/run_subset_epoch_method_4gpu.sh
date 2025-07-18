#!/bin/bash
#SBATCH --job-name=subset_epoch_method_4gpu
#SBATCH --partition=H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=15
#SBATCH --mem=400G
#SBATCH --time=24:00:00

#SBATCH --output=/netscratch/nrauscher/projects/BA-hydra/logs/slurm_%j_subset_epoch.out
#SBATCH --error=/netscratch/nrauscher/projects/BA-hydra/logs/slurm_%j_subset_epoch.err

# =============================================================================
# Subset-Epoch Method: 4-GPU H100 T5 Pre-training (3 epochs on 30% data)
# Multiple epochs for convergence analysis and learning curves
# =============================================================================

echo "=============================================="
echo "Starting Subset-Epoch Method T5 Pre-training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"  
echo "Start time: $(date)"
echo "=============================================="

# Check if fscratch is mounted
if [ ! -d "/fscratch" ]; then
    echo "ERROR: /fscratch not mounted!"
    exit 1
fi

if [ ! -d "/fscratch/nrauscher/projects/BA-hydra/data/cleaned_sliding_windows/text_2025-07-14" ]; then
    echo "ERROR: Training data not found"
    exit 1
fi

echo "✅ fscratch mounted and data accessible"
echo ""

# Critical environment variables for GPU visibility
export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS}
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS}
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo "GPU Environment Variables:"
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# Change to project directory
cd /netscratch/nrauscher/projects/BA-hydra

# Log system information
echo "=============================================="
echo "System Information:"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "Lightning: $(python -c 'import lightning; print(lightning.__version__)')"
echo "GPUs available: $(python -c 'import torch; print(torch.cuda.device_count())')"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "=============================================="

# Subset-Epoch Method Configuration
echo "Subset-Epoch Method Pre-training Configuration:"
echo "- Method: Multi-epoch convergence analysis"
echo "- Training Epochs: 3 full epochs"
echo "- Dataset: 30% of full data (~78M documents)"
echo "- Learning Rate: 5e-4 (standard T5)"
echo "- Batch Size: 512 sequences (32*4*4)"
echo "- Optimizer: Adafactor (standard T5)"
echo "- Scheduler: inverse-sqrt with 5k warmup"
echo "- Expected Duration: ~3-5 days"
echo "- Validation: Every epoch + 4x per epoch"
echo "- Checkpoints: Every 10k steps + after each epoch"
echo ""

# Run Subset-Epoch Method T5 pre-training
echo "Starting Subset-Epoch Method T5 training..."
echo "Configuration: configs/experiment/subset_epoch_method_4gpu.yaml"

srun -K \
    --container-image=/netscratch/nrauscher/containers/hydra_pytorch_25.05.25-final.sqsh \
    --container-mounts=/netscratch:/netscratch,/fscratch:/fscratch \
    --container-workdir=/netscratch/nrauscher/projects/BA-hydra \
    /bin/bash -c "
        source .venv_pretraining/bin/activate && \
        python src/train.py experiment=subset_epoch_method_4gpu
    "

# Log completion
echo "=============================================="
echo "Subset-Epoch Method Training completed at: $(date)"
echo "Exit code: $?"
echo "=============================================="

# Show final GPU status
echo "Final GPU memory status:"
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv

echo "Subset-Epoch Method Training completed. Check logs at:"
echo "- SLURM logs: /netscratch/nrauscher/projects/BA-hydra/logs/slurm_${SLURM_JOB_ID}_subset_epoch.*"
echo "- Training logs: outputs/[timestamp]/logs/"
echo "- Wandb: https://wandb.ai/[your-username]/BA-thesis-t5-final-runs"
echo "- Method: 3 epochs on 30% data for convergence analysis"