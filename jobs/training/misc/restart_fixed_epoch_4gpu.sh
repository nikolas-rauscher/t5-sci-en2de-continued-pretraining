#!/bin/bash
#SBATCH --job-name=restart_fixed_epoch_4gpu
#SBATCH --partition=H100-PCI,H100-SLT,H200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=600G
#SBATCH --time=24:00:00

#SBATCH --output=/netscratch/nrauscher/projects/BA-hydra/logs/slurm_%j_restart_fixed_epoch_4gpu.out
#SBATCH --error=/netscratch/nrauscher/projects/BA-hydra/logs/slurm_%j_restart_fixed_epoch_4gpu.err

# =============================================================================
# RESTART with Fixed DataLoader Resume from Step 145k - Correct Epochs
# =============================================================================

echo "=============================================="
echo "Starting RESTART with Fixed DataLoader Resume"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"  
echo "Start time: $(date)"
echo "Restarting from: step-145000.ckpt"
echo "Expected: Correct Epoch 0 display"
echo "=============================================="

# Check if fscratch is mounted
if [ ! -d "/fscratch" ]; then
    echo "ERROR: /fscratch not mounted!"
    exit 1
fi

if [ ! -d "/fscratch/nrauscher/projects/BA-hydra/data/validated_sliding_windows/validated" ]; then
    echo "ERROR: Training data not found"
    exit 1
fi

echo "✅ fscratch mounted and data accessible"

# Check if checkpoint exists
CHECKPOINT_PATH="pretraining_logs/train/runs/2025-07-24_02-18-57/checkpoints/steps/last.ckpt"
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

echo "✅ Checkpoint found: $CHECKPOINT_PATH"
RESUME_CMD="ckpt_path=$CHECKPOINT_PATH"
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

# Run T5 pre-training with fixed DataLoader resume
echo "Starting T5 training with FIXED DataLoader resume..."
echo "Configuration: configs/experiment/100_4GPU_10mio_doc.yaml"
echo "Resume from: step-145000.ckpt (before epoch bug)"
echo "Target: 600k steps with correct epoch counting"
echo "Effective batch size: 384 (48*4*2)"
echo "Expected: Epoch 0 display, correct progression"

srun -K \
    --container-image=/netscratch/nrauscher/containers/hydra_pytorch_25.05.25-final.sqsh \
    --container-mounts=/netscratch:/netscratch,/fscratch:/fscratch \
    --container-workdir=/netscratch/nrauscher/projects/BA-hydra \
    /bin/bash -c "
        source .venv_pretraining/bin/activate && \
        python src/train.py experiment=100_4GPU_10mio_doc trainer.log_every_n_steps=1 $RESUME_CMD
    "

# Log completion
echo "=============================================="
echo "Training completed at: $(date)"
echo "Exit code: $?"
echo "=============================================="

# Show final GPU status
echo "Final GPU memory status:"
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv

echo "Training completed. Check logs at:"
echo "- SLURM logs: /netscratch/nrauscher/projects/BA-hydra/logs/slurm_${SLURM_JOB_ID}_restart_fixed_epoch_4gpu.*"
echo "- Training logs: pretraining_logs/train/runs/[timestamp]/"
echo "- Wandb: Check for correct epoch progression starting from Epoch 0"