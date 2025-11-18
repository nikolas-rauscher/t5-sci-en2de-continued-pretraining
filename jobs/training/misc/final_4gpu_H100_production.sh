#!/bin/bash
#SBATCH --job-name=fixed_4gpu_H100_production
#SBATCH --partition=H100-PCI,H100-SLT,H200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=600G
#SBATCH --time=24:00:00

#SBATCH --output=/netscratch/nrauscher/projects/BA-hydra/logs/slurm_%j_fixed_4gpu_H100_production.out
#SBATCH --error=/netscratch/nrauscher/projects/BA-hydra/logs/slurm_%j_fixed_4gpu_H100_production.err

# =============================================================================
# FINAL 4-GPU H100/H200 T5 200k Steps Production Run - Bachelor Thesis
# =============================================================================

echo "=============================================="
echo "Starting FIXED 4-GPU H100/H200 Production Run"
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

# Run T5 pre-training test with 4-GPU configuration
echo "Starting T5 training test with 4-GPU configuration..."
echo "Configuration: configs/experiment/100_4GPU_10mio_doc.yaml"
echo "Expected duration: <30 minutes"
echo "Target: 1 epoch over 10k samples"
echo "Effective batch size: 384 (48*4*2)"

srun -K \
    --container-image=/netscratch/nrauscher/containers/hydra_pytorch_25.05.25-final.sqsh \
    --container-mounts=/netscratch:/netscratch,/fscratch:/fscratch \
    --container-workdir=/netscratch/nrauscher/projects/BA-hydra \
    /bin/bash -c "
        source .venv_pretraining/bin/activate && \
        python src/train.py experiment=100_4GPU_10mio_doc trainer.log_every_n_steps=1
    "

# Log completion
echo "=============================================="
echo "Test completed at: $(date)"
echo "Exit code: $?"
echo "=============================================="

# Show final GPU status
echo "Final GPU memory status:"
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv

echo "Test completed. Check logs at:"
echo "- SLURM logs: /netscratch/nrauscher/projects/BA-hydra/logs/slurm_${SLURM_JOB_ID}_fixed_4gpu_H100_production.*"
echo "- Training logs: outputs/[timestamp]/logs/"
echo "- Wandb: https://wandb.ai/[your-username]/BA-thesis-t5-final-runs"