#!/bin/bash
#SBATCH --job-name=smart_resume_4gpu
#SBATCH --partition=H100-PCI,H100-SLT,H200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=600G
#SBATCH --time=24:00:00

#SBATCH --output=/netscratch/nrauscher/projects/BA-hydra/logs/slurm_%j_smart_resume_4gpu.out
#SBATCH --error=/netscratch/nrauscher/projects/BA-hydra/logs/slurm_%j_smart_resume_4gpu.err

echo "=============================================="
echo "Smart Resume 4-GPU H100 Production Run"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"  
echo "Start time: $(date)"
echo "=============================================="

# Function to find latest checkpoint
find_latest_checkpoint() {
    # Find all pretraining run directories
    RUN_DIRS=$(find pretraining_logs/train/runs -name "20*" -type d 2>/dev/null | sort -r)
    
    if [[ -n "$RUN_DIRS" ]]; then
        # Try each run directory from newest to oldest
        for RUN_DIR in $RUN_DIRS; do
            # Check for checkpoints in order of preference
            if [[ -f "$RUN_DIR/checkpoints/steps/last.ckpt" ]]; then
                echo "$RUN_DIR/checkpoints/steps/last.ckpt"
                return
            elif [[ -f "$RUN_DIR/checkpoints/best/last.ckpt" ]]; then
                echo "$RUN_DIR/checkpoints/best/last.ckpt"
                return
            fi
        done
        
        # Fallback: find latest numbered checkpoint
        LATEST_CKPT=$(find pretraining_logs/train/runs -name "step-*.ckpt" 2>/dev/null | sort -V | tail -1)
        echo "$LATEST_CKPT"
    fi
}

# Find latest checkpoint
echo "Looking for latest checkpoint..."
LATEST_CKPT=$(find_latest_checkpoint)

if [[ -z "$LATEST_CKPT" ]]; then
    echo "No checkpoint found - starting fresh training"
    RESUME_CMD=""
else
    echo "Found checkpoint: $LATEST_CKPT"
    RESUME_CMD="ckpt_path=$LATEST_CKPT"
fi

# Check data access
if [ ! -d "/fscratch/nrauscher/projects/BA-hydra/data/cleaned_sliding_windows/text_2025-07-14" ]; then
    echo "ERROR: Training data not found"
    exit 1
fi

echo "✅ fscratch mounted and data accessible"

# Environment variables
export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS}
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS}
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

cd /netscratch/nrauscher/projects/BA-hydra

echo "Starting T5 training with smart resume..."
echo "Configuration: configs/experiment/100_4GPU_10mio_doc.yaml"
echo "Effective batch size: 384 (48*4*2)"

srun -K \
    --container-image=/netscratch/nrauscher/containers/hydra_pytorch_25.05.25-final.sqsh \
    --container-mounts=/netscratch:/netscratch,/fscratch:/fscratch \
    --container-workdir=/netscratch/nrauscher/projects/BA-hydra \
    /bin/bash -c "
        source .venv_pretraining/bin/activate && \
        python src/train.py experiment=100_4GPU_10mio_doc trainer.log_every_n_steps=1 $RESUME_CMD
    "

echo "=============================================="
echo "Smart resume completed at: $(date)"
echo "Exit code: $?"
echo "=============================================="