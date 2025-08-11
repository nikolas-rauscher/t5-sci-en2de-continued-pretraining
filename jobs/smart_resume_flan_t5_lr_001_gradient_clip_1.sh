#!/bin/bash
#SBATCH --job-name=smart_resume_flan_t5_lr001
#SBATCH --partition=H200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=600G
#SBATCH --time=24:00:00

#SBATCH --output=/netscratch/nrauscher/projects/BA-hydra/logs/slurm_%j_smart_resume_flan_t5_lr001.out
#SBATCH --error=/netscratch/nrauscher/projects/BA-hydra/logs/slurm_%j_smart_resume_flan_t5_lr001.err

echo "=============================================="
echo "Smart Resume FLAN-T5 - LR 0.001, Gradient Clip 1.0"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"  
echo "Start time: $(date)"
echo "=============================================="

# Function to find latest checkpoint
find_latest_checkpoint() {
    # Find all FLAN-T5 run directories for this experiment
    RUN_DIRS=$(find flan_t5_logs_lr_001_gradient_clip_1_with_inverse_sqrt_schedule/train/runs -name "20*" -type d 2>/dev/null | sort -r)
    
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
        LATEST_CKPT=$(find flan_t5_logs_lr_001_gradient_clip_1_with_inverse_sqrt_schedule/train/runs -name "step-*.ckpt" 2>/dev/null | sort -V | tail -1)
        echo "$LATEST_CKPT"
    fi
}

# Find latest checkpoint
echo "Looking for latest FLAN-T5 checkpoint..."
LATEST_CKPT=$(find_latest_checkpoint)

if [[ -z "$LATEST_CKPT" ]]; then
    echo "No checkpoint found - starting fresh training"
    RESUME_CMD=""
else
    echo "Found checkpoint: $LATEST_CKPT"
    RESUME_CMD="ckpt_path=$LATEST_CKPT"
fi

# Check data access
if [ ! -d "/fscratch/nrauscher/projects/BA-hydra/data/validated_sliding_windows/validated/" ]; then
    echo "ERROR: Training data not found"
    exit 1
fi

echo "✅ fscratch mounted and data accessible"

# Environment variables
export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS}
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS}
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# Memory optimization (no config changes needed)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Clear GPU memory before start
export CUDA_LAUNCH_BLOCKING=1
export SINGULARITYENV_CUDA_LAUNCH_BLOCKING=1

cd /netscratch/nrauscher/projects/BA-hydra

echo "Starting FLAN-T5 training with smart resume..."
echo "Configuration: configs/experiment/flan_t5_lr_001_gradient_clip_1_with_inverse_sqrt_schedule.yaml"
echo "Model: google/flan-t5-base"
echo "Learning Rate: 0.001 with 20k warmup"
echo "Gradient Clip: 1.0"
echo "Effective batch size: 384 (48*4*2)"

srun -K \
    --container-image=/netscratch/nrauscher/containers/hydra_pytorch_25.05.25-final.sqsh \
    --container-mounts=/netscratch:/netscratch,/fscratch:/fscratch \
    --container-workdir=/netscratch/nrauscher/projects/BA-hydra \
    /bin/bash -c "
        # Clear any residual GPU memory
        nvidia-smi --gpu-reset || true && \
        source .venv_pretraining/bin/activate && \
        python src/train.py experiment=flan_t5_lr_001_gradient_clip_1_with_inverse_sqrt_schedule trainer.log_every_n_steps=1 $RESUME_CMD
    "

echo "=============================================="
echo "Smart resume completed at: $(date)"
echo "Exit code: $?"
echo "=============================================="