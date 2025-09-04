#!/bin/bash
# Smart resume script for LR=0.001 OPTIMIZED clean-restart (inverse-sqrt)

#SBATCH --job-name=smart_resume_lr001_OPTIMIZED
#SBATCH --partition=H100-PCI,H100-SLT,H200,H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=5
#SBATCH --mem=600G
#SBATCH --time=24:00:00

#SBATCH --output=/netscratch/nrauscher/projects/BA-hydra/logs/slurm_%j_smart_resume_lr001_OPTIMIZED.out
#SBATCH --error=/netscratch/nrauscher/projects/BA-hydra/logs/slurm_%j_smart_resume_lr001_OPTIMIZED.err

echo "=============================================="
echo "Smart Resume T5 Continued Pretraining - OPTIMIZED COLLATOR"
echo "LR 0.001, Clip 1.0, Clean Restart"
echo "Collator: T5-formula (1.109x expansion, target=114)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="

# Function to find latest checkpoint under the experiment's Hydra run dir
find_latest_checkpoint() {
    ROOT_DIR="pretraining_logs_lr_001_OPTIMIZED_clean_restart/train/runs"

    # Find all pretraining run directories for this experiment
    RUN_DIRS=$(find "$ROOT_DIR" -name "20*" -type d 2>/dev/null | sort -r)
    
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
        LATEST_CKPT=$(find "$ROOT_DIR" -name "step-*.ckpt" 2>/dev/null | sort -V | tail -1)
        echo "$LATEST_CKPT"
    fi
}

# Find latest checkpoint
echo "Looking for latest OPTIMIZED checkpoint..."
LATEST_CKPT=$(find_latest_checkpoint)

if [[ -z "$LATEST_CKPT" ]]; then
    echo "No checkpoint found - starting fresh OPTIMIZED training"
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

cd /netscratch/nrauscher/projects/BA-hydra

echo "Starting T5 OPTIMIZED training with smart resume..."
echo "Configuration: configs/experiment/t5_continued_pretraining_lr_001_OPTIMIZED_clean_restart.yaml"
echo "Learning Rate: 0.001 with 20k warmup"
echo "Gradient Clip: 1.0"
echo "Effective batch size: 384 (48*4*2)"
echo "Optimization: T5-formula (35% more efficient than 1.5x heuristic)"

srun -K \
    --container-image=/netscratch/nrauscher/containers/hydra_pytorch_25.05.25-final.sqsh \
    --container-mounts=/netscratch:/netscratch,/fscratch:/fscratch \
    --container-workdir=/netscratch/nrauscher/projects/BA-hydra \
    /bin/bash -c "
        source .venv_pretraining/bin/activate && \
        python src/train.py experiment=t5_continued_pretraining_lr_001_OPTIMIZED_clean_restart trainer.log_every_n_steps=1 $RESUME_CMD
    "

echo "=============================================="
echo "Smart resume completed at: $(date)"
echo "Exit code: $?"
echo "=============================================="