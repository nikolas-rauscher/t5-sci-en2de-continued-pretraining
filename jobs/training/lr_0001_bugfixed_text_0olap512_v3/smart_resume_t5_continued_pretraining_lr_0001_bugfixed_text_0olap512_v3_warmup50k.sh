#!/bin/bash
# Smart resume script for T5 continued pretraining on 0% overlap text windows (512, v3) with warmup=50k

#SBATCH --job-name=smart_resume_lr0001_text0olap512v3_50k
#SBATCH --partition=H100-PCI,H100-SLT,H200,H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=5
#SBATCH --mem=600G
#SBATCH --time=24:00:00

#SBATCH --output=/netscratch/nrauscher/projects/BA-hydra/logs/slurm_%j_smart_resume_lr0001_text0olap512v3_50k.out
#SBATCH --error=/netscratch/nrauscher/projects/BA-hydra/logs/slurm_%j_smart_resume_lr0001_text0olap512v3_50k.err

echo "=============================================="
echo "Smart Resume T5 Continued Pretraining - LR 0.0001, Warmup 50k, 0% overlap (512, v3)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="

# Function to find latest checkpoint under this experiment's Hydra run dir
find_latest_checkpoint() {
    ROOT_DIR="pretraining_logs_text_0olap512_v3_lr_0001_bugfixed/train/runs"

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
echo "Looking for latest checkpoint..."
LATEST_CKPT=$(find_latest_checkpoint)

if [[ -z "$LATEST_CKPT" ]]; then
    echo "No checkpoint found - starting fresh training"
    RESUME_CMD=""
else
    echo "Found checkpoint: $LATEST_CKPT"
    RESUME_CMD="ckpt_path=$LATEST_CKPT"
fi

# Check data access for this run's dataset
DATA_DIR="/fscratch/nrauscher/projects/BA-hydra/data/windows_text_0olap_512_v3/enriched_windows"
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Training data not found at $DATA_DIR"
    exit 1
fi

echo "✅ fscratch mounted and data accessible: $DATA_DIR"

# Environment variables
export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS}
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS}
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

cd /netscratch/nrauscher/projects/BA-hydra

echo "Starting T5 training with smart resume..."
echo "Configuration: configs/experiment/t5_continued_pretraining_lr_0001_bugfixed_text_0olap512_v3_warmup50k.yaml"
echo "Learning Rate: 0.0001 with 50k warmup"
echo "Gradient Clip: 1.0"
echo "Effective batch size: 384 (48*4*2)"

srun -K \
    --container-image=/netscratch/nrauscher/containers/hydra_pytorch_25.05.25-final.sqsh \
    --container-mounts=/netscratch:/netscratch,/fscratch:/fscratch \
    --container-workdir=/netscratch/nrauscher/projects/BA-hydra \
    /bin/bash -c "
        source .venv_pretraining/bin/activate && \
        python src/train.py experiment=t5_continued_pretraining_lr_0001_bugfixed_text_0olap512_v3_warmup50k trainer.log_every_n_steps=1 $RESUME_CMD
    "

EXIT_CODE=$?
echo "=============================================="
echo "Smart resume completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=============================================="

