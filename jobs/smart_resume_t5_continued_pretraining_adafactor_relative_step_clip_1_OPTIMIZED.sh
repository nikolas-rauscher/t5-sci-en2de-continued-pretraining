#!/bin/bash
#SBATCH --job-name=resume_t5_ada_relstep_OPT
#SBATCH --partition=H100,H100-PCI,H100-SLT,H200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=600G
#SBATCH --time=24:00:00

# Logs
#SBATCH --output=/netscratch/nrauscher/projects/BA-hydra/logs/slurm_%j_resume_t5_ada_relstep_OPT.out
#SBATCH --error=/netscratch/nrauscher/projects/BA-hydra/logs/slurm_%j_resume_t5_ada_relstep_OPT.err

echo "=============================================="
echo "SMART RESUME: T5 Adafactor OPTIMIZED - 50% Overlap"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="

# Check data access
DATA_DIR="/fscratch/nrauscher/projects/BA-hydra/data/validated_sliding_windows/validated/"
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Training data not found at $DATA_DIR"
    exit 1
fi

echo "✅ fscratch mounted and data accessible (50% overlap dataset)"

# Environment variables
export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS}
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS}
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

cd /netscratch/nrauscher/projects/BA-hydra

# Find the latest run directory
RUN_DIR=$(find pretraining_logs_adafactor_relative_step_clip1_OPTIMIZED/train/runs -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)

if [ -z "$RUN_DIR" ]; then
    echo "No previous run found. Starting fresh training..."
    
    srun -K \
        --container-image=/netscratch/nrauscher/containers/hydra_pytorch_25.05.25-final.sqsh \
        --container-mounts=/netscratch:/netscratch,/fscratch:/fscratch \
        --container-workdir=/netscratch/nrauscher/projects/BA-hydra \
        /bin/bash -c "
            source .venv_pretraining/bin/activate && \
            python src/train.py experiment=t5_continued_pretraining_adafactor_relative_step_clip_1_OPTIMIZED trainer.log_every_n_steps=1
        "
else
    echo "Found previous run: $RUN_DIR"
    
    # Look for checkpoint in order of preference
    CKPT_PATH=""
    
    # 1. Check for last.ckpt in steps directory
    if [ -f "$RUN_DIR/checkpoints/steps/last.ckpt" ]; then
        CKPT_PATH="$RUN_DIR/checkpoints/steps/last.ckpt"
        echo "Found last.ckpt in steps directory"
    # 2. Check for last.ckpt in best directory
    elif [ -f "$RUN_DIR/checkpoints/best/last.ckpt" ]; then
        CKPT_PATH="$RUN_DIR/checkpoints/best/last.ckpt"
        echo "Found last.ckpt in best directory"
    # 3. Find the latest step checkpoint
    else
        LATEST_STEP=$(find "$RUN_DIR/checkpoints" -name "step-*.ckpt" | sort -V | tail -1)
        if [ ! -z "$LATEST_STEP" ]; then
            CKPT_PATH="$LATEST_STEP"
            echo "Found checkpoint: $(basename $LATEST_STEP)"
        fi
    fi
    
    if [ -z "$CKPT_PATH" ]; then
        echo "No checkpoint found. Starting fresh training..."
        
        srun -K \
            --container-image=/netscratch/nrauscher/containers/hydra_pytorch_25.05.25-final.sqsh \
            --container-mounts=/netscratch:/netscratch,/fscratch:/fscratch \
            --container-workdir=/netscratch/nrauscher/projects/BA-hydra \
            /bin/bash -c "
                source .venv_pretraining/bin/activate && \
                python src/train.py experiment=t5_continued_pretraining_adafactor_relative_step_clip_1_OPTIMIZED trainer.log_every_n_steps=1
            "
    else
        echo "Resuming from checkpoint: $CKPT_PATH"
        echo "Checkpoint size: $(ls -lh $CKPT_PATH | awk '{print $5}')"
        
        # Resume training with the checkpoint
        srun -K \
            --container-image=/netscratch/nrauscher/containers/hydra_pytorch_25.05.25-final.sqsh \
            --container-mounts=/netscratch:/netscratch,/fscratch:/fscratch \
            --container-workdir=/netscratch/nrauscher/projects/BA-hydra \
            /bin/bash -c "
                source .venv_pretraining/bin/activate && \
                python src/train.py experiment=t5_continued_pretraining_adafactor_relative_step_clip_1_OPTIMIZED \
                    ckpt_path=\"$CKPT_PATH\" \
                    trainer.log_every_n_steps=1
            "
    fi
fi

EXIT_CODE=$?
echo "=============================================="
echo "Completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=============================================="