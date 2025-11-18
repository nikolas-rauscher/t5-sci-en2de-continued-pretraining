#!/bin/bash
#SBATCH --job-name=smart_resume_scifive_v2_4gpu
#SBATCH --partition=H100-PCI,H100-SLT,H200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=600G
#SBATCH --time=24:00:00

#SBATCH --output=/netscratch/nrauscher/projects/BA-hydra/logs/slurm_%j_smart_resume_scifive_v2_4gpu.out
#SBATCH --error=/netscratch/nrauscher/projects/BA-hydra/logs/slurm_%j_smart_resume_scifive_v2_4gpu.err

echo "=============================================="
echo "Smart Resume SciFive-Style V2 4-GPU H100 Run"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"  
echo "Start time: $(date)"
echo "=============================================="

# Function to find latest SciFive-style v2 checkpoint
find_latest_scifive_v2_checkpoint() {
    # Find all SciFive-style v2 run directories
    RUN_DIRS=$(find scifive_style_v2_logs/train/runs -name "20*" -type d 2>/dev/null | sort -r)
    
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
        LATEST_CKPT=$(find scifive_style_v2_logs/train/runs -name "step-*.ckpt" 2>/dev/null | sort -V | tail -1)
        echo "$LATEST_CKPT"
    fi
}

# Find latest SciFive-style v2 checkpoint
echo "Looking for latest SciFive-style v2 checkpoint..."
LATEST_CKPT=$(find_latest_scifive_v2_checkpoint)

if [[ -z "$LATEST_CKPT" ]]; then
    echo "No SciFive-style v2 checkpoint found - starting fresh training"
    RESUME_CMD=""
else
    echo "Found SciFive-style v2 checkpoint: $LATEST_CKPT"
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

echo "Starting SciFive-Style V2 T5 training with smart resume..."
echo "Configuration: configs/experiment/scifive_style_4gpu_h100_v2.yaml"
echo "Learning Rate: 0.001 (constant, no scheduler - SciFive proven method)"
echo "Gradient Clip: 1.0 (updated from 0.5)"
echo "Effective batch size: 384 (48*4*2)"

srun -K \
    --container-image=/netscratch/nrauscher/containers/hydra_pytorch_25.05.25-final.sqsh \
    --container-mounts=/netscratch:/netscratch,/fscratch:/fscratch \
    --container-workdir=/netscratch/nrauscher/projects/BA-hydra \
    /bin/bash -c "
        source .venv_pretraining/bin/activate && \
        python src/train.py experiment=scifive_style_4gpu_h100_v2 trainer.log_every_n_steps=1 $RESUME_CMD
    "

echo "=============================================="
echo "SciFive-style v2 smart resume completed at: $(date)"
echo "Exit code: $?"
echo "=============================================="