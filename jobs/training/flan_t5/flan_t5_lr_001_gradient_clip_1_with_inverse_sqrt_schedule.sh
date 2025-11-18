#!/bin/bash
#SBATCH --job-name=flan_t5_lr001_clip1
#SBATCH --partition=H200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=600G
#SBATCH --time=24:00:00

#SBATCH --output=/netscratch/nrauscher/projects/BA-hydra/logs/slurm_%j_flan_t5_lr001_clip1.out
#SBATCH --error=/netscratch/nrauscher/projects/BA-hydra/logs/slurm_%j_flan_t5_lr001_clip1.err

echo "=============================================="
echo "FLAN-T5 Training - LR 0.001, Gradient Clip 1.0"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"  
echo "Start time: $(date)"
echo "=============================================="

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

echo "Starting FLAN-T5 training from scratch..."
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
        source .venv_pretraining/bin/activate && \
        python src/train.py experiment=flan_t5_lr_001_gradient_clip_1_with_inverse_sqrt_schedule trainer.log_every_n_steps=1
    "

echo "=============================================="
echo "FLAN-T5 training completed at: $(date)"
echo "Exit code: $?"
echo "=============================================="