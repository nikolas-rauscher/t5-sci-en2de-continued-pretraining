#!/bin/bash
#SBATCH --job-name=t5_lr001_OPTIMIZED_restart
#SBATCH --partition=H100,H100-PCI,H100-SLT,H200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=600G
#SBATCH --time=24:00:00

#SBATCH --output=/netscratch/nrauscher/projects/BA-hydra/logs/slurm_%j_t5_lr001_OPTIMIZED_restart.out
#SBATCH --error=/netscratch/nrauscher/projects/BA-hydra/logs/slurm_%j_t5_lr001_OPTIMIZED_restart.err

echo "=============================================="
echo "T5 Continued Pretraining - Clean Restart with OPTIMIZED COLLATOR"
echo "LR: 0.001, Warmup: 20k, Gradient Clip: 1.0"
echo "Collator: T5-formula (1.109x expansion, target=114)"
echo "Dataset: Validated sliding windows (50% overlap, legacy)"
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

echo "Starting T5 continued pretraining (OPTIMIZED COLLATOR) ..."
echo "Configuration: configs/experiment/t5_continued_pretraining_lr_001_OPTIMIZED_clean_restart.yaml"
echo "Learning Rate: 0.001 with T5-formula optimization"
echo "Gradient Clip: 1.0"
echo "Effective batch size: 384 (48*4*2)"
echo "Optimization: 35% more efficient than 1.5x heuristic"

srun -K \
    --container-image=/netscratch/nrauscher/containers/hydra_pytorch_25.05.25-final.sqsh \
    --container-mounts=/netscratch:/netscratch,/fscratch:/fscratch \
    --container-workdir=/netscratch/nrauscher/projects/BA-hydra \
    /bin/bash -c "
        source .venv_pretraining/bin/activate && \
        python src/train.py experiment=t5_continued_pretraining_lr_001_OPTIMIZED_clean_restart trainer.log_every_n_steps=1
    "

EXIT_CODE=$?
echo "=============================================="
echo "T5 continued pretraining completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=============================================="