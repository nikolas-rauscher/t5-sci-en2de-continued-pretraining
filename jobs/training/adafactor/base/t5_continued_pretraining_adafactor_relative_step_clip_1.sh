#!/bin/bash
#SBATCH --job-name=t5_continued_relstep_clip1
#SBATCH --partition=H100,H100-PCI,H100-SLT,H200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=600G
#SBATCH --time=24:00:00

# Logs
#
# Example:
#  /netscratch/nrauscher/projects/BA-hydra/logs/slurm_<JOBID>_t5_continued_relstep_clip1.out
#  /netscratch/nrauscher/projects/BA-hydra/logs/slurm_<JOBID>_t5_continued_relstep_clip1.err
#
#SBATCH --output=/netscratch/nrauscher/projects/BA-hydra/logs/slurm_%j_t5_continued_relstep_clip1.out
#SBATCH --error=/netscratch/nrauscher/projects/BA-hydra/logs/slurm_%j_t5_continued_relstep_clip1.err

echo "=============================================="
echo "T5 Continued Pretraining - Adafactor relative_step, Clip 1.0"
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

echo "✅ fscratch mounted and data accessible"

# Environment variables
export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS}
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS}
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

cd /netscratch/nrauscher/projects/BA-hydra

echo "System info:"
python --version
python -c 'import torch; print("torch", torch.__version__)'
python -c 'import lightning; print("lightning", lightning.__version__)'
python -c 'import torch; print("GPUs", torch.cuda.device_count())'
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

echo "Starting T5 continued pretraining (Adafactor relative_step) ..."
echo "Configuration: configs/experiment/t5_continued_pretraining_adafactor_relative_step_clip_1.yaml"
echo "Effective batch size: 384 (48*4*2)"

srun -K \
    --container-image=/netscratch/nrauscher/containers/hydra_pytorch_25.05.25-final.sqsh \
    --container-mounts=/netscratch:/netscratch,/fscratch:/fscratch \
    --container-workdir=/netscratch/nrauscher/projects/BA-hydra \
    /bin/bash -c "
        source .venv_pretraining/bin/activate && \
        python src/train.py experiment=t5_continued_pretraining_adafactor_relative_step_clip_1 trainer.log_every_n_steps=1
    "

EXIT_CODE=$?
echo "=============================================="
echo "Completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=============================================="

