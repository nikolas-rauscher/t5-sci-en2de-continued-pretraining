#!/bin/bash
#SBATCH --job-name=t5_0olap512_v3_lr002_OPT
#SBATCH --partition=H100,H100-PCI,H100-SLT,H200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=600G
#SBATCH --time=24:00:00

# Logs
#SBATCH --output=/netscratch/nrauscher/projects/BA-hydra/logs/slurm_%j_t5_0olap512_v3_lr002_OPTIMIZED.out
#SBATCH --error=/netscratch/nrauscher/projects/BA-hydra/logs/slurm_%j_t5_0olap512_v3_lr002_OPTIMIZED.err

echo "=============================================="
echo "T5 Continued Pretraining (OPTIMIZED COLLATOR)"
echo "Dataset: text windows (0% overlap, 512 tokens, v3)"
echo "LR: 0.02, Warmup: 50k steps, Gradient Clip: 1.0"
echo "Collator: T5-formula (1.109x expansion, target=114)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="

# Check data access
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

echo "System info:"
python --version
python -c 'import torch; print("torch", torch.__version__)'
python -c 'import lightning; print("lightning", lightning.__version__)'
python -c 'import torch; print("GPUs", torch.cuda.device_count())'
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

echo "Starting T5 continued pretraining (OPTIMIZED - T5 Formula Mode, LR 0.02) ..."
echo "Experiment: configs/experiment/t5_continued_pretraining_lr_002_OPTIMIZED_text_0olap512_v3_warmup50k.yaml"
echo "Effective batch size: 384 (48*4*2)"
echo "Target optimization: 1.109x expansion instead of 1.5x (35% more efficient)"

srun -K \
    --container-image=/netscratch/nrauscher/containers/hydra_pytorch_25.05.25-final.sqsh \
    --container-mounts=/netscratch:/netscratch,/fscratch:/fscratch \
    --container-workdir=/netscratch/nrauscher/projects/BA-hydra \
    /bin/bash -c "
        source .venv_pretraining/bin/activate && \
        python src/train.py experiment=t5_continued_pretraining_lr_002_OPTIMIZED_text_0olap512_v3_warmup50k trainer.log_every_n_steps=1
    "

EXIT_CODE=$?
echo "=============================================="
echo "Completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=============================================="