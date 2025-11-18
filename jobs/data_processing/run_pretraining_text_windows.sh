#!/bin/bash
#SBATCH --job-name=pretrain_text_windows_a100
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --partition=A100-80GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nikolas.rauscher@dfki.de
#SBATCH --output=logs/slurm-pretrain-text-windows-a100-%j.out
#SBATCH --error=logs/slurm-pretrain-text-windows-a100-%j.err

echo "Starting T5 pretraining with text windows at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $SLURM_GPUS_PER_NODE"
echo "Memory: $SLURM_MEM_PER_NODE MB"

# Change to project directory
cd /netscratch/nrauscher/projects/BA-hydra

srun -K \
  --cpus-per-task=32 \
  --gres=gpu:1 \
  --container-mounts=/netscratch:/netscratch,/home/$USER:/home/$USER,/ds-slt:/ds-slt \
  --container-image=/netscratch/$USER/containers/hydra_pytorch_25.05.25-final.sqsh \
  --container-workdir="$(pwd)" \
  bash -c "source .venv/bin/activate && pip install wandb && python src/train.py \
    experiment=a100_text_windows_production \
    logger.wandb.name=t5-production-a100-$(date +%Y%m%d-%H%M) \
    +logger.wandb.tags='[t5-pretraining,text-windows,a100,production]'"