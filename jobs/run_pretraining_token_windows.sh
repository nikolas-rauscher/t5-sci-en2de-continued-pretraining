#!/bin/bash
#SBATCH --job-name=pretrain_token_windows
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --partition=A100-80GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nikolas.rauscher@dfki.de
#SBATCH --output=logs/slurm-pretrain-token-windows-%j.out
#SBATCH --error=logs/slurm-pretrain-token-windows-%j.err

echo "Starting T5 pretraining with token windows at $(date)"
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
  bash -c "source .venv_pretraining/bin/activate && pip install wandb && python src/train.py \
    experiment=t5_pretraining \
    data=t5_pretraining_token_windows \
    trainer.max_steps=1000 \
    trainer.devices=1 \
    logger.wandb.group=token-window-comparison \
    logger.wandb.name=t5-pretraining-token-windows \
    +logger.wandb.tags='[t5-pretraining,token-windows,production,comparison]'"