#!/bin/bash
#SBATCH --job-name=pretrain_test_a100
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --partition=A100-80GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nikolas.rauscher@dfki.de
#SBATCH --output=logs/slurm-pretrain-test-a100-%j.out
#SBATCH --error=logs/slurm-pretrain-test-a100-%j.err

echo "Starting T5 test pretraining with text windows at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $SLURM_GPUS_PER_NODE"
echo "Memory: $SLURM_MEM_PER_NODE MB"

# Change to project directory
cd /netscratch/nrauscher/projects/BA-hydra

srun -K \
  --cpus-per-task=24 \
  --gres=gpu:1 \
  --container-mounts=/netscratch:/netscratch,/home/$USER:/home/$USER,/ds-slt:/ds-slt,/fscratch:/fscratch \
  --container-image=/netscratch/$USER/containers/hydra_pytorch_25.05.25-final.sqsh \
  --container-workdir="$(pwd)" \
  bash -c "source .venv/bin/activate && pip install wandb && python src/train.py \
    experiment=a100_text_windows_test \
    logger.wandb.name=t5-test-a100-$(date +%Y%m%d-%H%M) \
    +logger.wandb.tags='[t5-pretraining,text-windows,a100,test]'"