#!/bin/bash
#SBATCH --job-name=clean_data_h100
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=77
#SBATCH --mem=150G
#SBATCH --partition=H100
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nikolas.rauscher@dfki.de
#SBATCH --output=logs/slurm-clean-data-h100-%j.out
#SBATCH --error=logs/slurm-clean-data-h100-%j.err

echo "Starting H100 CLEAN DATA job at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"

cd /netscratch/nrauscher/projects/BA-hydra

srun -K \
  --cpus-per-task=75 \
  --gres=gpu:0 \
  --container-mounts=/netscratch:/netscratch,/home/$USER:/home/$USER,/ds-slt:/ds-slt \
  --container-image=/netscratch/$USER/containers/hydra_pytorch_25.05.25-final.sqsh \
  --container-workdir="$(pwd)" \
  bash -c "pip install wandb && python src/dataprep/pipelines/clean_data.py"