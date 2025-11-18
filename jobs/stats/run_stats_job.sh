#!/bin/bash
#SBATCH --job-name=ba-hydra-stats-optimized
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=100G
#SBATCH --partition=batch
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nikolas.rauscher@dfki.de
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

echo "Starting BA-hydra stats computation job at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"

# Change to project directory
cd /netscratch/nrauscher/projects/BA-hydra

# Run the stats computation with optimized settings
srun -K \
  --container-mounts=/netscratch:/netscratch,/home/$USER:/home/$USER,/ds-slt:/ds-slt \
  --container-image=/netscratch/$USER/containers/hydra_pytorch_25.05.25-final.sqsh \
  --container-workdir="$(pwd)" \
  --partition=batch \
  --gres=gpu:0 \
  --cpus-per-task=40 \
  --mem=100G \
  --exclusive \
  --mail-type=BEGIN,END,FAIL \
  --mail-user=nikolas.rauscher@dfki.de \
  --output=logs/slurm-%j.out \
  --error=logs/slurm-%j.err \
  make stats-all

echo "Job completed at $(date)"