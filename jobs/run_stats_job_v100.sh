#!/bin/bash
#SBATCH --job-name=hydra-stats
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=50
#SBATCH --mem=125G
#SBATCH --partition=batch
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nikolas.rauscher@dfki.de
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

echo "Starting BA-hydra stats computation job  at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"

# Change to project directory
cd /netscratch/nrauscher/projects/BA-hydra

# Run the stats computation - explicit CPU-only
srun -K \
  --cpus-per-task=50 \
  --gres=gpu:0 \
  --container-mounts=/netscratch:/netscratch,/home/$USER:/home/$USER,/ds-slt:/ds-slt \
  --container-image=/netscratch/$USER/containers/hydra_pytorch_25.05.25-final.sqsh \
  --container-workdir="$(pwd)" \
  make stats-all

echo "Stats computation completed at $(date)"