#!/bin/bash
#SBATCH --job-name=hydra-stats-production-chained
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --partition=V100-16GB
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nikolas.rauscher@dfki.de
#SBATCH --output=logs/slurm-production-chained-%j.out
#SBATCH --error=logs/slurm-production-chained-%j.err

echo "Starting BA-hydra PRODUCTION CHAINED stats computation job at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"


# Change to project directory
cd /netscratch/nrauscher/projects/BA-hydra

# Create output directory
mkdir -p data/statistics_production/
mkdir -p logs/


# Run the production chained stats pipeline 
srun -K \
  --cpus-per-task=64 \
  --gres=gpu:0 \
  --container-mounts=/netscratch:/netscratch,/home/$USER:/home/$USER,/ds-slt:/ds-slt \
  --container-image=/netscratch/$USER/containers/hydra_pytorch_25.05.25-final.sqsh \
  --container-workdir="$(pwd)" \
  make stats-production-chained

