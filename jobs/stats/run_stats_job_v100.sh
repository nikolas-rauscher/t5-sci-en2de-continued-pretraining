#!/bin/bash
#SBATCH --job-name=hydra-stats
#SBATCH   --time=24:00:00
#SBATCH --cpus-per-task=76
#SBATCH --mem=200G
#SBATCH --partition=V100-16GB,V100-32GB,A100-80GB,A100-40GB,H100,H200,H100-PCI
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nikolas.rauscher@dfki.de
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

echo "Starting BA-hydra spaCy stats computation job (second step) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"

# Change to project directory
cd /netscratch/nrauscher/projects/BA-hydra

# Run the spaCy stats computation only (second step) - explicit CPU-only
srun -K \
  --cpus-per-task=50 \
  --gres=gpu:0 \
  --container-mounts=/netscratch:/netscratch,/home/$USER:/home/$USER,/ds-slt:/ds-slt,/fscratch:/fscratch \
  --container-image=/netscratch/$USER/containers/hydra_pytorch_25.05.25-final.sqsh \
  --container-workdir="$(pwd)" \
  make stats_p2

echo "spaCy stats computation completed at $(date)"