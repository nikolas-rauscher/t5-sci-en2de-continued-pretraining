#!/bin/bash
#SBATCH --job-name=fasttext_stats_original_2025-07-08
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=76   # 75 CPUs wie im Original
#SBATCH --mem=150G           # Gleiche Memory wie Original
#SBATCH --partition=A100-80GB,A100-40GB,RTXA6000,V100-32GB,V100-16GB
#SBATCH --gres=gpu:0         # No GPUs needed for stats
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nikolas.rauscher@dfki.de
#SBATCH --output=logs/slurm-fasttext-stats-2025-07-08-%j.out
#SBATCH --error=logs/slurm-fasttext-stats-2025-07-08-%j.err

echo "Starting BA-hydra FASTTEXT STATISTICS job at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"

# Change to project directory
cd /netscratch/nrauscher/projects/BA-hydra

# Run FastText statistics aggregation on full dataset
srun -K \
  --cpus-per-task=75 \
  --gres=gpu:0 \
  --container-mounts=/netscratch:/netscratch,/home/$USER:/home/$USER,/ds-slt:/ds-slt \
  --container-image=/netscratch/$USER/containers/hydra_pytorch_25.05.25-final.sqsh \
  --container-workdir="$(pwd)" \
  bash -c "pip install wandb regex && python src/dataprep/pipelines/run_fasttext_stats.py fasttext_stats.tasks=75 fasttext_stats.workers=75"

echo "FastText statistics job completed at $(date)"