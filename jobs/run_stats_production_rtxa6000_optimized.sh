#!/bin/bash
#SBATCH --job-name=hydra-stats-rtxa6000-opt
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=76   # 75 workers + 5 buffer für system/overhead
#SBATCH --mem=150G           # Mehr RAM für spaCy (memory-intensiv)
#SBATCH --partition=RTXA6000 # RTXA6000 für beste Performance
#SBATCH --gres=gpu:0         # Explizit keine GPUs
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nikolas.rauscher@dfki.de
#SBATCH --output=logs/slurm-production-rtxa6000-opt-%j.out
#SBATCH --error=logs/slurm-production-rtxa6000-opt-%j.err

echo "Starting BA-hydra PRODUCTION OPTIMIZED stats computation job at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"


# Change to project directory
cd /netscratch/nrauscher/projects/BA-hydra

# Create output directory
mkdir -p data/statistics_production2/
mkdir -p logs/


srun -K \
  --cpus-per-task=76 \
  --gres=gpu:0 \
  --container-mounts=/netscratch:/netscratch,/home/$USER:/home/$USER,/ds-slt:/ds-slt \
  --container-image=/netscratch/$USER/containers/hydra_pytorch_25.05.25-final.sqsh \
  --container-workdir="$(pwd)" \
  .venv_spacy_stats/bin/python scripts/run_spacy_stats.py \
    stats.paths.input_folder=data/statistics_production/enriched_documents_statistics_v1/ \
    stats.paths.output_folder=data/statistics_production2/ \
    stats.tasks=75 \
    stats.workers=75
