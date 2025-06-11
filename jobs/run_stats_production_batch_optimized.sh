#!/bin/bash
#SBATCH --job-name=hydra-stats-batch-max
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=70   # Fast ganze Node nutzen (72 total, 10 belegt = 62 frei)
#SBATCH --mem=150G           # Mehr RAM für bessere Performance
#SBATCH --partition=batch    # CPU-optimierte partition
#SBATCH --gres=gpu:0         # Explizit keine GPUs
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nikolas.rauscher@dfki.de
#SBATCH --output=logs/slurm-production-batch-max-%j.out
#SBATCH --error=logs/slurm-production-batch-max-%j.err

echo "Starting BA-hydra PRODUCTION CHAINED stats computation job (BATCH MAX PERFORMANCE) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"


# Change to project directory
cd /netscratch/nrauscher/projects/BA-hydra

# Create output directory
mkdir -p data/statistics_production/
mkdir -p logs/

# Run the production chained stats pipeline with MAXIMUM batch performance
srun -K \
  --cpus-per-task=68 \
  --gres=gpu:0 \
  --container-mounts=/netscratch:/netscratch,/home/$USER:/home/$USER,/ds-slt:/ds-slt \
  --container-image=/netscratch/$USER/containers/hydra_pytorch_25.05.25-final.sqsh \
  --container-workdir="$(pwd)" \
  .venv_spacy_stats/bin/python src/dataprep/pipelines/run_spacy_stats.py stats.paths.input_folder=data/statistics_production/enriched_documents_statistics_v1/ stats.paths.output_folder=data/statistics_production/ stats.tasks=200 stats.workers=69

