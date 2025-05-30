#!/bin/bash
#SBATCH --job-name=hydra-stats-production-batch
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=60  # Weniger als die 70 in batch partition
#SBATCH --mem=100G  
#SBATCH --partition=batch   # Verwendet batch partition statt V100-16GB
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nikolas.rauscher@dfki.de
#SBATCH --output=logs/slurm-production-batch-%j.out
#SBATCH --error=logs/slurm-production-batch-%j.err

echo "Starting BA-hydra PRODUCTION stats computation job (BATCH PARTITION) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"

# Change to project directory
cd /netscratch/nrauscher/projects/BA-hydra

# Create output directory
mkdir -p data/statistics_production/
mkdir -p logs/

# Run the production chained stats pipeline on batch partition
srun -K \
  --cpus-per-task=56 \
  --gres=gpu:0 \
  --container-mounts=/netscratch:/netscratch,/home/$USER:/home/$USER,/ds-slt:/ds-slt \
  --container-image=/netscratch/$USER/containers/hydra_pytorch_25.05.25-final.sqsh \
  --container-workdir="$(pwd)" \
  .venv_spacy_stats/bin/python scripts/run_spacy_stats.py stats.paths.input_folder=data/statistics_production/enriched_documents_statistics_v1/ stats.paths.output_folder=data/statistics_production/ stats.tasks=150 stats.workers=59

echo "==================================="
echo "✅ Production stats computation completed at $(date)"
echo "📁 Results available in: data/statistics_production/"
echo "📝 v2 enriched docs: data/statistics_production/enriched_documents_statistics_v2/"
echo "===================================" 