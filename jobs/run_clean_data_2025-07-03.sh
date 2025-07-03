#!/bin/bash
#SBATCH --job-name=clean_data_2025-07-03
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=77   # 75 CPUs auf dem V100-16GB-Node nutzen
#SBATCH --mem=150G            # 100G RAM (angepasst für 75 CPUs)
#SBATCH --partition=A100-80GB  # A100-80GB partition
#SBATCH --gres=gpu:0         # Explizit keine GPUs
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nikolas.rauscher@dfki.de
#SBATCH --output=logs/slurm-clean-data-2025-07-03-%j.out
#SBATCH --error=logs/slurm-clean-data-2025-07-03-%j.err

echo "Starting BA-hydra CLEAN DATA 2025-07-03 job at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"


# Change to project directory
cd /netscratch/nrauscher/projects/BA-hydra



srun -K \
  --cpus-per-task=75 \
  --gres=gpu:0 \
  --container-mounts=/netscratch:/netscratch,/home/$USER:/home/$USER,/ds-slt:/ds-slt \
  --container-image=/netscratch/$USER/containers/hydra_pytorch_25.05.25-final.sqsh \
  --container-workdir="$(pwd)" \
  bash -c "pip install wandb && python src/dataprep/pipelines/clean_data.py"