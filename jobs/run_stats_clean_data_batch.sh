#!/bin/bash
#SBATCH --job-name=clean_data_production_run_on_all_data
#SBATCH --time=11:00:00
#SBATCH --cpus-per-task=76   # 75 CPUs auf dem H100-Node nutzen
#SBATCH --mem=100G            # 100G RAM (angepasst für 75 CPUs, H100 hat viel)
#SBATCH --partition=H100     # H100: 224 CPUs, wenig Queue, fast idle node!
#SBATCH --gres=gpu:0         # Explizit keine GPUs
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nikolas.rauscher@dfki.de
#SBATCH --output=logs/slurm-clean-data-batch-%j.out
#SBATCH --error=logs/slurm-clean-data-batch-%j.err

echo "Starting BA-hydra CLEAN DATA BATCH job at $(date)"
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
  bash -c "pip install wandb && make clean-data"
