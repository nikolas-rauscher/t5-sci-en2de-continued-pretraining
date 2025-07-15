#!/bin/bash
#SBATCH --job-name=sliding_windows_text
#SBATCH --time=11:00:00
#SBATCH --cpus-per-task=77   # 75 CPUs auf dem H100-Node nutzen
#SBATCH --mem=160G            # 100G RAM (angepasst für 75 CPUs, H100 hat viel)
#SBATCH --partition=A100-80GB,A100-40GB,RTXA6000,V100-32GB,V100-16GB
#SBATCH --gres=gpu:0         # Explizit keine GPUs
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nikolas.rauscher@dfki.de
#SBATCH --output=logs/slurm-sliding-windows-text-%j.out
#SBATCH --error=logs/slurm-sliding-windows-text-%j.err

echo "Starting BA-hydra SLIDING WINDOWS TEXT job at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"


# Change to project directory
cd /netscratch/nrauscher/projects/BA-hydra



srun -K \
  --cpus-per-task=75 \
  --gres=gpu:0 \
  --container-mounts=/netscratch:/netscratch,/home/$USER:/home/$USER,/ds-slt:/ds-slt,/fscratch:/fscratch \
  --container-image=/netscratch/$USER/containers/hydra_pytorch_25.05.25-final.sqsh \
  --container-workdir="$(pwd)" \
  bash -c "pip install wandb && python src/dataprep/pipelines/run_sliding_windows.py sliding_windows.paths.input_folder=/fscratch/nrauscher/projects/BA-hydra/data/after_cleaning/cleand_data_stats_enriched_V2/enriched_documents_statistics_v2 sliding_windows.paths.output_folder=/fscratch/nrauscher/projects/BA-hydra/data/cleaned_sliding_windows/text_2025-07-14 sliding_windows.window_config.output_format=text sliding_windows.paths.logging_dir=logs/sliding_windows_text_cleaned_2025-07-14"