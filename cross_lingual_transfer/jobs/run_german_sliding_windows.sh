#!/bin/bash
#SBATCH --job-name=german_sliding_windows
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=60   # 60 CPUs für 60 parquet files (1 CPU pro file)
#SBATCH --mem=64G            
#SBATCH --partition=A100-80GB,A100-40GB,RTXA6000,V100-32GB,H100,H100-SLT
#SBATCH --gres=gpu:0         # Keine GPUs nötig
#SBATCH --output=/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/logs/slurm-german-sliding-windows-%j.out
#SBATCH --error=/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/logs/slurm-german-sliding-windows-%j.err

echo "Starting German SLIDING WINDOWS job at $(date)"
echo "Job ID: $SLURM_JOB_ID"  
echo "Running on node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"

# Change to project directory
cd /netscratch/nrauscher/projects/BA-hydra

# Create output directories
mkdir -p cross_lingual_transfer/data/german/sliding_windows
mkdir -p cross_lingual_transfer/logs

echo "Processing German parquet files with sliding windows..."
echo "Input: cross_lingual_transfer/data/german/raw_parquet"
echo "Output: cross_lingual_transfer/data/german/sliding_windows"
echo "Tokenizer: German Wechsel-transferred tokenizer"

srun -K \
  --cpus-per-task=60 \
  --gres=gpu:0 \
  .venv/bin/python src/dataprep/pipelines/run_sliding_windows.py \
    sliding_windows.paths.input_folder=/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/data/german/raw_parquet \
    sliding_windows.paths.output_folder=/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/data/german/sliding_windows \
    sliding_windows.tokenizer.name_or_path=/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/models/german_T5_Optimized_50Olap_clean_restart_487k/tokenizer \
    sliding_windows.window_config.output_format=text \
    sliding_windows.window_config.target_tokens=512 \
    sliding_windows.window_config.overlap_ratio=0.5 \
    sliding_windows.normalization.enable=true \
    sliding_windows.filters.enable=true \
    sliding_windows.filters.min_actual_tokens=100 \
    sliding_windows.filters.min_char_length=200 \
    sliding_windows.stats.enable=true \
    sliding_windows.stats.save_enriched_docs=false \
    sliding_windows.paths.logging_dir=cross_lingual_transfer/logs/german_sliding_windows \
    sliding_windows.execution.tasks=60 \
    sliding_windows.execution.workers=60 \
    sliding_windows.log_to_wandb=false

echo "German sliding windows processing completed at $(date)"