#!/bin/bash
# Test job: Create 0%-overlap 512-token sliding windows on /fscratch with DocStats and per-window filtering

#SBATCH --job-name=sliding_windows_text_test
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=75
#SBATCH --mem=160G
#SBATCH --partition=A100-80GB,A100-40GB,RTXA6000,V100-32GB,H100,H200
#SBATCH --gres=gpu:0
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nikolas.rauscher@dfki.de
#SBATCH --output=logs/slurm-sliding-windows-text-test-%j.out
#SBATCH --error=logs/slurm-sliding-windows-text-test-%j.err

echo "Starting BA-hydra SLIDING WINDOWS TEXT (TEST) job at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node(s): $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK | Mem: $SLURM_MEM_PER_NODE MB"

# Project root
cd /netscratch/nrauscher/projects/BA-hydra || exit 1

# Input on /fscratch (adjust if needed)
INPUT_DIR="/fscratch/nrauscher/projects/BA-hydra/data/after_cleaning/cleand_data_stats_enriched_V2/enriched_documents_statistics_v2"

# Output on /fscratch (v3 test folder)
OUTPUT_DIR="/fscratch/nrauscher/projects/BA-hydra/data/windows_text_0olap_512_v3"
LOG_DIR="logs/sliding_windows_text_test_v3_$(date +%Y-%m-%d_%H-%M-%S)"

mkdir -p "$LOG_DIR"

echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Logs:   $LOG_DIR"

# Run sliding windows with:
# - 0% overlap (stride=512)
# - 512 target tokens
# - text output
# - per-window normalization + filtering
# - DocStats enabled (enriched windows written)
srun -K \
  --cpus-per-task=$SLURM_CPUS_PER_TASK \
  --gres=gpu:0 \
  --container-mounts=/netscratch:/netscratch,/home/$USER:/home/$USER,/ds-slt:/ds-slt,/fscratch:/fscratch \
  --container-image=/netscratch/$USER/containers/hydra_pytorch_25.05.25-final.sqsh \
  --container-workdir="$(pwd)" \
  bash -c "pip install -q wandb && python src/dataprep/pipelines/run_sliding_windows.py \
    sliding_windows.paths.input_folder=$INPUT_DIR \
    sliding_windows.paths.output_folder=$OUTPUT_DIR \
    sliding_windows.paths.logging_dir=$LOG_DIR \
    sliding_windows.window_config.output_format=text \
    sliding_windows.window_config.target_tokens=512 \
    sliding_windows.window_config.overlap_ratio=0.0 \
    sliding_windows.normalization.enable=true \
    sliding_windows.filters.enable=true \
    sliding_windows.filters.min_actual_tokens=384 \
    sliding_windows.filters.min_char_length=800 \
    sliding_windows.filters.max_punctuation_ratio=0.20 \
    sliding_windows.filters.max_non_alpha_digit_ratio=0.35 \
    sliding_windows.filters.max_uppercase_ratio=0.15 \
    sliding_windows.stats.enable=true \
    'sliding_windows.stats.modules={doc_stats: {_target_: datatrove.pipeline.stats.DocStats, output_folder: doc_stats, groups_to_compute: [summary, histogram], histogram_round_digits: 3}}' \
    sliding_windows.stats.save_enriched_docs=true"

EXIT_CODE=$?
echo "Finished at: $(date) | Exit code: $EXIT_CODE"
exit $EXIT_CODE
