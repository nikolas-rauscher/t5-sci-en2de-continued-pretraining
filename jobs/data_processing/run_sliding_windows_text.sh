#!/bin/bash
#SBATCH --job-name=sliding_windows_text
#SBATCH --time=11:00:00
#SBATCH --cpus-per-task=77   # 75 CPUs auf dem H100-Node nutzen
#SBATCH --mem=160G            # 100G RAM (angepasst für 75 CPUs, H100 hat viel)
#SBATCH --partition=A100-80GB,A100-40GB,RTXA6000,V100-32GB,H100,H100-SLT
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
  .venv/bin/python src/dataprep/pipelines/run_sliding_windows.py \
    sliding_windows.paths.input_folder=/fscratch/nrauscher/projects/BA-hydra/data/after_cleaning/cleand_data_stats_enriched_V2/enriched_documents_statistics_v2 \
    sliding_windows.paths.output_folder=/fscratch/nrauscher/projects/BA-hydra/data/windows_text_0olap_512_v3 \
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
    sliding_windows.stats.save_enriched_docs=true \
    sliding_windows.paths.logging_dir=logs/sliding_windows_text_0olap_512_v3
