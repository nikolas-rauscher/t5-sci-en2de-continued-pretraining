#!/bin/bash
#SBATCH --job-name=scilons-t5-baseline-improved
#SBATCH --partition=A100-80GB,H100-SLT,H100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=128GB
#SBATCH --output=/netscratch/nrauscher/projects/BA-hydra/scilons_eval_runs/logs/scilons_t5_baseline_improved_%j.out
#SBATCH --error=/netscratch/nrauscher/projects/BA-hydra/scilons_eval_runs/logs/scilons_t5_baseline_improved_%j.err

set -euo pipefail

# User-editable inputs
SCILONS_DATA="/netscratch/abu/Scilons/scibert/data"  # Must follow SciBERT data structure
CONTAINER_IMG="/netscratch/abu/scilons_eval_updated.sqsh"

WORKDIR="$(pwd)"
# For baseline, use cached T5-base to avoid HF authentication issues
HF_MODEL_DIR="/netscratch/${USER}/.cache/huggingface/hub/models--t5-base/snapshots/latest"
# Fallback to download if not cached
if [ ! -d "${HF_MODEL_DIR}" ]; then
  HF_MODEL_DIR="t5-base"  # Will try to download from HuggingFace hub
fi
# Timestamped output root to avoid overwriting previous runs
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${WORKDIR}/scilons_eval_runs/t5_baseline_improved/${TS}"

# Run inside container so we reuse colleagues' environment
srun -K \
  --container-image="${CONTAINER_IMG}" \
  --container-workdir="${WORKDIR}" \
  --container-mounts="/netscratch/${USER}:/netscratch/${USER},${WORKDIR}:${WORKDIR},/netscratch/abu/Scilons/scibert/data:/netscratch/abu/Scilons/scibert/data:ro" \
  bash -lc "set -euo pipefail
    export PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=false
    export HF_HOME=/netscratch/${USER}/.cache/huggingface
    export HF_DATASETS_CACHE=/netscratch/${USER}/.cache/huggingface/datasets
    export PYTHONNOUSERSITE=1
    export PYTHONPATH=\"${WORKDIR}:\${PYTHONPATH:-}\"  # make 'src' importable for checkpoint pickles
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python  # avoid compiled protobuf requirement

    mkdir -p \"${OUT_DIR}\"

    # Launch improved scilons-eval with proper W&B logging
    # Using t5-base directly from HuggingFace (no conversion needed)
    cd \"${OUT_DIR}\"
    python3 ${WORKDIR}/external/scilons-eval/evaluate_all_tasks_improved.py \
      --model \"${HF_MODEL_DIR}\" \
      --tokenizer \"${HF_MODEL_DIR}\" \
      --data \"${SCILONS_DATA}\" \
      --hf_token \"${HF_TOKEN:-}\" \
      --max_length 512 \
      --output_dir \"${OUT_DIR}\" \
      --per_device_train_batch_size 32 \
      --num_train_epochs 4 \
      --learning_rate 2e-5 \
      --seq_to_seq_model true \
      --report_to wandb \
      --logging_steps 50 \
      --save_strategy epoch \
      --evaluation_strategy epoch \
      --load_best_model_at_end true \
      --metric_for_best_model eval_loss \
      --greater_is_better false
  "
