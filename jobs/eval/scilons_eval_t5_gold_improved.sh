#!/bin/bash
#SBATCH --job-name=scilons-t5-gold-improved
#SBATCH --partition=A100-80GB,H100-SLT,H100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=128GB
#SBATCH --output=/netscratch/nrauscher/projects/BA-hydra/scilons_eval_runs/logs/scilons_t5_gold_improved_%j.out
#SBATCH --error=/netscratch/nrauscher/projects/BA-hydra/scilons_eval_runs/logs/scilons_t5_gold_improved_%j.err

set -euo pipefail

# User-editable inputs
GOLD_CKPT="/netscratch/nrauscher/projects/BA-hydra/pretraining_logs_lr_001_OPTIMIZED_clean_restart/train/runs/2025-09-08_02-33-22/checkpoints/best/step-487500-val_ppl-3.72168.ckpt"
SCILONS_DATA="/netscratch/abu/Scilons/scibert/data"  # Must follow SciBERT data structure
CONTAINER_IMG="/netscratch/abu/scilons_eval_updated.sqsh"

WORKDIR="$(pwd)"
# Store converted model and outputs inside repo root under scilons_eval_runs/
HF_MODEL_DIR="${WORKDIR}/converted_models/unknown-run-models/t5-base-unknown-487k-steps"
# Timestamped output root to avoid overwriting previous runs
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${WORKDIR}/scilons_eval_runs/t5_gold_clean_restart_487k_improved/${TS}"

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

    mkdir -p \"${OUT_DIR}\" \"${HF_MODEL_DIR}\"

    # 1) Convert Lightning checkpoint -> HuggingFace format if missing
    if [ ! -f \"${HF_MODEL_DIR}/config.json\" ]; then
      echo '[convert] Converting checkpoint to HF format:' \"${GOLD_CKPT}\"
      python3 task_finetuning/convert_checkpoint_for_eval.py \
        --checkpoint_path \"${GOLD_CKPT}\" \
        --output_dir \"${HF_MODEL_DIR}\" \
        --base_model t5-base
    else
      echo '[convert] Existing HF model found at' \"${HF_MODEL_DIR}\" '— skipping conversion.'
    fi

    # 2) Launch improved scilons-eval with proper W&B logging
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
      --logging_steps 1 \
      --save_strategy epoch \
      --evaluation_strategy epoch \
      --load_best_model_at_end true \
      --metric_for_best_model eval_loss \
      --greater_is_better false
  "
