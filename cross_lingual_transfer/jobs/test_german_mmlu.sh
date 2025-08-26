#!/bin/bash

#SBATCH --job-name=test_german_mmlu
#SBATCH --partition=RTXA6000
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

cd /netscratch/nrauscher/projects/BA-hydra/lm-evaluation-harness

source ../.venv_eval/bin/activate

python -m lm_eval \
  --model hf \
  --model_args pretrained=/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/models/german_transferred \
  --tasks m_mmlu_de \
  --device cuda \
  --batch_size 4 \
  --limit 50 \
  --output_path ../cross_lingual_transfer/results/german_mmlu_test.json