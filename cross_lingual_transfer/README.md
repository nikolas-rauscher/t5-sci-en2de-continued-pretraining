# Cross-Lingual Transfer: English → German

## Overview
Transfer the best English T5 checkpoint to German using the Wechsel library for embedding replacement and continued pretraining on German scientific text.

## Directory Structure
- `configs/` - Configuration files for German pretraining and evaluation
- `scripts/` - Implementation scripts for Wechsel integration and training
- `data/` - German dataset preparation and analysis
- `models/` - Checkpoint management and model artifacts
- `evaluation/` - Cross-lingual evaluation pipelines

## Best English Checkpoint
**Source**: `pretraining_logs_lr_001_gradient_clip_1_with_inverse_sqrt_schedule/train/runs/2025-08-13_23-20-56/checkpoints/steps/step-step=640000.ckpt`
**Performance**: MMLU score 0.2703 (+17.6% vs baseline)

## German Dataset
**Source**: `scilons/texts_pq_3` (deu_Latn split)
**Size**: 594k examples (~8.8GB scientific German text)

## Process Flow
1. **Wechsel Embedding Transfer** → Replace English tokenizer/embeddings with German
2. **German Continued Pretraining** → Fine-tune on German scientific data (10k-20k steps)  
3. **Cross-Lingual Evaluation** → Test knowledge transfer effectiveness

## Implementation Status
🟡 In Progress - Setting up project structure and configurations