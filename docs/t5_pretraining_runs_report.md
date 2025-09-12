# T5 Pretraining Runs — Inventory and Launch Guide

This document summarizes all current T5 continued‑pretraining experiments in this repo, the datasets they use, where logs/checkpoints are written, and how to start or smart‑resume each run.

## Datasets

- New 0% overlap windows (512 tokens, v3)
  - Path: `/fscratch/nrauscher/projects/BA-hydra/data/windows_text_0olap_512_v3/enriched_windows`
  - Size: ~129,218,744 windows (512 tokens target, 0% overlap)
- Validated windows (legacy/pre‑existing)
  - Path: `/fscratch/nrauscher/projects/BA-hydra/data/validated_sliding_windows/validated`


  - old Dataset 50% overlap sliding‑window dataset (legacy)

## Current Gold Models

- English Gold Model:
  - Name: lr_001_OPTIMIZED_50olap_487500k_vppl_3.72168_gold
  - Path: `pretraining_logs_lr_001_OPTIMIZED_clean_restart/train/runs/2025-09-08_02-33-22/checkpoints/best/step-487500-val_ppl-3.72168.ckpt`
  - Performance: MMLU 0.2687 (+17.3% vs T5‑Base, EN 0‑shot)
  - Method: T5‑Base → continued pretraining for 487k steps on validated windows (50% overlap), LR 0.001, OPTIMIZED collator
  - diffrents vs other runs: uses legacy 50% overlap dataset and the OPTIMIZED T5‑formula collator (exact target_length=114 tokens)

- German transferd Gold Model:
  - Name: german_T5_Optimized_50Olap_clean_restart_487k
  - Path: `cross_lingual_transfer/models/german_T5_Optimized_50Olap_clean_restart_487k/`
  - Performance: TBD (evaluation running)
  - Method: Wechsel transfer from the English Gold checkpoint (keeps all EN weights; swaps embeddings)

## Runs on 0% overlap dataset (512, v3)

### NEW: OPTIMIZED COLLATOR RUNS (T5-Formula Mode)

#### LR 0.0001 — OPTIMIZED warmup 50k [NEW - ready to start]
- Config: `configs/experiment/t5_continued_pretraining_lr_0001_OPTIMIZED_text_0olap512_v3_warmup50k.yaml`
- Start job: `jobs/t5_continued_pretraining_lr_0001_OPTIMIZED_text_0olap512_v3.sh`
- Smart‑resume: `jobs/smart_resume_t5_continued_pretraining_lr_0001_OPTIMIZED_text_0olap512_v3_warmup50k.sh`
- Hydra root (logs/ckpts): `pretraining_logs_text_0olap512_v3_lr_0001_OPTIMIZED/train/runs`
- W&B: group `H100-4GPU-continued-pretraining-OPTIMIZED`, name `lr-0001-OPTIMIZED-text-0olap512-v3-warmup50k`
- Dataset: 0% overlap (512, v3)
- **NEW**: Uses T5-formula collator (1.109x expansion, target_length=114)
- **Optimization**: 35% less CPU overhead vs 1.5x heuristic
- Key hyperparams:
  - Optimizer: Adafactor (lr=1e-4, relative_step=false)
  - Scheduler: inverse sqrt (`num_warmup_steps: 50000`)
  - Gradient clip: 1.0, precision: bf16‑mixed
  - Effective batch: 384 (48 × 4 GPUs × accumulate=2)
  - Split: `train_val_split: [0.999226, 0.000774]` (~100k validation windows)

#### LR 0.001 — OPTIMIZED warmup 50k [still running]
- Config: `configs/experiment/t5_continued_pretraining_lr_001_OPTIMIZED_text_0olap512_v3_warmup50k.yaml`
- Start job: `jobs/t5_continued_pretraining_lr_001_OPTIMIZED_text_0olap512_v3.sh`
- Smart‑resume: `jobs/smart_resume_t5_continued_pretraining_lr_001_OPTIMIZED_text_0olap512_v3_warmup50k.sh`
- Hydra root: `pretraining_logs_text_0olap512_v3_lr_001_OPTIMIZED/train/runs`
- W&B: group `H100-4GPU-continued-pretraining-OPTIMIZED`, name `lr-001-OPTIMIZED-text-0olap512-v3-warmup50k`
- Dataset: 0% overlap (512, v3)
- **NEW**: Uses T5-formula collator (1.109x expansion, target_length=114)
- **Optimization**: 35% less CPU overhead vs 1.5x heuristic
- Key hyperparams: as above, but lr=1e‑3

#### Adafactor relative_step — OPTIMIZED (no scheduler) [still running]
- Config: `configs/experiment/t5_continued_pretraining_adafactor_relative_step_clip_1_OPTIMIZED_text_0olap512_v3.yaml`
- Start job: `jobs/t5_continued_pretraining_adafactor_relative_step_clip_1_OPTIMIZED_text_0olap512_v3.sh`
- Smart‑resume: `jobs/smart_resume_t5_continued_pretraining_adafactor_relative_step_clip_1_OPTIMIZED_text_0olap512_v3.sh`
- Hydra root: `pretraining_logs_adafactor_relative_step_clip1_OPTIMIZED_text_0olap512_v3/train/runs`
- dir: `pretraining_logs_adafactor_relative_step_clip1_OPTIMIZED_text_0olap512_v3/train/runs/${now:%Y-%m-%d_%H-%M-%S}`
- Dataset: 0% overlap (512, v3)
- Notes: Adafactor with `relative_step: true`, gradient clip 1.0, optimized collator (target_length≈114)

### LEGACY RUNS (1.5x Heuristic - Backwards Compatible)

#### LR 0.0001 — warmup 50k [still running]
- Config: `configs/experiment/t5_continued_pretraining_lr_0001_bugfixed_text_0olap512_v3_warmup50k.yaml`
- Start job: `jobs/t5_continued_pretraining_lr_0001_bugfixed_text_0olap512_v3.sh`
- Smart‑resume: `jobs/smart_resume_t5_continued_pretraining_lr_0001_bugfixed_text_0olap512_v3_warmup50k.sh`
- Hydra root (logs/ckpts): `pretraining_logs_text_0olap512_v3_lr_0001_bugfixed/train/runs`
- dir: `pretraining_logs_text_0olap512_v3_lr_0001_bugfixed/train/runs/${now:%Y-%m-%d_%H-%M-%S}`
- W&B: group `H100-4GPU-continued-pretraining-text-0olap512-v3`, name `lr-0001-bugfixed-text-0olap512-v3-warmup50k`
- Dataset: 0% overlap (512, v3)
- Key hyperparams:
  - Optimizer: Adafactor (lr=1e-4, relative_step=false)
  - Scheduler: inverse sqrt (`num_warmup_steps: 50000`)
  - Gradient clip: 1.0, precision: bf16‑mixed
  - Effective batch: 384 (48 × 4 GPUs × accumulate=2)
  - Split: `train_val_split: [0.999226, 0.000774]` (~100k validation windows)
  

#### LR 0.001 — warmup 50k [still running]
- Config: `configs/experiment/t5_continued_pretraining_lr_001_bugfixed_text_0olap512_v3_warmup50k.yaml`
- Start job: `jobs/t5_continued_pretraining_lr_001_bugfixed_text_0olap512_v3.sh`
- Smart‑resume: `jobs/smart_resume_t5_continued_pretraining_lr_001_bugfixed_text_0olap512_v3_warmup50k.sh`
- Hydra root: `pretraining_logs_text_0olap512_v3_lr_001_bugfixed/train/runs`
- dir: `pretraining_logs_text_0olap512_v3_lr_001_bugfixed/train/runs/${now:%Y-%m-%d_%H-%M-%S}`
- W&B: group `H100-4GPU-continued-pretraining-text-0olap512-v3`, name `lr-001-bugfixed-text-0olap512-v3-warmup50k`
- Dataset: 0% overlap (512, v3)
- Key hyperparams: as above, but lr=1e‑3
- **NOTE**: Can run parallel to legacy versions for comparison

## The runs where the Critical bug pretraining code got fixed, The Runs on validated windows dataset (legacy) old dataset with 50% overlap but with the fixed pretraining

### LR 0.0001 — bugfixed clean‑restart [still running]
- Config: `configs/experiment/t5_continued_pretraining_lr_0001_bugfixed_clean_restart.yaml`
- Start job: `jobs/t5_continued_pretraining_lr_0001_bugfixed_clean_restart.sh`
- Smart‑resume: `jobs/smart_resume_t5_continued_pretraining_lr_0001_bugfixed_clean_restart.sh`
- Hydra root: `pretraining_logs_lr_0001_bugfixed_clean_restart/train/runs`
- dir: `pretraining_logs_lr_0001_bugfixed_clean_restart/train/runs/${now:%Y-%m-%d_%H-%M-%S}`
- W&B: name `lr-0001-bugfixed-clean-restart-warmup100k` (note: earlier 15k/100k variants used)
- Dataset: validated windows (50% overlap)

### LR 0.001 — bugfixed clean‑restart [still running]
- Config: `configs/experiment/t5_continued_pretraining_lr_001_bugfixed_clean_restart.yaml`
- Start job: `jobs/t5_continued_pretraining_lr_001_bugfixed_clean_restart.sh`
- Smart‑resume: `jobs/smart_resume_t5_continued_pretraining_lr_001_bugfixed_clean_restart.sh`
- Hydra root: `pretraining_logs_lr_001_bugfixed_clean_restart/train/runs`
- dir: `pretraining_logs_lr_001_bugfixed_clean_restart/train/runs/${now:%Y-%m-%d_%H-%M-%S}`
- W&B: name `lr-001-bugfixed-clean-restart-warmup60k`
- Dataset: validated windows (50% overlap)


### LR 0.0005 — inverse sqrt schedule [not running, was just a test]
- Config: `configs/experiment/t5_continued_pretraining_lr_0005_gradient_clip_1_with_inverse_sqrt_schedule.yaml`
- Start job: `jobs/t5_continued_pretraining_lr_0005_gradient_clip_1_with_inverse_sqrt_schedule.sh`
- Smart‑resume: (not present)
- Hydra root: `pretraining_logs_lr_0005_gradient_clip_1_with_inverse_sqrt_schedule/train/runs`
- dir: `pretraining_logs_lr_0005_gradient_clip_1_with_inverse_sqrt_schedule/train/runs/${now:%Y-%m-%d_%H-%M-%S}`
- Dataset: validated windows (50% overlap)

### LR 0.00005 — inverse sqrt schedule [still running]
- Config: `configs/experiment/t5_continued_pretraining_lr_00005_gradient_clip_1_with_inverse_sqrt_schedule.yaml`
- Start job: `jobs/t5_continued_pretraining_lr_00005_gradient_clip_1_with_inverse_sqrt_schedule.sh`
- Smart‑resume: `jobs/smart_resume_t5_continued_pretraining_lr_00005_gradient_clip_1.sh`
- Hydra root: `pretraining_logs_lr_00005_gradient_clip_1_with_inverse_sqrt_schedule/train/runs`
- dir: `pretraining_logs_lr_00005_gradient_clip_1_with_inverse_sqrt_schedule/train/runs/${now:%Y-%m-%d_%H-%M-%S}`
- Dataset: validated windows (50% overlap)

### Adafactor relative_step (no external scheduler) [still running]
- Config: `configs/experiment/t5_continued_pretraining_adafactor_relative_step_clip_1.yaml`
- Start job: `jobs/t5_continued_pretraining_adafactor_relative_step_clip_1.sh`
- Smart‑resume: `jobs/smart_resume_t5_continued_pretraining_adafactor_relative_step_clip_1.sh`
- Hydra root: `pretraining_logs_adafactor_relative_step_clip1/train/runs`
- dir: `pretraining_logs_adafactor_relative_step_clip1/train/runs/${now:%Y-%m-%d_%H-%M-%S}`
- Dataset: validated windows (50% overlap)

### Adafactor relative_step — OPTIMIZED collator (no scheduler) [still running]
- Config: `configs/experiment/t5_continued_pretraining_adafactor_relative_step_clip_1_OPTIMIZED.yaml`
- Start job: `jobs/t5_continued_pretraining_adafactor_relative_step_clip_1_OPTIMIZED.sh`
- Smart‑resume: `jobs/smart_resume_t5_continued_pretraining_adafactor_relative_step_clip_1_OPTIMIZED.sh`
- Hydra root: `pretraining_logs_adafactor_relative_step_clip1_OPTIMIZED/train/runs`
- dir: `pretraining_logs_adafactor_relative_step_clip1_OPTIMIZED/train/runs/${now:%Y-%m-%d_%H-%M-%S}`
- Dataset: validated windows (50% overlap)
- Notes: Same setup as above, but using the OPTIMIZED collator (T5‑formula, target_length≈114)

### Runs which we trained one full epoch on our data, Original our best and final runs but we found a critical bug so they are deprecated too 
For these runs I already did a lot of evaluation like the heatmaps over all checkpoints and also benchmarking them on multiple benchmarks like MMLU 

### Clean restart (fixed sampler baseline) e.g. Red Run
- Config: `configs/experiment/clean_restart_4gpu_h100.yaml`
- Start job: `jobs/clean_restart_4gpu_h100.sh`
- Smart‑resume: `jobs/smart_resume_4gpu.sh`
- Hydra root: `clean_restart_logs/train/runs`
- dir: `clean_restart_logs/train/runs/${now:%Y-%m-%d_%H-%M-%S}`
- W&B: group `H100-4GPU-clean-restart`, name `clean-restart-600k-15k-warmup-fixed-sampler-260M-docs`
- Notes: legacy baseline with gradient_clip=0.5, warmup=15k
- Dataset: validated windows (50% overlap)

### LR 0.001 — inverse sqrt schedule (production variant) e.g. Green Run 
- dir: pretraining_logs_lr_001_gradient_clip_1_with_inverse_sqrt_schedule/train/runs/${now:%Y-%m-%d_%H-%M-%S}
- Config: `configs/experiment/t5_continued_pretraining_lr_001_gradient_clip_1_with_inverse_sqrt_schedule.yaml`
- Start job: `jobs/t5_continued_pretraining_lr_001_gradient_clip_1_with_inverse_sqrt_schedule.sh`
- Smart‑resume: `jobs/smart_resume_t5_continued_pretraining_lr_001_gradient_clip_1.sh`
- Hydra root: `pretraining_logs_lr_001_gradient_clip_1_with_inverse_sqrt_schedule/train/runs`
- Dataset: validated windows (50% overlap)
- Notes: lr=1e‑3, warmup=20k, validated windows 50% overlap 

### LR 0.001 — OPTIMIZED clean‑restart [still running]
- Config: `configs/experiment/t5_continued_pretraining_lr_001_OPTIMIZED_clean_restart.yaml`
- Start job: `jobs/t5_continued_pretraining_lr_001_OPTIMIZED_clean_restart.sh`
- Smart‑resume: `jobs/smart_resume_t5_continued_pretraining_lr_001_OPTIMIZED_clean_restart.sh`
- Hydra root: `pretraining_logs_lr_001_OPTIMIZED_clean_restart/train/runs`
- dir: `pretraining_logs_lr_001_OPTIMIZED_clean_restart/train/runs/${now:%Y-%m-%d_%H-%M-%S}`
- Dataset: validated windows (50% overlap)
- Notes: Clean‑restart family with OPTIMIZED collator (T5‑formula, exact target_length=114) for apples‑to‑apples comparison.
- Gold checkpoint used for transfer/eval: `pretraining_logs_lr_001_OPTIMIZED_clean_restart/train/runs/2025-09-08_02-33-22/checkpoints/best/step-487500-val_ppl-3.72168.ckpt` (MMLU 0.2687, +17.3% vs T5‑Base)

### FLAN‑T5 LR 0.001 — inverse sqrt schedule e.g. Yellow Run 
- Config: `configs/experiment/flan_t5_lr_001_gradient_clip_1_with_inverse_sqrt_schedule.yaml`
- Start job: `jobs/flan_t5_lr_001_gradient_clip_1_with_inverse_sqrt_schedule.sh`
- Smart‑resume: `jobs/smart_resume_flan_t5_lr_001_gradient_clip_1.sh`
- Hydra root: `flan_t5_logs_lr_001_gradient_clip_1_with_inverse_sqrt_schedule/train/runs`
- dir: `flan_t5_logs_lr_001_gradient_clip_1_with_inverse_sqrt_schedule/train/runs/${now:%Y-%m-%d_%H-%M-%S}`
- W&B: group `H100-4GPU-FLAN-T5`, name `flan-t5-lr-001-clip-1.0-inverse-sqrt-679k-full-epoch-scifive-optimized`
- Dataset: validated windows (50% overlap)


## Old Runs which got deprecated since they were not stable or did not work good 

### SciFive‑style LR 0.001 — constant LR (no scheduler)
- Config: `configs/experiment/scifive_style_4gpu_h100.yaml`
- Start job: `jobs/scifive_style_4gpu_h100.sh`
- Smart‑resume: `jobs/smart_resume_scifive_4gpu.sh`
- Hydra root: `scifive_style_logs/train/runs`
- dir: `scifive_style_logs/train/runs/${now:%Y-%m-%d_%H-%M-%S}`
- Dataset: validated windows (50% overlap)

### SciFive‑style LR 0.002 — constant LR (no scheduler)
- Config: `configs/experiment/scifive_style_lr002_4gpu_h100.yaml`
- Start job: `jobs/scifive_style_lr002_4gpu_h100.sh`
- Smart‑resume: (not present)
- Hydra root: `scifive_style_lr002_logs/train/runs`
- dir: `scifive_style_lr002_logs/train/runs/${now:%Y-%m-%d_%H-%M-%S}`
- Dataset: validated windows (50% overlap)

### SciFive‑style v2 LR 0.001 — constant LR, clip=1.0
- Config: `configs/experiment/scifive_style_4gpu_h100_v2.yaml`
- Start job: `jobs/scifive_style_4gpu_h100_v2.sh`
- Smart‑resume: `jobs/smart_resume_scifive_v2_4gpu.sh`
- Hydra root: `scifive_style_v2_logs/train/runs`
- dir: `scifive_style_v2_logs/train/runs/${now:%Y-%m-%d_%H-%M-%S}`
- Dataset: validated windows (50% overlap)

## Old testing runs 

### First 4‑GPU production (pretraining_logs) was only for testing the first runs 
- Config: `configs/experiment/100_4GPU_10mio_doc.yaml`
- Start job: `jobs/final_4gpu_H100_production.sh` or `jobs/run_4gpu_10Mio.sh`
- Smart‑resume / restart: `jobs/restart_fixed_epoch_4gpu.sh`
- Hydra root: `pretraining_logs/train/runs`
- dir: `pretraining_logs/train/runs/${now:%Y-%m-%d_%H-%M-%S}`
- Dataset: validated windows (50% overlap)

## How to launch

### OPTIMIZED Collator Runs (NEW - Recommended)
```bash
# Start new optimized runs (T5-formula, 35% more efficient)
sbatch jobs/t5_continued_pretraining_lr_001_OPTIMIZED_text_0olap512_v3.sh
sbatch jobs/t5_continued_pretraining_lr_0001_OPTIMIZED_text_0olap512_v3.sh

# Resume optimized runs
sbatch jobs/smart_resume_t5_continued_pretraining_lr_001_OPTIMIZED_text_0olap512_v3_warmup50k.sh
sbatch jobs/smart_resume_t5_continued_pretraining_lr_0001_OPTIMIZED_text_0olap512_v3_warmup50k.sh
```

### Legacy Runs (Backwards Compatible)
```bash
# Resume existing legacy runs (can continue running alongside optimized)
sbatch jobs/smart_resume_t5_continued_pretraining_lr_001_bugfixed_text_0olap512_v3_warmup50k.sh
sbatch jobs/smart_resume_t5_continued_pretraining_lr_0001_bugfixed_text_0olap512_v3_warmup50k.sh
```

## OPTIMIZED vs LEGACY Collator Differences

| Aspect | Legacy (1.5x Heuristic) | OPTIMIZED (T5-Formula) |
|--------|-------------------------|------------------------|
| **Expansion Ratio** | 1.5x (768 tokens) | 1.109x (568 tokens) |
| **CPU Overhead** | Baseline | 35% reduction |
| **Target Length** | 512 (forced) | 114 (T5-correct) |
| **OLM Compliance** | Non-standard | Exact T5 implementation |
| **Backwards Compatible** | ✅ All existing runs | ✅ Opt-in via flag |
| **Validation Loss** | May have jumps | Clean, consistent |
| **W&B Group** | `text-0olap512-v3` | `continued-pretraining-OPTIMIZED` |

**Recommendation**: Start new OPTIMIZED runs for best performance, keep legacy runs for continuity.

## Notes & toggles

- Label smoothing and dropout overrides are supported in code but disabled by default:
  - `model.label_smoothing` (default 0.0)
  - `model.t5_model.dropout_rate` (default null → model default 0.1)
- Validation split fractions:
  - ~50k val: `0.000385`
  - ~100k val: `0.000774` (used in 0% overlap runs)
- Checkpoint discovery in smart‑resume scripts searches `checkpoints/steps/last.ckpt` → `checkpoints/best/last.ckpt` → latest `step-*.ckpt` under the run root.

## Misc

- Early tests (2‑GPU/V100 resume experiments)
  - Hydra root: `logs/train/runs`
  - dir: `logs/train/runs/${now:%Y-%m-%d_%H-%M-%S}`
  - Config: (no dedicated experiment YAML; ad‑hoc tests)
