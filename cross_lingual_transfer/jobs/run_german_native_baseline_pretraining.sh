#!/bin/bash
#SBATCH --job-name=german_native_baseline_pretraining
#SBATCH --partition=H100,H100-PCI,H100-SLT,H200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=6:00:00

#SBATCH --output=/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/logs/slurm_%j_german_native_baseline.out
#SBATCH --error=/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/logs/slurm_%j_german_native_baseline.err

echo "=============================================="
echo "German NATIVE BASELINE Continued Pretraining"
echo "FOR SCIENTIFIC CROSS-LINGUAL TRANSFER EVALUATION"
echo "Steps: 15k, LR: 0.001, Gradient Clip: 1.0"
echo "Model: GermanT5/t5-efficient-gc4-german-base-nl36 (ORIGINAL)"
echo "Tokenizer: GermanT5/t5-efficient-gc4-german-base-nl36 (ORIGINAL)"
echo "Purpose: True baseline for Wechsel transfer comparison"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="

# Check German sliding windows data
if [ ! -d "/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/data/german/sliding_windows" ]; then
    echo "ERROR: German sliding windows data not found"
    echo "Expected: /netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/data/german/sliding_windows"
    echo "Please run German sliding windows preparation first"
    exit 1
fi

echo "German sliding windows data found"

# Environment variables
export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS}
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS}
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

cd /netscratch/nrauscher/projects/BA-hydra

echo "Starting German NATIVE BASELINE continued pretraining..."
echo "Configuration: configs/experiment/german_native_baseline_continued_pretraining.yaml"
echo "Model: GermanT5/t5-efficient-gc4-german-base-nl36 (ORIGINAL HF model)"
echo "Tokenizer: GermanT5/t5-efficient-gc4-german-base-nl36 (ORIGINAL German tokenizer)"
echo "Learning Rate: 0.001 (identical to transfer experiment)"
echo "Gradient Clip: 1.0 (identical to transfer experiment)"
echo "Effective batch size: 48 (24*1*2 with gradient accumulation)"
echo "Using OPTIMIZED T5-formula collator (1.109x expansion, target_length=114)"
echo "Validation: 0.0385% for speed (identical to transfer)"
echo ""
echo "SCIENTIFIC COMPARISON SETUP:"
echo "  Transfer Model: English Scientific T5 → Wechsel → German + Transfer Tokenizer"
echo "  Native Baseline: German HF T5 + Original German Tokenizer"
echo "  Same Training: 15k steps on German scientific corpus"
echo "  Hypothesis: Transfer should be better due to English scientific knowledge"
echo ""

srun -K \
    --container-image=/netscratch/nrauscher/containers/hydra_pytorch_25.05.25-final.sqsh \
    --container-mounts=/netscratch:/netscratch,/fscratch:/fscratch \
    --container-workdir=/netscratch/nrauscher/projects/BA-hydra \
    /bin/bash -c "
        source .venv_pretraining/bin/activate && \
        python src/train.py experiment=german_native_baseline_continued_pretraining
    "

EXIT_CODE=$?
echo "=============================================="
echo "German native baseline pretraining completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo ""
echo "NEXT STEPS FOR SCIENTIFIC EVALUATION:"
echo "1. Compare val_ppl: Transfer vs Native Baseline"
echo "2. Run downstream German scientific QA evaluation"
echo "3. Analyze training dynamics and convergence speed"
echo "4. Test on different German scientific domains"
echo "=============================================="