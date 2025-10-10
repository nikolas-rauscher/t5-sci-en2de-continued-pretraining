#!/bin/bash
#SBATCH --job-name=german_continued_pretraining
#SBATCH --partition=H100,H100-PCI,H100-SLT,H200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=6:00:00

#SBATCH --output=/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/logs/slurm_%j_german_pretraining.out
#SBATCH --error=/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/logs/slurm_%j_german_pretraining.err

echo "=============================================="
echo "German Continued Pretraining after Wechsel Transfer"
echo "Steps: 15k, LR: 0.001, Gradient Clip: 1.0"
echo "Using OPTIMIZED T5-formula collator (same as English)"
echo "Dataset: German scientific texts with sliding windows"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="

# Check if transfer model exists
if [ ! -d "/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/models/german_T5_Optimized_50Olap_clean_restart_487k" ]; then
    echo "ERROR: Wechsel transferred model not found"
    echo "Please run Wechsel transfer first"
    exit 1
fi

echo "✅ Wechsel transferred model found"

# Check German sliding windows data
if [ ! -d "/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/data/german/sliding_windows" ]; then
    echo "ERROR: German sliding windows data not found at expected location"
    echo "Expected: /netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/data/german/sliding_windows"
    echo "Please run German sliding windows preparation first:"
    echo "  sbatch /netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/jobs/run_german_sliding_windows.sh"
    exit 1
fi

echo "✅ German sliding windows data found"

# Environment variables
export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS}
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS}
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

cd /netscratch/nrauscher/projects/BA-hydra

echo "Starting German continued pretraining..."
echo "Configuration: configs/experiment/german_continued_pretraining.yaml"
echo "Learning Rate: 0.001 (proven from English experiments)"
echo "Gradient Clip: 1.0 (proven from English experiments)"
echo "Effective batch size: 48 (48*1*1)"
echo "Using OPTIMIZED T5-formula collator (1.109x expansion, target_length=114)"
echo "Validation: 0.0385% for speed (same as English)"

srun -K \
    --container-image=/netscratch/nrauscher/containers/hydra_pytorch_25.05.25-final.sqsh \
    --container-mounts=/netscratch:/netscratch,/fscratch:/fscratch \
    --container-workdir=/netscratch/nrauscher/projects/BA-hydra \
    /bin/bash -c "
        source .venv_pretraining/bin/activate && \
        python src/train.py experiment=german_continued_pretraining
    "

EXIT_CODE=$?
echo "=============================================="
echo "German continued pretraining completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=============================================="