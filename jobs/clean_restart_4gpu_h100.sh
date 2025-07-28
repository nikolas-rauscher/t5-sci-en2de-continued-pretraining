#!/bin/bash
#SBATCH --job-name=H100_clean_restart
#SBATCH --partition=H100-PCI,H100-SLT,H200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=600G
#SBATCH --time=24:00:00

#SBATCH --output=/netscratch/nrauscher/projects/BA-hydra/logs/slurm_%j_clean_restart_4gpu_h100.out
#SBATCH --error=/netscratch/nrauscher/projects/BA-hydra/logs/slurm_%j_clean_restart_4gpu_h100.err

if [ ! -d "/fscratch" ]; then
    echo "ERROR: /fscratch not mounted!"; exit 1; fi
if [ ! -d "/fscratch/nrauscher/projects/BA-hydra/data/cleaned_sliding_windows/text_2025-07-14" ]; then
    echo "ERROR: Training data not found"; exit 1; fi

export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS}
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS}
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

cd /netscratch/nrauscher/projects/BA-hydra
python --version
python -c 'import torch; print("torch", torch.__version__)'
python -c 'import lightning; print("lightning", lightning.__version__)'
python -c 'import torch; print("GPUs", torch.cuda.device_count())'
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

srun -K \
    --container-image=/netscratch/nrauscher/containers/hydra_pytorch_25.05.25-final.sqsh \
    --container-mounts=/netscratch:/netscratch,/fscratch:/fscratch \
    --container-workdir=/netscratch/nrauscher/projects/BA-hydra \
    /bin/bash -c "source .venv_pretraining/bin/activate && python src/train.py experiment=clean_restart_4gpu_h100"

EXIT_CODE=$?
echo "Exit code: $EXIT_CODE"
echo "/netscratch/nrauscher/projects/BA-hydra/logs/slurm_${SLURM_JOB_ID}_clean_restart_4gpu_h100.*"
echo "outputs/[timestamp]/logs/"
echo "https://wandb.ai/[your-username]/BA-thesis-t5-final-runs" 