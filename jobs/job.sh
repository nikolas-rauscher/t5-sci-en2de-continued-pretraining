#!/bin/bash
#SBATCH --job-name=clean_data_job
#SBATCH --time=05:00:00
#SBATCH --partition=V100-16GB
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nikolas.rauscher@dfki.de

PROJECT_ROOT="/netscratch/nrauscher/projects/BA-hydra"



srun --container-mounts=/netscratch:/netscratch \
     --container-image=/netscratch/nrauscher/containers/hydra_pytorch_23.05-final.sqsh \
     --container-workdir="$PROJECT_ROOT" \
     --partition=V100-16GB \
     python src/dataprep/pipelines/clean_data.py \
     ++cleaning.cleaning.tasks=32 \
     ++cleaning.cleaning.workers=28 \
     ++cleaning.limit_documents=-1
