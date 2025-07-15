#!/bin/bash
#SBATCH --job-name=copy_data_to_fscratch
#SBATCH --time=02:00:00  # 2 hours should be plenty for the copy
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=batch
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nikolas.rauscher@dfki.de
#SBATCH --output=logs/slurm-copy-data-%j.out
#SBATCH --error=logs/slurm-copy-data-%j.err

# --- DATA COPY SCRIPT ---
echo "Starting data copy job at $(date)"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"

# 1. Define Source and Destination Paths
SOURCE_DIR="/netscratch/nrauscher/projects/BA-hydra/data/cleaned_pretraining_OPTIMIZED_3T_fix"
DEST_PARENT_DIR="/fscratch/nrauscher"
DEST_DIR="${DEST_PARENT_DIR}/projects/BA-hydra/data/cleaned-data-final-09-07-2025/"

echo "Source:      $SOURCE_DIR"
echo "Destination: $DEST_DIR"

# 2. Check if destination already exists to prevent accidental re-copying
if [ -d "$DEST_DIR" ]; then
    echo "Error: Destination directory $DEST_DIR already exists. Please remove it if you want to re-copy." >&2
    exit 1
fi

# 3. Create complete directory structure and start the copy
echo "Creating destination directory: $DEST_DIR"
mkdir -p "$DEST_DIR"

echo "Starting rsync copy process..."
start_time=$SECONDS

# Zeige die Anzahl der zu kopierenden Dateien
echo "Files to copy:"
find "${SOURCE_DIR}" -type f | wc -l

# Verwende rsync mit detaillierteren Optionen
rsync -avh --info=progress2 --stats "${SOURCE_DIR}/" "${DEST_DIR}/"

end_time=$SECONDS
copy_duration=$((end_time - start_time))

# 4. Final check and report
if [ $? -eq 0 ]; then
    # Überprüfe, ob Dateien im Zielverzeichnis existieren
    dest_file_count=$(find "${DEST_DIR}" -type f | wc -l)
    
    if [ "$dest_file_count" -gt 0 ]; then
        echo "-------------------------------------------------"
        echo "✅ Data copy completed successfully!"
        echo "Total time: $copy_duration seconds."
        echo "Files copied: $dest_file_count"
        echo "Source size: $(du -sh ${SOURCE_DIR} | cut -f1)"
        echo "Destination size: $(du -sh ${DEST_DIR} | cut -f1)"
        echo "Data is now available at: $DEST_DIR"
        echo "-------------------------------------------------"
    else
        echo "-------------------------------------------------"
        echo "⚠️ Warning: rsync completed but no files were found in the destination."
        echo "Please check if the source directory contains files."
        echo "-------------------------------------------------"
        exit 2
    fi
else
    echo "-------------------------------------------------"
    echo "❌ Data copy failed. Check the error messages above." >&2
    echo "-------------------------------------------------"
    exit 1
fi
