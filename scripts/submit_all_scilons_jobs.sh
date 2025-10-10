#!/bin/bash

echo "🚀 Submitting all Scilons evaluation jobs..."

# Count total jobs
total_jobs=$(find jobs/scilons_evals -name "scilons_*.sh" | wc -l)
echo "Found $total_jobs job scripts"

if [ $total_jobs -eq 0 ]; then
    echo "No Scilons job scripts found in jobs/scilons_evals/"
    echo "Run the modular_scilons_evaluator.py first to create job scripts"
    exit 1
fi

# Submit all jobs
submitted=0
for job in jobs/scilons_evals/*/scilons_*.sh; do
    if [ -f "$job" ]; then
        echo "Submitting: $job"
        sbatch "$job"
        submitted=$((submitted + 1))
    fi
done

echo ""
echo "✅ Submitted $submitted jobs!"
echo "Check status with: squeue -u $USER"