#!/bin/bash

# =============================================================================
# Add Job to Chain - Finds newest job and adds a dependent job
# Usage: ./add_to_chain.sh [number_of_jobs_to_add]
# =============================================================================

# Number of jobs to add (default 1)
NUM_JOBS=${1:-1}

echo "Adding $NUM_JOBS job(s) to chain..."

# Find the latest job ID for the current user
LATEST_JOB=$(squeue -u $USER --format="%.10i" --noheader | sort -n | tail -1 | tr -d ' ')

if [[ -z "$LATEST_JOB" ]]; then
    echo "ERROR: No jobs found in queue for user $USER"
    echo "Start a job first, then use this script to add to the chain"
    exit 1
fi

echo "Latest job found: $LATEST_JOB"

# Add the requested number of jobs
PREV_JOB=$LATEST_JOB
for ((i=1; i<=NUM_JOBS; i++)); do
    echo "Adding job $i/$NUM_JOBS dependent on job $PREV_JOB..."
    
    NEW_JOB=$(sbatch --parsable --dependency=afterany:$PREV_JOB jobs/smart_resume_4gpu.sh 2>/dev/null | head -1)
    NEW_JOB=$(echo "$NEW_JOB" | grep -o '^[0-9]*')
    
    if [[ -n "$NEW_JOB" ]]; then
        echo "✅ Submitted job $NEW_JOB (dependent on $PREV_JOB)"
        PREV_JOB=$NEW_JOB
    else
        echo "❌ Failed to submit job $i"
        exit 1
    fi
    
    # Small delay to avoid submission issues
    sleep 1
done

echo ""
echo "✅ Successfully added $NUM_JOBS job(s) to chain!"
echo "Final job chain ends with: $PREV_JOB"
echo ""
echo "Check status with: squeue -u \$USER"