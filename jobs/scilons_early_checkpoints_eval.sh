#!/bin/bash
#SBATCH --job-name=scilons_early_checkpoints
#SBATCH --output=logs/scilons_early_checkpoints_%j.out
#SBATCH --error=logs/scilons_early_checkpoints_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=A100-80GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Activate environment
source .venv_eval/bin/activate

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="scilons_eval_runs/early_checkpoints_comparison_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

echo "==========================================="
echo "Starting Scilons Evaluation of Early Checkpoints"
echo "Time: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "==========================================="

# Evaluate each model on Scilons
MODELS=("t5_clean_restart_130k" "t5_clean_restart_150k" "t5_clean_restart_252k")

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "==========================================="
    echo "Evaluating: $MODEL"
    echo "==========================================="
    
    MODEL_PATH="evaluation/models/$MODEL"
    OUTPUT_FILE="$OUTPUT_DIR/${MODEL}_${TIMESTAMP}.csv"
    
    python external/scilons-eval/scilons-eval.py \
        --model_path $MODEL_PATH \
        --output_csv $OUTPUT_FILE \
        --batch_size 16 \
        --device cuda
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully evaluated $MODEL"
        echo "  Results saved to: $OUTPUT_FILE"
    else
        echo "✗ Failed to evaluate $MODEL"
    fi
done

echo ""
echo "==========================================="
echo "All evaluations complete!"
echo "Time: $(date)"
echo "Results directory: $OUTPUT_DIR"
echo "==========================================="

# Run comparison analysis
echo ""
echo "Running comparison analysis..."
python -c "
import pandas as pd
import os

output_dir = '$OUTPUT_DIR'
models = ['t5_clean_restart_130k', 't5_clean_restart_150k', 't5_clean_restart_252k']

print('\\n' + '='*60)
print('SCILONS EVALUATION SUMMARY')
print('='*60)

for model in models:
    files = [f for f in os.listdir(output_dir) if f.startswith(model)]
    if files:
        csv_path = os.path.join(output_dir, files[0])
        df = pd.read_csv(csv_path)
        
        # Calculate average scores
        avg_micro = df['score 1'].dropna().mean()
        avg_macro = df['score 2'].mean()
        
        # Extract step number from model name
        step = model.split('_')[-1].replace('k', '000')
        
        print(f'\\n{model} (Step {step}):')
        print(f'  Average Micro F1: {avg_micro:.4f}' if avg_micro == avg_micro else '  Average Micro F1: N/A')
        print(f'  Average Macro F1: {avg_macro:.4f}')
        print(f'  Tasks evaluated: {len(df)}')
"