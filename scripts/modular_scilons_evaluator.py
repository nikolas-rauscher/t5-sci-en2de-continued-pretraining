#!/usr/bin/env python3
"""
Modular Scilons Evaluator
Automatically converts and evaluates all checkpoints at specified intervals for Scilons tasks
Creates SEPARATE SLURM jobs for PARALLEL execution
"""

import argparse
import os
import sys
from pathlib import Path
import re
from typing import List, Dict
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

sys.path.append('/netscratch/nrauscher/projects/BA-hydra')

class ScilonsCheckpointEvaluator:
    def __init__(self, run_root: str, eval_interval: int = 50000):
        """
        Initialize the Scilons evaluator
        
        Args:
            run_root: Root directory of the training run
            eval_interval: Evaluate every N steps (default 50k)
        """
        self.run_root = Path(run_root)
        self.eval_interval = eval_interval
        self.run_name = self.run_root.name
        
        # Find all available checkpoints
        self.checkpoints = self._find_checkpoints()
        
    def _find_checkpoints(self) -> List[Dict]:
        """Find all checkpoints in the run directory"""
        checkpoints = []
        
        # Pattern for checkpoint files
        patterns = [
            "train/runs/*/checkpoints/best/*.ckpt",
            "train/runs/*/checkpoints/steps/*.ckpt",
            "train/runs/*/checkpoints/*.ckpt"
        ]
        
        for pattern in patterns:
            for ckpt_path in self.run_root.glob(pattern):
                # Extract step number
                step_match = re.search(r'step[-=](\d+)', str(ckpt_path))
                if step_match:
                    step = int(step_match.group(1))
                    
                    # Extract perplexity if available
                    ppl_match = re.search(r'val_ppl[-=]([\d.]+)', str(ckpt_path))
                    if ppl_match:
                        try:
                            # Remove trailing dots
                            ppl_str = ppl_match.group(1).rstrip('.')
                            ppl = float(ppl_str)
                        except:
                            ppl = None
                    else:
                        ppl = None
                    
                    checkpoints.append({
                        'path': str(ckpt_path),
                        'step': step,
                        'ppl': ppl,
                        'name': f"{self.run_name}_step{step//1000}k"
                    })
        
        # Sort by step
        checkpoints.sort(key=lambda x: x['step'])
        return checkpoints
    
    def filter_checkpoints(self, interval: int = None) -> List[Dict]:
        """Filter checkpoints by interval"""
        if interval is None:
            interval = self.eval_interval
            
        filtered = []
        last_included = -interval  # Ensure first checkpoint is included
        
        for ckpt in self.checkpoints:
            # Include checkpoints at approximately the interval
            if ckpt['step'] - last_included >= interval:
                filtered.append(ckpt)
                last_included = ckpt['step']
            # Also include best/final checkpoints
            elif 'best' in ckpt['path'] or 'last' in ckpt['path']:
                # Check if not too close to last included
                if ckpt['step'] - last_included >= interval // 2:
                    filtered.append(ckpt)
                    last_included = ckpt['step']
                
        return filtered
    
    def convert_checkpoint(self, checkpoint: Dict) -> str:
        """Convert Lightning checkpoint to HuggingFace format"""
        output_dir = f"/netscratch/nrauscher/projects/BA-hydra/evaluation/models/{checkpoint['name']}"
        
        # Skip if already converted
        if Path(output_dir).exists() and (Path(output_dir) / "config.json").exists():
            print(f"  ✓ Already converted: {checkpoint['name']}")
            return output_dir
        
        print(f"  Converting: {checkpoint['name']}...")
        
        try:
            # Load checkpoint
            ckpt = torch.load(checkpoint['path'], map_location='cpu', weights_only=False)
            state_dict = ckpt['state_dict']
            
            # Clean state dict
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.model.'):
                    new_key = key[12:]  # Remove 'model.model.' prefix
                    new_state_dict[new_key] = value
                elif key.startswith('model.'):
                    new_key = key[6:]  # Remove 'model.' prefix
                    new_state_dict[new_key] = value
            
            # Load base T5 model and tokenizer
            model = T5ForConditionalGeneration.from_pretrained('t5-base')
            tokenizer = T5Tokenizer.from_pretrained('t5-base')
            
            # Load the new weights
            model.load_state_dict(new_state_dict, strict=True)
            
            # Save
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            print(f"  ✓ Converted: {checkpoint['name']}")
            return output_dir
            
        except Exception as e:
            print(f"  ❌ Failed to convert {checkpoint['name']}: {e}")
            return None
    
    def generate_scilons_job(self, checkpoint: Dict, model_dir: str) -> str:
        """Generate INDIVIDUAL Scilons evaluation job script for PARALLEL execution"""
        job_name = f"scilons_{checkpoint['name']}"
        
        # Create subdirectory for organized job scripts
        job_subdir = f"jobs/scilons_evals/{self.run_name}"
        os.makedirs(job_subdir, exist_ok=True)
        
        script_path = f"{job_subdir}/{job_name}.sh"
        
        script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=A100-80GB,H100-SLT,H100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=128GB
#SBATCH --output=/netscratch/nrauscher/projects/BA-hydra/scilons_eval_runs/logs/{job_name}_%j.out
#SBATCH --error=/netscratch/nrauscher/projects/BA-hydra/scilons_eval_runs/logs/{job_name}_%j.err

set -euo pipefail

# Configuration
HF_MODEL_DIR="{model_dir}"
SCILONS_DATA="/netscratch/abu/Scilons/scibert/data"
CONTAINER_IMG="/netscratch/abu/scilons_eval_updated.sqsh"

WORKDIR="$(pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${{WORKDIR}}/scilons_eval_runs/{checkpoint['name']}_improved/${{TS}}"

# Run inside container with task finetuning + evaluation
srun -K \\
  --container-image="${{CONTAINER_IMG}}" \\
  --container-workdir="${{WORKDIR}}" \\
  --container-mounts="/netscratch/${{USER}}:/netscratch/${{USER}},${{WORKDIR}}:${{WORKDIR}},/netscratch/abu/Scilons/scibert/data:/netscratch/abu/Scilons/scibert/data:ro" \\
  bash -lc "set -euo pipefail
    export PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=false
    export HF_HOME=/netscratch/${{USER}}/.cache/huggingface
    export HF_DATASETS_CACHE=/netscratch/${{USER}}/.cache/huggingface/datasets
    export PYTHONNOUSERSITE=1
    export PYTHONPATH=\\\"${{WORKDIR}}:\\${{PYTHONPATH:-}}\\\"
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

    mkdir -p \\\"${{OUT_DIR}}\\\"

    echo '==========================================='
    echo 'Scilons Evaluation (PARALLEL JOB)'
    echo 'Model: {checkpoint['name']}'
    echo 'Step: {checkpoint['step']}'
    echo 'Val PPL: {checkpoint['ppl'] if checkpoint['ppl'] else 'N/A'}'
    echo 'Time:' \\$(date)
    echo '==========================================='

    # Run improved scilons-eval with task finetuning
    cd \\\"${{OUT_DIR}}\\\"
    python3 ${{WORKDIR}}/external/scilons-eval/evaluate_all_tasks_improved.py \\
      --model \\\"${{HF_MODEL_DIR}}\\\" \\
      --tokenizer \\\"${{HF_MODEL_DIR}}\\\" \\
      --data \\\"${{SCILONS_DATA}}\\\" \\
      --hf_token \\\"${{HF_TOKEN:-}}\\\" \\
      --max_length 512 \\
      --output_dir \\\"${{OUT_DIR}}\\\" \\
      --per_device_train_batch_size 32 \\
      --num_train_epochs 4 \\
      --learning_rate 2e-5 \\
      --seq_to_seq_model true \\
      --report_to wandb \\
      --logging_steps 50 \\
      --save_strategy epoch \\
      --evaluation_strategy epoch \\
      --load_best_model_at_end true \\
      --metric_for_best_model eval_loss \\
      --greater_is_better false
    
    echo '==========================================='
    echo 'Evaluation Complete!'
    echo 'Results saved to:' \\\"${{OUT_DIR}}\\\"
    echo '==========================================='
  "
"""
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        return script_path
    
    def print_summary(self, checkpoints: List[Dict]):
        """Print summary of checkpoints to evaluate"""
        print("=" * 80)
        print(f"Scilons PARALLEL Evaluation Plan for {self.run_name}")
        print("=" * 80)
        print(f"Total checkpoints found: {len(self.checkpoints)}")
        print(f"Checkpoints to evaluate: {len(checkpoints)}")
        print(f"Interval: Every {self.eval_interval//1000}k steps")
        print(f"Execution: PARALLEL (separate SLURM jobs)")
        print("\nCheckpoints selected for Scilons (task finetuning + eval):")
        print("-" * 50)
        
        for ckpt in checkpoints:
            ppl_str = f"(val_ppl: {ckpt['ppl']:.3f})" if ckpt['ppl'] else ""
            print(f"  Step {ckpt['step']:7d} {ppl_str:<20} → {ckpt['name']}")
        
        print("\n📊 Parallel Execution Benefits:")
        print(f"  - {len(checkpoints)} jobs run simultaneously")
        print(f"  - Total time: ~4-6 hours (instead of {len(checkpoints)*4}-{len(checkpoints)*6} hours)")
        print(f"  - Each job gets dedicated GPU")
    
    def run(self, auto_submit: bool = False, convert_only: bool = False):
        """Run the complete evaluation pipeline"""
        # Filter checkpoints
        checkpoints = self.filter_checkpoints()
        
        if not checkpoints:
            print(f"No checkpoints found at {self.eval_interval//1000}k intervals!")
            return
        
        # Show summary
        self.print_summary(checkpoints)
        
        print("\n" + "=" * 80)
        print("Starting Conversion and Job Generation")
        print("=" * 80)
        
        job_scripts = []
        
        for i, ckpt in enumerate(checkpoints, 1):
            print(f"\n[{i}/{len(checkpoints)}] Processing {ckpt['name']}...")
            
            # Convert checkpoint
            model_dir = self.convert_checkpoint(ckpt)
            
            if model_dir and not convert_only:
                # Generate job script
                script_path = self.generate_scilons_job(ckpt, model_dir)
                job_scripts.append(script_path)
                print(f"  ✓ Job script: {script_path}")
        
        if convert_only:
            print(f"\n✓ Converted {len(checkpoints)} checkpoints")
            print("  Run without --convert-only to create and submit jobs")
            return
        
        # Show submission commands
        print("\n" + "=" * 80)
        print("PARALLEL Job Submission")
        print("=" * 80)
        
        if auto_submit:
            print(f"\n🚀 Submitting {len(job_scripts)} jobs in PARALLEL...")
            for i, script in enumerate(job_scripts, 1):
                cmd = f"sbatch {script}"
                print(f"  [{i}/{len(job_scripts)}] {cmd}")
                os.system(cmd)
            print(f"\n✅ All {len(job_scripts)} jobs submitted!")
            print("  Jobs will run in PARALLEL on different GPUs")
            print("  Check status with: squeue -u $USER")
        else:
            print("\n📋 To submit all jobs in PARALLEL, run:")
            print("-" * 40)
            print("# Submit each job individually:")
            for script in job_scripts:
                print(f"sbatch {script}")
            print("\n# Or submit all at once with a loop:")
            print(f"for job in jobs/scilons_evals/{self.run_name}/scilons_*.sh; do sbatch $job; done")
            print("\n💡 Add --submit flag to auto-submit all jobs")


def main():
    parser = argparse.ArgumentParser(
        description="Modular Scilons evaluator - Creates PARALLEL jobs for fast evaluation"
    )
    parser.add_argument('run_root', help='Root directory of the training run')
    parser.add_argument('--interval', type=int, default=50000,
                       help='Evaluate every N steps (default: 50000)')
    parser.add_argument('--submit', action='store_true',
                       help='Automatically submit all jobs in PARALLEL')
    parser.add_argument('--convert-only', action='store_true',
                       help='Only convert checkpoints, do not create job scripts')
    
    # Quick presets
    parser.add_argument('--every-50k', action='store_true',
                       help='Evaluate every 50k steps')
    parser.add_argument('--every-100k', action='store_true', 
                       help='Evaluate every 100k steps')
    
    args = parser.parse_args()
    
    # Handle presets
    if args.every_50k:
        args.interval = 50000
    elif args.every_100k:
        args.interval = 100000
    
    # Create evaluator
    evaluator = ScilonsCheckpointEvaluator(
        run_root=args.run_root,
        eval_interval=args.interval
    )
    
    # Run evaluation
    evaluator.run(
        auto_submit=args.submit,
        convert_only=args.convert_only
    )


if __name__ == "__main__":
    main()