#!/usr/bin/env python3

import os
import re
from pathlib import Path
from datetime import datetime

def extract_step_from_filename(filename):
    """Extract step number from checkpoint filename."""
    if 'step-step=' in filename:
        match = re.search(r'step-step=(\d+)\.ckpt', filename)
    elif 'step-' in filename:
        match = re.search(r'step-(\d+)-val_ppl', filename)
    else:
        return None
    
    return int(match.group(1)) if match else None

def extract_ppl_value(filename):
    """Extract perplexity value from best checkpoint filename."""
    match = re.search(r'val_ppl-([0-9.]+)', filename)
    if match:
        ppl_str = match.group(1).rstrip('.')
        try:
            return float(ppl_str)
        except ValueError:
            return None
    return None

def get_unique_checkpoints_for_run(base_dir):
    """Get all unique checkpoints for a specific run (prefer best over step)."""
    checkpoints = []
    step_to_checkpoint = {}  # step -> checkpoint info
    
    if not Path(base_dir).exists():
        return checkpoints
        
    for run_dir in sorted(Path(base_dir).glob('train/runs/*')):
        if not run_dir.is_dir():
            continue
            
        sub_run_name = run_dir.name
        step_dir = run_dir / 'checkpoints' / 'steps'
        best_dir = run_dir / 'checkpoints' / 'best'
        
        # First, collect step checkpoints
        if step_dir.exists():
            for ckpt in step_dir.glob('step-step=*.ckpt'):
                step = extract_step_from_filename(ckpt.name)
                if step:
                    step_to_checkpoint[step] = {
                        'step': step,
                        'type': 'step',
                        'sub_run': sub_run_name,
                        'filename': ckpt.name,
                        'full_path': str(ckpt),
                        'ppl_value': None,
                        'priority': 1  # Lower priority
                    }
        
        # Then, collect best checkpoints (they override step checkpoints)
        if best_dir.exists():
            for ckpt in best_dir.glob('step-*.ckpt'):
                if 'last.ckpt' not in ckpt.name:
                    step = extract_step_from_filename(ckpt.name)
                    if step:
                        ppl_value = extract_ppl_value(ckpt.name)
                        step_to_checkpoint[step] = {
                            'step': step,
                            'type': 'best',
                            'sub_run': sub_run_name,
                            'filename': ckpt.name,
                            'full_path': str(ckpt),
                            'ppl_value': ppl_value,
                            'priority': 0  # Higher priority
                        }
    
    # Convert to sorted list
    return sorted(step_to_checkpoint.values(), key=lambda x: x['step'])

def generate_config_for_run(base_dir, run_name, run_short, group_name):
    """Generate evaluation config for a specific run."""
    checkpoints = get_unique_checkpoints_for_run(base_dir)
    
    if not checkpoints:
        return None
    
    config_lines = []
    config_lines.append("# @package _global_")
    config_lines.append("")
    config_lines.append(f"# {run_name} - Complete Checkpoint Evaluation")
    config_lines.append(f"# All unique checkpoints from {run_name}")
    config_lines.append("")
    config_lines.append("# Weights & Biases configuration")
    config_lines.append("logger:")
    config_lines.append("  wandb:")
    config_lines.append("    project: \"eval-mmlu\"")
    config_lines.append("    entity: \"nikolas-rauscher-dfki\"")
    config_lines.append(f"    group: \"{group_name}\"")
    config_lines.append(f"    tags: [\"mmlu\", \"{run_short}\", \"complete-eval\", \"all-checkpoints\"]")
    config_lines.append("")
    config_lines.append("# All unique checkpoints")
    config_lines.append("models:")
    
    # Add all checkpoints
    for i, ckpt in enumerate(checkpoints, 1):
        step_k = ckpt['step'] // 1000
        
        # Create model name
        if ckpt['type'] == 'best' and ckpt['ppl_value']:
            ppl_str = str(ckpt['ppl_value']).replace('.', '')
            model_name = f"{run_short}_{ckpt['type']}{i:02d}_{step_k}k_ppl{ppl_str}"
        else:
            model_name = f"{run_short}_{ckpt['type']}{i:02d}_{step_k}k"
        
        config_lines.append(f"  - name: \"{model_name}\"")
        config_lines.append(f"    path: \"{ckpt['full_path']}\"")
        config_lines.append("")
    
    # Add baseline models for comparison
    config_lines.append("  # Baseline models for comparison")
    config_lines.append("  - name: \"t5_base_original\"")
    config_lines.append("    path: \"t5-base\"")
    config_lines.append("")
    
    # Add FLAN-T5 base for FLAN-T5 run
    if run_short == 'flan_t5':
        config_lines.append("  - name: \"flan_t5_base_original\"")
        config_lines.append("    path: \"google/flan-t5-base\"")
        config_lines.append("")
    
    # MMLU benchmark configuration
    config_lines.append("# MMLU benchmark - 0-shot only for faster evaluation")
    config_lines.append("benchmarks:")
    config_lines.append("  - name: \"mmlu\"")
    config_lines.append("    shots: [0]")
    config_lines.append("")
    
    # Evaluation configuration
    config_lines.append("# Evaluation configuration")
    config_lines.append("batch_size: auto")
    config_lines.append("max_length: 512")
    config_lines.append("temperature: 0.0")
    config_lines.append("device: \"auto\"")
    config_lines.append("seed: 42")
    config_lines.append("")
    
    # Export configuration
    config_lines.append("# Export results")
    config_lines.append("export_csv: true")
    config_lines.append(f"csv_filename: \"{run_short}_complete_eval_results.csv\"")
    
    return "\n".join(config_lines)

def main():
    """Generate all three run-specific configs."""
    runs = [
        {
            'base_dir': 'clean_restart_logs',
            'run_name': 'Clean Restart Run (Conservative LR 0.0001)',
            'run_short': 'clean_restart',
            'group_name': 'Clean-Restart-Complete',
            'filename': 'clean_restart_complete_eval.yaml'
        },
        {
            'base_dir': 'pretraining_logs_lr_001_gradient_clip_1_with_inverse_sqrt_schedule',
            'run_name': 'Green Run (Optimized LR 0.001)',
            'run_short': 'green_run',
            'group_name': 'Green-Run-Complete',
            'filename': 'green_run_complete_eval.yaml'
        },
        {
            'base_dir': 'flan_t5_logs_lr_001_gradient_clip_1_with_inverse_sqrt_schedule',
            'run_name': 'FLAN-T5 Run (Instruction-Tuned)',
            'run_short': 'flan_t5',
            'group_name': 'FLAN-T5-Complete',
            'filename': 'flan_t5_complete_eval.yaml'
        }
    ]
    
    configs_dir = Path('configs/experiment')
    configs_dir.mkdir(exist_ok=True)
    
    for run_config in runs:
        print(f"Generating config for {run_config['run_name']}...")
        
        config_content = generate_config_for_run(
            run_config['base_dir'],
            run_config['run_name'],
            run_config['run_short'],
            run_config['group_name']
        )
        
        if config_content:
            output_path = configs_dir / run_config['filename']
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(config_content)
            
            # Count checkpoints
            checkpoints = get_unique_checkpoints_for_run(run_config['base_dir'])
            step_count = len([c for c in checkpoints if c['type'] == 'step'])
            best_count = len([c for c in checkpoints if c['type'] == 'best'])
            
            print(f"✓ Created {output_path}")
            print(f"  - {len(checkpoints)} total checkpoints ({step_count} step, {best_count} best)")
        else:
            print(f"✗ No checkpoints found for {run_config['run_name']}")
        
        print()
    
    print("All configs generated!")
    print(f"Files created in {configs_dir}/")

if __name__ == '__main__':
    main()