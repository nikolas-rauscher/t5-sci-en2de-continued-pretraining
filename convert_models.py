#!/usr/bin/env python3
"""
Robust Checkpoint to HuggingFace Conversion Pipeline

Features:
- Single checkpoint or batch processing
- Automatic metadata extraction and naming
- Organized output structure
- Validation and upload support
- Good practices built-in

Usage:
    source .venv_eval/bin/activate
    python convert_models.py --checkpoint step-640000.ckpt
    python convert_models.py --batch "logs/**/checkpoints/*.ckpt"
    python convert_models.py --checkpoint step-640000.ckpt --upload username/model-name
"""

import sys
import os
import json
import argparse
import glob
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config

# Add project paths
sys.path.insert(0, '/netscratch/nrauscher/projects/BA-hydra')

class CheckpointAnalyzer:
    """Extract metadata from Lightning checkpoints"""
    
    @staticmethod
    def identify_run_type(path_str: str) -> Dict[str, str]:
        """Identify which of the three main runs this checkpoint belongs to"""
        
        path_lower = path_str.lower()
        
        if 'pretraining_logs_lr_001_gradient_clip_1_with_inverse_sqrt_schedule' in path_lower:
            return {
                'run_name': 'green-run',
                'run_type': 'green',
                'base_model': 't5-base',
                'base_model_hf': 't5-base',
                'learning_rate': 0.001,
                'gradient_clip': 1.0,
                'scheduler': 'inverse_sqrt',
                'warmup_steps': 20000,
                'config_name': 't5_continued_pretraining_lr_001_gradient_clip_1_with_inverse_sqrt_schedule.yaml',
                'full_name_template': 'green-run-t5-base-sci-cp-en-{steps}k-steps-lr0001-clip1'
            }
        elif 'clean_restart_logs' in path_lower:
            return {
                'run_name': 'red-run', 
                'run_type': 'red',
                'base_model': 't5-base',
                'base_model_hf': 't5-base',
                'learning_rate': 0.0001,
                'gradient_clip': 0.5,
                'scheduler': 'inverse_sqrt',
                'warmup_steps': 15000,
                'config_name': 'clean_restart_4gpu_h100.yaml',
                'full_name_template': 'red-run-t5-base-sci-cp-en-{steps}k-steps-lr00001-clip0_5'
            }
        elif 'flan_t5_logs_lr_001_gradient_clip_1_with_inverse_sqrt_schedule' in path_lower:
            return {
                'run_name': 'yellow-run',
                'run_type': 'yellow', 
                'base_model': 'flan-t5-base',
                'base_model_hf': 'google/flan-t5-base',
                'learning_rate': 0.001,
                'gradient_clip': 1.0,
                'scheduler': 'inverse_sqrt',
                'warmup_steps': 20000,
                'config_name': 'flan_t5_lr_001_gradient_clip_1_with_inverse_sqrt_schedule.yaml',
                'full_name_template': 'yellow-run-flan-t5-base-sci-cp-en-{steps}k-steps-lr0001-clip1'
            }
        else:
            return {
                'run_name': 'unknown-run',
                'run_type': 'unknown',
                'base_model': 't5-base',
                'base_model_hf': 't5-base',
                'learning_rate': None,
                'gradient_clip': None,
                'scheduler': 'unknown',
                'warmup_steps': None,
                'config_name': 'unknown',
                'full_name_template': 't5-base-unknown-{steps}k-steps'
            }

    @staticmethod
    def analyze(checkpoint_path: Path) -> Dict[str, Any]:
        """Extract comprehensive metadata from checkpoint"""
        
        metadata = {
            'source_path': str(checkpoint_path),
            'file_size_mb': checkpoint_path.stat().st_size / (1024 * 1024),
            'analysis_time': datetime.now().isoformat()
        }
        
        try:
            # Load checkpoint metadata only (not full weights)
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Basic training info
            metadata.update({
                'global_step': checkpoint.get('global_step'),
                'epoch': checkpoint.get('epoch'),
                'lightning_version': checkpoint.get('pytorch-lightning_version')
            })
            
            # Extract step from filename if available
            filename = checkpoint_path.name
            if 'step=' in filename:
                import re
                match = re.search(r'step=(\d+)', filename)
                if match:
                    metadata['step_from_filename'] = int(match.group(1))
            
            # Hyperparameters
            if 'hyper_parameters' in checkpoint:
                hp = checkpoint['hyper_parameters']
                metadata.update({
                    'learning_rate': hp.get('lr', hp.get('learning_rate')),
                    'model_name': hp.get('model_name'),
                    'batch_size': hp.get('batch_size')
                })
            
            # Identify training run type FIRST
            path_str = str(checkpoint_path)
            run_info = CheckpointAnalyzer.identify_run_type(path_str)
            metadata['run_info'] = run_info
            metadata['base_model'] = run_info['base_model']
            metadata['training_type'] = 'continued_pretraining'
            
            # Override with run-specific info
            metadata['learning_rate'] = run_info['learning_rate']
            metadata['gradient_clip'] = run_info['gradient_clip']
            metadata['scheduler'] = run_info['scheduler']
            metadata['warmup_steps'] = run_info['warmup_steps']
            metadata['config_name'] = run_info['config_name']
            
            # Extract validation perplexity from filename if available
            if 'val_ppl' in filename:
                import re
                match = re.search(r'val_ppl-([0-9.]+)', filename)
                if match:
                    metadata['val_perplexity'] = float(match.group(1))
                
        except Exception as e:
            metadata['analysis_error'] = str(e)
            
        return metadata

class ModelConverter:
    """Convert Lightning checkpoints to HuggingFace format"""
    
    def __init__(self, output_base_dir: str = "converted_models"):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(exist_ok=True)
    

    def generate_model_name(self, metadata: Dict) -> str:
        """Generate descriptive model name from metadata"""
        
        run_info = metadata.get('run_info', {})
        
        # Use full name template if available
        if run_info and 'full_name_template' in run_info:
            step = metadata.get('global_step') or metadata.get('step_from_filename')
            if step:
                steps_k = step // 1000 if step >= 1000 else step
                base_name = run_info['full_name_template'].format(steps=steps_k)
                
                # Add validation perplexity if available (for best checkpoints)
                val_ppl = metadata.get('val_perplexity')
                if val_ppl:
                    ppl_str = f"ppl-{val_ppl:.3f}".replace('.', '')
                    base_name += f"-{ppl_str}"
                
                return base_name
        
        # Fallback to simple naming
        parts = []
        
        # Run identification
        if run_info:
            parts.append(run_info.get('run_name', 'unknown-run'))
        
        # Steps
        step = metadata.get('global_step') or metadata.get('step_from_filename')
        if step:
            if step >= 1000:
                parts.append(f"step-{step//1000}k")
            else:
                parts.append(f"step-{step}")
        
        # Validation perplexity if available
        val_ppl = metadata.get('val_perplexity')
        if val_ppl:
            ppl_str = f"ppl-{val_ppl:.3f}".replace('.', '')
            parts.append(ppl_str)
        
        return '-'.join(parts)
    
    def convert_checkpoint(self, checkpoint_path: Path, metadata: Dict) -> Path:
        """Convert single checkpoint to HuggingFace format"""
        
        # Generate structured output directory
        run_info = metadata.get('run_info', {})
        run_type = run_info.get('run_type', 'unknown')
        
        model_name = self.generate_model_name(metadata)
        
        # Create run-specific subdirectory
        run_dir = self.output_base_dir / f"{run_type}-run-models" 
        output_dir = run_dir / model_name
        
        print(f"Converting: {checkpoint_path.name}")
        print(f"Output: {output_dir}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['state_dict']
        
        # Clean Lightning prefixes
        clean_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.model.'):
                clean_key = key[len('model.model.'):]
            elif key.startswith('model.'):
                clean_key = key[len('model.'):]
            else:
                clean_key = key
            clean_state_dict[clean_key] = value
        
        # Load base model and config
        run_info = metadata.get('run_info', {})
        base_model = run_info.get('base_model_hf', 't5-base')
        config = T5Config.from_pretrained(base_model)
        model = T5ForConditionalGeneration(config)
        tokenizer = T5Tokenizer.from_pretrained(base_model)
        
        # Load weights with validation
        load_result = model.load_state_dict(clean_state_dict, strict=False)
        missing_keys = len(load_result.missing_keys) if load_result.missing_keys else 0
        unexpected_keys = len(load_result.unexpected_keys) if load_result.unexpected_keys else 0
        
        if missing_keys > 0 or unexpected_keys > 0:
            print(f"WARNING: Load issues: {missing_keys} missing, {unexpected_keys} unexpected keys")
        
        # Save model
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save metadata
        conversion_info = {
            **metadata,
            'conversion_time': datetime.now().isoformat(),
            'model_name': model_name,
            'output_path': str(output_dir),
            'base_model_used': base_model,
            'missing_keys': missing_keys,
            'unexpected_keys': unexpected_keys
        }
        
        with open(output_dir / 'conversion_info.json', 'w') as f:
            json.dump(conversion_info, f, indent=2)
        
        # Test loading
        try:
            test_model = T5ForConditionalGeneration.from_pretrained(output_dir)
            test_tokenizer = T5Tokenizer.from_pretrained(output_dir)
            print("SUCCESS: Model loads successfully")
        except Exception as e:
            print(f"ERROR: Model loading test failed: {e}")
        
        return output_dir
    
    def validate_model(self, model_path: Path) -> bool:
        """Quick validation of converted model"""
        try:
            model = T5ForConditionalGeneration.from_pretrained(model_path)
            tokenizer = T5Tokenizer.from_pretrained(model_path)
            
            # Test inference
            test_input = "Question: What is 2+2? Answer:"
            inputs = tokenizer(test_input, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=20)
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Test generation: '{result}'")
            return True
            
        except Exception as e:
            print(f"ERROR: Validation failed: {e}")
            return False

class ModelRepository:
    """Organize and manage converted models"""
    
    def __init__(self, base_dir: str = "converted_models"):
        self.base_dir = Path(base_dir)
    
    def list_models(self) -> List[Dict]:
        """List all converted models with metadata"""
        models = []
        
        # Search in both flat structure and nested run directories
        search_dirs = [self.base_dir]
        
        # Add run-specific directories if they exist
        for run_dir in self.base_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.endswith('-run-models'):
                search_dirs.append(run_dir)
        
        for search_dir in search_dirs:
            for model_dir in search_dir.iterdir():
                if model_dir.is_dir():
                    info_file = model_dir / 'conversion_info.json'
                    if info_file.exists():
                        with open(info_file) as f:
                            info = json.load(f)
                        models.append({
                            'name': model_dir.name,
                            'path': str(model_dir),
                            'size_gb': sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file()) / (1024**3),
                            **info
                        })
        
        return sorted(models, key=lambda x: x.get('global_step', 0), reverse=True)
    
    def print_summary(self, sort_by: str = 'steps'):
        """Print repository summary with sorting options"""
        models = self.list_models()
        
        print(f"\n{'='*80}")
        print("MODEL REPOSITORY SUMMARY")
        print(f"{'='*80}")
        print(f"Location: {self.base_dir}")
        print(f"Total models: {len(models)}")
        
        if models:
            total_size = sum(m['size_gb'] for m in models)
            print(f"Total size: {total_size:.2f} GB")
            
            # Group by run type
            runs = {}
            for model in models:
                run_info = model.get('run_info', {})
                run_type = run_info.get('run_type', 'unknown')
                if run_type not in runs:
                    runs[run_type] = []
                runs[run_type].append(model)
            
            # Sort within each run
            for run_type in runs:
                if sort_by == 'steps':
                    runs[run_type].sort(key=lambda x: x.get('global_step', 0), reverse=True)
                elif sort_by == 'ppl':
                    runs[run_type].sort(key=lambda x: x.get('val_perplexity', 999.0))
                elif sort_by == 'size':
                    runs[run_type].sort(key=lambda x: x.get('size_gb', 0), reverse=True)
            
            # Print by run type
            for run_type, run_models in runs.items():
                if run_models:
                    run_info = run_models[0].get('run_info', {})
                    base_model = run_info.get('base_model', 'unknown')
                    lr = run_info.get('learning_rate', 'unknown')
                    clip = run_info.get('gradient_clip', 'unknown')
                    
                    print(f"\n{run_type.upper()} RUN - {base_model} lr={lr} clip={clip}")
                    print("-" * 80)
                    print(f"{'Model Name':<50} {'Steps':<8} {'PPL':<8} {'Size':<8} {'Status'}")
                    print("-" * 80)
                    
                    for model in run_models:
                        step = model.get('global_step', 'N/A')
                        ppl = f"{model.get('val_perplexity', 0):.3f}" if model.get('val_perplexity') else 'N/A'
                        size = f"{model['size_gb']:.1f}GB"
                        status = "OK" if model.get('missing_keys', 0) == 0 else "WARN"
                        print(f"{model['name']:<50} {step:<8} {ppl:<8} {size:<8} {status}")
            
            print(f"\nSorted by: {sort_by}")
            print("Available sort options: --sort steps|ppl|size")

def upload_to_hub(model_path: Path, hub_model_id: str):
    """Upload model to HuggingFace Hub"""
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        
        print(f"Uploading {model_path.name} to {hub_model_id}")
        
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        
        model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)
        
        print(f"Successfully uploaded to: https://huggingface.co/{hub_model_id}")
        
    except Exception as e:
        print(f"ERROR: Upload failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert Lightning checkpoints to HuggingFace format")
    
    parser.add_argument('--checkpoint', type=str, help='Single checkpoint path')
    parser.add_argument('--batch', type=str, help='Glob pattern for batch processing')
    parser.add_argument('--output-dir', type=str, default='converted_models', help='Output directory')
    parser.add_argument('--upload', type=str, help='HuggingFace Hub model ID for upload')
    parser.add_argument('--list', action='store_true', help='List existing converted models')
    parser.add_argument('--sort', type=str, default='steps', choices=['steps', 'ppl', 'size'], help='Sort models by steps, validation perplexity, or size')
    parser.add_argument('--validate', type=str, help='Validate specific converted model')
    
    args = parser.parse_args()
    
    # Initialize components
    analyzer = CheckpointAnalyzer()
    converter = ModelConverter(args.output_dir)
    repository = ModelRepository(args.output_dir)
    
    if args.list:
        repository.print_summary(sort_by=args.sort)
        return
    
    if args.validate:
        model_path = Path(args.output_dir) / args.validate
        if model_path.exists():
            converter.validate_model(model_path)
        else:
            print(f"ERROR: Model not found: {model_path}")
        return
    
    # Collect checkpoints to process
    checkpoints = []
    
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            checkpoints.append(checkpoint_path)
        else:
            print(f"ERROR: Checkpoint not found: {checkpoint_path}")
            return
    
    if args.batch:
        batch_paths = glob.glob(args.batch, recursive=True)
        checkpoints.extend([Path(p) for p in batch_paths if Path(p).suffix == '.ckpt'])
        print(f"Found {len(checkpoints)} checkpoints for batch processing")
    
    if not checkpoints:
        print("ERROR: No checkpoints specified. Use --checkpoint or --batch")
        return
    
    # Process checkpoints
    converted_models = []
    
    for checkpoint_path in checkpoints:
        try:
            print(f"\n{'='*60}")
            print(f"Processing: {checkpoint_path.name}")
            print(f"{'='*60}")
            
            # Analyze checkpoint
            metadata = analyzer.analyze(checkpoint_path)
            print(f"Steps: {metadata.get('global_step', 'N/A')}")
            print(f"Model: {metadata.get('base_model', 'N/A')}")
            
            # Convert
            output_path = converter.convert_checkpoint(checkpoint_path, metadata)
            converted_models.append(output_path)
            
            # Upload if requested
            if args.upload:
                upload_to_hub(output_path, args.upload)
            
        except Exception as e:
            print(f"ERROR: Failed to process {checkpoint_path.name}: {e}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully converted: {len(converted_models)} models")
    
    if converted_models:
        repository.print_summary(sort_by=args.sort)

if __name__ == "__main__":
    main()