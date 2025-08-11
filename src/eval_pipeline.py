#!/usr/bin/env python3
"""
Universal Evaluation Pipeline with Hydra

A flexible, configurable pipeline for evaluating language models on various benchmarks.
Uses Hydra for configuration management and automatically extracts metadata from checkpoints.

Usage:
    python src/eval_pipeline.py experiment=three_models
    python src/eval_pipeline.py experiment=checkpoint_progression
    python src/eval_pipeline.py models=[t5-base,path/to/checkpoint.ckpt] benchmarks=[mmlu]
"""

import os
import json
import re
import logging
import subprocess
import time
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple

import hydra
import rootutils
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

# Setup root directory
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


def convert_dictconfig_to_dict(obj):
    """Recursively convert DictConfig objects to regular dicts for JSON serialization"""
    if isinstance(obj, DictConfig):
        return {k: convert_dictconfig_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        return {k: convert_dictconfig_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_dictconfig_to_dict(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_dictconfig_to_dict(item) for item in obj)
    elif hasattr(obj, '__dict__'):
        # Handle dataclasses and other objects with __dict__
        try:
            return convert_dictconfig_to_dict(obj.__dict__)
        except:
            return str(obj)  # Fallback to string representation
    else:
        return obj


class CheckpointAnalyzer:
    """Analyzes checkpoints to extract metadata automatically"""
    
    @staticmethod
    def extract_step_and_ppl(checkpoint_path: Path) -> Tuple[Optional[int], Optional[float]]:
        """Extract step number and validation perplexity from checkpoint filename/path"""
        
        filename = checkpoint_path.name
        
        # Pattern for best checkpoints: step-262500-val_ppl-1.402.ckpt
        best_pattern = r'step-(\d+)-val_ppl-([0-9.]+)\.ckpt'
        match = re.search(best_pattern, filename)
        
        if match:
            step = int(match.group(1))
            ppl = float(match.group(2))
            return step, ppl
        
        # Pattern for step checkpoints: step-step=375000.ckpt
        step_pattern = r'step-step=(\d+)\.ckpt'
        match = re.search(step_pattern, filename)
        
        if match:
            step = int(match.group(1))
            return step, None
            
        # Pattern for epoch checkpoints: epoch-1-step-10000.ckpt
        epoch_step_pattern = r'epoch-\d+-step-(\d+)\.ckpt'
        match = re.search(epoch_step_pattern, filename)
        
        if match:
            step = int(match.group(1))
            return step, None
        
        return None, None
    
    @staticmethod
    def extract_run_info(checkpoint_path: Path) -> Dict[str, Any]:
        """Extract training run information from checkpoint path"""
        
        metadata = {}
        
        # Extract step and validation perplexity
        step, val_ppl = CheckpointAnalyzer.extract_step_and_ppl(checkpoint_path)
        if step:
            metadata['training_steps'] = step
        if val_ppl:
            metadata['val_perplexity'] = val_ppl
            
        # Extract run date from path (format: 2025-08-01_20-57-52)
        path_str = str(checkpoint_path)
        date_pattern = r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})'
        match = re.search(date_pattern, path_str)
        if match:
            date_str = match.group(1)
            time_str = match.group(2)
            metadata['run_date'] = date_str
            metadata['run_time'] = time_str.replace('-', ':')
            
        # Detect training style from path
        if 'scifive' in path_str.lower():
            metadata['training_style'] = 'scifive'
            metadata['learning_rate'] = 0.001  # SciFive uses constant 0.001
        elif 'clean_restart' in path_str.lower():
            metadata['training_style'] = 'restart'
            
        # Extract run directory name
        run_dir_pattern = r'runs/([^/]+)'
        match = re.search(run_dir_pattern, path_str)
        if match:
            metadata['run_directory'] = match.group(1)
            
        return metadata
    
    @staticmethod
    def analyze_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
        """Comprehensive checkpoint analysis"""
        
        metadata = CheckpointAnalyzer.extract_run_info(checkpoint_path)
        
        # Try to load checkpoint for additional metadata
        try:
            if checkpoint_path.exists():
                # Load checkpoint header without full weights
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                
                # Extract hyperparameters if available
                if 'hyper_parameters' in checkpoint:
                    hp = checkpoint['hyper_parameters']
                    metadata['learning_rate'] = hp.get('lr', hp.get('learning_rate'))
                    metadata['batch_size'] = hp.get('batch_size')
                    metadata['model_name'] = hp.get('model_name')
                    
                # Extract training info
                if 'global_step' in checkpoint:
                    metadata['actual_global_step'] = checkpoint['global_step']
                if 'epoch' in checkpoint:
                    metadata['epoch'] = checkpoint['epoch']
                    
                # Extract Lightning version info
                if 'pytorch-lightning_version' in checkpoint:
                    metadata['lightning_version'] = checkpoint['pytorch-lightning_version']
                    
        except Exception as e:
            log.warning(f"Could not load checkpoint metadata from {checkpoint_path}: {e}")
            
        return metadata


class ModelManager:
    """Handles model loading, conversion, and management with automatic metadata extraction"""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.project_root = Path(cfg.paths.root_dir)
        self.temp_dir = self.project_root / "evaluation" / "models" / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_model(self, model_config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Prepare model for evaluation and return path + enriched metadata"""
        
        source_type = model_config.get('source_type', 'auto')
        source_path = model_config['source_path']
        
        # Auto-detect source type if not specified
        if source_type == 'auto':
            source_type = self._detect_source_type(source_path)
            
        # Prepare metadata
        metadata = model_config.get('metadata', {})
        
        if source_type == 'huggingface':
            # Use HuggingFace model directly
            model_path = source_path
            metadata.update({
                'source_type': 'huggingface',
                'original_source': source_path
            })
            
        elif source_type == 'converted_path':
            # Use existing converted model
            path = Path(source_path)
            if not path.is_absolute():
                path = self.project_root / path
            if not path.exists():
                raise FileNotFoundError(f"Converted model not found: {path}")
            
            model_path = str(path)
            metadata.update({
                'source_type': 'converted_path',
                'original_source': source_path
            })
            
        elif source_type == 'checkpoint':
            # Convert checkpoint and extract metadata
            ckpt_path = Path(source_path)
            if not ckpt_path.is_absolute():
                ckpt_path = self.project_root / ckpt_path
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
                
            # Extract metadata from checkpoint
            checkpoint_metadata = CheckpointAnalyzer.analyze_checkpoint(ckpt_path)
            metadata.update(checkpoint_metadata)
            metadata.update({
                'source_type': 'checkpoint',
                'original_checkpoint': str(ckpt_path)
            })
            
            # Generate model name based on metadata
            model_name = self._generate_model_name(model_config, checkpoint_metadata)
            
            # Convert checkpoint
            output_path = self.temp_dir / model_name
            model_path = self._convert_checkpoint(ckpt_path, output_path, metadata)
            
        else:
            raise ValueError(f"Unknown source_type: {source_type}")
            
        return model_path, metadata
    
    def _detect_source_type(self, source_path: str) -> str:
        """Auto-detect source type from path"""
        
        if source_path.endswith('.ckpt'):
            return 'checkpoint'
        elif '/' in source_path and not source_path.startswith('/'):
            # Looks like HuggingFace model name
            return 'huggingface'
        elif Path(source_path).exists() or '/' in source_path:
            # Existing path or absolute path
            return 'converted_path'
        else:
            # Default to HuggingFace
            return 'huggingface'
    
    def _generate_model_name(self, model_config: Dict, metadata: Dict) -> str:
        """Generate descriptive model name from source path structure"""
        
        # Use explicit name if provided
        if 'name' in model_config and model_config['name']:
            return model_config['name']
        
        source_path = model_config['source_path']
        name_parts = []
        
        # Handle HuggingFace models
        if source_path == 't5-base':
            return "t5-base"
        elif source_path.startswith('t5-') or source_path.startswith('google/'):
            return source_path.replace('/', '_')
        
        # Extract meaningful parts from path structure
        path_parts = Path(source_path).parts
        
        # Look for key identifiers in path
        for part in path_parts:
            part_lower = part.lower()
            
            # Training type identifiers
            if 'scifive' in part_lower:
                name_parts.append('scifive_style')
            elif 'clean_restart' in part_lower:
                name_parts.append('clean_restart')
            elif 'pretrained' in part_lower:
                name_parts.append('pretrained')
            
            # Model base identifiers  
            if 't5-base-pretrained' in part_lower:
                if not any('t5' in existing for existing in name_parts):
                    name_parts.append('t5_base_pretrained')
            elif 't5' in part_lower and len(part_lower) > 2:  # Avoid matching just "t5"
                if not any('t5' in existing for existing in name_parts):
                    name_parts.append('t5_base')
        
        # If no specific identifiers found, use generic path-based name
        if not name_parts:
            # Use parent directory name
            if len(path_parts) > 1:
                parent_dir = path_parts[-2] if path_parts[-1].endswith('.ckpt') else path_parts[-1]
                if parent_dir not in ['checkpoints', 'best', 'steps']:
                    name_parts.append(parent_dir.replace('-', '_'))
        
        # Add training steps for disambiguation
        if 'training_steps' in metadata:
            steps = metadata['training_steps']
            if steps >= 1000:
                name_parts.append(f"{steps//1000}k")
            else:
                name_parts.append(f"{steps}")
        
        # Add validation perplexity if available (for best checkpoints)
        if 'val_perplexity' in metadata:
            ppl = metadata['val_perplexity']
            # Format: 1.397 → ppl1397, 1.40 → ppl140
            ppl_str = f"ppl{ppl:.3f}".replace('.', '')
            name_parts.append(ppl_str)
        
        # Fallback to timestamp if nothing found
        if not name_parts:
            name_parts.append(f"model_{int(time.time())}")
        
        return "_".join(name_parts)
    
    def _convert_checkpoint(self, ckpt_path: Path, output_path: Path, metadata: Dict) -> str:
        """Convert Lightning checkpoint to HuggingFace format"""
        
        log.info(f"Converting checkpoint: {ckpt_path}")
        
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        # Extract model state dict
        state_dict = checkpoint['state_dict']
        
        # Remove 'model.' prefix from keys (Lightning adds this)
        clean_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                clean_key = key[6:]  # Remove 'model.' prefix
                clean_state_dict[clean_key] = value
            else:
                clean_state_dict[key] = value
                
        # Load base T5 config and tokenizer
        config = T5Config.from_pretrained('t5-base')
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        
        # Create model and load state dict
        model = T5ForConditionalGeneration(config)
        model.load_state_dict(clean_state_dict, strict=False)
        
        # Save in HuggingFace format
        output_path.mkdir(parents=True, exist_ok=True)
        
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        # Save comprehensive metadata
        conversion_metadata = {
            'conversion_time': datetime.now().isoformat(),
            'original_checkpoint': str(ckpt_path),
            'extracted_metadata': metadata,
            'pipeline_temp_model': True
        }
        
        with open(output_path / 'conversion_metadata.json', 'w') as f:
            json.dump(conversion_metadata, f, indent=2)
            
        log.info(f"Converted model saved to: {output_path}")
        return str(output_path)
    
    def cleanup_temp_models(self):
        """Remove temporary converted models"""
        if self.cfg.eval.get('cleanup_temp_models', True):
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                log.info(f"Cleaned up temporary models: {self.temp_dir}")


class BenchmarkRunner:
    """Handles benchmark execution"""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.project_root = Path(cfg.paths.root_dir)
        self.setup_environment()
        
    def setup_environment(self):
        """Setup environment variables for cached datasets"""
        data_dir = self.project_root / self.cfg.paths.data_dir
        mmlu_cache = data_dir / "mmlu_datasets"
        hf_cache = data_dir / "huggingface_cache"
        
        os.environ['HF_DATASETS_CACHE'] = str(mmlu_cache)
        os.environ['HF_HOME'] = str(hf_cache)
        os.environ['HF_HUB_CACHE'] = str(hf_cache / "hub")
        os.environ['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output
        
        log.info(f"Using MMLU datasets cache: {mmlu_cache}")
        log.info(f"Using HuggingFace cache: {hf_cache}")
    
    def run_benchmark(self, model_path: str, model_name: str, benchmark_config: Dict) -> Dict:
        """Run a benchmark evaluation on a model"""
        
        results = {}
        output_base = self.project_root / self.cfg.paths.output_dir / f"evaluation/results/universal/{self.cfg.experiment_name}"
        output_base.mkdir(parents=True, exist_ok=True)
        
        shots = benchmark_config.get('shots', [0, 5])
        
        for shot in shots:
            log.info(f"Running {benchmark_config['name']} {shot}-shot evaluation for {model_name}")
            
            # Create output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = output_base / f"{model_name}_{benchmark_config['name']}_{shot}shot_{timestamp}"
            
            # Build command
            cmd = self._build_evaluation_command(
                model_path, model_name, benchmark_config, shot, output_dir
            )
            
            # Run evaluation
            start_time = time.time()
            try:
                # Check if wandb_args is present - if so, use shell=True for proper quoting
                if "--wandb_args" in cmd:
                    # Find the wandb_args value and quote it properly
                    wandb_idx = cmd.index("--wandb_args")
                    wandb_value = cmd[wandb_idx + 1]
                    # Reconstruct command as string with quoted wandb_args
                    cmd_parts = cmd[:wandb_idx] + ["--wandb_args", f'"{wandb_value}"'] + cmd[wandb_idx + 2:]
                    cmd_string = " ".join(cmd_parts)
                    log.info(f"Executing (shell=True): {cmd_string}")
                    
                    result = subprocess.run(
                        cmd_string,
                        cwd=self.project_root,
                        capture_output=True,
                        text=True,
                        check=True,
                        shell=True,
                        env=os.environ.copy()
                    )
                    
                    # lm-eval handles W&B integration natively
                else:
                    # Use normal list-based execution for non-wandb commands
                    log.info(f"Executing: {' '.join(cmd)}")
                    
                    result = subprocess.run(
                        cmd,
                        cwd=self.project_root,
                        capture_output=True,
                        text=True,
                        check=True,
                        env=os.environ.copy()
                    )
                
                duration = time.time() - start_time
                
                # Parse results
                eval_results = self._parse_results(output_dir, benchmark_config)
                eval_results.update({
                    'duration_seconds': duration,
                    'duration_minutes': duration / 60,
                    'command': ' '.join(cmd),
                    'output_dir': str(output_dir),
                    'status': 'success'
                })
                
                results[f"{shot}_shot"] = eval_results
                
                log.info(f"Completed {shot}-shot in {duration/60:.1f} minutes")
                
            except subprocess.CalledProcessError as e:
                log.error(f"Evaluation failed: {e}")
                log.error(f"Command: {' '.join(cmd)}")
                log.error(f"STDOUT: {e.stdout}")
                log.error(f"STDERR: {e.stderr}")
                results[f"{shot}_shot"] = {
                    'status': 'failed',
                    'error': str(e),
                    'stdout': e.stdout,
                    'stderr': e.stderr,
                    'command': ' '.join(cmd)
                }
                
        return results
    
    def _build_evaluation_command(self, model_path: str, model_name: str, benchmark_config: Dict, 
                                shots: int, output_dir: Path) -> List[str]:
        """Build the evaluation command for lm-eval with native W&B integration"""
        
        cmd = [
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path}",
            "--tasks", benchmark_config['name'],
            "--device", benchmark_config.get('device', 'cuda'),
            "--batch_size", benchmark_config.get('batch_size', 'auto'),
            "--output_path", str(output_dir),
            "--num_fewshot", str(shots)
        ]
        
        # Add specific tasks if configured
        if 'tasks' in benchmark_config:
            cmd[cmd.index("--tasks") + 1] = ",".join(benchmark_config['tasks'])
        
        # ✅ NATIVE W&B INTEGRATION - Syntax validated and working!
        if self.cfg.logger.get('wandb', {}).get('project'):
            # Use the actual model name from config for W&B run naming
            model_name_for_wandb = model_name
            
            # Build W&B args using the validated comma-separated syntax
            wandb_args_parts = [
                f"project={self.cfg.logger.wandb.project}",
            ]
            
            # Add entity if configured
            if self.cfg.logger.wandb.get('entity'):
                wandb_args_parts.append(f"entity={self.cfg.logger.wandb.entity}")
                
            # Add group if configured
            if self.cfg.logger.wandb.get('group'):
                wandb_args_parts.append(f"group={self.cfg.logger.wandb.group}")
                
            # Add descriptive run name with model and benchmark info
            run_name = f"{model_name_for_wandb}_{benchmark_config['name']}_{shots}shot"
            wandb_args_parts.append(f"name={run_name}")
            
            # Skip tags in wandb_args - they cause parsing issues
            # We'll add better W&B organization after the evaluation completes
            
            # Combine into single comma-separated string
            wandb_args_string = ",".join(wandb_args_parts)
            
            # Add wandb_args without quotes (subprocess handles arguments correctly)
            cmd.extend(["--wandb_args", wandb_args_string])
            
            log.info(f"🚀 Using native W&B integration: {wandb_args_string}")
        else:
            log.info("📊 W&B logging disabled (no project configured)")
            
        # Add extra arguments
        extra_args = benchmark_config.get('extra_args', {})
        for key, value in extra_args.items():
            cmd.extend([f"--{key}", str(value)])
            
        return cmd
    
    def _parse_results(self, output_dir: Path, benchmark_config: Dict) -> Dict:
        """Parse evaluation results from output directory"""
        
        # lm-eval creates subdirectory with model name, so search for results file
        results_file = None
        
        # First try direct path
        direct_results = output_dir / "results.json"
        if direct_results.exists():
            results_file = direct_results
        else:
            # Search in subdirectories for results_*.json files
            for subdir in output_dir.iterdir():
                if subdir.is_dir():
                    for json_file in subdir.glob("results_*.json"):
                        results_file = json_file
                        break
                    if results_file:
                        break
        
        if not results_file:
            return {'parse_status': 'failed', 'error': 'Results file not found'}
            
        try:
            with open(results_file) as f:
                data = json.load(f)
                
            # Extract results based on benchmark type
            if benchmark_config['name'] == 'mmlu':
                return self._parse_mmlu_results(data)
            else:
                # Generic parsing for other benchmarks
                return self._parse_generic_results(data, benchmark_config['name'])
                
        except Exception as e:
            return {'parse_status': 'failed', 'error': f'Failed to parse results: {e}'}
    
    def _parse_mmlu_results(self, data: Dict) -> Dict:
        """Parse MMLU-specific results (handles both full MMLU and subsets)"""
        try:
            results = data['results']
            
            # Check if we have the full MMLU aggregated results
            if 'mmlu' in results:
                # Full MMLU evaluation
                return {
                    'parse_status': 'success',
                    'overall_accuracy': results['mmlu']['acc,none'],
                    'overall_stderr': results['mmlu']['acc_stderr,none'],
                    'humanities': results.get('mmlu_humanities', {}).get('acc,none', 'N/A'),
                    'social_sciences': results.get('mmlu_social_sciences', {}).get('acc,none', 'N/A'),
                    'stem': results.get('mmlu_stem', {}).get('acc,none', 'N/A'),
                    'other': results.get('mmlu_other', {}).get('acc,none', 'N/A'),
                    'raw_results': data
                }
            else:
                # MMLU subset evaluation - calculate average across available tasks
                mmlu_tasks = {k: v for k, v in results.items() if k.startswith('mmlu_')}
                
                if not mmlu_tasks:
                    return {
                        'parse_status': 'failed',
                        'error': 'No MMLU tasks found in results',
                        'raw_results': data
                    }
                
                # Calculate average accuracy across subset tasks
                accuracies = []
                task_results = {}
                
                for task_name, task_result in mmlu_tasks.items():
                    if 'acc,none' in task_result:
                        accuracy = task_result['acc,none']
                        accuracies.append(accuracy)
                        task_results[task_name] = accuracy
                
                if not accuracies:
                    return {
                        'parse_status': 'failed',
                        'error': 'No accuracy metrics found in MMLU tasks',
                        'raw_results': data
                    }
                
                avg_accuracy = sum(accuracies) / len(accuracies)
                
                return {
                    'parse_status': 'success',
                    'overall_accuracy': avg_accuracy,
                    'overall_stderr': 'N/A',  # Can't calculate stderr for subset
                    'humanities': 'N/A',
                    'social_sciences': 'N/A', 
                    'stem': 'N/A',
                    'other': 'N/A',
                    'subset_tasks': task_results,
                    'num_tasks': len(accuracies),
                    'raw_results': data
                }
                
        except KeyError as e:
            return {
                'parse_status': 'failed', 
                'error': f'Missing key in MMLU results: {e}',
                'raw_results': data
            }
        except Exception as e:
            return {
                'parse_status': 'failed',
                'error': f'Error parsing MMLU results: {e}',
                'raw_results': data
            }
    
    def _parse_generic_results(self, data: Dict, benchmark_name: str) -> Dict:
        """Parse generic benchmark results"""
        try:
            results = data['results']
            
            # Find the main metric for this benchmark
            if benchmark_name in results:
                main_result = results[benchmark_name]
                # Common metric names
                for metric in ['acc,none', 'acc_norm,none', 'exact_match,none']:
                    if metric in main_result:
                        return {
                            'parse_status': 'success',
                            'primary_metric': metric,
                            'primary_score': main_result[metric],
                            'raw_results': data
                        }
            
            return {
                'parse_status': 'partial',
                'error': 'Could not find primary metric',
                'raw_results': data
            }
            
        except Exception as e:
            return {
                'parse_status': 'failed',
                'error': f'Failed to parse {benchmark_name} results: {e}',
                'raw_results': data
            }


class WandBLogger:
    """W&B logging coordinator - works with native lm-eval integration"""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.run = None
        self.native_integration_active = False
        
    def start_experiment(self):
        """Initialize W&B experiment with native lm-eval integration"""
        
        if not self.cfg.logger.get('wandb', {}).get('project'):
            log.info("W&B logging disabled (no project configured)")
            return
        
        log.info("🚀 Using native lm-eval W&B integration")
        self.native_integration_active = True
        
    def log_model_results(self, model_name: str, model_metadata: Dict, 
                         benchmark_name: str, results: Dict):
        """Log results - enhanced metadata and plots with native integration"""
        
        # If using native integration, only log additional metadata
        if self.native_integration_active:
            # Try to get the active run from lm-eval
            if not self.run:
                self.run = wandb.run
                
            if self.run and model_metadata:
                # Log additional metadata that lm-eval doesn't capture
                metadata_metrics = {}
                
                if 'training_steps' in model_metadata:
                    metadata_metrics[f"{model_name}/metadata/training_steps"] = model_metadata['training_steps']
                if 'val_perplexity' in model_metadata:
                    metadata_metrics[f"{model_name}/metadata/val_perplexity"] = model_metadata['val_perplexity']
                if 'source_type' in model_metadata:
                    metadata_metrics[f"{model_name}/metadata/source_type"] = model_metadata['source_type']
                if 'training_style' in model_metadata:
                    metadata_metrics[f"{model_name}/metadata/training_style"] = model_metadata['training_style']
                if 'run_date' in model_metadata:
                    metadata_metrics[f"{model_name}/metadata/run_date"] = model_metadata['run_date']
                if 'learning_rate' in model_metadata:
                    metadata_metrics[f"{model_name}/metadata/learning_rate"] = model_metadata['learning_rate']
                if 'run_directory' in model_metadata:
                    metadata_metrics[f"{model_name}/metadata/run_directory"] = model_metadata['run_directory']
                
                # Add performance enhancements and create plots
                for shot_key, shot_results in results.items():
                    if shot_results.get('status') == 'success' and 'duration_minutes' in shot_results:
                        shot_num = shot_key.replace('_shot', '')
                        metadata_metrics[f"{model_name}/{benchmark_name}/{shot_num}shot/duration_min"] = shot_results['duration_minutes']
                        
                        # Add improvement over random baseline for MMLU
                        if benchmark_name == 'mmlu' and 'overall_accuracy' in shot_results:
                            random_baseline = 0.25  # 25% for 4-choice questions
                            improvement = shot_results['overall_accuracy'] - random_baseline
                            metadata_metrics[f"{model_name}/{benchmark_name}/{shot_num}shot/improvement_over_random"] = improvement
                        
                        # 🎨 CREATE W&B PLOTS for MMLU subtasks/categories
                        if benchmark_name == 'mmlu':
                            self._create_mmlu_plots(model_name, shot_num, shot_results)
                
                if metadata_metrics:
                    log.info(f"📊 Logging {len(metadata_metrics)} enhanced metadata fields to W&B for {model_name}")
                    self.run.log(metadata_metrics)
                else:
                    log.info(f"ℹ️ No additional metadata to log for {model_name} (using native lm-eval integration)")
    
    def _create_mmlu_plots(self, model_name: str, shot_num: str, shot_results: Dict):
        """Create W&B plots for MMLU results"""
        
        if not self.run:
            return
            
        try:
            # 📊 MMLU Categories Bar Chart
            categories = ['humanities', 'social_sciences', 'stem', 'other']
            category_data = []
            category_labels = []
            
            for category in categories:
                value = shot_results.get(category)
                if value is not None and value != "N/A" and isinstance(value, (int, float)):
                    category_data.append(value)
                    category_labels.append(category.replace('_', ' ').title())
            
            if category_data:
                # Create bar chart data for W&B
                data = [[label, value] for label, value in zip(category_labels, category_data)]
                table = wandb.Table(data=data, columns=["Category", "Accuracy"])
                
                bar_chart = wandb.plot.bar(
                    table, "Category", "Accuracy",
                    title=f"{model_name} - MMLU Categories ({shot_num}-shot)"
                )
                
                self.run.log({f"{model_name}/plots/mmlu_categories_{shot_num}shot": bar_chart})
                log.info(f"📊 Created MMLU categories plot for {model_name} ({shot_num}-shot)")
            
            # 📊 MMLU Subtasks Bar Chart (if available)
            if 'subset_tasks' in shot_results:
                subtask_data = []
                for task_name, accuracy in shot_results['subset_tasks'].items():
                    clean_name = task_name.replace('mmlu_', '').replace('_', ' ').title()
                    subtask_data.append([clean_name, accuracy])
                
                if subtask_data:
                    # Sort by accuracy for better visualization
                    subtask_data.sort(key=lambda x: x[1], reverse=True)
                    
                    table = wandb.Table(data=subtask_data, columns=["Task", "Accuracy"])
                    
                    bar_chart = wandb.plot.bar(
                        table, "Task", "Accuracy", 
                        title=f"{model_name} - MMLU Subtasks ({shot_num}-shot)"
                    )
                    
                    self.run.log({f"{model_name}/plots/mmlu_subtasks_{shot_num}shot": bar_chart})
                    log.info(f"📊 Created MMLU subtasks plot for {model_name} ({shot_num}-shot) with {len(subtask_data)} tasks")
            
            # 📊 Performance vs Random Baseline
            if 'overall_accuracy' in shot_results:
                overall_acc = shot_results['overall_accuracy']
                random_baseline = 0.25
                
                comparison_data = [
                    ["Random Baseline", random_baseline],
                    [f"{model_name}", overall_acc],
                    ["Improvement", overall_acc - random_baseline]
                ]
                
                table = wandb.Table(data=comparison_data, columns=["Metric", "Value"])
                
                bar_chart = wandb.plot.bar(
                    table, "Metric", "Value",
                    title=f"{model_name} - Performance vs Baseline ({shot_num}-shot)"
                )
                
                self.run.log({f"{model_name}/plots/performance_vs_baseline_{shot_num}shot": bar_chart})
                log.info(f"📊 Created performance comparison plot for {model_name} ({shot_num}-shot)")
                
        except Exception as e:
            log.warning(f"Failed to create plots for {model_name}: {e}")
    
    def finish_experiment(self):
        """Finish W&B experiment - handled natively by lm-eval"""
        log.info("🚀 Native W&B integration - run lifecycle managed by lm-eval")


@task_wrapper
def evaluate(cfg: DictConfig) -> Dict[str, Any]:
    """Main evaluation function"""
    
    log.info(f"Starting Universal Evaluation Pipeline: {cfg.experiment_name}")
    log.info(f"Models: {len(cfg.models)}")
    log.info(f"Benchmarks: {len(cfg.benchmarks)}")
    
    # Initialize components
    model_manager = ModelManager(cfg)
    benchmark_runner = BenchmarkRunner(cfg)
    wandb_logger = WandBLogger(cfg)
    
    # Start W&B experiment
    wandb_logger.start_experiment()
    
    all_results = {}
    
    try:
        # Process each model
        for i, model_config in enumerate(cfg.models):
            # Generate model name if not provided
            if 'name' in model_config and model_config['name']:
                model_name = model_config['name']
            else:
                # Use auto-generation based on source path
                model_name = f"model_{i+1}"  # Temporary, will be updated after metadata extraction
            
            log.info(f"\n{'='*60}")
            log.info(f"Processing model: {model_name}")
            log.info(f"Source: {model_config['source_path']}")
            log.info(f"{'='*60}")
            
            try:
                # Prepare model (convert if necessary) and extract metadata
                model_path, metadata = model_manager.prepare_model(model_config)
                
                # Generate proper model name from metadata if not provided
                if 'name' not in model_config or not model_config['name']:
                    model_name = model_manager._generate_model_name(model_config, metadata)
                
                log.info(f"Model name: {model_name}")
                log.info(f"Model path: {model_path}")
                log.info(f"Extracted metadata: {metadata}")
                
                model_results = {
                    'model_config': model_config,
                    'model_path': model_path,
                    'metadata': metadata,
                    'benchmarks': {}
                }
                
                # Run each benchmark
                for benchmark_config in cfg.benchmarks:
                    log.info(f"\nRunning benchmark: {benchmark_config['name']}")
                    
                    benchmark_results = benchmark_runner.run_benchmark(
                        model_path, model_name, benchmark_config
                    )
                    
                    model_results['benchmarks'][benchmark_config['name']] = benchmark_results
                    
                    # Log enhanced metadata to complement native lm-eval W&B integration
                    wandb_logger.log_model_results(
                        model_name, metadata, benchmark_config['name'], benchmark_results
                    )
                
                all_results[model_name] = model_results
                
            except Exception as e:
                log.error(f"Failed to process model {model_name}: {e}")
                all_results[model_name] = {
                    'model_config': model_config,
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Save results
        save_results(cfg, all_results)
        
        # Save compact summary
        save_compact_summary(cfg, all_results)
        
        # Print summary
        print_summary(cfg, all_results)
        
    finally:
        # Cleanup
        model_manager.cleanup_temp_models()
        wandb_logger.finish_experiment()
        
    log.info("Pipeline completed!")
    
    # Return results in format expected by task_wrapper
    return {}, all_results


def save_results(cfg: DictConfig, results: Dict):
    """Save results to JSON file"""
    
    output_dir = Path(cfg.paths.root_dir) / cfg.paths.output_dir / "evaluation/results/universal"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"{cfg.experiment_name}_{timestamp}.json"
    
    # Add metadata
    results_with_metadata = {
        'experiment_metadata': {
            'name': cfg.experiment_name,
            'description': cfg.get('description', ''),
            'timestamp': timestamp,
            'config': OmegaConf.to_container(cfg, resolve=True)
        },
        'results': results
    }
    
    with open(results_file, 'w') as f:
        json.dump(convert_dictconfig_to_dict(results_with_metadata), f, indent=2)
        
    log.info(f"Results saved to: {results_file}")


def export_results_to_csv(summary_data: Dict, csv_file: Path):
    """Export evaluation results to CSV format for easy analysis"""
    
    try:
        rows = []
        
        # Extract experiment info
        exp_info = summary_data.get('experiment_info', {})
        experiment_name = exp_info.get('name', 'unknown')
        timestamp = exp_info.get('timestamp', 'unknown')
        
        # Process each model's results
        for model_name, model_data in summary_data.get('models', {}).items():
            if model_data.get('status') != 'success':
                continue
                
            metadata = model_data.get('metadata', {})
            benchmarks = model_data.get('benchmarks', {})
            
            # Create base row with model info
            base_row = {
                'experiment_name': experiment_name,
                'timestamp': timestamp,
                'model_name': model_name,
                'source_path': model_data.get('source_path', ''),
                'source_type': metadata.get('source_type', ''),
                'training_steps': metadata.get('training_steps', ''),
                'val_perplexity': metadata.get('val_perplexity', ''),
                'learning_rate': metadata.get('learning_rate', ''),
                'run_date': metadata.get('run_date', ''),
                'training_style': metadata.get('training_style', '')
            }
            
            # Add benchmark results
            for benchmark_name, benchmark_data in benchmarks.items():
                for shot_type, shot_data in benchmark_data.items():
                    if isinstance(shot_data, dict) and 'overall_accuracy' in shot_data:
                        row = base_row.copy()
                        row.update({
                            'benchmark': benchmark_name,
                            'shot_type': shot_type,
                            'overall_accuracy': shot_data['overall_accuracy'],
                            'status': shot_data.get('status', 'success')
                        })
                        
                        # Add category scores
                        categories = shot_data.get('categories', {})
                        for cat_name, cat_score in categories.items():
                            row[f'{cat_name}_accuracy'] = cat_score
                            
                        rows.append(row)
        
        if not rows:
            log.warning("No successful results to export to CSV")
            return
            
        # Write CSV
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if rows:
                fieldnames = rows[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
                
        log.info(f"📄 CSV results exported to: {csv_file}")
        
    except Exception as e:
        log.error(f"Failed to export CSV: {e}")


def save_compact_summary(cfg: DictConfig, results: Dict):
    """Save compact evaluation summary with key metrics and metadata"""
    
    output_dir = Path(cfg.paths.root_dir) / cfg.paths.output_dir / "evaluation/results/universal"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_dir / f"{cfg.experiment_name}_summary_{timestamp}.json"
    
    # Build compact summary
    compact_summary = {
        'experiment_info': {
            'name': cfg.experiment_name,
            'description': cfg.get('description', ''),
            'timestamp': timestamp,
            'evaluation_date': datetime.now().isoformat(),
            'total_models': len(results),
            'benchmarks': [b['name'] for b in cfg.benchmarks],
            'shots_evaluated': [b.get('shots', [0, 5]) for b in cfg.benchmarks]
        },
        'models': {},
        'benchmark_rankings': {},
        'performance_summary': {}
    }
    
    # Extract compact model info and results
    all_benchmark_scores = {}  # For rankings
    
    for model_name, model_results in results.items():
        if model_results.get('status') == 'failed':
            compact_summary['models'][model_name] = {
                'status': 'failed',
                'error': model_results.get('error', 'Unknown error'),
                'metadata': model_results.get('metadata', {})
            }
            continue
            
        # Model metadata
        metadata = model_results.get('metadata', {})
        model_info = {
            'status': 'success',
            'source_path': model_results.get('model_config', {}).get('source_path', 'unknown'),
            'metadata': {
                'training_steps': metadata.get('training_steps'),
                'val_perplexity': metadata.get('val_perplexity'),
                'training_style': metadata.get('training_style'),
                'learning_rate': metadata.get('learning_rate'),
                'run_date': metadata.get('run_date'),
                'source_type': metadata.get('source_type')
            },
            'benchmarks': {}
        }
        
        # Extract benchmark results
        if 'benchmarks' in model_results:
            for benchmark_name, benchmark_results in model_results['benchmarks'].items():
                model_info['benchmarks'][benchmark_name] = {}
                
                for shot_key, shot_results in benchmark_results.items():
                    if shot_results.get('status') == 'success':
                        shot_num = shot_key.replace('_shot', '')
                        
                        # Core metrics
                        shot_summary = {
                            'overall_accuracy': shot_results.get('overall_accuracy'),
                            'status': 'success'
                        }
                        
                        # MMLU specific metrics (clean, no redundant raw data)
                        if benchmark_name == 'mmlu':
                            # Only include non-null category scores
                            categories = {}
                            for cat in ['humanities', 'social_sciences', 'stem', 'other']:
                                value = shot_results.get(cat)
                                if value is not None and value != "N/A":
                                    categories[cat] = value
                            
                            if categories:
                                shot_summary['categories'] = categories
                            
                            # Include subset tasks summary if available
                            if shot_results.get('subset_tasks'):
                                shot_summary['subset_tasks'] = shot_results['subset_tasks']
                                shot_summary['num_tasks'] = shot_results.get('num_tasks')
                        
                        # Add to rankings data
                        ranking_key = f"{benchmark_name}_{shot_num}shot"
                        if ranking_key not in all_benchmark_scores:
                            all_benchmark_scores[ranking_key] = {}
                        
                        score = shot_results.get('overall_accuracy') or shot_results.get('primary_score')
                        if score is not None:
                            all_benchmark_scores[ranking_key][model_name] = score
                        
                        model_info['benchmarks'][benchmark_name][shot_key] = shot_summary
                    else:
                        model_info['benchmarks'][benchmark_name][shot_key] = {
                            'status': 'failed',
                            'error': shot_results.get('error', 'Unknown error')
                        }
        
        compact_summary['models'][model_name] = model_info
    
    # Generate rankings
    for benchmark_key, scores in all_benchmark_scores.items():
        if len(scores) > 1:
            sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            compact_summary['benchmark_rankings'][benchmark_key] = [
                {
                    'rank': rank,
                    'model': model_name,
                    'score': score,
                    'improvement_over_last': score - sorted_models[rank][1] if rank > 0 else 0.0
                }
                for rank, (model_name, score) in enumerate(sorted_models)
            ]
    
    # Performance summary statistics
    if all_benchmark_scores:
        for benchmark_key, scores in all_benchmark_scores.items():
            if scores:
                compact_summary['performance_summary'][benchmark_key] = {
                    'best_score': max(scores.values()),
                    'worst_score': min(scores.values()),
                    'average_score': sum(scores.values()) / len(scores),
                    'score_range': max(scores.values()) - min(scores.values()),
                    'models_count': len(scores)
                }
    
    # Save compact summary
    with open(summary_file, 'w') as f:
        json.dump(compact_summary, f, indent=2, default=str)
        
    log.info(f"📊 Compact summary saved to: {summary_file}")
    
    # Create CSV export for easy analysis
    csv_file = output_dir / f"{cfg.experiment_name}_results_{timestamp}.csv"
    export_results_to_csv(compact_summary, csv_file)
    
    return summary_file


def print_summary(cfg: DictConfig, results: Dict):
    """Enhanced evaluation summary with comparisons and task details"""
    
    log.info(f"\n{'='*80}")
    log.info("EVALUATION SUMMARY")
    log.info(f"{'='*80}")
    
    # Print table header
    header = f"{'Model':<30}"
    for benchmark_config in cfg.benchmarks:
        shots = benchmark_config.get('shots', [0, 5])
        for shot in shots:
            header += f" {benchmark_config['name']}_{shot}shot"[:12].ljust(12)
    header += " Status"
    
    log.info(header)
    log.info("-" * len(header))
    
    # Print results for each model
    model_scores = {}  # For ranking
    
    for model_name, model_results in results.items():
        if 'benchmarks' in model_results:
            row = f"{model_name:<30}"
            
            for benchmark_config in cfg.benchmarks:
                benchmark_name = benchmark_config['name']
                shots = benchmark_config.get('shots', [0, 5])
                
                if benchmark_name in model_results['benchmarks']:
                    benchmark_results = model_results['benchmarks'][benchmark_name]
                    
                    for shot in shots:
                        shot_key = f"{shot}_shot"
                        if shot_key in benchmark_results:
                            result = benchmark_results[shot_key]
                            if result.get('status') == 'success':
                                if 'overall_accuracy' in result:
                                    score = result['overall_accuracy']
                                    row += f" {score:.3f}"
                                    
                                    # Store for ranking
                                    key = f"{benchmark_name}_{shot}shot"
                                    if key not in model_scores:
                                        model_scores[key] = {}
                                    model_scores[key][model_name] = score
                                    
                                elif 'primary_score' in result:
                                    score = result['primary_score']
                                    row += f" {score:.3f}"
                                    
                                    # Store for ranking
                                    key = f"{benchmark_name}_{shot}shot"
                                    if key not in model_scores:
                                        model_scores[key] = {}
                                    model_scores[key][model_name] = score
                                else:
                                    row += f" {'N/A':>11}"
                            else:
                                row += f" {'Failed':>11}"
                        else:
                            row += f" {'N/A':>11}"
                else:
                    for shot in shots:
                        row += f" {'N/A':>11}"
            
            status = "Success" if model_results.get('status') != 'failed' else "Failed"
            row += f" {status}"
            
            log.info(row)
        else:
            row = f"{model_name:<30}"
            # Fill with N/A for all benchmarks
            total_cols = sum(len(b.get('shots', [0, 5])) for b in cfg.benchmarks)
            row += " N/A" * (total_cols * 12) + " Failed"
            log.info(row)
    
    # ✨ NEW: Print rankings and comparisons
    log.info(f"\n{'='*80}")
    log.info("MODEL RANKINGS & COMPARISONS")
    log.info(f"{'='*80}")
    
    for benchmark_key, scores in model_scores.items():
        if len(scores) > 1:
            sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            log.info(f"\n📊 {benchmark_key.upper()}:")
            for rank, (model_name, score) in enumerate(sorted_models, 1):
                if rank == 1:
                    log.info(f"  🥇 #{rank}: {model_name:<25} {score:.3f}")
                elif rank == 2:
                    log.info(f"  🥈 #{rank}: {model_name:<25} {score:.3f} ({score-sorted_models[0][1]:+.3f})")
                elif rank == 3:
                    log.info(f"  🥉 #{rank}: {model_name:<25} {score:.3f} ({score-sorted_models[0][1]:+.3f})")
                else:
                    log.info(f"  📍 #{rank}: {model_name:<25} {score:.3f} ({score-sorted_models[0][1]:+.3f})")
    
    # ✨ DYNAMIC: Print task details and all available metrics
    log.info(f"\n{'='*80}")
    log.info("DETAILED METRICS")
    log.info(f"{'='*80}")
    
    for model_name, model_results in results.items():
        if 'benchmarks' in model_results and model_results.get('status') != 'failed':
            log.info(f"\n📋 {model_name.upper()}:")
            
            for benchmark_name, benchmark_results in model_results['benchmarks'].items():
                for shot_key, shot_results in benchmark_results.items():
                    if shot_results.get('status') == 'success':
                        shot_num = shot_key.replace('_shot', '')
                        
                        # Core metrics
                        if 'overall_accuracy' in shot_results:
                            log.info(f"  {benchmark_name} {shot_num}-shot:")
                            log.info(f"    🎯 Overall Accuracy:    {shot_results['overall_accuracy']:.3f}")
                            
                            # ✨ DYNAMIC: Show all available category scores
                            mmlu_categories = ['humanities', 'social_sciences', 'stem', 'other']
                            found_categories = []
                            
                            for category in mmlu_categories:
                                value = shot_results.get(category)
                                if value is not None and value != "N/A" and isinstance(value, (int, float)):
                                    found_categories.append((category, value))
                            
                            if found_categories:
                                log.info(f"    📊 Categories:")
                                for category, value in found_categories:
                                    log.info(f"      • {category.replace('_', ' ').title():<15} {value:.3f}")
                            
                            # Show subset task details if available
                            if 'subset_tasks' in shot_results:
                                log.info(f"    📝 Tasks (subset of {shot_results.get('num_tasks', 0)}):")
                                
                                for task_name, accuracy in shot_results['subset_tasks'].items():
                                    task_short = task_name.replace('mmlu_', '')
                                    log.info(f"      • {task_short:<20} {accuracy:.3f}")
                            
                            # ✨ DYNAMIC: Show any additional metrics that were found
                            extra_metrics = []
                            for key, value in shot_results.items():
                                if (isinstance(value, (int, float)) and 
                                    key not in ['overall_accuracy', 'overall_stderr', 'duration_seconds', 'duration_minutes', 'num_tasks'] and
                                    key not in mmlu_categories and
                                    key not in ['subset_tasks'] and
                                    not key.startswith('_') and
                                    value != "N/A"):
                                    extra_metrics.append((key, value))
                            
                            if extra_metrics:
                                log.info(f"    ➕ Additional Metrics:")
                                for key, value in extra_metrics:
                                    clean_key = key.replace('_', ' ').title()
                                    log.info(f"      • {clean_key:<20} {value:.3f}")
                            
                            # Performance info
                            duration = shot_results.get('duration_minutes', 0)
                            log.info(f"    ⏱️  Duration:            {duration:.1f} min")
                            
                        elif 'primary_score' in shot_results:
                            log.info(f"  {benchmark_name} {shot_num}-shot:")
                            log.info(f"    🎯 Primary Score:       {shot_results['primary_score']:.3f}")
                            
                            duration = shot_results.get('duration_minutes', 0)
                            log.info(f"    ⏱️  Duration:            {duration:.1f} min")
    
    log.info(f"\n{'='*80}")
    log.info("EVALUATION COMPLETED")
    log.info(f"{'='*80}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_pipeline")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point"""
    
    # Execute evaluation
    results = evaluate(cfg)
    
    # Results are now automatically added to W&B run summaries during evaluation
    # for easy comparison in W&B's run comparison views
    
    # Return metric for hyperparameter optimization
    return None


if __name__ == "__main__":
    main()