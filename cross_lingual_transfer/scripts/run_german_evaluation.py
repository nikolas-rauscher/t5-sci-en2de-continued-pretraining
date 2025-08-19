"""
German cross-lingual evaluation script
Evaluates German-adapted models on German benchmarks
"""

import logging
import torch
from pathlib import Path
import json
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Dict, Any
import sys
import os

# Add project root to path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GermanEvaluationPipeline:
    def __init__(self, 
                 results_dir: str = "./evaluation/results",
                 device: str = "auto"):
        """
        Initialize German evaluation pipeline
        
        Args:
            results_dir: Directory to save evaluation results
            device: Device for evaluation (auto, cpu, cuda)
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
    
    def load_model_and_tokenizer(self, model_path: str, model_name: str):
        """Load model and tokenizer for evaluation"""
        logger.info(f"Loading {model_name} from {model_path}")
        
        try:
            if model_path.startswith("google/"):
                # HuggingFace model
                model = T5ForConditionalGeneration.from_pretrained(model_path)
                tokenizer = T5Tokenizer.from_pretrained(model_path)
            elif model_path.endswith(".ckpt"):
                # Lightning checkpoint
                # Load base model architecture first
                model = T5ForConditionalGeneration.from_pretrained("google/mt5-base")
                
                # Load checkpoint weights
                checkpoint = torch.load(model_path, map_location="cpu")
                if 'state_dict' in checkpoint:
                    state_dict = {}
                    for key, value in checkpoint['state_dict'].items():
                        if key.startswith('model.'):
                            new_key = key[6:]  # Remove 'model.' prefix
                            state_dict[new_key] = value
                    model.load_state_dict(state_dict, strict=False)
                
                tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")
            else:
                # Directory path
                model = T5ForConditionalGeneration.from_pretrained(model_path)
                tokenizer_path = Path(model_path).parent / "tokenizer"
                if tokenizer_path.exists():
                    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
                else:
                    tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")
            
            model.to(self.device)
            model.eval()
            
            logger.info(f"Successfully loaded {model_name}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            return None, None
    
    def evaluate_german_mmlu_sample(self, model, tokenizer, question: str, choices: List[str], answer: str):
        """Evaluate a single German MMLU sample"""
        
        # Format question for T5
        prompt = f"Frage: {question}\\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}: {choice}\\n"
        prompt += "Antwort:"
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate answer
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode prediction
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = prediction.replace(prompt, "").strip()
        
        # Check if prediction matches answer (A, B, C, D)
        correct = prediction.upper().startswith(answer.upper())
        
        return {
            "question": question,
            "prediction": prediction,
            "answer": answer,
            "correct": correct
        }
    
    def create_mock_german_mmlu(self, num_samples: int = 50):
        """Create mock German MMLU samples for testing"""
        logger.info(f"Creating {num_samples} mock German MMLU samples...")
        
        mock_samples = []
        for i in range(num_samples):
            sample = {
                "question": f"Was ist die Hauptstadt von Deutschland? (Frage {i+1})",
                "choices": ["Paris", "Berlin", "Madrid", "Rom"],
                "answer": "B",
                "subject": "geography"
            }
            mock_samples.append(sample)
        
        return mock_samples
    
    def run_evaluation(self, model_configs: List[Dict[str, str]], tasks: List[str] = None):
        """Run evaluation on specified models and tasks"""
        
        if tasks is None:
            tasks = ["german_mmlu_mock"]  # Default to mock task
        
        results = {}
        
        for model_config in model_configs:
            model_name = model_config["name"]
            model_path = model_config["path"]
            
            logger.info(f"Evaluating {model_name}...")
            
            # Load model
            model, tokenizer = self.load_model_and_tokenizer(model_path, model_name)
            if model is None:
                continue
            
            model_results = {}
            
            for task in tasks:
                logger.info(f"Running {task} for {model_name}...")
                
                if task == "german_mmlu_mock":
                    # Run mock German MMLU evaluation
                    samples = self.create_mock_german_mmlu(50)
                    task_results = []
                    
                    for sample in samples:
                        result = self.evaluate_german_mmlu_sample(
                            model, tokenizer,
                            sample["question"],
                            sample["choices"], 
                            sample["answer"]
                        )
                        task_results.append(result)
                    
                    # Calculate accuracy
                    correct = sum(1 for r in task_results if r["correct"])
                    accuracy = correct / len(task_results)
                    
                    model_results[task] = {
                        "accuracy": accuracy,
                        "correct": correct,
                        "total": len(task_results),
                        "samples": task_results
                    }
                    
                    logger.info(f"{model_name} - {task}: {accuracy:.3f} ({correct}/{len(task_results)})")
            
            results[model_name] = model_results
            
            # Clean up memory
            del model, tokenizer
            torch.cuda.empty_cache()
        
        # Save results
        self.save_results(results)
        return results
    
    def save_results(self, results: Dict[str, Any]):
        """Save evaluation results to files"""
        logger.info("Saving evaluation results...")
        
        # Save detailed results as JSON
        with open(self.results_dir / "detailed_results.json", "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Create summary table
        summary_data = []
        for model_name, model_results in results.items():
            for task_name, task_results in model_results.items():
                summary_data.append({
                    "model": model_name,
                    "task": task_name,
                    "accuracy": task_results["accuracy"],
                    "correct": task_results["correct"],
                    "total": task_results["total"]
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.results_dir / "evaluation_summary.csv", index=False)
        
        logger.info(f"Results saved to {self.results_dir}")


def main():
    """Main evaluation execution"""
    
    # Setup evaluation pipeline  
    evaluator = GermanEvaluationPipeline(
        results_dir="/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/evaluation/results"
    )
    
    # Model configurations for evaluation
    model_configs = [
        {
            "name": "mt5_base_baseline", 
            "path": "google/mt5-base",
            "description": "Multilingual T5 baseline"
        },
        # Add more models here once they are trained
        # {
        #     "name": "german_transferred_5k",
        #     "path": "cross_lingual_transfer/logs/german_continued_pretraining/checkpoints/step-5000.ckpt"
        # }
    ]
    
    # Run evaluation
    results = evaluator.run_evaluation(model_configs, tasks=["german_mmlu_mock"])
    
    # Print summary
    print("\\nEvaluation Summary:")
    for model_name, model_results in results.items():
        for task_name, task_results in model_results.items():
            accuracy = task_results["accuracy"]
            print(f"{model_name} - {task_name}: {accuracy:.3f}")


if __name__ == "__main__":
    main()