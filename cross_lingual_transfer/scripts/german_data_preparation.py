"""
German dataset preparation for continued pretraining
Downloads and preprocesses scilons/texts_pq_3 deu_Latn split
"""

import logging
from pathlib import Path
from datasets import load_dataset
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GermanDataPreparation:
    def __init__(self, output_dir: str = "./data/german"):
        """
        Initialize German data preparation
        
        Args:
            output_dir: Directory to save processed German data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_german_dataset(self):
        """Download German scientific dataset"""
        logger.info("Downloading German scientific dataset...")
        
        try:
            # Load German split from scilons dataset
            dataset = load_dataset(
                "scilons/texts_pq_3",
                name="deu_Latn", 
                split="train",
                streaming=False  # Download full dataset
            )
            
            logger.info(f"Downloaded {len(dataset)} German documents")
            return dataset
            
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            logger.info("You may need to authenticate with: huggingface-cli login")
            raise
    
    def analyze_dataset(self, dataset):
        """Analyze German dataset statistics"""
        logger.info("Analyzing German dataset...")
        
        # Convert to pandas for analysis
        df = dataset.to_pandas()
        
        # Basic statistics
        total_docs = len(df)
        
        if 'text' in df.columns:
            text_lengths = df['text'].str.len()
            avg_length = text_lengths.mean()
            total_chars = text_lengths.sum()
            
            stats = {
                "total_documents": total_docs,
                "average_document_length": avg_length,
                "total_characters": total_chars,
                "estimated_tokens": total_chars // 4,  # Rough estimate
                "columns": list(df.columns)
            }
        else:
            # Check what columns exist
            stats = {
                "total_documents": total_docs,
                "columns": list(df.columns),
                "sample_data": df.head().to_dict()
            }
        
        logger.info(f"Dataset statistics: {stats}")
        
        # Save statistics
        import json
        with open(self.output_dir / "dataset_statistics.json", "w", encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
            
        return stats
    
    def prepare_training_data(self, dataset, validation_split=0.01):
        """Prepare German data for continued pretraining"""
        logger.info("Preparing training data...")
        
        # Split into train/validation
        split_dataset = dataset.train_test_split(
            test_size=validation_split, 
            seed=42,
            shuffle=True
        )
        
        train_dataset = split_dataset['train']
        val_dataset = split_dataset['test']
        
        logger.info(f"Train: {len(train_dataset)} docs, Validation: {len(val_dataset)} docs")
        
        # Save datasets in parquet format for efficient loading
        train_dataset.to_parquet(self.output_dir / "train.parquet")
        val_dataset.to_parquet(self.output_dir / "validation.parquet")
        
        logger.info("German training data prepared successfully")
        
        return {
            "train_size": len(train_dataset),
            "validation_size": len(val_dataset)
        }
    
    def run_preparation(self):
        """Complete data preparation pipeline"""
        logger.info("Starting German data preparation...")
        
        # Download dataset
        dataset = self.download_german_dataset()
        
        # Analyze dataset
        stats = self.analyze_dataset(dataset)
        
        # Prepare training data
        split_info = self.prepare_training_data(dataset)
        
        logger.info("German data preparation completed!")
        return {**stats, **split_info}


def main():
    """Main data preparation execution"""
    
    output_dir = "/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/data/german"
    
    # Execute preparation
    prep = GermanDataPreparation(output_dir)
    result = prep.run_preparation()
    
    print(f"Preparation completed! Result: {result}")


if __name__ == "__main__":
    main()