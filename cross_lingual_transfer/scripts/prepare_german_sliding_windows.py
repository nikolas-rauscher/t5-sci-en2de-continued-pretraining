#!/usr/bin/env python3
"""
Prepare German text data with sliding windows for continued pretraining.
Based on English sliding window preparation but adapted for German data.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict
import pickle
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from transformers import AutoTokenizer
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GermanSlidingWindowPreparation:
    """Prepare German text data with sliding windows matching English pipeline."""
    
    def __init__(
        self,
        output_dir: str = "/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/data/german/sliding_windows",
        tokenizer_path: str = "/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/models/german_T5_Optimized_50Olap_clean_restart_487k/tokenizer",
        window_size: int = 512,
        overlap: float = 0.5,  # 50% overlap like English
        min_window_tokens: int = 100  # Minimum tokens for a window to be valid
    ):
        """
        Initialize German sliding window preparation.
        
        Args:
            output_dir: Directory to save processed windows
            tokenizer_path: Path to German tokenizer (from Wechsel transfer)
            window_size: Size of sliding window in tokens
            overlap: Overlap fraction between windows (0.5 = 50%)
            min_window_tokens: Minimum tokens required for valid window
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer_path = tokenizer_path
        self.window_size = window_size
        self.stride = int(window_size * (1 - overlap))  # Step size between windows
        self.min_window_tokens = min_window_tokens
        
        # Load tokenizer
        logger.info(f"Loading German tokenizer from: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info(f"Tokenizer loaded with vocab size: {len(self.tokenizer)}")
        
        self.documents = []
        self.metadata = {
            "window_size": window_size,
            "overlap": overlap,
            "stride": self.stride,
            "tokenizer": tokenizer_path,
            "total_windows": 0,
            "total_documents": 0,
            "min_window_tokens": min_window_tokens
        }
    
    def load_german_dataset(self, dataset_name: str = "scilons/texts_pq_3", split: str = "deu_Latn"):
        """
        Load German text dataset from HuggingFace.
        
        Args:
            dataset_name: HuggingFace dataset name
            split: Dataset split to load (deu_Latn for German)
        """
        logger.info(f"Loading German dataset: {dataset_name} split: {split}")
        
        # First check if we have local parquet files
        local_parquet_path = Path("/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/data/german/parquet")
        if local_parquet_path.exists():
            logger.info("Found local parquet files, loading from disk...")
            return self.load_from_parquet(local_parquet_path)
        
        try:
            # Set cache directory for HuggingFace
            import os
            os.environ['HF_HOME'] = '/netscratch/nrauscher/projects/BA-hydra/.hf_cache'
            
            # Try to load dataset from HuggingFace
            logger.info("Attempting to load from HuggingFace Hub...")
            
            # Use cache_dir explicitly
            cache_dir = "/netscratch/nrauscher/projects/BA-hydra/.hf_cache"
            dataset = load_dataset(
                dataset_name, 
                split=split, 
                streaming=False,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            
            # Convert to list of documents
            documents = []
            for example in tqdm(dataset, desc="Processing documents"):
                if 'text' in example:
                    text = example['text'].strip()
                    if text:  # Skip empty documents
                        documents.append(text)
            
            logger.info(f"Loaded {len(documents)} German documents")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load dataset from HuggingFace: {e}")
            logger.info("Attempting to load from local files...")
            # Fallback to local files if needed
            return self.load_local_german_data()
    
    def load_local_german_data(self):
        """Load German data from local files as fallback."""
        local_path = Path("/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/data/german/raw")
        
        if not local_path.exists():
            logger.error(f"Local data path does not exist: {local_path}")
            return []
        
        documents = []
        for file_path in local_path.glob("*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if text:
                    documents.append(text)
        
        logger.info(f"Loaded {len(documents)} documents from local files")
        return documents
    
    def create_sliding_windows(self, text: str) -> List[Dict]:
        """
        Create sliding windows from a single document.
        
        Args:
            text: Input text document
            
        Returns:
            List of window dictionaries with text and metadata
        """
        # Tokenize the full document
        tokens = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        
        if len(tokens) < self.min_window_tokens:
            return []  # Skip documents too short for meaningful windows
        
        windows = []
        position = 0
        
        while position < len(tokens):
            # Extract window
            window_tokens = tokens[position:position + self.window_size]
            
            # Only keep windows with minimum tokens
            if len(window_tokens) >= self.min_window_tokens:
                # Decode back to text
                window_text = self.tokenizer.decode(window_tokens, skip_special_tokens=True)
                
                windows.append({
                    'text': window_text,
                    'num_tokens': len(window_tokens),
                    'start_pos': position,
                    'end_pos': position + len(window_tokens)
                })
            
            # Move to next position
            position += self.stride
            
            # Break if remaining tokens less than minimum
            if position + self.min_window_tokens > len(tokens):
                break
        
        return windows
    
    def process_documents(self, documents: List[str]):
        """
        Process all documents into sliding windows.
        
        Args:
            documents: List of text documents
        """
        logger.info(f"Processing {len(documents)} documents into sliding windows...")
        
        all_windows = []
        
        for doc_idx, doc in enumerate(tqdm(documents, desc="Creating windows")):
            windows = self.create_sliding_windows(doc)
            
            # Add document index to each window
            for window in windows:
                window['doc_id'] = doc_idx
                all_windows.append(window)
        
        self.metadata['total_windows'] = len(all_windows)
        self.metadata['total_documents'] = len(documents)
        
        logger.info(f"Created {len(all_windows)} sliding windows from {len(documents)} documents")
        logger.info(f"Average windows per document: {len(all_windows) / len(documents):.2f}")
        
        return all_windows
    
    def save_windows(self, windows: List[Dict], split: str = "train"):
        """
        Save sliding windows to disk in format compatible with T5MaterializedWindowDataModule.
        
        Args:
            windows: List of window dictionaries
            split: Data split name (train/val)
        """
        split_dir = self.output_dir / split
        split_dir.mkdir(exist_ok=True)
        
        # Save in batches like English pipeline
        batch_size = 10000  # Windows per file
        num_batches = (len(windows) + batch_size - 1) // batch_size
        
        logger.info(f"Saving {len(windows)} windows to {num_batches} files...")
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(windows))
            batch_windows = windows[start_idx:end_idx]
            
            # Save as pickle (compatible with English pipeline)
            batch_file = split_dir / f"windows_batch_{batch_idx:04d}.pkl"
            with open(batch_file, 'wb') as f:
                pickle.dump(batch_windows, f)
            
            logger.info(f"Saved batch {batch_idx + 1}/{num_batches}: {len(batch_windows)} windows")
        
        # Save metadata
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Metadata saved to: {metadata_file}")
    
    def prepare_train_val_split(self, windows: List[Dict], val_ratio: float = 0.000385):
        """
        Split windows into train and validation sets.
        
        Args:
            windows: List of all windows
            val_ratio: Validation set ratio (0.000385 = 0.0385% like English)
        """
        # Shuffle windows for random split
        import random
        random.seed(42)
        random.shuffle(windows)
        
        # Calculate split
        num_val = int(len(windows) * val_ratio)
        num_val = max(num_val, 100)  # Minimum 100 validation windows
        
        val_windows = windows[:num_val]
        train_windows = windows[num_val:]
        
        logger.info(f"Split: {len(train_windows)} train, {len(val_windows)} validation windows")
        logger.info(f"Validation ratio: {len(val_windows) / len(windows):.4%}")
        
        return train_windows, val_windows
    
    def run_preparation(self):
        """Run the complete German data preparation pipeline."""
        logger.info("Starting German sliding window data preparation...")
        logger.info(f"Configuration:")
        logger.info(f"  Window size: {self.window_size} tokens")
        logger.info(f"  Overlap: {self.overlap * 100}%")
        logger.info(f"  Stride: {self.stride} tokens")
        logger.info(f"  Min window tokens: {self.min_window_tokens}")
        
        # Load German documents
        documents = self.load_german_dataset()
        
        if not documents:
            logger.error("No documents loaded! Please check dataset configuration.")
            return
        
        # Create sliding windows
        all_windows = self.process_documents(documents)
        
        # Split into train/val
        train_windows, val_windows = self.prepare_train_val_split(all_windows)
        
        # Save windows
        self.save_windows(train_windows, "train")
        self.save_windows(val_windows, "val")
        
        # Update metadata with final stats
        self.metadata['train_windows'] = len(train_windows)
        self.metadata['val_windows'] = len(val_windows)
        
        # Save updated metadata
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info("=" * 50)
        logger.info("German sliding window preparation completed!")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Total windows: {len(all_windows)}")
        logger.info(f"Train windows: {len(train_windows)}")
        logger.info(f"Validation windows: {len(val_windows)}")
        logger.info("=" * 50)


def main():
    """Main execution."""
    
    # Configuration
    output_dir = "/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/data/german/sliding_windows"
    tokenizer_path = "/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/models/german_T5_Optimized_50Olap_clean_restart_487k/tokenizer"
    
    # Check if tokenizer exists (Wechsel transfer must be done first)
    if not Path(tokenizer_path).exists():
        logger.error(f"Tokenizer not found at: {tokenizer_path}")
        logger.error("Please run Wechsel transfer first to create German tokenizer!")
        sys.exit(1)
    
    # Run preparation
    preparation = GermanSlidingWindowPreparation(
        output_dir=output_dir,
        tokenizer_path=tokenizer_path,
        window_size=512,
        overlap=0.5,  # 50% overlap like English
        min_window_tokens=100
    )
    
    preparation.run_preparation()


if __name__ == "__main__":
    main()