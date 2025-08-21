"""
German T5 DataModule for Cross-Lingual Transfer
Based on main T5DataModule but adapted for German dataset
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import T5TokenizerFast
import datasets

# Import base components (reuse from main project)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.data.components.optimized_t5_collator import DataCollatorForT5MLM
from src.data.components.deterministic_sampler import DeterministicGlobalSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GermanT5Dataset(Dataset):
    """German dataset for T5 pretraining"""
    
    def __init__(self, data_path: str, text_column: str = "text", id_column: str = "id"):
        """
        Args:
            data_path: Path to parquet file with German data
            text_column: Column name containing text
            id_column: Column name containing document IDs
        """
        self.data_path = Path(data_path)
        self.text_column = text_column
        self.id_column = id_column
        
        # Load dataset
        if self.data_path.suffix == '.parquet':
            self.df = pd.read_parquet(self.data_path)
        else:
            # Load HuggingFace dataset
            self.dataset = datasets.load_from_disk(str(self.data_path))
            self.df = self.dataset.to_pandas()
            
        logger.info(f"Loaded {len(self.df)} German documents from {self.data_path}")
        
        # Ensure required columns exist
        if self.text_column not in self.df.columns:
            raise ValueError(f"Text column '{self.text_column}' not found in dataset")
            
        if self.id_column not in self.df.columns:
            # Generate IDs if missing
            self.df[self.id_column] = range(len(self.df))
            logger.info(f"Generated {self.id_column} column with sequential IDs")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        return {
            "text": str(row[self.text_column]),
            "id": str(row[self.id_column])
        }


class GermanT5DataModule(LightningDataModule):
    """German T5 DataModule for cross-lingual transfer"""
    
    def __init__(
        self,
        data_dir: str,
        tokenizer_name_or_path: str = "google/mt5-base",  # German/multilingual tokenizer
        batch_size: int = 32,
        num_workers: int = 4, 
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 4,
        train_val_split: Tuple[float, float] = (0.99, 0.01),  # Small val for efficiency
        max_length: int = 512,
        corruption_rate: float = 0.15,  # T5 span corruption rate
        mean_span_length: int = 3,
        text_column: str = "text",
        id_column: str = "id",
        train_filename: str = "train.parquet",
        val_filename: str = "validation.parquet"
    ):
        super().__init__()
        
        # Store hyperparameters
        self.save_hyperparameters(logger=False)
        
        # Paths
        self.data_dir = Path(data_dir) 
        self.train_path = self.data_dir / train_filename
        self.val_path = self.data_dir / val_filename
        
        # Tokenizer and collator will be set in setup()
        self.tokenizer = None
        self.collator = None
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        
        logger.info(f"German T5 DataModule initialized")
        logger.info(f"Data dir: {self.data_dir}")
        logger.info(f"Tokenizer: {tokenizer_name_or_path}")
        logger.info(f"Batch size: {batch_size}, Workers: {num_workers}")
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets and tokenizer"""
        
        # Initialize tokenizer
        if self.tokenizer is None:
            logger.info(f"Loading tokenizer: {self.hparams.tokenizer_name_or_path}")
            self.tokenizer = T5TokenizerFast.from_pretrained(
                self.hparams.tokenizer_name_or_path
            )
            
            # Initialize collator with German tokenizer
            self.collator = DataCollatorForT5MLM(
                tokenizer=self.tokenizer,
                max_length=self.hparams.max_length,
                mlm_probability=self.hparams.corruption_rate,
                mean_span_length=self.hparams.mean_span_length,
                return_tensors="pt"
            )
            
            logger.info("German tokenizer and collator initialized")
        
        # Setup datasets
        if stage == "fit" or stage is None:
            
            # Check if processed files exist
            if self.train_path.exists() and self.val_path.exists():
                logger.info("Using preprocessed train/validation split")
                self.train_dataset = GermanT5Dataset(
                    self.train_path,
                    self.hparams.text_column,
                    self.hparams.id_column
                )
                self.val_dataset = GermanT5Dataset(
                    self.val_path,
                    self.hparams.text_column, 
                    self.hparams.id_column
                )
            else:
                logger.info("No preprocessed split found, creating from raw data")
                # Load full dataset and split
                raw_data_path = self.data_dir / "raw_parquet"
                if raw_data_path.exists():
                    # Load all chunks and combine
                    chunk_files = list(raw_data_path.glob("*.parquet"))
                    if chunk_files:
                        dfs = [pd.read_parquet(f) for f in chunk_files]
                        combined_df = pd.concat(dfs, ignore_index=True)
                        logger.info(f"Combined {len(combined_df)} documents from {len(chunk_files)} chunks")
                        
                        # Create train/val split
                        from sklearn.model_selection import train_test_split
                        train_df, val_df = train_test_split(
                            combined_df,
                            test_size=self.hparams.train_val_split[1],
                            random_state=42
                        )
                        
                        # Save splits
                        processed_dir = self.data_dir / "processed"
                        processed_dir.mkdir(exist_ok=True)
                        train_df.to_parquet(processed_dir / "train.parquet", index=False)
                        val_df.to_parquet(processed_dir / "validation.parquet", index=False)
                        
                        # Create datasets
                        self.train_dataset = GermanT5Dataset(
                            processed_dir / "train.parquet",
                            self.hparams.text_column,
                            self.hparams.id_column
                        )
                        self.val_dataset = GermanT5Dataset(
                            processed_dir / "validation.parquet",
                            self.hparams.text_column,
                            self.hparams.id_column
                        )
                    else:
                        raise FileNotFoundError(f"No parquet files found in {raw_data_path}")
                else:
                    raise FileNotFoundError(f"German data not found in {self.data_dir}")
            
            logger.info(f"German datasets ready: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}")
    
    def train_dataloader(self) -> DataLoader:
        """German training dataloader"""
        
        # Use deterministic sampler for reproducible training
        sampler = DeterministicGlobalSampler(
            dataset_length=len(self.train_dataset),
            seed=42,
            shuffle=True
        )
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            sampler=sampler,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collator,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None
        )
    
    def val_dataloader(self) -> DataLoader:
        """German validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collator,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None
        )


if __name__ == "__main__":
    # Test German DataModule
    data_dir = "/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/data/german"
    
    dm = GermanT5DataModule(
        data_dir=data_dir,
        batch_size=4,
        num_workers=2
    )
    
    dm.setup("fit")
    
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()
    
    print(f"Train batches: {len(train_dl)}")
    print(f"Val batches: {len(val_dl)}")
    
    # Test batch
    batch = next(iter(train_dl))
    print(f"Batch keys: {batch.keys()}")
    print(f"Input shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    
    print("German DataModule test successful!")