from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from transformers import (
    PreTrainedTokenizerFast,
    T5TokenizerFast,
)

# Import optimized collator
from .components.optimized_t5_collator import DataCollatorForT5MLM

from .components.t5_dataset import T5ParquetDataset, T5MaterializedWindowDataset


class T5DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        tokenizer_name_or_path: str = "t5-base",  # Use pretrained T5 tokenizer 
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_val_split: Tuple[float, float] = (0.95, 0.05),
        max_length: int = 512,
        corruption_rate: float = 0.15,
        mean_span_length: int = 3,
        shuffle_buffer_size: int = 10_000,
        use_materialized_windows: bool = True,  # Use materialized windows by default 
        limit_files: int = -1,
        limit_documents: int = -1,  # Limit total number of windows/documents
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_dir = Path(data_dir)
        self.tokenizer_name_or_path = tokenizer_name_or_path

        self.train_dataset = None
        self.val_dataset = None


    # Tokenizer utilities old 
    def prepare_data(self): 
        # skip 
        pass
            
            
    def _load_tokenizer(self) -> PreTrainedTokenizerFast:
        tokenizer = T5TokenizerFast.from_pretrained(self.tokenizer_name_or_path)
        tokenizer.model_max_length = self.hparams.max_length
        return tokenizer
    
    # Setup datasets
    def setup(self, stage: str | None = None): 
        if self.train_dataset is not None:
            return  # Already initialised

        parquet_files = list(self.data_dir.rglob("*.parquet"))
        if not parquet_files:
            raise RuntimeError(f"No Parquet files found in {self.data_dir!s}")

        # Apply file limit if specified
        if self.hparams.limit_files > 0:
            parquet_files = parquet_files[:self.hparams.limit_files]

        # Load tokenizer first (needed for dataset)
        self.tokenizer = self._load_tokenizer()

        from src.utils.pylogger import RankedLogger
        log = RankedLogger(__name__, rank_zero_only=True)
        
        log.info(f"Using materialized windows from: {self.data_dir}")
        
        full_dataset = T5MaterializedWindowDataset(
            parquet_files=parquet_files,
            limit_documents=self.hparams.limit_documents
        )

        # Compute split lengths
        train_ratio, val_ratio = self.hparams.train_val_split
        train_len = int(len(full_dataset) * train_ratio)
        val_len = len(full_dataset) - train_len
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
        )

        # Create optimized collator with all required parameters
        self.collator = DataCollatorForT5MLM(
            tokenizer=self.tokenizer,
            noise_density=self.hparams.corruption_rate,
            mean_noise_span_length=self.hparams.mean_span_length,
            input_length=self.hparams.max_length,
            target_length=self.hparams.max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            decoder_start_token_id=self.tokenizer.pad_token_id,  # T5 uses pad_token_id for decoder start
        )


    # Dataloaders
    def train_dataloader(self): 
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.collator,
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=4,        # Prefetch 4 batches per worker
            drop_last=True,           # More consistent batch sizes
        )

    def val_dataloader(self): 
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collator,
            persistent_workers=True,  # Keep workers alive 
            prefetch_factor=4,        # Prefetch 4 batches per worker
            drop_last=False,          # Keep all validation data
        ) 