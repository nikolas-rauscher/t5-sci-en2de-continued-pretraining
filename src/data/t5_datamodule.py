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

from typing import Dict, List
import torch
from torch.nn.utils.rnn import pad_sequence
from .components.span_masking import apply_span_corruption

class DataCollatorForT5MLM:  
    """Efficient T5 Data Collator for text-only windows with batch tokenization."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        noise_density: float = 0.15,
        mean_noise_span_length: int = 3,
        input_length: int = 512,
        target_length: int | None = None,
        **kwargs  # Ignore extra args from config
    ) -> None:
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.input_length = input_length
        self.target_length = target_length or input_length

    def _pad(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        return pad_sequence(
            sequences,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id or 0,
        )

    def __call__(self, batch):
        input_tensors: List[torch.Tensor] = []
        label_tensors: List[torch.Tensor] = []

        for item in batch:
            # Handle both text-only and legacy pre-tokenized
            if "tokens" in item and item["tokens"] and item.get("text") is None:
                # Legacy pre-tokenized sequence
                ids = item["tokens"][:self.input_length]  # Truncate if needed
            else:
                # Text-only window - tokenize efficiently
                text = item.get("text", "")
                if not text:
                    # Fallback for empty text
                    ids = [self.tokenizer.pad_token_id or 0] * 10
                else:
                    ids = self.tokenizer.encode(
                        text,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=self.input_length,
                    )

            # Apply span corruption (same as before)
            inp_ids, lbl_ids = apply_span_corruption(
                ids,
                self.tokenizer,
                corruption_rate=self.noise_density,
                mean_span_length=self.mean_noise_span_length,
            )

            input_tensors.append(torch.tensor(inp_ids, dtype=torch.long))
            label_tensors.append(torch.tensor(lbl_ids, dtype=torch.long))

        batch_inputs = self._pad(input_tensors)
        batch_labels = self._pad(label_tensors)
        attention_mask = (batch_inputs != (self.tokenizer.pad_token_id or 0)).long()

        return {
            "input_ids": batch_inputs,
            "attention_mask": attention_mask,
            "labels": batch_labels,
        }

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
        
        full_dataset = T5MaterializedWindowDataset(parquet_files=parquet_files)

        # Compute split lengths
        train_ratio, val_ratio = self.hparams.train_val_split
        train_len = int(len(full_dataset) * train_ratio)
        val_len = len(full_dataset) - train_len
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
        )

        # Create efficient text-only collator
        self.collator = DataCollatorForT5MLM(
            tokenizer=self.tokenizer,
            noise_density=self.hparams.corruption_rate,
            mean_noise_span_length=self.hparams.mean_span_length,
            input_length=self.hparams.max_length,
            target_length=self.hparams.max_length,
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
        )

    def val_dataloader(self): 
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collator,
        ) 