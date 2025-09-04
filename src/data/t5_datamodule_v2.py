"""
T5 DataModule with backwards compatibility support.
Can use either legacy (1.5x heuristic) or optimized (exact T5 formula) collator.
"""

from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from lightning import LightningDataModule
from transformers import AutoTokenizer
from src.data.components.optimized_t5_collator_v2 import (
    DataCollatorForT5MLM, 
    compute_t5_input_and_target_lengths
)
from src.data.components.t5_dataset import (
    T5MaterializedWindowDataset, 
    T5ProcessedDatasetMemoryMapped,
    DeterministicGlobalSampler
)
import os
import logging

log = logging.getLogger(__name__)


class T5DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        tokenizer_name_or_path: str = "t5-base",
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 4,
        train_val_split: Tuple[float, float] = (0.95, 0.05),
        max_length: int = 512,
        corruption_rate: float = 0.15,
        mean_span_length: int = 3,
        shuffle_buffer_size: int = 10_000,
        use_materialized_windows: bool = True,
        limit_files: int = -1,
        limit_documents: int = -1,
        seed: int = 42,
        use_legacy_collator: bool = True,  # Default to legacy (1.5x) for backwards compatibility
        use_optimized_collator: bool = False,  # Explicitly opt-in to T5 formula mode
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            use_fast=True,
            model_max_length=max_length,
        )
        self.tokenizer.model_max_length = max_length

        self.sampler = None
        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.collator = None
        self.resume_global_start = 0

    def setup(self, stage: Optional[str] = None):
        """Load and split dataset; create optimized collator."""
        if self.train_dataset is not None:
            return  # Already setup
        
        log.info(f"[rank: {os.environ.get('LOCAL_RANK', 0)}] Using materialized windows from: {self.hparams.data_dir}")
        
        full_dataset = T5MaterializedWindowDataset(
            data_dir=self.hparams.data_dir,
            tokenizer=self.tokenizer,
            max_length=self.hparams.max_length,
            shuffle_on_load=False,
            limit_files=self.hparams.limit_files,
            limit_documents=self.hparams.limit_documents,
            seed=self.hparams.seed,
        )
        
        self.full_dataset = full_dataset

        # Compute split lengths
        train_ratio, val_ratio = self.hparams.train_val_split
        train_len = int(len(full_dataset) * train_ratio)
        val_len = len(full_dataset) - train_len
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
        )

        # Create collator based on mode (check optimized flag first)
        if self.hparams.use_optimized_collator:
            # Optimized mode: calculate correct target_length
            log.info("[Optimized Mode] Using exact T5 formula for collator")
            _, target_length = compute_t5_input_and_target_lengths(
                self.hparams.max_length,
                self.hparams.corruption_rate,
                self.hparams.mean_span_length
            )
            log.info(f"Calculated target_length: {target_length} (for input_length={self.hparams.max_length})")
            use_legacy = False
        else:
            # Legacy mode (default): use 1.5x heuristic with target_length=512
            log.info("[Legacy Mode] Using 1.5x heuristic collator (default for backwards compatibility)")
            target_length = self.hparams.max_length  # 512
            use_legacy = True
        
        self.collator = DataCollatorForT5MLM(
            tokenizer=self.tokenizer,
            noise_density=self.hparams.corruption_rate,
            mean_noise_span_length=self.hparams.mean_span_length,
            input_length=self.hparams.max_length,
            target_length=target_length,
            pad_token_id=self.tokenizer.pad_token_id,
            decoder_start_token_id=self.tokenizer.pad_token_id,
            use_legacy_heuristic=use_legacy,  # Pass the computed flag
        )

    def set_resume_global_start(self, pos: int):
        """Set the global start position for exact resume (applied in train_dataloader)."""
        try:
            self.resume_global_start = int(pos)
        except Exception:
            self.resume_global_start = pos


    # Dataloaders
    def train_dataloader(self):
        # We use DeterministicGlobalSampler for training
        sampler = DeterministicGlobalSampler(
            dataset_length=len(self.train_dataset),
            seed=42,
            resume_global_start=self.resume_global_start,
        )
        self.sampler = sampler
        
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            sampler=sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
            persistent_workers=self.hparams.persistent_workers if self.hparams.num_workers > 0 else False,
        )
        
        return dataloader


    def val_dataloader(self):
        # Use deterministic sampler for validation
        sampler = DeterministicGlobalSampler(
            dataset_length=len(self.val_dataset),
            seed=99,  # Different seed for validation
            shuffle=False,
        )
        
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            sampler=sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
            persistent_workers=self.hparams.persistent_workers if self.hparams.num_workers > 0 else False,
        )
        
        return dataloader
            
    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Apply label masking after batch is on correct device."""
        # Move label padding to -100 if not already done
        if "labels" in batch:
            # Make sure padding tokens are set to -100 for loss computation
            batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100
        return batch

    def state_dict(self):
        """Save sampler state for resuming."""
        state = {}
        if self.sampler and hasattr(self.sampler, 'get_state'):
            state['sampler_state'] = self.sampler.get_state()
        return state

    def load_state_dict(self, state_dict):
        """Restore sampler state."""
        if 'sampler_state' in state_dict and self.sampler and hasattr(self.sampler, 'set_state'):
            self.sampler.set_state(state_dict['sampler_state'])