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

try:
    from transformers import DataCollatorForT5MLM  
except ImportError:  # pragma: no cover
    try:
        from transformers.data.data_collator import DataCollatorForT5MLM  
    except ImportError:  # pragma: no 
        from typing import Dict, List

        import torch
        from torch.nn.utils.rnn import pad_sequence

        from .components.span_masking import apply_span_corruption  

        class DataCollatorForT5MLM:  
            """Simple re-implementation of DataCollatorForT5MLM."""

            def __init__(
                self,
                tokenizer: PreTrainedTokenizerFast,
                noise_density: float = 0.15,
                mean_noise_span_length: int = 3,
                input_length: int = 512,
                target_length: int | None = None,
            ) -> None:
                self.tokenizer = tokenizer
                self.noise_density = noise_density
                self.mean_noise_span_length = mean_noise_span_length
                self.input_length = input_length
                self.target_length = target_length or input_length

            # -------------------------------------------------------------
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
                    ids: List[int] = self.tokenizer.encode(
                        item["text"],
                        add_special_tokens=False,
                        truncation=True,
                        max_length=self.input_length,
                    )

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

                attention_mask = (batch_inputs != self.tokenizer.pad_token_id).long()

                return {
                    "input_ids": batch_inputs,
                    "attention_mask": attention_mask,
                    "labels": batch_labels,
                }

from .components.t5_dataset import T5ParquetDataset


class T5DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        tokenizer_path: str,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_val_split: Tuple[float, float] = (0.95, 0.05),
        max_length: int = 512,
        corruption_rate: float = 0.15,
        mean_span_length: int = 3,
        shuffle_buffer_size: int = 10_000,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_dir = Path(data_dir)
        self.tokenizer_path = Path(tokenizer_path)
        # Ensure the tokenizer directory exists
        self.tokenizer_path.mkdir(parents=True, exist_ok=True)

        self.train_dataset = None
        self.val_dataset = None


    # Tokenizer utilities
    def prepare_data(self): 
        if not (self.tokenizer_path / "sentencepiece.model").exists():
            self._train_sentencepiece_tokenizer()

    def _train_sentencepiece_tokenizer(self):
        """Train a SentencePiece tokenizer on the *text* column of the corpus."""
        from sentencepiece import SentencePieceTrainer

        parquet_files = list(self.data_dir.rglob("*.parquet"))
        if not parquet_files:
            raise RuntimeError(f"No Parquet files found in {self.data_dir!s}")

        # Extract raw text into a temporary file for sentencepiece training.
        tmp_txt = self.tokenizer_path / "train_data.txt"
        with tmp_txt.open("w", encoding="utf-8") as out_f:
            for pf in parquet_files:
                import pyarrow.parquet as pq

                parquet_file = pq.ParquetFile(pf)
                for batch in parquet_file.iter_batches(batch_size=10_000, columns=["text"]):
                    chunk = batch.to_pandas()
                    for line in chunk["text"].astype(str):
                        out_f.write(line.replace("\n", " ") + "\n")

        model_prefix = str(self.tokenizer_path / "sentencepiece")
        SentencePieceTrainer.train(
            input=str(tmp_txt),
            model_prefix=model_prefix,
            vocab_size=32_000,
            model_type="unigram",
            character_coverage=0.9995,
            unk_id=0,
            pad_id=1,
            bos_id=-1,
            eos_id=2,
            user_defined_symbols=[f"<extra_id_{i}>" for i in range(100)],
        )

        # Create tokenizer.json file for compatibility with T5TokenizerFast
        self._create_tokenizer_json_files(model_prefix)

    def _create_tokenizer_json_files(self, model_prefix: str):
        """Create additional files needed for T5TokenizerFast compatibility."""
        import json
        
        # Create a simple tokenizer config file
        config = {
            "model_type": "t5",
            "tokenizer_class": "T5TokenizerFast",
        }
        
        with open(self.tokenizer_path / "config.json", "w") as f:
            json.dump(config, f)
            
            
    def _load_tokenizer(self) -> PreTrainedTokenizerFast:
        tokenizer = T5TokenizerFast.from_pretrained(str(self.tokenizer_path))
        tokenizer.model_max_length = self.hparams.max_length
        return tokenizer
    
    # Setup datasets
    def setup(self, stage: str | None = None): 
        if self.train_dataset is not None:
            return  # Already initialised

        parquet_files = list(self.data_dir.rglob("*.parquet"))
        if not parquet_files:
            raise RuntimeError(f"No Parquet files found in {self.data_dir!s}")

        full_dataset = T5ParquetDataset(parquet_files)

        # Compute split lengths
        train_ratio, val_ratio = self.hparams.train_val_split
        train_len = int(len(full_dataset) * train_ratio)
        val_len = len(full_dataset) - train_len
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
        )

        self.tokenizer = self._load_tokenizer()
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