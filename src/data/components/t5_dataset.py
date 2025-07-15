from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class _ParquetTextFile:
    """Random access to parquet file with batch caching."""

    def __init__(self, path: Path):
        self.file = pq.ParquetFile(path)
        self.num_rows = self.file.metadata.num_rows
        # Disable caching in multiprocessing environments to avoid deadlocks
        import os
        self._use_cache = os.getenv('PYTORCH_DATALOADER_WORKERS', '0') == '0'
        if self._use_cache:
            self._cache = {}  # Cache for row groups
            self._cache_size = 0
            self._max_cache_size = 2  # Cache up to 2 row groups

    def __len__(self):
        return self.num_rows

    def __getitem__(self, idx: int) -> dict:
        # find row group and local index
        row_group, row_index = self._locate(idx)
        
        if self._use_cache:
            # Check cache first
            if row_group not in self._cache:
                # Read entire row group
                batch: pa.Table = self.file.read_row_group(row_group, columns=["text", "id", "metadata"])
                batch_data = batch.to_pylist()
                
                # Manage cache size
                if self._cache_size >= self._max_cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                    self._cache_size -= 1
                
                self._cache[row_group] = batch_data
                self._cache_size += 1
            
            return self._cache[row_group][row_index]
        else:
            # No caching - direct read (safer for multiprocessing)
            batch: pa.Table = self.file.read_row_group(row_group, columns=["text", "id", "metadata"])
            row = batch.slice(row_index, 1).to_pylist()[0]
            return row

    def _locate(self, global_idx: int):
        rows_cum = 0
        for rg_idx in range(self.file.num_row_groups):
            rg_rows = self.file.metadata.row_group(rg_idx).num_rows
            if rows_cum + rg_rows > global_idx:
                return rg_idx, global_idx - rows_cum
            rows_cum += rg_rows
        raise IndexError("Row index out of range")


class T5ParquetDataset(Dataset):
    """Concatenated parquet files dataset."""

    def __init__(self, parquet_files: List[str | Path]):
        super().__init__()
        self.files = [_ParquetTextFile(Path(f)) for f in parquet_files]
        self.cumulative_lengths = []
        total = 0
        for f in self.files:
            total += len(f)
            self.cumulative_lengths.append(total)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def _find_file(self, idx: int):
        for file_idx, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                prev_cum = 0 if file_idx == 0 else self.cumulative_lengths[file_idx - 1]
                return file_idx, idx - prev_cum
        raise IndexError("Index out of range")

    def __getitem__(self, idx: int):
        file_idx, local_idx = self._find_file(idx)
        row_data = self.files[file_idx][local_idx]
        return row_data






class T5MaterializedWindowDataset(Dataset):
    """Text-only sliding windows dataset for efficient batch tokenization."""
    
    def __init__(self, parquet_files: List[str | Path], limit_documents: int = -1):
        super().__init__()
        
        self.base_dataset = T5ParquetDataset(parquet_files)
        self._verify_text_windows()
        
        # Apply document limit if specified
        self.limit_documents = limit_documents
        if limit_documents > 0 and limit_documents < len(self.base_dataset):
            self.actual_length = limit_documents
            log.info(f"T5MaterializedWindowDataset created with {limit_documents} text windows (limited from {len(self.base_dataset)})")
        else:
            self.actual_length = len(self.base_dataset)
            log.info(f"T5MaterializedWindowDataset created with {len(self.base_dataset)} text windows")
        
    def _verify_text_windows(self):
        if len(self.base_dataset) == 0:
            raise ValueError("No data found in parquet files")
        
        sample_size = min(5, len(self.base_dataset))
        text_window_count = 0
        
        for i in range(sample_size):
            doc_data = self.base_dataset[i]
            metadata = doc_data.get("metadata", {})
            
            # Check for text window markers
            doc_type = metadata.get("document_type")
            version = metadata.get("preprocessing_version")
            
            if doc_type == "text_sliding_window" and version in ["v3_text_only_windows", "v6_precise_text_windows"]:
                text_window_count += 1
            elif doc_type == "token_sliding_window" and version == "v5_configurable_windows":
                # Token-based sliding windows are also acceptable
                text_window_count += 1
                log.info(f"Found token-based sliding window at index {i}")
            elif doc_type == "t5_sliding_window" and version == "v2_materialized_windows":
                # Legacy pre-tokenized windows are also acceptable
                text_window_count += 1
                log.info(f"Found legacy pre-tokenized window at index {i}")
            
            # Verify text exists
            text = doc_data.get("text", "")
            if not text.strip():
                log.warning(f"Document {i} has empty text")
        
        # Check if we found valid windows
        if text_window_count == 0:
            raise ValueError(
                "No sliding windows found! "
                "Expected metadata.document_type='text_sliding_window' or 't5_sliding_window'. "
                "Run sliding window creation first: "
                "python src/dataprep/pipelines/run_sliding_windows.py"
            )
        
        window_ratio = text_window_count / sample_size
        log.info(f"✅ Verified sliding windows: {text_window_count}/{sample_size} ({window_ratio:.1%})")
        
        if window_ratio < 1.0:
            log.warning(f"⚠️ Only {window_ratio:.1%} of documents are sliding windows. "
                       f"Consider filtering or re-running window creation.")
    
    def __len__(self):
        return self.actual_length
    
    def __getitem__(self, idx: int) -> dict:
        """Get text window - much faster than pre-tokenized version."""
        
        if idx >= self.actual_length:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.actual_length}")
        
        # Apply document limit check
        if self.limit_documents > 0 and idx >= self.limit_documents:
            raise IndexError(f"Index {idx} exceeds document limit {self.limit_documents}")
        
        # Get window data (direct access, no caching needed)
        doc_data = self.base_dataset[idx]
        text = doc_data.get("text", "")
        metadata = doc_data.get("metadata", {})
        
        # Check if this is legacy pre-tokenized data
        if metadata.get("document_type") == "t5_sliding_window":
            # Legacy: try to parse as tokens first, fallback to text
            try:
                tokens = [int(x) for x in text.split()[:10]]  # Test first 10
                # If successful, it's pre-tokenized - convert back to text
                # This is a fallback for compatibility
                return {
                    "tokens": [int(x) for x in text.split()],
                    "text": None,  # Signal this is pre-tokenized
                    "type": "legacy_t5_materialized_window",
                    "doc_id": doc_data.get("id", f"unknown_{idx}"),
                    "original_doc_id": metadata.get("original_doc_id"),
                    "window_idx": metadata.get("window_idx", 0),
                    "original_metadata": metadata.get("original_metadata", {})
                }
            except ValueError:
                # Not pre-tokenized, treat as text
                pass
        
        # Modern text-only window (fast path)
        return {
            "text": text.strip(),
            "type": "text_sliding_window", 
            "doc_id": doc_data.get("id", f"unknown_{idx}"),
            "original_doc_id": metadata.get("original_doc_id"),
            "window_idx": metadata.get("window_idx", 0),
            "original_metadata": metadata.get("original_metadata", {})
        }


 