from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class _ParquetTextFile:
    """Random access to parquet file."""

    def __init__(self, path: Path):
        self.file = pq.ParquetFile(path)
        self.num_rows = self.file.metadata.num_rows

    def __len__(self):
        return self.num_rows

    def __getitem__(self, idx: int) -> dict:
        # find row group and local index
        row_group, row_index = self._locate(idx)
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
    """Pre-tokenized sliding windows dataset."""
    
    def __init__(self, parquet_files: List[str | Path]):
        super().__init__()
        
        self.base_dataset = T5ParquetDataset(parquet_files)
        
        self._verify_materialized_windows()
        
        log.info(f"T5MaterializedWindowDataset created with {len(self.base_dataset)} pre-tokenized windows")
        
    def _verify_materialized_windows(self):
        if len(self.base_dataset) == 0:
            raise ValueError("No data found in parquet files")
        
        sample_size = min(5, len(self.base_dataset))
        materialized_count = 0
        
        for i in range(sample_size):
            doc_data = self.base_dataset[i]
            metadata = doc_data.get("metadata", {})
            
            # Check for materialized window markers
            doc_type = metadata.get("document_type")
            version = metadata.get("preprocessing_version")
            
            if doc_type == "t5_sliding_window" and version == "v2_materialized_windows":
                materialized_count += 1
            
            # Verify text format 
            text = doc_data.get("text", "")
            if text and not self._is_token_sequence(text):
                log.warning(f"Document {i} text doesn't look like token sequence: {text[:50]}...")
        
        # check if we found materialized windows
        if materialized_count == 0:
            raise ValueError(
                "No materialized windows found! "
                "Expected metadata.document_type='t5_sliding_window' and "
                "metadata.preprocessing_version='v2_materialized_windows'. "
                "Run sliding window materialization first: "
                "python src/dataprep/pipelines/run_sliding_windows.py"
            )
        
        materialized_ratio = materialized_count / sample_size
        log.info(f"✅ Verified materialized windows: {materialized_count}/{sample_size} ({materialized_ratio:.1%})")
        
        if materialized_ratio < 1.0:
            log.warning(f"⚠️ Only {materialized_ratio:.1%} of documents are materialized windows. "
                       f"Consider filtering or re-running materialization.")
    
    def _is_token_sequence(self, text: str) -> bool:
        parts = text.strip().split()
        if len(parts) < 10:  # Too short to be a meaningful token sequence
            return False
        
        # Check if first 10 parts are integers
        try:
            for part in parts[:10]:
                int(part)
            return True
        except ValueError:
            return False
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> dict:
        """Get pre-tokenized window."""
        
        if idx >= len(self.base_dataset):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.base_dataset)}")
        
        # Get window data
        doc_data = self.base_dataset[idx]
        text = doc_data.get("text", "")
        metadata = doc_data.get("metadata", {})
        
        # parse pre-tokenized sequence
        try:
            tokens = [int(x) for x in text.split()]
        except ValueError as e:
            log.error(f"Failed to parse tokens from text: {text[:100]}...")
            raise ValueError(f"Invalid token sequence in document {idx}: {e}")
        
        return {
            "tokens": tokens,
            "type": "t5_materialized_window",
            "doc_id": doc_data.get("id", f"unknown_{idx}"),
            "original_doc_id": metadata.get("original_doc_id"),
            "window_idx": metadata.get("window_idx", 0),
            "original_metadata": metadata.get("original_metadata", {})
        }


 