from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class _ParquetTextFile:
    """Lightweight wrapper providing random access to parquet file with text, id, and metadata."""

    def __init__(self, path: Path):
        self.file = pq.ParquetFile(path)
        self.num_rows = self.file.metadata.num_rows

    def __len__(self):
        return self.num_rows

    def __getitem__(self, idx: int) -> dict:
        # Find row group and local index
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
    """Concat multiple Parquet files containing text, id, and metadata columns into one dataset."""

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
        return row_data  # Now returns full row with text, id, metadata


class T5TokenCountDataset(Dataset):
    """
    T5 Dataset that computes sliding windows on-the-fly from stored T5 SentencePiece token counts.
    
    Expects documents processed with T5 SentencePiece token counts in metadata:
    - metadata.t5_sentencepiece_token_count: Number of tokens with T5 SentencePiece
    - metadata.t5_sliding_window_config: Window configuration for T5
    
    Computes windows deterministically:
    - window_count = max(1, (token_count - overlap_size) // stride + 1)
    - start_pos = window_idx * stride  
    - end_pos = min(start_pos + max_length, token_count)
    """
    
    def __init__(
        self, 
        parquet_files: List[str | Path], 
        tokenizer, 
        max_length: int = 512,
        overlap_size: int = 256,  # 50% overlap
        verify_config: bool = True
    ):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.overlap_size = overlap_size
        self.stride = max_length - overlap_size
        self.verify_config = verify_config
        
        # Create base dataset for document access
        self.base_dataset = T5ParquetDataset(parquet_files)
        
        log.info(f"Building window index from token counts for {len(self.base_dataset)} documents")
        log.info(f"Window config: max_length={max_length}, overlap={overlap_size}, stride={self.stride}")
        
        # Build index from token counts
        self._build_token_count_index()
        
        log.info(f"Created index for {len(self.window_index)} windows from token counts")
        
    def _verify_window_config(self, config: dict) -> bool:
        """Verify that stored window config matches expected configuration."""
        
        if not config:
            log.warning("No sliding window config found in metadata")
            return False
        
        expected_max_length = config.get("max_length", 0)
        expected_overlap = config.get("overlap_size", 0)
        
        if expected_max_length != self.max_length:
            log.warning(f"Max length mismatch: expected {self.max_length}, got {expected_max_length}")
            return False
        
        expected_overlap_ratio = expected_overlap / expected_max_length if expected_max_length > 0 else 0
        if abs(expected_overlap_ratio - 0.5) > 0.01:  # Allow 1% tolerance
            log.warning(f"Overlap ratio mismatch: expected ~50%, got {expected_overlap_ratio:.1%}")
            return False
        
        return True
    
    def _compute_window_count(self, token_count: int) -> int:
        """Compute number of windows for given token count."""
        if token_count <= self.max_length:
            return 1
        else:
            return max(1, (token_count - self.overlap_size) // self.stride + 1)
        
    def _build_token_count_index(self):
        """Build index from token count metadata."""
        
        self.window_index = []
        verified_config = None
        total_windows = 0
        docs_with_token_counts = 0
        
        for doc_idx in range(len(self.base_dataset)):
            # Get document to access metadata
            doc_data = self.base_dataset[doc_idx]
            
            # Handle case where metadata might be in different location
            metadata = None
            if isinstance(doc_data, dict):
                metadata = doc_data.get("metadata")
            
            if not metadata:
                log.warning(f"No metadata found for document {doc_idx}. Using fallback T5 SentencePiece tokenization.")
                # Fallback: tokenize on-the-fly to get T5 SentencePiece token count
                text = doc_data.get("text", "") if isinstance(doc_data, dict) else doc_data["text"]
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                token_count = len(tokens)
                window_count = self._compute_window_count(token_count)
                
                # Add windows for this document
                for window_idx in range(window_count):
                    self.window_index.append((doc_idx, window_idx, token_count))
                total_windows += window_count
                continue
            
            # Verify configuration on first document with metadata
            if verified_config is None and self.verify_config:
                window_config = metadata.get("t5_sliding_window_config", {})
                if self._verify_window_config(window_config):
                    verified_config = window_config
                    log.info(f"✅ Verified T5 window config: {window_config}")
                else:
                    log.warning("⚠️ T5 configuration verification failed - proceeding with current config")
                    verified_config = {}
            
            # Get T5 SentencePiece token count from metadata
            token_count = metadata.get("t5_sentencepiece_token_count")
            
            if token_count is None:
                log.warning(f"No T5 SentencePiece token count found for document {doc_idx}. Using fallback T5 SentencePiece tokenization.")
                # Fallback: tokenize on-the-fly with T5 SentencePiece
                text = doc_data.get("text", "") if isinstance(doc_data, dict) else doc_data["text"]
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                token_count = len(tokens)
            else:
                docs_with_token_counts += 1
            
            # Compute window count
            window_count = self._compute_window_count(token_count)
            
            # Add all windows for this document
            for window_idx in range(window_count):
                self.window_index.append((doc_idx, window_idx, token_count))
            
            total_windows += window_count
            
            if doc_idx < 5:  # Log first few documents for debugging
                log.info(f"Document {doc_idx}: {token_count} T5 SentencePiece tokens → {window_count} windows")
        
        log.info(f"📊 Index built: {len(self.base_dataset)} docs → {total_windows} windows")
        log.info(f"📈 Docs with T5 SentencePiece token counts: {docs_with_token_counts}/{len(self.base_dataset)} ({docs_with_token_counts/len(self.base_dataset):.1%})")
        if docs_with_token_counts < len(self.base_dataset):
            log.warning(f"⚠️ {len(self.base_dataset) - docs_with_token_counts} documents missing T5 SentencePiece token counts, using fallback T5 SentencePiece tokenization")
    
    def __len__(self):
        return len(self.window_index)
    
    def __getitem__(self, idx: int) -> dict:
        """Get window computed on-the-fly from token count."""
        
        if idx >= len(self.window_index):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.window_index)}")
        
        doc_idx, window_idx, token_count = self.window_index[idx]
        
        # Get document text
        doc_data = self.base_dataset[doc_idx]
        text = doc_data.get("text", "") if isinstance(doc_data, dict) else doc_data["text"]
        
        # Tokenize document
        full_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Verify T5 SentencePiece token count if available
        if len(full_tokens) != token_count:
            log.warning(f"T5 SentencePiece token count mismatch for doc {doc_idx}: stored={token_count}, actual={len(full_tokens)}")
            # Use actual token count
            token_count = len(full_tokens)
        
        # Compute window boundaries on-the-fly
        if token_count <= self.max_length:
            # Short document: single window with padding
            window_tokens = full_tokens
            start_pos = 0
            end_pos = len(full_tokens)
        else:
            # Long document: extract specific window
            start_pos = window_idx * self.stride
            end_pos = min(start_pos + self.max_length, token_count)
            window_tokens = full_tokens[start_pos:end_pos]
        
        # Pad if needed
        if len(window_tokens) < self.max_length:
            window_tokens.extend([self.tokenizer.pad_token_id or 0] * (self.max_length - len(window_tokens)))
        
        # Ensure exactly max_length tokens
        window_tokens = window_tokens[:self.max_length]
        
        return {
            "text": "",  # Empty - we return tokens directly
            "tokens": window_tokens,
            "type": "t5_sentencepiece_on_the_fly",
            "doc_idx": doc_idx,
            "window_idx": window_idx,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "t5_sentencepiece_token_count": token_count
        }


class T5PrecomputedWindowDataset(Dataset):
    """
    DEPRECATED: Use T5TokenCountDataset with token count preprocessing instead.
    
    This approach stored full window boundaries which was inefficient.
    The new approach stores only token counts and computes windows on-the-fly.
    """
    
    def __init__(self, *args, **kwargs):
        log.warning("⚠️ T5PrecomputedWindowDataset is DEPRECATED. Use T5TokenCountDataset instead.")
        log.warning("   This class stored full window boundaries which was inefficient.")
        log.warning("   Use T5TokenCountDataset with token count preprocessing for better performance.")
        
        # Fallback implementation for backwards compatibility
        super().__init__()
        raise NotImplementedError(
            "T5PrecomputedWindowDataset is deprecated. "
            "Use T5TokenCountDataset with token count preprocessing instead."
        )


class T5SlidingWindowDataset(Dataset):
    """
    DEPRECATED: Use T5PrecomputedWindowDataset with preprocessing pipeline instead.
    
    This class is kept for backwards compatibility but should not be used for new implementations.
    Instead, use:
    1. SlidingWindowProcessor to precompute windows
    2. T5PrecomputedWindowDataset to read precomputed windows
    """
    
    def __init__(self, *args, **kwargs):
        log.warning("⚠️ T5SlidingWindowDataset is DEPRECATED. Use T5PrecomputedWindowDataset with preprocessing pipeline instead.")
        log.warning("   1. Run: python src/dataprep/pipelines/run_sliding_windows.py")
        log.warning("   2. Use: T5PrecomputedWindowDataset for training")
        
        # Fallback to simple implementation for backwards compatibility
        super().__init__()
        raise NotImplementedError(
            "T5SlidingWindowDataset is deprecated. "
            "Use token count preprocessing pipeline + T5TokenCountDataset instead."
        ) 