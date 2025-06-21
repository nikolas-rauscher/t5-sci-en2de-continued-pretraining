from __future__ import annotations

from pathlib import Path
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import Dataset


class _ParquetTextFile:
    """Lightweight wrapper providing random access to the *text* column."""

    def __init__(self, path: Path):
        self.file = pq.ParquetFile(path)
        self.num_rows = self.file.metadata.num_rows

    def __len__(self):
        return self.num_rows

    def __getitem__(self, idx: int) -> str:
        # Find row group and local index
        row_group, row_index = self._locate(idx)
        batch: pa.Table = self.file.read_row_group(row_group, columns=["text"])
        return batch.column(0)[row_index].as_py()

    def _locate(self, global_idx: int):
        rows_cum = 0
        for rg_idx in range(self.file.num_row_groups):
            rg_rows = self.file.metadata.row_group(rg_idx).num_rows
            if rows_cum + rg_rows > global_idx:
                return rg_idx, global_idx - rows_cum
            rows_cum += rg_rows
        raise IndexError("Row index out of range")


class T5ParquetDataset(Dataset):
    """Concat multiple Parquet files containing a *text* column into one dataset."""

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
        text = self.files[file_idx][local_idx]
        return {"text": text} 