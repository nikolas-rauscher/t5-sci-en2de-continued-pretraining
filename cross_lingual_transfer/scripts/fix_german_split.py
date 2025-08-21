"""
Fix German dataset train/validation split without scikit-learn
Use numpy-only solution to avoid binary compatibility issues
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_train_val_split_numpy(data_dir: str, val_ratio: float = 0.01):
    """Create train/validation split using only numpy (no sklearn)"""
    
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw_parquet"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(exist_ok=True)
    
    logger.info("Loading all German chunks...")
    
    # Load all chunks
    chunk_files = sorted(raw_dir.glob("german_scientific_chunk_*.parquet"))
    logger.info(f"Found {len(chunk_files)} chunk files")
    
    all_dfs = []
    total_docs = 0
    
    for chunk_file in chunk_files:
        df = pd.read_parquet(chunk_file)
        all_dfs.append(df)
        total_docs += len(df)
        logger.info(f"Loaded {chunk_file.name}: {len(df)} documents")
    
    logger.info(f"Total documents: {total_docs}")
    
    # Combine all chunks
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined_df)} documents")
    
    # Create train/val split using numpy only
    np.random.seed(42)  # For reproducibility
    total_samples = len(combined_df)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    # Calculate split point
    val_size = int(total_samples * val_ratio)
    train_size = total_samples - val_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    logger.info(f"Split: Train={len(train_indices)}, Validation={len(val_indices)}")
    
    # Create splits
    train_df = combined_df.iloc[train_indices].reset_index(drop=True)
    val_df = combined_df.iloc[val_indices].reset_index(drop=True)
    
    # Save splits
    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    val_df.to_parquet(processed_dir / "validation.parquet", index=False)
    
    logger.info(f"Saved train.parquet: {len(train_df)} documents")
    logger.info(f"Saved validation.parquet: {len(val_df)} documents")
    
    # Calculate statistics
    if 'text' in combined_df.columns:
        text_lengths = combined_df['text'].astype(str).str.len()
        avg_length = text_lengths.mean()
        total_chars = text_lengths.sum()
        
        stats = {
            "total_documents": total_samples,
            "train_documents": len(train_df),
            "validation_documents": len(val_df),
            "validation_ratio": val_ratio,
            "average_document_length": float(avg_length),
            "total_characters": int(total_chars),
            "estimated_tokens": int(total_chars // 4),
            "columns": list(combined_df.columns)
        }
    else:
        stats = {
            "total_documents": total_samples,
            "train_documents": len(train_df),
            "validation_documents": len(val_df),
            "validation_ratio": val_ratio,
            "columns": list(combined_df.columns)
        }
    
    # Save metadata
    with open(data_dir / "dataset_metadata.json", "w", encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Dataset statistics: {stats}")
    logger.info("German dataset processing completed successfully!")
    
    return stats


def main():
    """Main execution"""
    
    data_dir = "/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/data/german"
    
    try:
        result = create_train_val_split_numpy(data_dir, val_ratio=0.01)
        logger.info("SUCCESS: German data ready for training!")
        
        # Show file sizes
        processed_dir = Path(data_dir) / "processed"
        for file in ["train.parquet", "validation.parquet"]:
            filepath = processed_dir / file
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024*1024)
                logger.info(f"  {file}: {size_mb:.1f} MB")
                
        return True
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)