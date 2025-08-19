"""
Prepare German dataset for DataTrove statistics pipeline
Downloads German data and converts to DataTrove-compatible format
"""

import logging
from pathlib import Path
from datasets import load_dataset
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GermanDataTrovePreparation:
    def __init__(self, base_dir: str = "cross_lingual_transfer/data/german"):
        """
        Initialize German DataTrove preparation
        
        Args:
            base_dir: Base directory for German data
        """
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw_parquet"
        self.processed_dir = self.base_dir / "processed"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def download_and_convert_german_dataset(self):
        """Download German dataset and convert to DataTrove format"""
        logger.info("Downloading German scientific dataset...")
        
        try:
            # Set HuggingFace token if available
            import os
            hf_token = os.getenv('HUGGING_FACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
            if hf_token:
                logger.info("Using HuggingFace token from environment")
                
            # Load German split from scilons dataset
            dataset = load_dataset(
                "scilons/texts_pq_3",
                name="deu_Latn", 
                split="train",
                streaming=False,
                token=hf_token
            )
            
            logger.info(f"Downloaded {len(dataset)} German documents")
            
            # Convert to pandas for processing
            df = dataset.to_pandas()
            logger.info(f"Dataset columns: {list(df.columns)}")
            
            # Prepare DataTrove-compatible format
            if 'text' in df.columns:
                # Standard case: text column exists
                datatrove_df = df.copy()
                
                # Ensure we have 'id' column
                if 'id' not in datatrove_df.columns:
                    datatrove_df['id'] = range(len(datatrove_df))
                    
            else:
                logger.error(f"No 'text' column found. Available columns: {list(df.columns)}")
                # Try to find text-like column
                text_candidates = [col for col in df.columns if 'text' in col.lower() or 'content' in col.lower()]
                if text_candidates:
                    logger.info(f"Using {text_candidates[0]} as text column")
                    datatrove_df = df.rename(columns={text_candidates[0]: 'text'}).copy()
                    if 'id' not in datatrove_df.columns:
                        datatrove_df['id'] = range(len(datatrove_df))
                else:
                    raise ValueError("No suitable text column found in dataset")
            
            # Basic statistics
            total_docs = len(datatrove_df)
            if 'text' in datatrove_df.columns:
                text_lengths = datatrove_df['text'].astype(str).str.len()
                avg_length = text_lengths.mean()
                total_chars = text_lengths.sum()
                
                stats = {
                    "total_documents": total_docs,
                    "average_document_length": float(avg_length),
                    "total_characters": int(total_chars),
                    "estimated_tokens": int(total_chars // 4),
                    "columns": list(datatrove_df.columns)
                }
                
                logger.info(f"Dataset statistics: {stats}")
            
            # Save as parquet files for DataTrove (chunked for parallel processing)
            chunk_size = 10000  # 10k documents per file
            num_chunks = (len(datatrove_df) + chunk_size - 1) // chunk_size
            
            logger.info(f"Splitting into {num_chunks} chunks of ~{chunk_size} documents")
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(datatrove_df))
                chunk_df = datatrove_df.iloc[start_idx:end_idx]
                
                # Save chunk as parquet
                chunk_file = self.raw_dir / f"german_scientific_chunk_{i:04d}.parquet"
                chunk_df.to_parquet(chunk_file, index=False)
                logger.info(f"Saved chunk {i+1}/{num_chunks}: {len(chunk_df)} docs -> {chunk_file}")
            
            # Save metadata
            import json
            metadata = {
                **stats,
                "num_chunks": num_chunks,
                "chunk_size": chunk_size,
                "source_dataset": "scilons/texts_pq_3",
                "source_split": "deu_Latn"
            }
            
            with open(self.base_dir / "dataset_metadata.json", "w", encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                
            logger.info("German dataset converted to DataTrove format successfully!")
            return metadata
            
        except Exception as e:
            logger.error(f"Error downloading/converting dataset: {e}")
            logger.info("You may need to authenticate with: huggingface-cli login")
            raise
    
    def create_train_validation_split(self, validation_ratio=0.01):
        """Create train/validation split from chunked data"""
        logger.info("Creating train/validation split...")
        
        # Load all chunks
        chunk_files = sorted(self.raw_dir.glob("german_scientific_chunk_*.parquet"))
        if not chunk_files:
            raise FileNotFoundError("No chunk files found. Run download_and_convert_german_dataset first.")
        
        # Read all chunks and combine
        all_dfs = []
        for chunk_file in chunk_files:
            df = pd.read_parquet(chunk_file)
            all_dfs.append(df)
            
        combined_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Combined {len(combined_df)} documents from {len(chunk_files)} chunks")
        
        # Create train/validation split
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            combined_df, 
            test_size=validation_ratio,
            random_state=42,
            shuffle=True
        )
        
        logger.info(f"Split: Train={len(train_df)}, Validation={len(val_df)}")
        
        # Save splits
        train_df.to_parquet(self.processed_dir / "train.parquet", index=False)
        val_df.to_parquet(self.processed_dir / "validation.parquet", index=False)
        
        logger.info("Train/validation split created successfully!")
        
        return {
            "train_size": len(train_df),
            "validation_size": len(val_df)
        }
    
    def run_preparation(self):
        """Complete German DataTrove preparation pipeline"""
        logger.info("Starting German DataTrove preparation...")
        
        # Download and convert
        metadata = self.download_and_convert_german_dataset()
        
        # Create splits
        split_info = self.create_train_validation_split()
        
        logger.info("German DataTrove preparation completed!")
        return {**metadata, **split_info}


def main():
    """Main preparation execution"""
    
    base_dir = "/netscratch/nrauscher/projects/BA-hydra/cross_lingual_transfer/data/german"
    
    # Execute preparation
    prep = GermanDataTrovePreparation(base_dir)
    result = prep.run_preparation()
    
    print(f"DataTrove preparation completed! Result: {result}")
    
    # Show next steps
    print("\\n=== Next Steps ===")
    print("1. Run DataTrove stats pipeline:")
    print("   python src/dataprep/pipelines/run_stats.py --config-path=../../cross_lingual_transfer/configs/stats --config-name=german_datatrove_stats")
    print("\\n2. Results will be saved in:")
    print("   cross_lingual_transfer/data/german/datatrove_stats/")


if __name__ == "__main__":
    main()