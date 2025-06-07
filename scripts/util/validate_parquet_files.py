#!/usr/bin/env python3
"""
Script to validate all parquet files in a directory for integrity and structure.
Checks if files can be read properly and have expected structure.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import logging
from datetime import datetime
import argparse
from typing import List, Tuple, Dict, Any
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parquet_validation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def validate_parquet_file(file_path: Path) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate a single parquet file.
    
    Args:
        file_path: Path to the parquet file
        
    Returns:
        Tuple of (is_valid, error_message, file_info)
    """
    file_info = {
        "file_name": file_path.name,
        "file_size_mb": file_path.stat().st_size / (1024 * 1024),
        "num_rows": None,
        "num_columns": None,
        "columns": None,
        "schema": None
    }
    
    try:
        # Try to read parquet file metadata first (faster)
        parquet_file = pq.ParquetFile(file_path)
        metadata = parquet_file.metadata
        schema = parquet_file.schema
        
        # Basic metadata checks
        file_info["num_rows"] = metadata.num_rows
        file_info["num_columns"] = metadata.num_columns
        file_info["schema"] = str(schema)
        file_info["columns"] = [field.name for field in schema]
        
        # Try to read first few rows to ensure data integrity
        # Use row_group instead of nrows for older PyArrow versions
        try:
            # Try newer API first
            table = parquet_file.read_row_group(0)
            # Limit to first 10 rows
            if table.num_rows > 10:
                table = table.slice(0, 10)
        except Exception:
            # Fallback: read entire file (slower but works)
            table = parquet_file.read()
            # Limit to first 10 rows
            if table.num_rows > 10:
                table = table.slice(0, 10)
        
        df_sample = table.to_pandas()
        
        # Basic structure validation for enriched documents
        # Updated to match actual flat structure (not nested metadata)
        required_columns = ['text', 'id']
        expected_stats_columns = ['length', 'token_count', 'fasttext_en', 'n_words', 'n_sentences']
        
        missing_required = [col for col in required_columns if col not in file_info["columns"]]
        missing_stats = [col for col in expected_stats_columns if col not in file_info["columns"]]
        
        if missing_required:
            return False, f"Missing required columns: {missing_required}", file_info
            
        # Check if we have some expected statistics columns
        if len(missing_stats) > len(expected_stats_columns) // 2:  # More than half missing
            logger.warning(f"File {file_path.name}: many expected stats columns missing: {missing_stats}")
        
        # Check data quality: ensure we have actual content
        if 'text' in df_sample.columns and len(df_sample) > 0:
            text_sample = df_sample['text'].iloc[0]
            if not text_sample or len(str(text_sample).strip()) == 0:
                return False, "Text column contains empty content", file_info
        
        logger.info(f"✓ {file_path.name}: Valid parquet file ({file_info['num_rows']:,} rows, {file_info['num_columns']} columns)")
        return True, "", file_info
        
    except pa.lib.ArrowInvalid as e:
        error_msg = f"Arrow/Parquet format error: {str(e)}"
        logger.error(f"✗ {file_path.name}: {error_msg}")
        return False, error_msg, file_info
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"✗ {file_path.name}: {error_msg}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return False, error_msg, file_info


def validate_directory(directory_path: Path) -> Dict[str, Any]:
    """
    Validate all parquet files in a directory.
    
    Args:
        directory_path: Path to directory containing parquet files
        
    Returns:
        Dictionary with validation results
    """
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
        
    # Find all parquet files
    parquet_files = list(directory_path.glob("*.parquet"))
    
    if not parquet_files:
        logger.warning(f"No parquet files found in {directory_path}")
        return {"total_files": 0, "valid_files": 0, "invalid_files": 0, "results": []}
    
    logger.info(f"Found {len(parquet_files)} parquet files to validate")
    
    results = []
    valid_count = 0
    invalid_count = 0
    total_size_mb = 0
    total_rows = 0
    
    # Sort files for consistent order
    parquet_files.sort()
    
    for i, file_path in enumerate(parquet_files, 1):
        logger.info(f"Validating file {i}/{len(parquet_files)}: {file_path.name}")
        
        is_valid, error_msg, file_info = validate_parquet_file(file_path)
        
        result = {
            "file_path": str(file_path),
            "is_valid": is_valid,
            "error_message": error_msg,
            **file_info
        }
        results.append(result)
        
        if is_valid:
            valid_count += 1
            total_size_mb += file_info["file_size_mb"]
            if file_info["num_rows"]:
                total_rows += file_info["num_rows"]
        else:
            invalid_count += 1
    
    # Summary
    summary = {
        "total_files": len(parquet_files),
        "valid_files": valid_count,
        "invalid_files": invalid_count,
        "total_size_mb": total_size_mb,
        "total_rows": total_rows,
        "results": results
    }
    
    return summary


def save_validation_report(summary: Dict[str, Any], output_path: Path):
    """Save validation report to JSON file."""
    import json
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_types(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, (list, tuple)):
            return [convert_types(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        return obj
    
    summary_json = convert_types(summary)
    
    with open(output_path, 'w') as f:
        json.dump(summary_json, f, indent=2, default=str)
    
    logger.info(f"Validation report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate parquet files for integrity")
    parser.add_argument(
        "directory", 
        nargs="?",
        default="data/statistics_data_gold/enriched_documents_statistics_v2",
        help="Directory containing parquet files to validate"
    )
    parser.add_argument(
        "--output-report",
        default="parquet_validation_report.json",
        help="Path to save validation report (JSON)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    directory_path = Path(args.directory)
    output_path = Path(args.output_report)
    
    logger.info(f"Starting parquet validation for directory: {directory_path}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    try:
        summary = validate_directory(directory_path)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total files: {summary['total_files']}")
        logger.info(f"Valid files: {summary['valid_files']}")
        logger.info(f"Invalid files: {summary['invalid_files']}")
        logger.info(f"Total size: {summary['total_size_mb']:.2f} MB")
        logger.info(f"Total rows: {summary['total_rows']:,}")
        
        if summary['invalid_files'] > 0:
            logger.error("\nINVALID FILES:")
            for result in summary['results']:
                if not result['is_valid']:
                    logger.error(f"  - {result['file_path']}: {result['error_message']}")
        
        # Save detailed report
        save_validation_report(summary, output_path)
        
        # Exit with error code if any files are invalid
        if summary['invalid_files'] > 0:
            logger.error(f"\n{summary['invalid_files']} invalid files found!")
            sys.exit(1)
        else:
            logger.info("\n✓ All parquet files are valid!")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main() 