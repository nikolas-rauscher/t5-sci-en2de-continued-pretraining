#!/usr/bin/env python3
"""
Search for a document by ID in all parquet files and save text to file.
Efficiently searches through large parquet files without loading everything into memory.
"""

import pyarrow.parquet as pq
import pandas as pd
import argparse
from pathlib import Path
import json
from datetime import datetime


def search_in_parquet_file(file_path: Path, search_id: str, verbose: bool = False):
    """
    Search for a specific ID in a parquet file.
    
    Args:
        file_path: Path to parquet file
        search_id: ID to search for
        verbose: Print search progress
        
    Returns:
        Dict with document data if found, None otherwise
    """
    if verbose:
        print(f"Searching in {file_path.name}...")
    
    try:
        pf = pq.ParquetFile(file_path)
        
        # Read file in chunks (row groups) to save memory
        for i in range(pf.num_row_groups):
            if verbose and i % 10 == 0:
                print(f"  Checking row group {i+1}/{pf.num_row_groups}")
                
            # Read one row group at a time
            table = pf.read_row_group(i)
            df = table.to_pandas()
            
            # Search for the ID
            matches = df[df['id'] == search_id]
            
            if not matches.empty:
                # Found the document!
                doc = matches.iloc[0]
                
                result = {
                    'found_in_file': str(file_path),
                    'row_group': i,
                    'id': doc['id'],
                    'text': doc['text'],
                    'metadata': doc['metadata'] if 'metadata' in doc else None
                }
                
                if verbose:
                    print(f"  ✓ Found in row group {i+1}")
                
                return result
                
    except Exception as e:
        if verbose:
            print(f"  ✗ Error reading {file_path.name}: {e}")
        return None
    
    return None


def search_all_files(directory: Path, search_id: str, verbose: bool = False):
    """
    Search for ID in all parquet files in directory.
    
    Args:
        directory: Directory containing parquet files
        search_id: ID to search for
        verbose: Print progress
        
    Returns:
        Document data if found, None otherwise
    """
    parquet_files = list(directory.glob("*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {directory}")
        return None
    
    parquet_files.sort()
    
    print(f"Searching for ID '{search_id}' in {len(parquet_files)} parquet files...")
    
    for i, file_path in enumerate(parquet_files, 1):
        if verbose:
            print(f"\n[{i}/{len(parquet_files)}] {file_path.name}")
        else:
            print(f"Checking {file_path.name}... ", end="", flush=True)
        
        result = search_in_parquet_file(file_path, search_id, verbose)
        
        if result:
            if not verbose:
                print("✓ FOUND!")
            return result
        else:
            if not verbose:
                print("not found")
    
    return None


def save_document(doc_data: dict, output_dir: Path, search_id: str):
    """
    Save document text and metadata to files.
    
    Args:
        doc_data: Document data dictionary
        output_dir: Output directory
        search_id: Original search ID
    """
    output_dir.mkdir(exist_ok=True)
    
    # Create safe filename
    safe_id = "".join(c for c in search_id if c.isalnum() or c in '-_.')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save text file
    text_file = output_dir / f"document_{safe_id}_{timestamp}.txt"
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(f"Document ID: {doc_data['id']}\n")
        f.write(f"Found in: {doc_data['found_in_file']}\n")
        f.write(f"Extracted on: {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")
        f.write(doc_data['text'])
    
    print(f"✓ Text saved to: {text_file}")
    
    # Save metadata if available
    if doc_data['metadata']:
        metadata_file = output_dir / f"metadata_{safe_id}_{timestamp}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                'id': doc_data['id'],
                'found_in_file': doc_data['found_in_file'],
                'row_group': doc_data['row_group'],
                'extracted_on': datetime.now().isoformat(),
                'metadata': doc_data['metadata']
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Metadata saved to: {metadata_file}")
        
        # Print some key statistics
        metadata = doc_data['metadata']
        print(f"\nDocument Statistics:")
        print(f"  Text length: {metadata.get('length', 'N/A'):,} characters")
        print(f"  Token count: {metadata.get('token_count', 'N/A'):,}")
        print(f"  Language score: {metadata.get('fasttext_en', 'N/A')}")
        print(f"  Authors: {metadata.get('authors', 'N/A')}")
        print(f"  Title: {metadata.get('title', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(
        description="Search for document by ID and save text to file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/find_document_by_id.py "part-1.parquet/4"
  python scripts/find_document_by_id.py "doc123" --directory data/other_parquet_files/
  python scripts/find_document_by_id.py "abc" --output extracted_docs/ --verbose
        """
    )
    
    parser.add_argument("id", help="Document ID to search for")
    parser.add_argument(
        "--directory", "-d",
        default="data/statistics_data_gold/enriched_documents_statistics_v2",
        help="Directory containing parquet files"
    )
    parser.add_argument(
        "--output", "-o",
        default="extracted_documents",
        help="Output directory for extracted files"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed search progress"
    )
    
    args = parser.parse_args()
    
    directory = Path(args.directory)
    output_dir = Path(args.output)
    
    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        return 1
    
    print(f"Searching for document with ID: '{args.id}'")
    print(f"Search directory: {directory}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    # Search for the document
    doc_data = search_all_files(directory, args.id, args.verbose)
    
    if doc_data:
        print(f"\n✓ Document found!")
        print(f"File: {doc_data['found_in_file']}")
        print(f"Text length: {len(doc_data['text']):,} characters")
        
        # Save to files
        save_document(doc_data, output_dir, args.id)
        
        print(f"\n✓ Extraction complete!")
        return 0
    else:
        print(f"\n✗ Document with ID '{args.id}' not found in any parquet file.")
        return 1


if __name__ == "__main__":
    exit(main()) 