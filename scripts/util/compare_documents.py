#!/usr/bin/env python3
"""
Compare documents by ID between gold data and cleaned data directories.
Shows text differences and saves comparison results.
"""

import pyarrow.parquet as pq
import pandas as pd
import argparse
from pathlib import Path
import json
from datetime import datetime
import difflib
import sys


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


def search_in_directory(directory: Path, search_id: str, verbose: bool = False):
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
        if verbose:
            print(f"No parquet files found in {directory}")
        return None
    
    parquet_files.sort()
    
    if verbose:
        print(f"Searching in {len(parquet_files)} files in {directory.name}/")
    
    for file_path in parquet_files:
        result = search_in_parquet_file(file_path, search_id, verbose)
        if result:
            if verbose:
                print(f"  ✓ Found in {file_path.name}")
            return result
    
    return None


def create_text_diff(text1: str, text2: str, name1: str = "Original", name2: str = "Modified"):
    """
    Create a unified diff between two texts.
    
    Args:
        text1: First text (original)
        text2: Second text (modified)
        name1: Label for first text
        name2: Label for second text
        
    Returns:
        Tuple of (unified_diff_string, stats_dict)
    """
    # Split texts into lines for better diff readability
    lines1 = text1.splitlines(keepends=True)
    lines2 = text2.splitlines(keepends=True)
    
    # Create unified diff
    diff = list(difflib.unified_diff(
        lines1, lines2,
        fromfile=name1,
        tofile=name2,
        lineterm=''
    ))
    
    # Calculate statistics
    added_lines = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
    removed_lines = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))
    
    stats = {
        'original_length': len(text1),
        'modified_length': len(text2),
        'original_lines': len(lines1),
        'modified_lines': len(lines2),
        'added_lines': added_lines,
        'removed_lines': removed_lines,
        'length_diff': len(text2) - len(text1),
        'has_changes': len(diff) > 0
    }
    
    return '\n'.join(diff), stats


def save_comparison_results(gold_doc, cleaned_doc, diff_text, stats, output_dir: Path, search_id: str):
    """
    Save comparison results to files.
    
    Args:
        gold_doc: Gold document data
        cleaned_doc: Cleaned document data  
        diff_text: Unified diff text
        stats: Comparison statistics
        output_dir: Output directory
        search_id: Search ID
    """
    output_dir.mkdir(exist_ok=True)
    
    # Create safe filename
    safe_id = "".join(c for c in search_id if c.isalnum() or c in '-_.')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save diff file
    diff_file = output_dir / f"diff_{safe_id}_{timestamp}.txt"
    with open(diff_file, 'w', encoding='utf-8') as f:
        f.write(f"Document Comparison Report\n")
        f.write(f"Document ID: {search_id}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"GOLD DATA:\n")
        f.write(f"  File: {gold_doc['found_in_file']}\n")
        f.write(f"  Length: {len(gold_doc['text']):,} characters\n\n")
        
        f.write(f"CLEANED DATA:\n")
        f.write(f"  File: {cleaned_doc['found_in_file']}\n")
        f.write(f"  Length: {len(cleaned_doc['text']):,} characters\n\n")
        
        f.write(f"COMPARISON STATISTICS:\n")
        f.write(f"  Length change: {stats['length_diff']:+,} characters\n")
        f.write(f"  Lines added: {stats['added_lines']}\n")
        f.write(f"  Lines removed: {stats['removed_lines']}\n")
        f.write(f"  Changes detected: {stats['has_changes']}\n\n")
        
        f.write("UNIFIED DIFF:\n")
        f.write("-" * 80 + "\n")
        f.write(diff_text)
    
    print(f"✓ Diff saved to: {diff_file}")
    
    # Save original texts for reference
    gold_file = output_dir / f"gold_{safe_id}_{timestamp}.txt"
    with open(gold_file, 'w', encoding='utf-8') as f:
        f.write(gold_doc['text'])
    
    cleaned_file = output_dir / f"cleaned_{safe_id}_{timestamp}.txt"
    with open(cleaned_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_doc['text'])
    
    print(f"✓ Original texts saved to: {gold_file.name}, {cleaned_file.name}")
    
    # Save metadata comparison
    if gold_doc['metadata'] or cleaned_doc['metadata']:
        metadata_file = output_dir / f"metadata_comparison_{safe_id}_{timestamp}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                'document_id': search_id,
                'comparison_date': datetime.now().isoformat(),
                'statistics': stats,
                'gold_metadata': gold_doc['metadata'],
                'cleaned_metadata': cleaned_doc['metadata'],
                'gold_file': gold_doc['found_in_file'],
                'cleaned_file': cleaned_doc['found_in_file']
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Metadata comparison saved to: {metadata_file}")


def print_comparison_summary(stats, diff_text):
    """Print a summary of the comparison to console."""
    
    print(f"\n" + "="*60)
    print(f"COMPARISON SUMMARY")
    print(f"="*60)
    
    if not stats['has_changes']:
        print(f"✓ NO CHANGES DETECTED - Documents are identical")
        return
    
    print(f"📊 STATISTICS:")
    print(f"   Original: {stats['original_length']:,} chars, {stats['original_lines']:,} lines")
    print(f"   Cleaned:  {stats['modified_length']:,} chars, {stats['modified_lines']:,} lines")
    print(f"   Change:   {stats['length_diff']:+,} chars ({stats['length_diff']/stats['original_length']*100:+.1f}%)")
    print(f"   Lines:    +{stats['added_lines']} added, -{stats['removed_lines']} removed")
    
    # Show preview of changes
    print(f"\n📝 DIFF PREVIEW (first 20 lines):")
    print(f"-" * 60)
    diff_lines = diff_text.split('\n')
    preview_lines = diff_lines[:20]
    for line in preview_lines:
        if line.startswith('+++') or line.startswith('---'):
            continue
        elif line.startswith('+'):
            print(f"\033[92m{line}\033[0m")  # Green for additions
        elif line.startswith('-'):
            print(f"\033[91m{line}\033[0m")  # Red for deletions
        elif line.startswith('@@'):
            print(f"\033[94m{line}\033[0m")  # Blue for context
        else:
            print(line)
    
    if len(diff_lines) > 20:
        print(f"... ({len(diff_lines)-20} more lines)")


def main():
    parser = argparse.ArgumentParser(
        description="Compare documents between gold and cleaned data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/compare_documents.py "part-1.parquet/4"
  python scripts/compare_documents.py "doc123" --gold-dir data/gold/ --cleaned-dir data/processed/
  python scripts/compare_documents.py "abc" --output comparisons/ --verbose
        """
    )
    
    parser.add_argument("id", help="Document ID to search for and compare")
    parser.add_argument(
        "--gold-dir",
        default="data/statistics_data_gold/enriched_documents_statistics_v2",
        help="Directory with gold/original parquet files"
    )
    parser.add_argument(
        "--cleaned-dir", 
        default="data/cleaned",
        help="Directory with cleaned parquet files"
    )
    parser.add_argument(
        "--output", "-o",
        default="document_comparisons",
        help="Output directory for comparison files"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed search progress"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save files, just show diff"
    )
    
    args = parser.parse_args()
    
    gold_dir = Path(args.gold_dir)
    cleaned_dir = Path(args.cleaned_dir)
    output_dir = Path(args.output)
    
    # Check directories exist
    if not gold_dir.exists():
        print(f"Error: Gold directory not found: {gold_dir}")
        return 1
    
    if not cleaned_dir.exists():
        print(f"Error: Cleaned directory not found: {cleaned_dir}")
        return 1
    
    print(f"Comparing document ID: '{args.id}'")
    print(f"Gold data: {gold_dir}")
    print(f"Cleaned data: {cleaned_dir}")
    print(f"Output: {output_dir}")
    print("-" * 60)
    
    # Search in both directories
    print("🔍 Searching in gold data...")
    gold_doc = search_in_directory(gold_dir, args.id, args.verbose)
    
    print("🔍 Searching in cleaned data...")
    cleaned_doc = search_in_directory(cleaned_dir, args.id, args.verbose)
    
    # Check if both documents were found
    if not gold_doc:
        print(f"\n✗ Document '{args.id}' not found in gold data directory")
        return 1
    
    if not cleaned_doc:
        print(f"\n✗ Document '{args.id}' not found in cleaned data directory")
        return 1
    
    print(f"\n✓ Document found in both directories!")
    
    # Create diff
    print("🔄 Comparing texts...")
    diff_text, stats = create_text_diff(
        gold_doc['text'], 
        cleaned_doc['text'],
        f"Gold ({Path(gold_doc['found_in_file']).name})",
        f"Cleaned ({Path(cleaned_doc['found_in_file']).name})"
    )
    
    # Show summary
    print_comparison_summary(stats, diff_text)
    
    # Save results
    if not args.no_save:
        print(f"\n💾 Saving comparison results...")
        save_comparison_results(gold_doc, cleaned_doc, diff_text, stats, output_dir, args.id)
        print(f"\n✓ Comparison complete!")
    
    return 0


if __name__ == "__main__":
    exit(main()) 