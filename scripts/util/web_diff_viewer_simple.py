#!/usr/bin/env python3
"""
Simple Web-based document diff viewer using difflib.HtmlDiff.
Clean, simple, reliable diff display.
"""

from flask import Flask, request, jsonify, make_response
import pyarrow.parquet as pq
import pandas as pd
import argparse
from pathlib import Path
import difflib
import logging
import html
import os
import threading
import time
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global configuration
config = {
    'gold_dir': None,
    'cleaned_dir': None,
    'document_cache': {},
    'available_docs': [],
    'loading_status': 'ready',
    'document_metadata': {}
}

# Cleaning-Methoden, die als Filter verwendet werden können
CLEANING_METHODS = [
    {'key': 'isolated_numeric_citations_citations_removed', 'label': 'Einzelne Nummern (1)'},
    {'key': 'semicolon_blocks_citations_removed', 'label': 'Semikolon-Listen'},
    {'key': 'figure_table_refs_citations_removed', 'label': 'Abbildungen/Tabellen'},
    {'key': 'consecutive_numeric_citations_citations_removed', 'label': 'Konsekutive Nummern'},
    {'key': 'eckige_klammern_numerisch_citations_removed', 'label': 'Eckige Klammern [1,2]'},
    {'key': 'autor_jahr_multi_klammer_citations_removed', 'label': 'Autor-Jahr (mehrere)'},
    {'key': 'autor_jahr_klammer_einzel_citations_removed', 'label': 'Autor-Jahr (einzeln)'},
    {'key': 'autor_jahr_eckig_einzel_citations_removed', 'label': 'Autor-Jahr [einzeln]'},
    {'key': 'autor_jahr_eckig_multi_citations_removed', 'label': 'Autor-Jahr [mehrere]'},
    {'key': 'citation_length_reduction', 'label': 'Zeichen entfernt'},
    {'key': 'citation_word_reduction', 'label': 'Wörter entfernt'},
    {'key': 'appendix_length_reduction', 'label': 'Anhang entfernt'},
    {'key': 'appendix_sections_removed', 'label': 'Anhang-Abschnitte'},
    {'key': 'figure_lines_removed', 'label': 'Abbildungszeilen'}
]


def quick_scan_documents(cleaned_dir: Path, gold_dir: Path, max_files: int = 50, max_docs: int = 1000):
    """Quick document discovery with metadata extraction."""
    logger.info(f"Quick scan starting...")
    config['loading_status'] = 'scanning'
    
    try:
        cleaned_files = list(cleaned_dir.glob("*.parquet"))[:max_files]
        gold_files = list(gold_dir.glob("*.parquet"))[:max_files]
        
        cleaned_ids = set()
        gold_ids = set()
        document_metadata = {}
        
        # Fast ID collection with metadata from cleaned files
        for file_path in cleaned_files:
            try:
                if file_path.stat().st_size == 0:
                    continue
                
                logger.info(f"Scanning cleaned file: {file_path}")
                
                # Read table with metadata columns
                table = pq.read_table(file_path)
                df = table.to_pandas()
                
                for _, row in df.iterrows():
                    doc_id = row['id']
                    if not doc_id:
                        continue
                    
                    cleaned_ids.add(doc_id)
                    
                    # Store metadata for citation stats
                    metadata = {}
                    
                    # Check if there's a 'metadata' column that contains JSON string
                    if 'metadata' in df.columns and not pd.isna(row['metadata']):
                        try:
                            import json
                            # Try to parse metadata as JSON
                            if isinstance(row['metadata'], str):
                                metadata = json.loads(row['metadata'])
                            elif isinstance(row['metadata'], dict):
                                metadata = row['metadata']
                            
                            # Debug: Log the metadata to check what we found
                            logger.info(f"Found metadata for {doc_id}: {str(metadata)[:100]}...")
                        except Exception as e:
                            logger.error(f"Error parsing metadata JSON: {e}: {str(row['metadata'])[:100]}")
                    
                    # If no metadata found or parsing failed, look for individual columns
                    if not metadata:
                        citation_cols = [col for col in df.columns if any(typ in col for typ in 
                                        ['citations_found', 'citations_removed', 'citations_rejected', 
                                        'length_reduction', 'had_'])]
                        
                        for col in citation_cols:
                            if col in row and not pd.isna(row[col]):
                                metadata[col] = row[col]
                    
                    document_metadata[doc_id] = metadata
                
                if len(cleaned_ids) > max_docs:
                    break
            except Exception as e:
                logger.error(f"Error reading cleaned file {file_path}: {e}")
                continue
        
        # ID collection from gold files
        for file_path in gold_files:
            try:
                if file_path.stat().st_size == 0:
                    continue
                
                logger.info(f"Scanning gold file: {file_path}")
                
                table = pq.read_table(file_path, columns=['id'])
                file_ids = [id for id in table['id'].to_pylist()[:100] if id]
                gold_ids.update(file_ids)
                if len(gold_ids) > max_docs:
                    break
            except Exception as e:
                logger.error(f"Error reading gold file {file_path}: {e}")
                continue
        
        # Find intersection
        available_doc_ids = list(cleaned_ids & gold_ids)
        logger.info(f"Found {len(available_doc_ids)} document IDs in both gold and cleaned")
        
        # Create document objects with metadata
        available_docs = []
        for doc_id in available_doc_ids[:max_docs]:
            doc_info = {
                'id': doc_id,
                'metadata': document_metadata.get(doc_id, {})
            }
            
            # Calculate total citations removed (all types)
            total_removed = 0
            for key, value in doc_info['metadata'].items():
                if '_citations_removed' in key:
                    try:
                        if isinstance(value, (int, float)):
                            total_removed += value
                        elif isinstance(value, str) and value.isdigit():
                            total_removed += int(value)
                    except (ValueError, TypeError):
                        pass
            
            doc_info['total_citations_removed'] = total_removed
            available_docs.append(doc_info)
        
        # Store in global config
        config['available_docs'] = available_docs
        config['document_metadata'] = document_metadata
        config['loading_status'] = 'complete'
        
        logger.info(f"Found {len(available_docs)} documents with metadata")
        return available_docs
        
    except Exception as e:
        logger.error(f"Scan error: {e}")
        config['loading_status'] = 'error'
        return []


def get_document_by_id(directory: Path, doc_id: str):
    """Simple document retrieval."""
    cache_key = f"{directory}:{doc_id}"
    if cache_key in config['document_cache']:
        return config['document_cache'][cache_key]
    
    logger.info(f"🔎 Searching for document {doc_id} in {directory}")
    parquet_files = list(directory.glob("*.parquet"))
    logger.info(f"Found {len(parquet_files)} parquet files in {directory}")
    
    for file_path in parquet_files:
        try:
            if file_path.stat().st_size == 0:
                continue
                
            # Versuche zuerst mit Filter (schneller wenn der Index funktioniert)
            try:
                table = pq.read_table(file_path, filters=[('id', '=', doc_id)])
                if len(table) > 0:
                    df = table.to_pandas()
                    text = df.iloc[0]['text']
                    config['document_cache'][cache_key] = text
                    logger.info(f"✅ Found document {doc_id} with filter in {file_path}")
                    return text
            except Exception as e:
                # Wenn der Filter nicht funktioniert, lese die gesamte Tabelle
                logger.warning(f"Filter failed, reading whole table for {file_path}: {e}")
                table = pq.read_table(file_path)
                df = table.to_pandas()
                
                # Suche nach der ID in der Tabelle
                if 'id' in df.columns:
                    matching_rows = df[df['id'] == doc_id]
                    if not matching_rows.empty:
                        text = matching_rows.iloc[0]['text']
                        config['document_cache'][cache_key] = text
                        logger.info(f"✅ Found document {doc_id} with full scan in {file_path}")
                        return text
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            continue
    
    logger.warning(f"❌ Document {doc_id} not found in {directory}")
    return None


# Hilfsfunktion für inline word-diff
def get_word_diff(old_text, new_text):
    """Erstelle einen Wort-für-Wort Diff zwischen zwei Texten."""
    # Text in Wörter aufteilen, Leerzeichen beibehalten
    pattern = r'(\s+|\S+)'
    old_words = re.findall(pattern, old_text)
    new_words = re.findall(pattern, new_text)
    
    # Diff auf Wortebene durchführen
    s = difflib.SequenceMatcher(None, old_words, new_words)
    
    # Ergebnisse für alte und neue Zeile
    old_result = []
    new_result = []
    
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'equal':
            old_result.extend(old_words[i1:i2])
            new_result.extend(new_words[j1:j2])
        elif tag == 'replace':
            old_result.append(f'<span class="word-deleted">{html.escape("".join(old_words[i1:i2]))}</span>')
            new_result.append(f'<span class="word-added">{html.escape("".join(new_words[j1:j2]))}</span>')
        elif tag == 'delete':
            old_result.append(f'<span class="word-deleted">{html.escape("".join(old_words[i1:i2]))}</span>')
        elif tag == 'insert':
            new_result.append(f'<span class="word-added">{html.escape("".join(new_words[j1:j2]))}</span>')
    
    return ''.join(old_result), ''.join(new_result)


def create_simple_diff(text1: str, text2: str, doc_id: str):
    """Create side-by-side Git-style diff with full document view."""
    
    if text1 == text2:
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Diff: {doc_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #f6f8fa; padding-bottom: 40px; }}
        .header {{ background: white; border-bottom: 1px solid #d1d9e0; padding: 16px 24px; position: sticky; top: 0; z-index: 10; }}
        .container {{ max-width: 1400px; margin: 20px auto; background: white; border-radius: 6px; border: 1px solid #d1d9e0; }}
        .file-header {{ background: #f6f8fa; padding: 8px 16px; border-bottom: 1px solid #d1d9e0; font-weight: 600; position: sticky; top: 60px; z-index: 5; }}
        .btn {{ padding: 8px 16px; background: #0969da; color: white; text-decoration: none; border-radius: 6px; margin-right: 8px; display: inline-block; }}
        .diff-content {{ padding: 16px; text-align: center; color: #656d76; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>{doc_id}</h2>
        <a href="/" class="btn">← Back</a>
        <a href="/raw/{doc_id}/gold" class="btn">Gold</a>
        <a href="/raw/{doc_id}/cleaned" class="btn">Cleaned</a>
    </div>
    <div class="container">
        <div class="file-header">No differences</div>
        <div class="diff-content">
            <p>Files are identical</p>
            <p>{len(text1):,} characters, {len(text1.splitlines()):,} lines</p>
        </div>
    </div>
</body>
</html>
        """
    
    # Get metadata for this document (if available)
    doc_metadata = config.get('document_metadata', {}).get(doc_id, {})
    
    # Create the diff using difflib
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()
    
    # Create line by line diff
    rows = []
    i1, i2 = 0, 0
    matcher = difflib.SequenceMatcher(None, lines1, lines2)
    
    for tag, i1start, i1end, i2start, i2end in matcher.get_opcodes():
        if tag == 'equal':
            for i in range(i1start, i1end):
                line1 = html.escape(lines1[i]).replace(" ", "<span class='space'> </span>").replace("\t", "<span class='tab'>\t</span>")
                rows.append(f"""<tr class="line"><td class="line-num">{i+1}</td><td class="content">{line1}</td><td class="line-num">{i+1}</td><td class="content">{line1}</td></tr>""")
            i1, i2 = i1end, i2end
        
        elif tag == 'replace':
            max_len = max(i1end - i1start, i2end - i2start)
            for i in range(max_len):
                i1idx = i1start + i if i1start + i < i1end else None
                i2idx = i2start + i if i2start + i < i2end else None
                
                if i1idx is not None and i2idx is not None:
                    # Wort-für-Wort-Diff für geänderte Zeilen
                    line1_raw = lines1[i1idx]
                    line2_raw = lines2[i2idx]
                    
                    # Nur Wort-Diff nutzen, wenn Zeilen ähnlich sind
                    if difflib.SequenceMatcher(None, line1_raw, line2_raw).ratio() > 0.5:
                        # Wort-für-Wort-Diff anwenden
                        line1, line2 = get_word_diff(line1_raw, line2_raw)
                    else:
                        # Andernfalls nur HTML-Escape und Leerzeichen-Formatierung
                        line1 = html.escape(line1_raw).replace(" ", "<span class='space'> </span>").replace("\t", "<span class='tab'>\t</span>")
                        line2 = html.escape(line2_raw).replace(" ", "<span class='space'> </span>").replace("\t", "<span class='tab'>\t</span>")
                    
                    rows.append(f"""<tr class="line changed"><td class="line-num">{i1idx+1}</td><td class="content old">{line1}</td><td class="line-num">{i2idx+1}</td><td class="content new">{line2}</td></tr>""")
                elif i1idx is not None:
                    line1 = html.escape(lines1[i1idx]).replace(" ", "<span class='space'> </span>").replace("\t", "<span class='tab'>\t</span>")
                    rows.append(f"""<tr class="line removed"><td class="line-num">{i1idx+1}</td><td class="content">{line1}</td><td class="line-num"></td><td class="content empty"></td></tr>""")
                elif i2idx is not None:
                    line2 = html.escape(lines2[i2idx]).replace(" ", "<span class='space'> </span>").replace("\t", "<span class='tab'>\t</span>")
                    rows.append(f"""<tr class="line added"><td class="line-num"></td><td class="content empty"></td><td class="line-num">{i2idx+1}</td><td class="content">{line2}</td></tr>""")
            
            i1, i2 = i1end, i2end
        
        elif tag == 'delete':
            for i in range(i1start, i1end):
                line1 = html.escape(lines1[i]).replace(" ", "<span class='space'> </span>").replace("\t", "<span class='tab'>\t</span>")
                rows.append(f"""<tr class="line removed"><td class="line-num">{i+1}</td><td class="content">{line1}</td><td class="line-num"></td><td class="content empty"></td></tr>""")
            i1 = i1end
        
        elif tag == 'insert':
            for i in range(i2start, i2end):
                line2 = html.escape(lines2[i]).replace(" ", "<span class='space'> </span>").replace("\t", "<span class='tab'>\t</span>")
                rows.append(f"""<tr class="line added"><td class="line-num"></td><td class="content empty"></td><td class="line-num">{i+1}</td><td class="content">{line2}</td></tr>""")
            i2 = i2end

    # Prepare metadata badges
    metadata_badges = []
    
    # Document info
    if 'title' in doc_metadata:
        metadata_badges.append(f'<div class="meta-item"><span class="meta-label">Titel:</span> {doc_metadata.get("title", "")}</div>')
    
    if 'authors' in doc_metadata:
        metadata_badges.append(f'<div class="meta-item"><span class="meta-label">Autoren:</span> {doc_metadata.get("authors", "")}</div>')
    
    # Citation stats
    citation_types = {
        'semicolon_blocks': 'Semikolon-Listen',
        'eckige_klammern_numerisch': 'Eckige Klammern [1,2]',
        'consecutive_numeric_citations': 'Konsekutive Nummern (1)(2)',
        'isolated_numeric_citations': 'Einzelne Nummern (1)',
        'autor_jahr_multi_klammer': 'Autor-Jahr (Smith, 2020; Jones, 2021)',
        'autor_jahr_klammer_einzel': 'Autor-Jahr (Smith, 2020)',
        'autor_jahr_eckig_einzel': 'Autor-Jahr [Smith, 2020]',
        'autor_jahr_eckig_multi': 'Autor-Jahr [Smith, 2020; Jones, 2021]',
        'ref_nummer': 'Referenz-Nummern (ref. 1)',
        'page_references': 'Seitenzahlen (p. 123)',
        'figure_table_refs': 'Abbildungen/Tabellen (Fig. 1)'
    }
    
    # Add summary metadata
    length_reduction = doc_metadata.get('citation_length_reduction', 0)
    word_reduction = doc_metadata.get('citation_word_reduction', 0)
    
    # Convert to integers if they are strings
    try:
        if isinstance(length_reduction, str):
            length_reduction = int(length_reduction)
        if isinstance(word_reduction, str):
            word_reduction = int(word_reduction)
    except ValueError:
        pass  # If conversion fails, keep as is
    
    if length_reduction or word_reduction:
        metadata_badges.append(f'<span class="meta-badge"><span class="meta-label">Zeichen:</span> {length_reduction:,}</span>')
        metadata_badges.append(f'<span class="meta-badge"><span class="meta-label">Wörter:</span> {word_reduction:,}</span>')
    
    # Add citation type stats
    for type_name, display_name in citation_types.items():
        # Get citation count, handle string values
        removed_key = f"{type_name}_citations_removed"
        if removed_key in doc_metadata:
            removed = doc_metadata[removed_key]
            # Convert to integer if it's a string
            if isinstance(removed, str):
                try:
                    removed = int(removed)
                except ValueError:
                    removed = 0
        else:
            removed = 0
        
        if removed > 0:
            metadata_badges.append(f'<span class="meta-badge"><span class="meta-label">{display_name}:</span> {removed:,}</span>')
    
    # Add metadata section to HTML
    meta_section = ""
    if metadata_badges:
        meta_section = f"""
        <div class="metadata-bar">
            <div class="meta-title">
                {metadata_badges[0] if len(metadata_badges) > 0 else ''}
                {metadata_badges[1] if len(metadata_badges) > 1 else ''}
            </div>
            <div class="meta-badges">
                {''.join(metadata_badges[2:]) if len(metadata_badges) > 2 else ''}
            </div>
        </div>
        """
    
    # Complete HTML structure
    html_output = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Diff: {html.escape(doc_id)}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #f6f8fa; padding-bottom: 40px; }}
        .header {{ background: white; border-bottom: 1px solid #d1d9e0; padding: 16px 24px; position: sticky; top: 0; z-index: 10; }}
        .container {{ max-width: 1400px; margin: 20px auto; background: white; border-radius: 6px; border: 1px solid #d1d9e0; }}
        .file-header {{ background: #f6f8fa; padding: 8px 16px; border-bottom: 1px solid #d1d9e0; font-weight: 600; position: sticky; top: 60px; z-index: 5; }}
        .btn {{ padding: 8px 16px; background: #0969da; color: white; text-decoration: none; border-radius: 6px; margin-right: 8px; display: inline-block; }}
        
        /* Diff table styling */
        .diff {{ width: 100%; border-collapse: collapse; font-family: SFMono-Regular, Consolas, Liberation Mono, Menlo, monospace; font-size: 12px; }}
        .diff td {{ padding: 0; vertical-align: top; }}
        .diff td.line-num {{ text-align: right; color: rgba(27, 31, 35, 0.3); padding: 0 5px; width: 30px; user-select: none; }}
        .diff tr.line {{ height: 20px; }}
        .diff tr:hover {{ background-color: #f8fafd; }}
        
        /* Line content */
        .diff td.content {{ white-space: pre-wrap; width: calc(50% - 30px); padding: 0 10px; }}
        .diff td.content.empty {{ background-color: #fafbfc; }}
        
        /* Colors for changes */
        .diff tr.added td.content {{ background-color: #e6ffec; }}
        .diff tr.removed td.content {{ background-color: #ffebe9; }}
        .diff tr.changed td.content.old {{ background-color: #ffebe9; }}
        .diff tr.changed td.content.new {{ background-color: #e6ffec; }}
        
        /* Word-level diff styling */
        .word-deleted {{ background-color: #fdb8c0; color: #82071e; border-radius: 2px; padding: 1px 0; }}
        .word-added {{ background-color: #abf2bc; color: #116329; border-radius: 2px; padding: 1px 0; }}
        
        /* Metadata styling */
        .metadata-bar {{ padding: 10px 16px; border-bottom: 1px solid #d1d9e0; background: #f6f8fa; }}
        .meta-title {{ margin-bottom: 8px; font-size: 14px; }}
        .meta-badges {{ display: flex; flex-wrap: wrap; gap: 8px; }}
        .meta-badge {{ display: inline-block; padding: 3px 8px; background: #ddf4ff; color: #0969da; border-radius: 12px; font-size: 12px; }}
        .meta-item {{ margin-bottom: 5px; }}
        .meta-label {{ font-weight: 600; }}
        
        /* Typography */
        h2 {{ margin: 0; color: #24292e; }}
        
        /* Special characters highlighting */
        .space {{ background-color: rgba(0, 0, 255, 0.05); position: relative; }}
        .space::after {{ content: "·"; position: absolute; color: rgba(0, 0, 255, 0.3); }}
        .tab {{ background-color: rgba(255, 0, 0, 0.05); position: relative; }}
        .tab::after {{ content: "→"; position: absolute; color: rgba(255, 0, 0, 0.3); }}
    </style>
</head>
<body>
    <div class="header">
        <h2>{html.escape(doc_id)}</h2>
        <a href="/" class="btn">← Back</a>
        <a href="/raw/{html.escape(doc_id)}/gold" class="btn">Gold</a>
        <a href="/raw/{html.escape(doc_id)}/cleaned" class="btn">Cleaned</a>
        </div>
    <div class="container">
        <div class="file-header">
            <span>Gold</span>
            <span style="float: right;">Cleaned</span>
            </div>
        {meta_section}
        <table class="diff">{''.join(rows)}</table>
    </div>
</body>
</html>
    """
    
    return html_output


@app.route('/')
def index():
    """Main page with documents list."""
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Document Diff Viewer</title>
    <meta charset="utf-8">
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
            margin: 0;
            background: #f6f8fa;
            line-height: 1.5;
        }
        
        .header {
            background: white;
            border-bottom: 1px solid #d1d9e0;
            padding: 16px 24px;
        }
        
        .container {
            max-width: 1200px;
            margin: 20px auto;
            background: white;
            border-radius: 6px;
            border: 1px solid #d1d9e0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .top-controls {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            padding: 16px;
            border-bottom: 1px solid #d1d9e0;
            background: #f6f8fa;
            align-items: center;
        }
        
        .filter-section {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            padding: 10px 16px;
            border-bottom: 1px solid #d1d9e0;
            background: #f6f8fa;
            align-items: center;
        }
        
        .filter-label {
            font-weight: 600;
            font-size: 14px;
            margin-right: 5px;
        }
        
        select, input {
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #d1d9e0;
            background: white;
        }
        
        .status {
            padding: 16px;
            border-bottom: 1px solid #d1d9e0;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th {
            text-align: left;
            padding: 12px 16px;
            border-bottom: 1px solid #d1d9e0;
            background: #f6f8fa;
            font-weight: 600;
            position: sticky;
            top: 0;
            z-index: 1;
        }
        
        td {
            padding: 12px 16px;
            border-bottom: 1px solid #eaecef;
            vertical-align: top;
        }
        
        tr:hover {
            background: #f6f8fa;
        }
        
        a {
            color: #0969da;
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        
        .document-link {
            font-weight: 500;
        }
        
        .stats-badge {
            display: inline-flex;
            align-items: center;
            padding: 3px 8px;
            background: #e6f2ff;
            color: #0969da;
            border-radius: 16px;
            font-size: 12px;
            margin-right: 6px;
            margin-bottom: 6px;
            white-space: nowrap;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        }
        
        .stats-badge.highlight {
            background: #0969da;
            color: white;
            font-weight: 500;
        }
        
        .badge-group {
            display: flex;
            flex-wrap: wrap;
            margin-top: 8px;
            gap: 4px;
        }
        
        .pagination {
            padding: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-top: 1px solid #d1d9e0;
        }
        
        .btn { 
            padding: 8px 16px;
            background: #0969da;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
        }
        
        .btn-secondary {
            background: #f6f8fa;
            color: #24292f;
            border: 1px solid #d0d7de;
        }
        
        .loading {
            display: flex;
            padding: 24px;
            justify-content: center;
            align-items: center;
            color: #57606a;
        }
        
        .spinner {
            border: 4px solid rgba(0, 0, 0, 0.1);
            width: 24px;
            height: 24px;
            border-radius: 50%;
            border-left-color: #0969da;
            animation: spin 1s linear infinite;
            margin-right: 12px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .truncate {
            max-width: 500px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            display: inline-block;
        }
        
        .sort-icon {
            display: inline-block;
            width: 0;
            height: 0;
            margin-left: 4px;
            vertical-align: middle;
        }
        
        .sort-icon.asc {
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-bottom: 4px solid #24292f;
        }
        
        .sort-icon.desc {
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 4px solid #24292f;
        }
        
        .hidden {
            display: none;
        }
        
        .table-container {
            overflow-x: auto;
        }
        
        .badge-citation {
            background-color: #ddf4ff;
        }
        
        .badge-length {
            background-color: #ffefb7;
            color: #9a6700;
        }
        
        .badge-word {
            background-color: #dafbe1;
            color: #116329;
        }
        
        .badge-figure {
            background-color: #ffddee;
            color: #bf3989;
        }
        
        .badge-appendix {
            background-color: #ddd;
            color: #444;
        }
        
        .document-title {
            font-weight: 600;
            margin-bottom: 3px;
            color: #24292f;
        }
        
        .document-authors {
            color: #57606a;
            margin-bottom: 8px;
            font-size: 13px;
        }
        
        /* Mobile responsive */
        @media (max-width: 768px) {
            .top-controls, .filter-section {
                flex-direction: column;
                align-items: stretch;
            }
            
            .truncate {
                max-width: 300px;
            }
            
            td, th {
                padding: 8px 12px;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Document Diff Viewer</h1>
    </div>
    
    <div class="container">
        <div class="top-controls">
            <div style="display: flex; gap: 8px; align-items: center;">
                <span class="filter-label">Sortieren nach:</span>
                <select id="sort-by">
                    <option value="id">Dokument ID</option>
                    <option value="citation_length_reduction">Zeichen entfernt</option>
                    <option value="citation_word_reduction">Wörter entfernt</option>
                    <option value="total_citations_removed">Zitierungen entfernt</option>
                </select>
                
                <select id="sort-order">
                    <option value="desc">Absteigend</option>
                    <option value="asc">Aufsteigend</option>
                </select>
        </div>
        
            <input type="text" id="search-input" placeholder="Suche nach ID, Titel oder Autoren..." style="flex-grow: 1;">
            
            <button class="btn btn-secondary" onclick="reloadDocuments()">Aktualisieren</button>
        </div>
        
        <div class="filter-section">
            <span class="filter-label">Filter nach Cleaning-Methode:</span>
            <select id="cleaning-method-filter">
                <option value="">Alle Methoden</option>
            </select>
            <input type="number" id="min-value" placeholder="Min. Wert" style="width: 100px;">
            <button class="btn btn-secondary" onclick="applyFilter()">Filter anwenden</button>
            <button class="btn btn-secondary" onclick="resetFilter()">Zurücksetzen</button>
        </div>
        
        <div id="status" class="status">
            <div id="loading" class="loading hidden">
                <div class="spinner"></div>
                <div>Dokumente werden geladen...</div>
            </div>
            <div id="status-text"></div>
        </div>
        
        <div class="table-container">
            <table id="documents-table">
                <thead>
                    <tr>
                        <th style="width: 20%;">Dokument ID</th>
                        <th>Metadaten</th>
                        <th style="width: 150px;">Aktionen</th>
                    </tr>
                </thead>
                <tbody id="documents-list">
                    <!-- Documents will be listed here -->
                </tbody>
            </table>
        </div>
        
        <div class="pagination">
            <div id="pagination-info">Dokumente werden geladen...</div>
        </div>
    </div>

    <script>
        // Cleaning-Methoden für Filter-Dropdown
        const cleaningMethods = [
            {key: 'isolated_numeric_citations_citations_removed', label: 'Einzelne Nummern (1)', type: 'citation'},
            {key: 'semicolon_blocks_citations_removed', label: 'Semikolon-Listen', type: 'citation'},
            {key: 'figure_table_refs_citations_removed', label: 'Abbildungen/Tabellen', type: 'citation'},
            {key: 'consecutive_numeric_citations_citations_removed', label: 'Konsekutive Nummern', type: 'citation'},
            {key: 'eckige_klammern_numerisch_citations_removed', label: 'Eckige Klammern [1,2]', type: 'citation'},
            {key: 'autor_jahr_multi_klammer_citations_removed', label: 'Autor-Jahr (mehrere)', type: 'citation'},
            {key: 'autor_jahr_klammer_einzel_citations_removed', label: 'Autor-Jahr (einzeln)', type: 'citation'},
            {key: 'autor_jahr_eckig_einzel_citations_removed', label: 'Autor-Jahr [einzeln]', type: 'citation'},
            {key: 'autor_jahr_eckig_multi_citations_removed', label: 'Autor-Jahr [mehrere]', type: 'citation'},
            {key: 'citation_length_reduction', label: 'Zeichen entfernt', type: 'length'},
            {key: 'citation_word_reduction', label: 'Wörter entfernt', type: 'word'},
            {key: 'appendix_length_reduction', label: 'Anhang entfernt', type: 'appendix'},
            {key: 'appendix_sections_removed', label: 'Anhang-Abschnitte', type: 'appendix'},
            {key: 'figure_lines_removed', label: 'Abbildungszeilen', type: 'figure'}
        ];
        
        // Dropdown füllen
        const methodSelect = document.getElementById('cleaning-method-filter');
        cleaningMethods.forEach(method => {
            const option = document.createElement('option');
            option.value = method.key;
            option.textContent = method.label;
            methodSelect.appendChild(option);
        });
        
        let currentPage = 1;
        const pageSize = 20;
        let allDocuments = [];
        let filteredDocuments = [];
        
        // Aktualisieren der Tabelle
        function updateTable() {
            const tableBody = document.getElementById('documents-list');
            tableBody.innerHTML = '';
            
            const start = (currentPage - 1) * pageSize;
            const end = Math.min(start + pageSize, filteredDocuments.length);
            
            for (let i = start; i < end; i++) {
                const doc = filteredDocuments[i];
                const row = document.createElement('tr');
                
                // ID-Zelle
                const idCell = document.createElement('td');
                const docLink = document.createElement('a');
                docLink.href = `/diff/${doc.id}`;
                docLink.className = 'document-link';
                docLink.textContent = doc.id;
                idCell.appendChild(docLink);
                row.appendChild(idCell);
                
                // Metadaten-Zelle
                const metaCell = document.createElement('td');
                
                // Titel und Autoren
                if (doc.metadata.title) {
                    const titleSpan = document.createElement('div');
                    titleSpan.className = 'document-title truncate';
                    titleSpan.textContent = doc.metadata.title;
                    titleSpan.title = doc.metadata.title;
                    metaCell.appendChild(titleSpan);
                }
                
                if (doc.metadata.authors) {
                    const authorSpan = document.createElement('div');
                    authorSpan.className = 'document-authors truncate';
                    authorSpan.textContent = doc.metadata.authors;
                    authorSpan.title = doc.metadata.authors;
                    metaCell.appendChild(authorSpan);
                }
                
                // Badges für wichtige Metriken
                const badgesDiv = document.createElement('div');
                badgesDiv.className = 'badge-group';
                
                // Zeichen/Wörter entfernt als größere Badges
                const lengthReduction = parseInt(doc.metadata.citation_length_reduction || 0);
                if (lengthReduction > 0) {
                    const lengthBadge = document.createElement('span');
                    lengthBadge.className = 'stats-badge badge-length';
                    lengthBadge.textContent = `${lengthReduction} Zeichen`;
                    badgesDiv.appendChild(lengthBadge);
                }
                
                const wordReduction = parseInt(doc.metadata.citation_word_reduction || 0);
                if (wordReduction > 0) {
                    const wordBadge = document.createElement('span');
                    wordBadge.className = 'stats-badge badge-word';
                    wordBadge.textContent = `${wordReduction} Wörter`;
                    badgesDiv.appendChild(wordBadge);
                }
                
                // Zitierungen
                if (doc.total_citations_removed > 0) {
                    const citationBadge = document.createElement('span');
                    citationBadge.className = 'stats-badge badge-citation';
                    citationBadge.textContent = `${doc.total_citations_removed} Zitierungen`;
                    badgesDiv.appendChild(citationBadge);
                }
                
                // Spezifische Zitierungstypen
                cleaningMethods.forEach(method => {
                    const count = parseInt(doc.metadata[method.key] || 0);
                    if (count > 0) {
                        const badge = document.createElement('span');
                        badge.className = `stats-badge badge-${method.type}`;
                        badge.textContent = `${method.label}: ${count}`;
                        badgesDiv.appendChild(badge);
                    }
                });
                
                metaCell.appendChild(badgesDiv);
                row.appendChild(metaCell);
                
                // Aktionen-Zelle
                const actionsCell = document.createElement('td');
                const viewBtn = document.createElement('a');
                viewBtn.href = `/diff/${doc.id}`;
                viewBtn.className = 'btn';
                viewBtn.textContent = 'Diff anzeigen';
                actionsCell.appendChild(viewBtn);
                
                row.appendChild(actionsCell);
                tableBody.appendChild(row);
            }
            
            // Pagination aktualisieren
            const paginationInfo = document.getElementById('pagination-info');
            if (filteredDocuments.length === 0) {
                paginationInfo.textContent = 'Keine Dokumente gefunden';
            } else {
                paginationInfo.textContent = `Zeige ${start + 1} bis ${end} von ${filteredDocuments.length} Dokumenten`;
            }
        }
        
        // Dokumente laden
        function loadDocuments() {
            document.getElementById('loading').classList.remove('hidden');
            document.getElementById('status-text').textContent = '';
            
            const sortBy = document.getElementById('sort-by').value;
            const sortOrder = document.getElementById('sort-order').value;
            
            fetch(`/api/documents?sort_by=${sortBy}&sort_order=${sortOrder}`)
                .then(response => response.json())
                .then(data => {
                        allDocuments = data.documents;
                filteredDocuments = [...allDocuments];
                updateTable();
                document.getElementById('loading').classList.add('hidden');
                })
                .catch(error => {
                console.error('Error:', error);
                document.getElementById('loading').classList.add('hidden');
                document.getElementById('status-text').textContent = 'Fehler beim Laden der Dokumente';
            });
        }
        
        // Filter anwenden
        function applyFilter() {
            const method = document.getElementById('cleaning-method-filter').value;
            const minValue = parseInt(document.getElementById('min-value').value || 0);
            
            if (!method) {
                filteredDocuments = [...allDocuments];
            } else {
                filteredDocuments = allDocuments.filter(doc => {
                    const value = parseInt(doc.metadata[method] || 0);
                    return value >= minValue;
                });
            }
            
            currentPage = 1;
            updateTable();
        }
        
        // Filter zurücksetzen
        function resetFilter() {
            document.getElementById('cleaning-method-filter').value = '';
            document.getElementById('min-value').value = '';
            filteredDocuments = [...allDocuments];
            currentPage = 1;
            updateTable();
        }
        
        // Dokumente neu laden
        function reloadDocuments() {
            loadDocuments();
        }
        
        // Suche
        document.getElementById('search-input').addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase();
            
            if (!searchTerm) {
                filteredDocuments = [...allDocuments];
            } else {
                filteredDocuments = allDocuments.filter(doc => 
                    doc.id.toLowerCase().includes(searchTerm) ||
                    (doc.metadata.title && doc.metadata.title.toLowerCase().includes(searchTerm)) ||
                    (doc.metadata.authors && doc.metadata.authors.toLowerCase().includes(searchTerm))
                );
            }
            
            currentPage = 1;
            updateTable();
        });
        
        // Sortierung ändern
        document.getElementById('sort-by').addEventListener('change', loadDocuments);
        document.getElementById('sort-order').addEventListener('change', loadDocuments);
        
        // Status prüfen und Dokumente laden
        function checkStatus() {
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => {
                    if (data.status === 'scanning') {
                        document.getElementById('loading').classList.remove('hidden');
                        document.getElementById('status-text').textContent = 'Dokumente werden gescannt...';
                        setTimeout(checkStatus, 1000);
                    } else {
                        document.getElementById('loading').classList.add('hidden');
                        loadDocuments();
                    }
                });
        }
        
        // Beim Laden starten
        checkStatus();
    </script>
</body>
</html>
    """
    return html


@app.route('/api/documents')
def api_documents():
    """API endpoint to get documents list."""
    if config['loading_status'] == 'ready' and not config['available_docs']:
        # Trigger background scan if no documents loaded yet
        def background_scan():
            gold_dir = Path(config['gold_dir']) if isinstance(config['gold_dir'], str) else config['gold_dir']
            cleaned_dir = Path(config['cleaned_dir']) if isinstance(config['cleaned_dir'], str) else config['cleaned_dir']
            quick_scan_documents(cleaned_dir, gold_dir)
        
        threading.Thread(target=background_scan).start()
        config['loading_status'] = 'scanning'
    
    # Handle sorting and filtering
    sort_by = request.args.get('sort_by', 'id')
    sort_order = request.args.get('sort_order', 'asc')
    citation_type = request.args.get('citation_type', '')
    
    docs = config['available_docs']
    
    # Filter by citation type if specified
    if citation_type:
        key = f"{citation_type}_citations_removed"
        docs = [doc for doc in docs if doc['metadata'].get(key, 0) > 0]
    
    # Sort the documents
    if sort_by == 'id':
        docs = sorted(docs, key=lambda d: d['id'], reverse=(sort_order == 'desc'))
    elif sort_by == 'total_citations_removed':
        docs = sorted(docs, key=lambda d: d.get('total_citations_removed', 0), reverse=(sort_order == 'desc'))
    else:
        # Sort by a specific metadata field
        def get_metadata_value(doc, key):
            value = doc['metadata'].get(key, 0)
            if isinstance(value, str) and value.lstrip('-').isdigit():
                return int(value)
            return 0
        
        docs = sorted(docs, key=lambda d: get_metadata_value(d, sort_by), reverse=(sort_order == 'desc'))
    
    # Return the response
    response = jsonify({
        'documents': docs,
        'total': len(docs),
        'status': config['loading_status']
    })
    return response


@app.route('/api/status')
def api_status():
    """API endpoint for status."""
    return jsonify({
        'success': True,
        'status': config['loading_status'],
        'loaded_count': len(config['available_docs'])
    })


@app.route('/diff/<path:doc_id>')
def show_diff(doc_id):
    """Show GitHub-style diff with citation metadata."""
    try:
        logger.info(f"📄 Creating diff for: {doc_id}")
        
        # Load documents
        gold_dir = config['gold_dir']
        cleaned_dir = config['cleaned_dir']
        
        gold_text = get_document_by_id(gold_dir, doc_id)
        cleaned_text = get_document_by_id(cleaned_dir, doc_id)
        
        if not gold_text:
            return f"<h1>Error</h1><p>Document '{doc_id}' not found in gold dataset</p>", 404
        
        if not cleaned_text:
            return f"<h1>Error</h1><p>Document '{doc_id}' not found in cleaned dataset</p>", 404
        
        # Find document metadata
        doc_metadata = {}
        for doc in config['available_docs']:
            if isinstance(doc, dict) and doc.get('id') == doc_id:
                doc_metadata = doc.get('metadata', {})
                break
        
        # Create diff with metadata
        html_diff = create_simple_diff(gold_text, cleaned_text, doc_id)
        
        # Add metadata section to HTML
        if doc_metadata:
            # Format citation types
            citation_types = {
                'semicolon_blocks': 'Semikolon-Listen',
                'eckige_klammern_numerisch': 'Eckige Klammern [1,2]',
                'consecutive_numeric_citations': 'Konsekutive Nummern (1)(2)',
                'isolated_numeric_citations': 'Einzelne Nummern (1)',
                'autor_jahr_multi_klammer': 'Autor-Jahr (Smith, 2020; Jones, 2021)',
                'autor_jahr_klammer_einzel': 'Autor-Jahr (Smith, 2020)',
                'autor_jahr_eckig_einzel': 'Autor-Jahr [Smith, 2020]',
                'autor_jahr_eckig_multi': 'Autor-Jahr [Smith, 2020; Jones, 2021]',
                'ref_nummer': 'Referenz-Nummern (ref. 1)',
                'page_references': 'Seitenzahlen (p. 123)',
                'figure_table_refs': 'Abbildungen/Tabellen (Fig. 1)'
            }
            
            # Create metadata table
            metadata_rows = []
            
            # Add document info
            if 'title' in doc_metadata:
                metadata_rows.append(f'<tr><td><strong>Titel:</strong></td><td colspan="3">{doc_metadata.get("title", "")}</td></tr>')
                
            if 'authors' in doc_metadata:
                metadata_rows.append(f'<tr><td><strong>Autoren:</strong></td><td colspan="3">{doc_metadata.get("authors", "")}</td></tr>')
            
            # Add summary metadata
            length_reduction = doc_metadata.get('citation_length_reduction', 0)
            word_reduction = doc_metadata.get('citation_word_reduction', 0)
            
            # Convert to integers if they are strings
            try:
                if isinstance(length_reduction, str):
                    length_reduction = int(length_reduction)
                if isinstance(word_reduction, str):
                    word_reduction = int(word_reduction)
            except ValueError:
                pass  # If conversion fails, keep as is
            
            if length_reduction or word_reduction:
                metadata_rows.append(f'<tr><td><strong>Zeichen entfernt:</strong></td><td>{length_reduction:,}</td><td><strong>Wörter entfernt:</strong></td><td>{word_reduction:,}</td></tr>')
            
            # Add citation type stats
            for type_name, display_name in citation_types.items():
                # Get citation count, handle string values
                removed_key = f"{type_name}_citations_removed"
                if removed_key in doc_metadata:
                    removed = doc_metadata[removed_key]
                    # Convert to integer if it's a string
                    if isinstance(removed, str):
                        try:
                            removed = int(removed)
                        except ValueError:
                            removed = 0
                else:
                    removed = 0
                
                if removed > 0:
                    metadata_rows.append(f'<tr><td><strong>{display_name}:</strong></td><td>{removed:,} entfernt</td><td colspan="2"></td></tr>')
            
            if metadata_rows:
                metadata_html = f'''
                <div class="metadata-container">
                    <h3>Zitierungsmetadaten</h3>
                    <table class="metadata-table">
                        {''.join(metadata_rows)}
                    </table>
                </div>
                '''
                
                # Insert metadata section after header
                html_diff = html_diff.replace('</div>\n    \n    <div class="container">', 
                                           f'</div>\n    \n    {metadata_html}\n    <div class="container">')
                
                # Add CSS for metadata
                metadata_css = '''
                .metadata-container {
                    max-width: 98%;
                    margin: 10px auto;
                    background: white;
                    border-radius: 6px;
                    border: 1px solid #d1d9e0;
                    padding: 16px;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
                }
                
                .metadata-container h3 {
                    margin-top: 0;
                    margin-bottom: 12px;
                    color: #0969da;
                    font-size: 18px;
                }
                
                .metadata-table {
                    width: 100%;
                    border-collapse: collapse;
                }
                
                .metadata-table td {
                    padding: 8px 12px;
                    border-bottom: 1px solid #f0f0f0;
                }
                
                .metadata-table tr:last-child td {
                    border-bottom: none;
                }
                '''
                
                # Add CSS to the head section
                html_diff = html_diff.replace('</style>', f'{metadata_css}\n    </style>')
        
        response = make_response(html_diff)
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        return response
        
    except Exception as e:
        logger.error(f"Error creating diff for {doc_id}: {e}")
        return f"<h1>Error</h1><p>Error creating diff: {str(e)}</p>", 500


@app.route('/raw/<path:doc_id>/<doc_type>')
def show_raw(doc_id, doc_type):
    """Show raw document text."""
    try:
        if doc_type == 'gold':
            text = get_document_by_id(config['gold_dir'], doc_id)
        elif doc_type == 'cleaned':
            text = get_document_by_id(config['cleaned_dir'], doc_id)
        else:
            return "Invalid document type. Use 'gold' or 'cleaned'.", 400
        
        if not text:
            return f"Document '{doc_id}' not found in {doc_type} dataset", 404
        
        # Simple raw view
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{doc_type.title()}: {doc_id}</title>
            <meta charset="utf-8">
            <style>
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
                    margin: 0; background: #f6f8fa; 
                }}
                .header {{ 
                    background: white; border-bottom: 1px solid #d1d9e0; padding: 16px 24px; 
                    display: flex; justify-content: space-between; align-items: center;
                }}
                .container {{ 
                    max-width: 1200px; margin: 20px auto; background: white; 
                    border-radius: 6px; border: 1px solid #d1d9e0; padding: 20px;
                }}
                .content {{ 
                    font-family: 'SFMono-Regular', Consolas, monospace; 
                    white-space: pre-wrap; line-height: 1.45; font-size: 14px;
                }}
                .btn {{ 
                    padding: 6px 12px; background: #f6f8fa; color: #1f2328; text-decoration: none; 
                    border-radius: 6px; border: 1px solid #d1d9e0;
                }}
                .stats {{ margin-bottom: 16px; color: #656d76; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>📄 {doc_type.title()}: {doc_id}</h2>
                <div>
                    <a href="javascript:history.back()" class="btn">← Back</a>
                    <a href="/diff/{doc_id}" class="btn">View Diff</a>
                </div>
            </div>
            
            <div class="container">
                <div class="stats">
                    {len(text):,} characters, {len(text.splitlines()):,} lines, {len(text.split()):,} words
                </div>
                <div class="content">{html.escape(text)}</div>
            </div>
        </body>
        </html>
        """
        
        return html
        
    except Exception as e:
        logger.error(f"Error showing raw document {doc_id}: {e}")
        return f"<h1>Error</h1><p>Error loading document: {str(e)}</p>", 500


def start_ngrok_tunnel(port, auth_token=None):
    """Start ngrok tunnel."""
    try:
        import pyngrok
        from pyngrok import ngrok
        
        if auth_token:
            ngrok.set_auth_token(auth_token)
        
        public_url = ngrok.connect(port)
        logger.info(f"🌐 ngrok tunnel started: {public_url}")
        return public_url
        
    except ImportError:
        logger.error("pyngrok not installed. Install with: pip install pyngrok")
        return None
    except Exception as e:
        logger.error(f"Failed to start ngrok tunnel: {e}")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple Web-based document diff viewer")
    
    parser.add_argument("--gold-dir", default="data/statistics_data_gold/enriched_documents_statistics_v2")
    parser.add_argument("--cleaned-dir", default="data/cleaned")
    parser.add_argument("--port", "-p", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--ngrok", action="store_true")
    parser.add_argument("--auth-token", default=os.environ.get('NGROK_AUTH_TOKEN'))
    parser.add_argument("--max-files", type=int, default=100, help="Maximum number of parquet files to scan")
    parser.add_argument("--max-docs", type=int, default=2000, help="Maximum number of documents to load")
    
    args = parser.parse_args()
    
    # Konvertiere Strings in Path-Objekte
    config['gold_dir'] = Path(args.gold_dir)
    config['cleaned_dir'] = Path(args.cleaned_dir)
    
    if not config['gold_dir'].exists():
        print(f"❌ Gold directory not found: {config['gold_dir']}")
        exit(1)
    
    if not config['cleaned_dir'].exists():
        print(f"❌ Cleaned directory not found: {config['cleaned_dir']}")
        exit(1)
    
    print(f"🚀 Starting Simple Document Diff Viewer")
    print(f"Gold data: {config['gold_dir']}")
    print(f"Cleaned data: {config['cleaned_dir']}")
    print(f"Local server: http://{args.host}:{args.port}")
    
    if args.ngrok:
        print(f"\n🚀 Starting ngrok tunnel...")
        
        def start_tunnel():
            time.sleep(2)
            start_ngrok_tunnel(args.port, args.auth_token)
        
        tunnel_thread = threading.Thread(target=start_tunnel)
        tunnel_thread.daemon = True
        tunnel_thread.start()
    
    print(f"\n🚀 Starting server...")
    
    # Starte Scan im Hintergrund, damit die API sofort verfügbar ist
    def background_scan():
        time.sleep(1)
        print(f"\n🔍 Starting document scan in background...")
        quick_scan_documents(config['cleaned_dir'], config['gold_dir'], 
                             max_files=args.max_files, max_docs=args.max_docs)
    
    scan_thread = threading.Thread(target=background_scan)
    scan_thread.daemon = True
    scan_thread.start()
    
    try:
        app.run(host=args.host, port=args.port, debug=False)
    except KeyboardInterrupt:
        print(f"\n👋 Simple viewer shutting down...")