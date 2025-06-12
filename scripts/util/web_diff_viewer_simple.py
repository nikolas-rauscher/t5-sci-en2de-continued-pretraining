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
import random
from datatrove.pipeline.readers import ParquetReader
import json
import duckdb
import pickle
import hashlib

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
    'document_metadata': {},
    'parquet_index': {},
    'list_categories': {},
    'current_list_category': None
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


def quick_scan_documents(cleaned_dir: Path, gold_dir: Path, max_files: int = None, max_docs: int = None):
    """Scan all parquet files and all IDs for full search, wie im Random-Mode."""
    logger.info(f"Full scan (alle Dateien/IDs) starting...")
    config['loading_status'] = 'scanning'
    try:
        cleaned_files = list(cleaned_dir.glob("*.parquet"))
        gold_files = list(gold_dir.glob("*.parquet"))
        cleaned_ids = set()
        gold_ids = set()
        document_metadata = {}
        # Collect all cleaned IDs and metadata
        for file_path in cleaned_files:
            try:
                if file_path.stat().st_size == 0:
                    continue
                logger.info(f"Scanning cleaned file: {file_path}")
                table = pq.read_table(file_path)
                df = table.to_pandas()
                for _, row in df.iterrows():
                    doc_id = row['id']
                    if not doc_id:
                        continue
                    cleaned_ids.add(doc_id)
                    metadata = {}
                    if 'metadata' in df.columns and not pd.isna(row['metadata']):
                        try:
                            import json
                            if isinstance(row['metadata'], str):
                                metadata = json.loads(row['metadata'])
                            elif isinstance(row['metadata'], dict):
                                metadata = row['metadata']
                        except Exception:
                            pass
                    if not metadata:
                        citation_cols = [col for col in df.columns if any(typ in col for typ in ['citations_found', 'citations_removed', 'citations_rejected', 'length_reduction', 'had_'])]
                        for col in citation_cols:
                            if col in row and not pd.isna(row[col]):
                                metadata[col] = row[col]
                    document_metadata[doc_id] = metadata
            except Exception as e:
                logger.error(f"Error reading cleaned file {file_path}: {e}")
                continue
        # Collect all gold IDs
        for file_path in gold_files:
            try:
                if file_path.stat().st_size == 0:
                    continue
                logger.info(f"Scanning gold file: {file_path}")
                table = pq.read_table(file_path, columns=['id'])
                file_ids = [id for id in table['id'].to_pylist() if id]
                gold_ids.update(file_ids)
            except Exception as e:
                logger.error(f"Error reading gold file {file_path}: {e}")
                continue
        # Find intersection
        available_doc_ids = list(cleaned_ids & gold_ids)
        logger.info(f"Found {len(available_doc_ids)} document IDs in both gold and cleaned (full scan)")
        # Create document objects with metadata
        available_docs = []
        for doc_id in available_doc_ids:
            doc_info = {
                'id': doc_id,
                'metadata': document_metadata.get(doc_id, {})
            }
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
        config['available_docs'] = available_docs
        config['document_metadata'] = document_metadata
        config['loading_status'] = 'complete'
        logger.info(f"Full scan: {len(available_docs)} documents with metadata")
        return available_docs
    except Exception as e:
        logger.error(f"Full scan error: {e}")
        config['loading_status'] = 'error'
        return []


def get_document_by_id(directory: Path, doc_id: str):
    """Document retrieval using optimized binary search and DataTrove."""
    cache_key = f"{directory}:{doc_id}"
    if cache_key in config['document_cache']:
        return config['document_cache'][cache_key]
    
    logger.info(f"🔎 Searching for document {doc_id} in {directory}")
    
    # Index-Key für dieses Verzeichnis
    index_key = f"{directory}_index".replace("/", "_").replace("\\", "_")
    index = config.get(index_key)
    
    # Wenn kein Index, erstelle ihn
    if not index:
        logger.info(f"Erstelle fehlenden Index für {directory}")
        config[index_key] = build_optimized_document_index(directory)
        index = config[index_key]
    
    # Effiziente binäre Suche mit Divide-and-Conquer
    file_path_str = binary_search_document(index, doc_id)
    
    if file_path_str:
        file_path = ensure_valid_path_for_datatrove(file_path_str)
        logger.info(f"🔍 Binäre Suche: Dokument gefunden in {Path(file_path).name}")
        
        try:
            # Schnelle PyArrow-Suche mit Filter (schneller als DataTrove)
            doc_id_str = str(doc_id).strip()
            
            # Lese mit Filter
            table = pq.read_table(file_path_str, filters=[('id', '=', doc_id)])
            
            if len(table) > 0:
                df = table.to_pandas()
                # Exakter String-Vergleich
                matches = df[df['id'].astype(str).str.strip() == doc_id_str]
                
                if not matches.empty:
                    text = matches.iloc[0]['text']
                    config['document_cache'][cache_key] = text
                    logger.info(f"✅ PyArrow-Filter: Dokument gefunden in {Path(file_path).name}")
                    return text
            
            # Falls Filter nicht funktioniert, versuche DataTrove
            logger.warning(f"PyArrow-Filter fehlgeschlagen, verwende DataTrove")
            
            # Mit DataTrove ParquetReader laden
            from datatrove.pipeline.readers import ParquetReader
            logger.info(f"Lade Dokument mit DataTrove aus {file_path}")
            
            reader = ParquetReader(file_path)
            
            # Dokument in der Datei finden
            for doc in reader.run():
                if str(doc.id).strip() == doc_id_str:
                    if hasattr(doc, 'text') and doc.text:
                        text = doc.text
                        config['document_cache'][cache_key] = text
                        logger.info(f"✅ Dokument gefunden mit DataTrove")
                        return text
            
            # Wenn nicht gefunden mit DataTrove, Fallback auf PyArrow Vollscan
            logger.warning(f"DataTrove konnte Dokument {doc_id} nicht finden, verwende PyArrow Vollscan")
            
            # Falls Filter nicht funktioniert, lese spezifisch die wichtigen Zeilen
            table = pq.read_table(file_path_str)
            df = table.to_pandas()
            
            # Verschiedene Vergleichsmethoden
            matches = df[df['id'].astype(str).str.strip() == doc_id_str]
            
            if matches.empty:
                # Versuche ohne Leerzeichen
                matches = df[df['id'].astype(str).str.replace(" ", "") == doc_id_str.replace(" ", "")]
            
            if not matches.empty:
                text = matches.iloc[0]['text']
                config['document_cache'][cache_key] = text
                logger.info(f"✅ Dokument gefunden mit PyArrow Vollscan")
                return text
                
        except Exception as e:
            logger.error(f"Fehler beim Laden des Dokuments aus {file_path}: {e}")
    
    # Fallback: Traditionelle Suche in allen Dateien
    logger.warning(f"Binäre Suche fehlgeschlagen, verwende sequenzielle Suche")
    
    # Versuche mit DataTrove alle Dateien zu durchsuchen
    try:
        from datatrove.pipeline.readers import ParquetReader
        logger.info(f"Versuche sequenzielle Suche mit DataTrove")
        
        doc_id_str = str(doc_id).strip()
        parquet_files = list(directory.glob("*.parquet"))
        
        for file_path in parquet_files:
            if file_path.stat().st_size == 0:
                continue
                
            try:
                reader = ParquetReader(str(file_path))
                for doc in reader.run():
                    if str(doc.id).strip() == doc_id_str:
                        if hasattr(doc, 'text') and doc.text:
                            text = doc.text
                            config['document_cache'][cache_key] = text
                            logger.info(f"✅ Dokument gefunden mit DataTrove sequenziell in {file_path}")
                            return text
            except Exception as e:
                logger.error(f"DataTrove-Fehler bei {file_path}: {e}")
                continue
    except Exception as e:
        logger.error(f"Fehler beim DataTrove-Fallback: {e}")
    
    # Wenn DataTrove fehlschlägt, PyArrow als letzten Ausweg
    logger.warning(f"DataTrove-Suche fehlgeschlagen, verwende PyArrow als letzten Ausweg")
    parquet_files = list(directory.glob("*.parquet"))
    
    for file_path in parquet_files:
        try:
            if file_path.stat().st_size == 0:
                continue
                
            # Mit Filter versuchen
            try:
                table = pq.read_table(file_path, filters=[('id', '=', doc_id)])
                if len(table) > 0:
                    df = table.to_pandas()
                    match = df[df['id'].astype(str).str.strip() == str(doc_id).strip()]
                    if not match.empty:
                        text = match.iloc[0]['text']
                        config['document_cache'][cache_key] = text
                        logger.info(f"✅ Found document {doc_id} with PyArrow filter in {file_path}")
                        return text
            except Exception as e:
                # Wenn Filter nicht funktioniert, lese die gesamte Tabelle
                logger.warning(f"Filter failed, reading whole table for {file_path}: {e}")
                table = pq.read_table(file_path)
                df = table.to_pandas()
                # Robust: vergleiche als String und strip
                if 'id' in df.columns:
                    match = df[df['id'].astype(str).str.strip() == str(doc_id).strip()]
                    if not match.empty:
                        text = match.iloc[0]['text']
                        config['document_cache'][cache_key] = text
                        logger.info(f"✅ Found document {doc_id} with PyArrow full scan in {file_path}")
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
    """Create side-by-side Git-style diff with full document view. Always align both sides to the same number of rows. Show reduction in percent."""
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
    
    # Create line by line diff, synchronisiert
    rows = []
    matcher = difflib.SequenceMatcher(None, lines1, lines2)
    i1, i2 = 0, 0
    for tag, i1start, i1end, i2start, i2end in matcher.get_opcodes():
        max_len = max(i1end - i1start, i2end - i2start)
        for i in range(max_len):
            l_idx = i1start + i if i1start + i < i1end else None
            r_idx = i2start + i if i2start + i < i2end else None
            if tag == 'equal':
                line = html.escape(lines1[l_idx]).replace(" ", "<span class='space'> </span>").replace("\t", "<span class='tab'>\t</span>")
                rows.append(f"<tr class='line'><td class='line-num'>{l_idx+1}</td><td class='content'>{line}</td><td class='line-num'>{r_idx+1}</td><td class='content'>{line}</td></tr>")
            elif tag == 'replace':
                if l_idx is not None and r_idx is not None:
                    # Wort-für-Wort-Diff für geänderte Zeilen
                    line1_raw = lines1[l_idx]
                    line2_raw = lines2[r_idx]
                    if difflib.SequenceMatcher(None, line1_raw, line2_raw).ratio() > 0.5:
                        line1, line2 = get_word_diff(line1_raw, line2_raw)
                    else:
                        line1 = html.escape(line1_raw).replace(" ", "<span class='space'> </span>").replace("\t", "<span class='tab'>\t</span>")
                        line2 = html.escape(line2_raw).replace(" ", "<span class='space'> </span>").replace("\t", "<span class='tab'>\t</span>")
                    rows.append(f"<tr class='line changed'><td class='line-num'>{l_idx+1}</td><td class='content old'>{line1}</td><td class='line-num'>{r_idx+1}</td><td class='content new'>{line2}</td></tr>")
                elif l_idx is not None:
                    line1 = html.escape(lines1[l_idx]).replace(" ", "<span class='space'> </span>").replace("\t", "<span class='tab'>\t</span>")
                    rows.append(f"<tr class='line removed'><td class='line-num'>{l_idx+1}</td><td class='content'>{line1}</td><td class='line-num'></td><td class='content empty'></td></tr>")
                elif r_idx is not None:
                    line2 = html.escape(lines2[r_idx]).replace(" ", "<span class='space'> </span>").replace("\t", "<span class='tab'>\t</span>")
                    rows.append(f"<tr class='line added'><td class='line-num'></td><td class='content empty'></td><td class='line-num'>{r_idx+1}</td><td class='content'>{line2}</td></tr>")
            elif tag == 'delete':
                if l_idx is not None:
                    line1 = html.escape(lines1[l_idx]).replace(" ", "<span class='space'> </span>").replace("\t", "<span class='tab'>\t</span>")
                    rows.append(f"<tr class='line removed'><td class='line-num'>{l_idx+1}</td><td class='content'>{line1}</td><td class='line-num'></td><td class='content empty'></td></tr>")
                else:
                    rows.append(f"<tr class='line removed'><td class='line-num'></td><td class='content empty'></td><td class='line-num'></td><td class='content empty'></td></tr>")
            elif tag == 'insert':
                if r_idx is not None:
                    line2 = html.escape(lines2[r_idx]).replace(" ", "<span class='space'> </span>").replace("\t", "<span class='tab'>\t</span>")
                    rows.append(f"<tr class='line added'><td class='line-num'></td><td class='content empty'></td><td class='line-num'>{r_idx+1}</td><td class='content'>{line2}</td></tr>")
                else:
                    rows.append(f"<tr class='line added'><td class='line-num'></td><td class='content empty'></td><td class='line-num'></td><td class='content empty'></td></tr>")

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
        
        .btn-warning {
            background: #ffc107;
            color: #212529;
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
            <button class="btn btn-secondary" onclick="fullScanDocuments()">Vollsuche</button>
            <button class="btn btn-warning" onclick="loadRandomDocuments()">🎲 Random 100</button>
            <button class="btn btn-info" onclick="showListMode()">📋 Listen-Modus</button>
            <span id="mode" style="margin-left:16px;color:#bf3989;"></span>
        </div>
        
        <div id="listModeControls" class="list-mode-controls" style="display: none; margin: 20px 0; padding: 15px; background: #e8f4fd; border-radius: 4px;">
            <strong>📋 Listen-Modus:</strong>
            <select id="categorySelect" onchange="loadListCategory()" style="margin: 0 10px; padding: 5px;">
                <option value="">Kategorie auswählen...</option>
            </select>
            <button class="btn btn-secondary" onclick="exitListMode()">Exit Listen-Modus</button>
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
        
        function loadRandomDocuments() {
            document.getElementById('loading').classList.remove('hidden');
            document.getElementById('status-text').textContent = 'Lade 100 zufällige Dokumente...';
            document.getElementById('mode').textContent = '🎲 Random 100 mode - loading...';
            fetch('/api/random_documents?size=100')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        monitorRandomLoading();
                    } else {
                        document.getElementById('status-text').textContent = 'Fehler beim Laden der Zufallsdokumente';
                        document.getElementById('mode').textContent = '';
                    }
                })
                .catch(error => {
                    document.getElementById('status-text').textContent = 'Fehler beim Laden der Zufallsdokumente';
                    document.getElementById('mode').textContent = '';
                });
        }
        
        function monitorRandomLoading() {
            function checkStatus() {
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => {
                        if (data.status === 'complete') {
                            loadDocuments();
                            document.getElementById('mode').textContent = '🎲 Random 100 mode';
                        } else if (data.status === 'scanning') {
                            setTimeout(checkStatus, 500);
                        } else if (data.status === 'error') {
                            document.getElementById('status-text').textContent = 'Fehler beim Laden der Zufallsdokumente';
                            document.getElementById('mode').textContent = '';
                        }
                    })
                    .catch(() => setTimeout(checkStatus, 1000));
            }
            checkStatus();
        }
        
        // Suche
        document.getElementById('search-input').addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase();
            if (!searchTerm) {
                filteredDocuments = [...allDocuments];
                currentPage = 1;
                updateTable();
            } else {
                // Globale Suche über alle Parquet-Dateien
                document.getElementById('loading').classList.remove('hidden');
                document.getElementById('status-text').textContent = 'Globale Suche läuft...';
                fetch(`/api/search_documents?q=${encodeURIComponent(searchTerm)}`)
                    .then(response => response.json())
                    .then(data => {
                        filteredDocuments = data.documents;
                        currentPage = 1;
                        updateTable();
                        document.getElementById('loading').classList.add('hidden');
                        document.getElementById('status-text').textContent = data.total === 0 ? 'Keine Treffer gefunden.' : '';
                    })
                    .catch(error => {
                        document.getElementById('loading').classList.add('hidden');
                        document.getElementById('status-text').textContent = 'Fehler bei der Suche.';
                    });
            }
        });
        
        // Sortierung ändern
        document.getElementById('sort-by').addEventListener('change', loadDocuments);
        document.getElementById('sort-order').addEventListener('change', loadDocuments);

        function fullScanDocuments() {
            document.getElementById('loading').classList.remove('hidden');
            document.getElementById('status-text').textContent = 'Vollsuche: Alle Parquet-Dateien und IDs werden geladen...';
            document.getElementById('mode').textContent = 'Vollsuche (alle Dokumente)';
            // Starte vollständigen Scan (wie reloadDocuments, aber explizit)
            loadDocuments();
        }
        
        function showListMode() {
            console.log("Frontend: showListMode() called.");
            
            // Show list mode controls
            document.getElementById('listModeControls').style.display = 'block';
            
            // Load categories
            fetch('/api/list_categories')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        const select = document.getElementById('categorySelect');
                        select.innerHTML = '<option value="">Kategorie auswählen...</option>';
                        
                        data.categories.forEach(category => {
                            const option = document.createElement('option');
                            option.value = category.name;
                            option.textContent = `${category.display_name} (${category.count} IDs)`;
                            select.appendChild(option);
                        });
                        
                        document.getElementById('mode').textContent = '📋 Listen-Modus - Kategorie auswählen';
                    } else {
                        console.error('Error loading categories:', data.error);
                        alert('Error loading categories: ' + data.error);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Error loading categories');
                });
        }
        
        function loadListCategory() {
            const categorySelect = document.getElementById('categorySelect');
            const selectedCategory = categorySelect.value;
            
            if (!selectedCategory) {
                return;
            }
            
            console.log("Frontend: loadListCategory() called for category:", selectedCategory);
            
            // Show loading indicator
            document.getElementById('loading').classList.remove('hidden');
            document.getElementById('status-text').textContent = '📋 Loading documents for category "' + selectedCategory + '"...';
            
            // Load documents for this category
            fetch(`/api/list_documents?category=${encodeURIComponent(selectedCategory)}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        allDocuments = data.documents;
                        filteredDocuments = [...allDocuments];
                        currentPage = 1;
                        updateTable();
                        document.getElementById('loading').classList.add('hidden');
                        document.getElementById('status-text').textContent = '';
                        document.getElementById('mode').textContent = `📋 Listen-Modus: ${data.category_display_name} (${data.count} documents)`;
                    } else {
                        console.error('Error loading list documents:', data.error);
                        document.getElementById('loading').classList.add('hidden');
                        document.getElementById('status-text').textContent = 'Error: ' + data.error;
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('loading').classList.add('hidden');
                    document.getElementById('status-text').textContent = 'Error loading list documents';
                });
        }
        
        function exitListMode() {
            console.log("Frontend: exitListMode() called.");
            
            // Hide list mode controls
            document.getElementById('listModeControls').style.display = 'none';
            
            // Reset category select
            document.getElementById('categorySelect').value = '';
            
            // Reset mode display
            document.getElementById('mode').textContent = '';
            
            // Load default documents
            loadDocuments();
        }
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


def get_random_documents(cleaned_dir: Path, gold_dir: Path, sample_size: int = 100, max_files: int = 20):
    """Ultra-schnell: Wähle zufällige Parquet-Datei und nimm alle IDs daraus."""
    logger.info(f"Ultra-schnelle Random sample scan starting... (size: {sample_size})")
    config['loading_status'] = 'scanning'
    
    # Verwende die optimierten Indizes wenn vorhanden
    cleaned_index_key = f"{cleaned_dir}_index".replace("/", "_").replace("\\", "_")
    gold_index_key = f"{gold_dir}_index".replace("/", "_").replace("\\", "_")
    
    cleaned_index = config.get(cleaned_index_key, {})
    gold_index = config.get(gold_index_key, {})
    
    # Wenn keine Indizes vorhanden, erstelle sie
    if not cleaned_index:
        logger.info("Erstelle cleaned Index für Random-Suche...")
        cleaned_index = build_optimized_document_index(cleaned_dir)
        config[cleaned_index_key] = cleaned_index
    
    if not gold_index:
        logger.info("Erstelle gold Index für Random-Suche...")
        gold_index = build_optimized_document_index(gold_dir)
        config[gold_index_key] = gold_index
    
    # Sammle alle verfügbaren IDs aus beiden Indizes
    all_cleaned_ids = list(cleaned_index['id_to_file'].keys())
    all_gold_ids = list(gold_index['id_to_file'].keys())
    
    # Finde Schnittmenge
    intersection_ids = set(all_cleaned_ids) & set(all_gold_ids)
    logger.info(f"Gefunden: {len(intersection_ids)} IDs in beiden Datasets")
    
    if len(intersection_ids) == 0:
        logger.error("Keine übereinstimmenden IDs gefunden!")
        config['loading_status'] = 'error'
        return []
    
    # Wähle zufällige Parquet-Datei aus cleaned directory
    cleaned_files = list(cleaned_dir.glob("*.parquet"))
    if not cleaned_files:
        logger.error("Keine Parquet-Dateien im cleaned directory gefunden!")
        config['loading_status'] = 'error'
        return []
    
    # Wähle zufällige Datei
    random_file = random.choice(cleaned_files)
    logger.info(f"Wähle zufällige Datei: {random_file.name}")
    
    # Lade alle IDs aus dieser Datei
    try:
        # Lade alle IDs und Metadaten aus der zufälligen Datei
        table = pq.read_table(random_file, columns=['id', 'metadata'])
        df = table.to_pandas()
        
        logger.info(f"Gefunden: {len(df)} Dokumente in {random_file.name}")
        
        # Filtere nur IDs, die auch in gold vorhanden sind
        available_docs = []
        for idx, row in df.iterrows():
            doc_id = str(row['id']).strip()
            
            # Prüfe ob ID auch in gold vorhanden ist
            gold_file = binary_search_document(gold_index, doc_id)
            if not gold_file:
                continue
            
            # Metadaten extrahieren
            metadata = {}
            if 'metadata' in row and not pd.isna(row['metadata']):
                meta = row['metadata']
                if isinstance(meta, str):
                    try:
                        import json
                        metadata = json.loads(meta)
                    except:
                        metadata = {}
                elif isinstance(meta, dict):
                    metadata = meta
            
            # Dokument-Info erstellen
            doc_info = {
                'id': doc_id,
                'metadata': metadata,
                'total_citations_removed': 0
            }
            
            # Citations-Removed zählen
            for key, value in metadata.items():
                if '_citations_removed' in key:
                    try:
                        if isinstance(value, (int, float)):
                            doc_info['total_citations_removed'] += value
                        elif isinstance(value, str) and value.isdigit():
                            doc_info['total_citations_removed'] += int(value)
                    except (ValueError, TypeError):
                        pass
            
            available_docs.append(doc_info)
            
            # Stoppe bei gewünschter Anzahl
            if len(available_docs) >= sample_size:
                break
        
        logger.info(f"Gefunden: {len(available_docs)} verfügbare Dokumente aus {random_file.name}")
        
    except Exception as e:
        logger.error(f"Fehler beim Laden aus {random_file}: {e}")
        config['loading_status'] = 'error'
        return []
    
    config['available_docs'] = available_docs
    config['loading_status'] = 'complete'
    logger.info(f"✅ Ultra-schnelle Random sample: {len(available_docs)} documents ready aus {random_file.name}")
    return available_docs


@app.route('/api/random_documents')
def api_random_documents():
    """API endpoint to get a random sample of documents."""
    try:
        sample_size = request.args.get('size', 100, type=int)
        logger.info(f"Random documents API request: {sample_size} docs")
        def background_random_scan():
            gold_dir = Path(config['gold_dir']) if isinstance(config['gold_dir'], str) else config['gold_dir']
            cleaned_dir = Path(config['cleaned_dir']) if isinstance(config['cleaned_dir'], str) else config['cleaned_dir']
            get_random_documents(cleaned_dir, gold_dir, sample_size=sample_size)
        threading.Thread(target=background_random_scan).start()
        config['loading_status'] = 'scanning'
        return jsonify({
            'success': True,
            'message': f'Loading {sample_size} random documents...',
            'status': 'scanning'
        })
    except Exception as e:
        logger.error(f"Error in random documents API: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def ensure_valid_path_for_datatrove(file_path):
    """Ensure the file path is in the correct format for DataTrove's ParquetReader."""
    # If it's already a Path object, convert to string
    if isinstance(file_path, Path):
        return str(file_path)
    
    # If it's a string but doesn't exist, try to convert to Path then back to string
    # This helps normalize paths
    if isinstance(file_path, str):
        try:
            path_obj = Path(file_path)
            if path_obj.exists():
                return str(path_obj.absolute())
        except:
            pass
    
    # Return as is if nothing else works
    return file_path


@app.route('/api/search_documents')
def api_search_documents():
    """Optimierte Suche: Primär Binary Search, dann DataTrove als Fallback."""
    try:
        query = request.args.get('q', '').strip()
        if not query:
            return jsonify({'documents': [], 'total': 0, 'status': 'empty'})
        
        # Verwende die optimierten Indizes wenn vorhanden
        cleaned_index_key = f"{config['cleaned_dir']}_index".replace("/", "_").replace("\\", "_")
        gold_index_key = f"{config['gold_dir']}_index".replace("/", "_").replace("\\", "_")
        
        cleaned_index = config.get(cleaned_index_key, {})
        gold_index = config.get(gold_index_key, {})
        
        # Finde passende Datei mit binärer Suche
        cleaned_file = binary_search_document(cleaned_index, query) if cleaned_index else None
        gold_file = binary_search_document(gold_index, query) if gold_index else None
        
        if not cleaned_file or not gold_file:
            return jsonify({'documents': [], 'total': 0, 'status': 'not_found'})
        
        # Stelle sicher, dass die Pfade für DataTrove korrekt formatiert sind
        cleaned_file = ensure_valid_path_for_datatrove(cleaned_file)
        gold_file = ensure_valid_path_for_datatrove(gold_file)
        
        # Schnelle PyArrow-Suche mit Filter (schneller als DataTrove)
        found_cleaned = None
        found_metadata = {}
        
        try:
            logger.info(f"Schnelle PyArrow-Suche in cleaned: {cleaned_file}")
            
            # PyArrow mit Filter (sehr schnell)
            table = pq.read_table(cleaned_file, filters=[('id', '=', query)])
            
            if len(table) > 0:
                df = table.to_pandas()
                matches = df[df['id'].astype(str).str.strip() == query]
                
                if not matches.empty:
                    found_cleaned = query
                    # Metadaten extrahieren
                    if 'metadata' in matches.columns and not pd.isna(matches.iloc[0]['metadata']):
                        metadata = matches.iloc[0]['metadata']
                        if isinstance(metadata, str):
                            try:
                                import json
                                found_metadata = json.loads(metadata)
                            except:
                                found_metadata = {}
                        elif isinstance(metadata, dict):
                            found_metadata = metadata
                    
                    logger.info(f"✅ PyArrow-Suche erfolgreich in cleaned")
                    
        except Exception as e:
            logger.error(f"PyArrow-Suche fehlgeschlagen, verwende DataTrove: {e}")
            
            # Fallback auf DataTrove
            try:
                from datatrove.pipeline.readers import ParquetReader
                reader = ParquetReader(cleaned_file)
                
                for doc in reader.run():
                    if str(doc.id).strip() == query:
                        found_cleaned = doc.id
                        # Metadaten extrahieren
                        if hasattr(doc, 'metadata') and doc.metadata:
                            if isinstance(doc.metadata, dict):
                                found_metadata = doc.metadata
                            elif isinstance(doc.metadata, str):
                                try:
                                    import json
                                    found_metadata = json.loads(doc.metadata)
                                except:
                                    found_metadata = {}
                        break
            except Exception as e2:
                logger.error(f"Auch DataTrove-Fallback fehlgeschlagen: {e2}")
        
        # Schnelle PyArrow-Suche in gold (nur prüfen ob vorhanden)
        found_gold = None
        try:
            logger.info(f"Schnelle PyArrow-Suche in gold: {gold_file}")
            
            # PyArrow mit Filter (sehr schnell)
            table = pq.read_table(gold_file, filters=[('id', '=', query)])
            
            if len(table) > 0:
                df = table.to_pandas()
                matches = df[df['id'].astype(str).str.strip() == query]
                
                if not matches.empty:
                    found_gold = query
                    logger.info(f"✅ PyArrow-Suche erfolgreich in gold")
                    
        except Exception as e:
            logger.error(f"PyArrow-Suche in gold fehlgeschlagen, verwende DataTrove: {e}")
            
            # Fallback auf DataTrove
            try:
                from datatrove.pipeline.readers import ParquetReader
                reader = ParquetReader(gold_file)
                
                for doc in reader.run():
                    if str(doc.id).strip() == query:
                        found_gold = doc.id
                        break
            except Exception as e2:
                logger.error(f"Auch DataTrove-Fallback in gold fehlgeschlagen: {e2}")
        
        # Nur wenn in beiden gefunden
        docs = []
        if found_cleaned and found_gold:
            doc_info = {
                'id': query,
                'metadata': found_metadata,
                'total_citations_removed': 0
            }
            for key, value in found_metadata.items():
                if '_citations_removed' in key:
                    try:
                        if isinstance(value, (int, float)):
                            doc_info['total_citations_removed'] += value
                        elif isinstance(value, str) and value.isdigit():
                            doc_info['total_citations_removed'] += int(value)
                    except (ValueError, TypeError):
                        pass
            docs.append(doc_info)
        
        return jsonify({'documents': docs, 'total': len(docs), 'status': 'ok'})
    except Exception as e:
        logger.error(f"Error in optimized search: {e}")
        return jsonify({'documents': [], 'total': 0, 'status': 'error', 'error': str(e)})


@app.route('/api/list_categories')
def api_list_categories():
    """API endpoint to get available list categories."""
    try:
        # Load categories if not already loaded
        if not config['list_categories']:
            load_list_categories()
        
        # Format for frontend
        categories = []
        for category_name, category_data in config['list_categories'].items():
            categories.append({
                'name': category_name,
                'display_name': category_data['display_name'],
                'count': category_data['count']
            })
        
        return jsonify({
            'success': True,
            'categories': categories
        })
        
    except Exception as e:
        logger.error(f"Error getting list categories: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/list_documents')
def api_list_documents():
    """API endpoint to get documents for a specific list category."""
    try:
        category_name = request.args.get('category', '').strip()
        
        if not category_name:
            return jsonify({
                'success': False,
                'error': 'Category parameter required'
            }), 400
        
        logger.info(f"📋 List documents request for category: {category_name}")
        
        # Load categories if not already loaded
        if not config['list_categories']:
            load_list_categories()
        
        # Get documents for this category
        documents = get_list_documents_for_category(category_name)
        
        config['current_list_category'] = category_name
        config['available_docs'] = documents
        config['loading_status'] = 'complete'
        
        return jsonify({
            'success': True,
            'documents': documents,
            'count': len(documents),
            'category': category_name,
            'category_display_name': config['list_categories'][category_name]['display_name']
        })
        
    except Exception as e:
        logger.error(f"Error getting list documents: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def get_min_max_id_from_metadata(file_path):
    pf = pq.ParquetFile(file_path)
    min_id = None
    max_id = None
    try:
        # Try to get min/max from statistics (fast, no data read)
        for i in [0, pf.num_row_groups - 1]:
            row_group = pf.metadata.row_group(i)
            col_idx = pf.schema_arrow.get_field_index('id')
            col_chunk = row_group.column(col_idx)
            stats = col_chunk.statistics
            if stats and stats.has_min_max:
                if i == 0:
                    min_id = stats.min
                else:
                    max_id = stats.max
    except Exception:
        pass
    # Fallback: read first/last row if stats not available
    if min_id is None:
        try:
            min_id = pf.read_row_group(0, columns=['id']).to_pandas()['id'].iloc[0]
        except Exception:
            min_id = None
    if max_id is None:
        try:
            max_id = pf.read_row_group(pf.num_row_groups-1, columns=['id']).to_pandas()['id'].iloc[-1]
        except Exception:
            max_id = None
    return str(min_id) if min_id is not None else None, str(max_id) if max_id is not None else None


def build_parquet_index(cleaned_dir: Path, gold_dir: Path):
    """Effizient: Nutze Parquet-Metadaten (Min/Max), sonst nur erste/letzte Zeile lesen."""
    logger.info("Starte ultraschnellen Parquet-Index-Build...")
    index = {'cleaned': {}, 'gold': {}}
    # Cleaned
    for file_path in cleaned_dir.glob('*.parquet'):
        try:
            min_id, max_id = get_min_max_id_from_metadata(file_path)
            if min_id and max_id:
                index['cleaned'][str(file_path)] = {'min_id': min_id, 'max_id': max_id}
                logger.info(f"Index cleaned: {file_path.name}: {min_id} ... {max_id}")
        except Exception as e:
            logger.error(f"Index-Fehler cleaned {file_path}: {e}")
    # Gold
    for file_path in gold_dir.glob('*.parquet'):
        try:
            min_id, max_id = get_min_max_id_from_metadata(file_path)
            if min_id and max_id:
                index['gold'][str(file_path)] = {'min_id': min_id, 'max_id': max_id}
                logger.info(f"Index gold: {file_path.name}: {min_id} ... {max_id}")
        except Exception as e:
            logger.error(f"Index-Fehler gold {file_path}: {e}")
    config['parquet_index'] = index
    logger.info("Ultraschneller Parquet-Index fertig.")


def load_list_categories():
    """Load all document IDs from the JSON tables, grouped by category."""
    logger.info("📋 Loading list categories from JSON tables...")
    
    # Define the JSON table files and their display names
    table_files = {
        'top_semicolon_blocks': 'wandb/run-20250612_141655-cpwqgv8t/files/media/table/tables/top_semicolon_blocks_documents_762153_504bad13379f3c14e9ee.table.json',
        'top_page_references': 'wandb/run-20250612_141655-cpwqgv8t/files/media/table/tables/top_page_references_documents_762173_729df18578e72c95f499.table.json',
        'top_autor_jahr_eckig_einzel': 'wandb/run-20250612_141655-cpwqgv8t/files/media/table/tables/top_autor_jahr_eckig_einzel_documents_762167_d62e21a74b2dba2b3f96.table.json',
        'top_autor_jahr_klammer_einzel': 'wandb/run-20250612_141655-cpwqgv8t/files/media/table/tables/top_autor_jahr_klammer_einzel_documents_762165_7a7cb633ebf379499338.table.json',
        'top_autor_jahr_multi_klammer': 'wandb/run-20250612_141655-cpwqgv8t/files/media/table/tables/top_autor_jahr_multi_klammer_documents_762163_34d98346c9d320c4e0dc.table.json',
        'top_combined_reduction': 'wandb/run-20250612_141655-cpwqgv8t/files/media/table/tables/top_combined_reduction_documents_762181_0380d0bfd4adaa11c076.table.json',
        'top_consecutive_numeric_citations': 'wandb/run-20250612_141655-cpwqgv8t/files/media/table/tables/top_consecutive_numeric_citations_documents_762158_35a5d33e81496cef646b.table.json',
        'top_eckige_klammern_numerisch': 'wandb/run-20250612_141655-cpwqgv8t/files/media/table/tables/top_eckige_klammern_numerisch_documents_762156_c69b5bc70406b3f661fc.table.json',
        'top_figure_line_removal': 'wandb/run-20250612_141655-cpwqgv8t/files/media/table/tables/top_figure_line_removal_documents_762180_68322fb758017a0a126d.table.json',
        'top_figure_table_refs': 'wandb/run-20250612_141655-cpwqgv8t/files/media/table/tables/top_figure_table_refs_documents_762176_ba1d9d436d0eadb046c6.table.json',
        'top_isolated_numeric_citations': 'wandb/run-20250612_141655-cpwqgv8t/files/media/table/tables/top_isolated_numeric_citations_documents_762160_074eac1439239bd642dd.table.json',
        'top_ref_nummer': 'wandb/run-20250612_141655-cpwqgv8t/files/media/table/tables/top_ref_nummer_documents_762171_21ec0516ec90921147be.table.json'
    }
    
    categories = {}
    
    for category_name, file_path in table_files.items():
        try:
            if not Path(file_path).exists():
                logger.warning(f"Table file not found: {file_path}")
                continue
            with open(file_path, 'r') as f:
                data = json.load(f)
            # Find the index of the Doc ID column
            columns = data.get('columns', [])
            if 'Doc ID' in columns:
                docid_idx = columns.index('Doc ID')
            else:
                logger.warning(f"No 'Doc ID' column in {category_name}")
                continue
            # Extract document IDs from the correct column
            doc_ids = []
            for row in data.get('data', []):
                if len(row) > docid_idx:
                    doc_id = row[docid_idx]
                    if doc_id and isinstance(doc_id, str):
                        doc_ids.append(doc_id)
            if doc_ids:
                categories[category_name] = {
                    'display_name': category_name.replace('_', ' ').title(),
                    'doc_ids': doc_ids,
                    'count': len(doc_ids)
                }
                logger.info(f"📋 Loaded {len(doc_ids)} IDs for category: {category_name}")
            else:
                logger.warning(f"No document IDs found in {category_name}")
        except Exception as e:
            logger.error(f"Error loading category {category_name}: {e}")
            continue
    
    config['list_categories'] = categories
    logger.info(f"📋 Loaded {len(categories)} list categories")
    return categories


def get_list_documents_for_category(category_name: str):
    """Get documents for a specific category using optimized binary search and PyArrow filters."""
    if category_name not in config['list_categories']:
        logger.error(f"Category {category_name} not found")
        return []
    
    category_data = config['list_categories'][category_name]
    all_doc_ids = category_data['doc_ids']
    
    logger.info(f"📋 Getting documents for category '{category_name}' ({len(all_doc_ids)} total IDs)")
    
    # Optimierte Indizes laden/erstellen
    cleaned_index_key = f"{config['cleaned_dir']}_index".replace("/", "_").replace("\\", "_")
    gold_index_key = f"{config['gold_dir']}_index".replace("/", "_").replace("\\", "_")
    
    # Indizes initialisieren, falls noch nicht vorhanden
    if cleaned_index_key not in config:
        logger.info(f"Erstelle optimierten Index für cleaned directory...")
        config[cleaned_index_key] = build_optimized_document_index(config['cleaned_dir'])
    
    if gold_index_key not in config:
        logger.info(f"Erstelle optimierten Index für gold directory...")
        config[gold_index_key] = build_optimized_document_index(config['gold_dir'])
    
    cleaned_index = config[cleaned_index_key]
    gold_index = config[gold_index_key]
    
    # Dokumente mit binärer Suche finden
    available_docs = []
    found_count = 0
    
    logger.info(f"Führe optimierte Suche für {len(all_doc_ids)} Dokument-IDs durch...")
    
    # Optimierter Prozess: Primär PyArrow Filter, dann DataTrove als Fallback
    for i, doc_id in enumerate(all_doc_ids):
        if i % 10 == 0:
            logger.info(f"Fortschritt: {i+1}/{len(all_doc_ids)} IDs verarbeitet...")
        
        # 1. Binäre Suche für die Datei
        cleaned_file_path = binary_search_document(cleaned_index, doc_id)
        if not cleaned_file_path:
            continue
            
        # 2. In gold vorhanden?
        gold_file_path = binary_search_document(gold_index, doc_id)
        if not gold_file_path:
            continue
        
        # 3. Schnelle PyArrow-Suche für Metadaten
        doc_id_str = str(doc_id).strip()
        metadata = {}
        found = False
        
        try:
            # PyArrow mit Filter (sehr schnell)
            table = pq.read_table(cleaned_file_path, filters=[('id', '=', doc_id)])
            
            if len(table) > 0:
                df = table.to_pandas()
                matches = df[df['id'].astype(str).str.strip() == doc_id_str]
                
                if not matches.empty:
                    # Metadaten extrahieren
                    if 'metadata' in matches.columns and not pd.isna(matches.iloc[0]['metadata']):
                        meta = matches.iloc[0]['metadata']
                        if isinstance(meta, str):
                            try:
                                import json
                                metadata = json.loads(meta)
                            except:
                                metadata = {}
                        elif isinstance(meta, dict):
                            metadata = meta
                    
                    found = True
                    logger.debug(f"✅ PyArrow-Suche erfolgreich für {doc_id}")
                    
        except Exception as e:
            logger.warning(f"PyArrow-Suche fehlgeschlagen für {doc_id}: {e}")
        
        # 4. Fallback auf DataTrove wenn PyArrow fehlschlägt
        if not found:
            try:
                from datatrove.pipeline.readers import ParquetReader
                cleaned_path = ensure_valid_path_for_datatrove(cleaned_file_path)
                
                logger.debug(f"DataTrove-Fallback für {doc_id}")
                reader = ParquetReader(cleaned_path)
                
                for doc in reader.run():
                    if str(doc.id).strip() == doc_id_str:
                        # Metadaten extrahieren
                        if hasattr(doc, 'metadata') and doc.metadata:
                            if isinstance(doc.metadata, dict):
                                metadata = doc.metadata
                            elif isinstance(doc.metadata, str):
                                try:
                                    import json
                                    metadata = json.loads(doc.metadata)
                                except:
                                    metadata = {}
                        found = True
                        break
                        
            except Exception as e:
                logger.error(f"DataTrove-Fallback fehlgeschlagen für {doc_id}: {e}")
        
        # 5. Dokument zum Ergebnis hinzufügen wenn gefunden
        if found:
            doc_info = {
                'id': doc_id,
                'metadata': metadata,
                'total_citations_removed': 0
            }
            
            # Citations-Removed zählen
            for key, value in metadata.items():
                if '_citations_removed' in key:
                    try:
                        if isinstance(value, (int, float)):
                            doc_info['total_citations_removed'] += value
                        elif isinstance(value, str) and value.isdigit():
                            doc_info['total_citations_removed'] += int(value)
                    except (ValueError, TypeError):
                        pass
            
            available_docs.append(doc_info)
            found_count += 1
    
    # Ergebnisse sortieren
    available_docs = sorted(available_docs, key=lambda x: x['id'])
    
    logger.info(f"📋 Category '{category_name}': {len(available_docs)}/{len(all_doc_ids)} Dokumente gefunden mit optimierter Suche")
    return available_docs


def build_fast_document_index(directory: Path):
    """
    Erstellt einen schnellen Index aller Dokument-IDs und ihrer Dateipfade.
    Liest nur die ID-Spalte für maximale Geschwindigkeit.
    """
    logger.info(f"Erstelle schnellen Dokument-Index für {directory}...")
    start_time = time.time()
    
    # Ergebnis-Index: {str(doc_id): file_path}
    doc_index = {}
    
    # Alle Parquet-Dateien scannen
    parquet_files = list(directory.glob("*.parquet"))
    total_files = len(parquet_files)
    
    # Batch-Verarbeitung mit Fortschrittsanzeige
    for i, file_path in enumerate(parquet_files):
        if i % 10 == 0:
            logger.info(f"Indexiere Datei {i+1}/{total_files} ({(i+1)/total_files*100:.1f}%)...")
        
        try:
            # Nur ID-Spalte einlesen (schnell)
            table = pq.read_table(file_path, columns=['id'])
            df = pd.DataFrame({'id': table['id'].to_numpy()})
            
            # Alle IDs als Strings speichern
            for doc_id in df['id']:
                doc_id_str = str(doc_id).strip()
                doc_index[doc_id_str] = file_path
                
        except Exception as e:
            logger.error(f"Fehler beim Indexieren von {file_path}: {e}")
    
    elapsed = time.time() - start_time
    logger.info(f"✅ Index für {directory} erstellt: {len(doc_index):,} Dokumente in {elapsed:.2f}s")
    return doc_index


def get_document_by_id_indexed(directory: Path, doc_id: str, file_index=None):
    """Sucht ein Dokument mit dem optimierten Index-Lookup."""
    cache_key = f"{directory}:{doc_id}"
    if cache_key in config['document_cache']:
        return config['document_cache'][cache_key]
    
    # Normalisierten ID-String erstellen
    doc_id_str = str(doc_id).strip()
    
    # Index verwenden, wenn verfügbar
    if file_index and doc_id_str in file_index:
        file_path = file_index[doc_id_str]
        logger.info(f"🔍 Direkter Index-Lookup für '{doc_id}' in {file_path.name}")
        
        try:
            # Zuerst mit Filter versuchen (schnell)
            table = pq.read_table(file_path, filters=[('id', '=', doc_id)])
            if len(table) > 0:
                # Robust: Stringvergleich mit strip()
                df = table.to_pandas()
                matches = df[df['id'].astype(str).str.strip() == doc_id_str]
                
                if not matches.empty:
                    text = matches.iloc[0]['text']
                    config['document_cache'][cache_key] = text
                    logger.info(f"✅ Index-Lookup: Dokument gefunden in {file_path.name}")
                    return text
            
            # Wenn Filter nicht funktioniert, ganzen Datensatz lesen (mit ID-Filterung)
            table = pq.read_table(file_path)
            df = table.to_pandas()
            matches = df[df['id'].astype(str).str.strip() == doc_id_str]
            
            if not matches.empty:
                text = matches.iloc[0]['text']
                config['document_cache'][cache_key] = text
                logger.info(f"✅ Index-Lookup (Volldatei): Dokument gefunden in {file_path.name}")
                return text
                
        except Exception as e:
            logger.error(f"Fehler bei Index-Lookup für {doc_id} in {file_path}: {e}")
    
    # Fallback: Standard-Suche
    logger.info(f"🔍 Index-Lookup fehlgeschlagen für {doc_id}, verwende Standardsuche")
    return get_document_by_id(directory, doc_id)


def build_optimized_document_index(directory: Path):
    """
    Erstellt einen hochoptimierten, sortierten Index aller Dokument-IDs für binäre Suche.
    Speichert pro Datei:
    1. Sortierte Liste aller IDs 
    2. Min/Max ID für schnelles Ausschließen
    """
    logger.info(f"Erstelle optimierten Divide-and-Conquer-Index für {directory}...")
    start_time = time.time()
    
    # Ergebnis-Struktur:
    # {
    #   'by_file': {file_path: {'ids': sorted_ids, 'min': min_id, 'max': max_id}},
    #   'id_to_file': {id_str: file_path}
    # }
    result = {'by_file': {}, 'id_to_file': {}}
    
    # Alle Parquet-Dateien scannen
    parquet_files = list(directory.glob("*.parquet"))
    total_files = len(parquet_files)
    total_ids = 0
    
    # Für jede Datei
    for i, file_path in enumerate(parquet_files):
        if i % 10 == 0 or i == total_files - 1:
            logger.info(f"Indexiere Datei {i+1}/{total_files} ({(i+1)/total_files*100:.1f}%)...")
        
        try:
            # Nur ID-Spalte einlesen (sehr schnell)
            table = pq.read_table(file_path, columns=['id'])
            ids = table['id'].to_numpy()
            
            # Alle IDs als normalisierte Strings
            id_strings = [str(id_val).strip() for id_val in ids]
            
            # Sortieren für binäre Suche
            sorted_ids = sorted(id_strings)
            
            if sorted_ids:
                # Pro Datei speichern
                result['by_file'][str(file_path)] = {
                    'ids': sorted_ids,
                    'min': sorted_ids[0],
                    'max': sorted_ids[-1]
                }
                
                # Lookup-Index für schnellen Zugriff
                for id_str in id_strings:
                    result['id_to_file'][id_str] = str(file_path)
                    
                total_ids += len(id_strings)
                
        except Exception as e:
            logger.error(f"Fehler beim Indexieren von {file_path}: {e}")
    
    elapsed = time.time() - start_time
    logger.info(f"✅ Optimierter Index erstellt: {total_ids:,} Dokumente in {len(parquet_files)} Dateien in {elapsed:.2f}s")
    
    # Index persistent speichern
    save_optimized_index(result, directory)
    
    return result


def binary_search_document(index, doc_id: str):
    """
    Effiziente Suche mit Divide-and-Conquer-Strategie:
    1. Direkter Lookup über id_to_file-Map
    2. Wenn nicht gefunden, überprüfe jede Datei mit Min/Max-Filter
    3. Binäre Suche in der sortierten ID-Liste jeder relevanten Datei
    """
    # Normalisierte ID
    doc_id_str = str(doc_id).strip()
    
    # 1. Direkter Lookup (O(1))
    if doc_id_str in index['id_to_file']:
        return index['id_to_file'][doc_id_str]
    
    # 2. Min/Max-Filterung mit binärer Suche in jeder Datei
    for file_path, file_data in index['by_file'].items():
        # Schneller Min/Max-Check
        if doc_id_str < file_data['min'] or doc_id_str > file_data['max']:
            continue
        
        # Binäre Suche in der sortierten ID-Liste
        sorted_ids = file_data['ids']
        left, right = 0, len(sorted_ids) - 1
        
        while left <= right:
            mid = (left + right) // 2
            mid_id = sorted_ids[mid]
            
            if mid_id == doc_id_str:
                return file_path
            elif mid_id < doc_id_str:
                left = mid + 1
            else:
                right = mid - 1
                
        # Noch ein linearer Scan für ähnliche IDs (Toleranz für Formatunterschiede)
        # Dies ist nur ein Backup, sollte selten genutzt werden
        for id_str in sorted_ids:
            if id_str.replace(" ", "") == doc_id_str.replace(" ", ""):
                return file_path
    
    # Nichts gefunden
    return None


def get_index_filename(directory: Path):
    """Generate a unique filename for the index based on directory path and content."""
    # Create a hash of the directory path to avoid filename conflicts
    dir_hash = hashlib.md5(str(directory.absolute()).encode()).hexdigest()[:8]
    return f"index_{dir_hash}.pkl"

def save_optimized_index(index, directory: Path):
    """Save the optimized index to disk."""
    try:
        index_file = Path("data") / "indexes" / get_index_filename(directory)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(index_file, 'wb') as f:
            pickle.dump(index, f)
        
        logger.info(f"✅ Index gespeichert: {index_file}")
        return True
    except Exception as e:
        logger.error(f"Fehler beim Speichern des Index: {e}")
        return False

def load_optimized_index(directory: Path):
    """Load the optimized index from disk if it exists."""
    try:
        index_file = Path("data") / "indexes" / get_index_filename(directory)
        
        if not index_file.exists():
            logger.info(f"Kein gespeicherter Index gefunden: {index_file}")
            return None
        
        # Check if index is still valid (directory hasn't changed significantly)
        if not is_index_valid(index_file, directory):
            logger.info(f"Index veraltet, wird neu erstellt: {index_file}")
            return None
        
        with open(index_file, 'rb') as f:
            index = pickle.load(f)
        
        logger.info(f"✅ Index geladen: {index_file}")
        return index
    except Exception as e:
        logger.error(f"Fehler beim Laden des Index: {e}")
        return None

def is_index_valid(index_file: Path, directory: Path):
    """Check if the saved index is still valid by comparing file counts and modification times."""
    try:
        # Get current parquet files
        current_files = set(f.name for f in directory.glob("*.parquet"))
        
        # Get index file modification time
        index_mtime = index_file.stat().st_mtime
        
        # Check if any parquet files are newer than the index
        for parquet_file in directory.glob("*.parquet"):
            if parquet_file.stat().st_mtime > index_mtime:
                logger.info(f"Parquet file {parquet_file.name} ist neuer als Index")
                return False
        
        return True
    except Exception as e:
        logger.error(f"Fehler bei Index-Validierung: {e}")
        return False


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
    parser.add_argument("--skip-index", action="store_true", help="Skip building the optimized index")
    
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
    
    print(f"🚀 Starting Document Diff Viewer with Divide-and-Conquer Search")
    print(f"Gold data: {config['gold_dir']}")
    print(f"Cleaned data: {config['cleaned_dir']}")
    print(f"Local server: http://{args.host}:{args.port}")
    
    # Build optimized index (optional, can be skipped with --skip-index)
    if not args.skip_index:
        print(f"\n🚀 Lade oder erstelle optimierte Divide-and-Conquer-Indizes...")
        try:
            # Parallel index building for both directories
            def build_cleaned_index():
                cleaned_index_key = f"{config['cleaned_dir']}_index".replace("/", "_").replace("\\", "_")
                # Versuche zuerst gespeicherten Index zu laden
                saved_index = load_optimized_index(config['cleaned_dir'])
                if saved_index:
                    config[cleaned_index_key] = saved_index
                    print(f"✅ Gespeicherter Index für cleaned geladen")
                else:
                    config[cleaned_index_key] = build_optimized_document_index(config['cleaned_dir'])
                
            def build_gold_index():
                gold_index_key = f"{config['gold_dir']}_index".replace("/", "_").replace("\\", "_")
                # Versuche zuerst gespeicherten Index zu laden
                saved_index = load_optimized_index(config['gold_dir'])
                if saved_index:
                    config[gold_index_key] = saved_index
                    print(f"✅ Gespeicherter Index für gold geladen")
                else:
                    config[gold_index_key] = build_optimized_document_index(config['gold_dir'])
                
            cleaned_thread = threading.Thread(target=build_cleaned_index)
            gold_thread = threading.Thread(target=build_gold_index)
            
            cleaned_thread.start()
            gold_thread.start()
            
            # Wait for both threads to complete
            cleaned_thread.join()
            gold_thread.join()
            
            print(f"✅ Optimierte Indizes erfolgreich geladen/erstellt!")
            
        except Exception as e:
            print(f"⚠️ Warnung: Konnte optimierte Indizes nicht erstellen: {e}")
    else:
        print(f"\n⚠️ Indexierung übersprungen (--skip-index)")
    
    # Load list categories at startup
    print(f"\n📋 Loading list categories...")
    try:
        load_list_categories()
        print(f"✅ Loaded {len(config['list_categories'])} list categories")
    except Exception as e:
        print(f"⚠️ Warning: Could not load list categories: {e}")
    
    if args.ngrok:
        print(f"\n🚀 Starting ngrok tunnel...")
        
        def start_tunnel():
            time.sleep(2)
            start_ngrok_tunnel(args.port, args.auth_token)
        
        tunnel_thread = threading.Thread(target=start_tunnel)
        tunnel_thread.daemon = True
        tunnel_thread.start()
    
    print(f"\n🚀 Starting server...")
    
    try:
        app.run(host=args.host, port=args.port, debug=False)
    except KeyboardInterrupt:
        print(f"\n👋 Viewer shutting down...")