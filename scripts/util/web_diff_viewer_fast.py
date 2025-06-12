#!/usr/bin/env python3
"""
FAST Web-based document diff viewer - optimized for quick startup with Datatrove.
Provides immediate UI response while loading documents in background using Datatrove ParquetReader.
"""

from flask import Flask, render_template, request, jsonify, make_response
import pyarrow.parquet as pq
import argparse
from pathlib import Path
import difflib
import json
import logging
from datetime import datetime
import html
import os
import threading
import time
import pandas as pd
import glob
import sys
import random

# Add project root to path for datatrove imports
script_dir = os.path.dirname(__file__)
proj_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, proj_root)

# Import Datatrove components
from datatrove.pipeline.readers import ParquetReader
from datatrove.data import Document

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global configuration
config = {
    'gold_dir': None,
    'cleaned_dir': None,
    'document_cache': {},
    'diff_cache': {},
    'available_docs': [],
    'loading_status': 'ready',
    'loaded_count': 0,
    'random_mode': False
}


def get_random_documents_datatrove(cleaned_dir: Path, gold_dir: Path, sample_size: int = 100):
    """
    Get random sample of documents using Datatrove.
    """
    logger.info(f"🎲 Datatrove random sample starting... (size: {sample_size})")
    config['loading_status'] = 'scanning'
    config['random_mode'] = True
    
    try:
        cleaned_ids = set()
        gold_ids = set()
        
        # Use Datatrove to get ALL cleaned IDs
        logger.info(f"📁 Scanning ALL cleaned documents with Datatrove...")
        try:
            reader = ParquetReader(
                data_folder=str(cleaned_dir),
                glob_pattern="*.parquet",
                limit=-1  # Read all documents
            )
            
            for document in reader():
                if document.id:
                    cleaned_ids.add(document.id)
                        
        except Exception as e:
            logger.warning(f"Datatrove scan error for cleaned: {e}")
        
        # Use Datatrove to get ALL gold IDs
        logger.info(f"📁 Scanning ALL gold documents with Datatrove...")
        try:
            reader = ParquetReader(
                data_folder=str(gold_dir),
                glob_pattern="*.parquet",
                limit=-1  # Read all documents
            )
            
            for document in reader():
                if document.id:
                    gold_ids.add(document.id)
                        
        except Exception as e:
            logger.warning(f"Datatrove scan error for gold: {e}")
        
        # Find intersection
        all_intersecting = list(cleaned_ids & gold_ids)
        
        if len(all_intersecting) == 0:
            logger.warning("No intersecting documents found")
            config['available_docs'] = []
            config['loaded_count'] = 0
            config['loading_status'] = 'complete'
            return []
        
        # Take random sample
        sample_size = min(sample_size, len(all_intersecting))
        random_docs = random.sample(all_intersecting, sample_size)
        random_docs = sorted(random_docs)  # Sort for consistent display
        
        config['available_docs'] = random_docs
        config['loaded_count'] = len(random_docs)
        config['loading_status'] = 'complete'
        
        logger.info(f"✅ Random sample: Found {len(cleaned_ids)} cleaned, {len(gold_ids)} gold, {len(all_intersecting)} intersecting, selected {len(random_docs)} random documents")
        return random_docs
        
    except Exception as e:
        logger.error(f"Random sample error: {e}")
        config['loading_status'] = 'error'
        config['random_mode'] = False
        return []


def quick_scan_documents_datatrove(cleaned_dir: Path, gold_dir: Path, max_docs: int = 1000, random_sample: bool = False, sample_size: int = 100):
    """
    Enhanced document discovery using Datatrove for better ID handling.
    """
    if random_sample:
        return get_random_documents_datatrove(cleaned_dir, gold_dir, sample_size)
    
    logger.info(f"⚡ Datatrove quick scan starting...")
    config['loading_status'] = 'scanning'
    config['random_mode'] = False
    
    try:
        cleaned_ids = set()
        gold_ids = set()
        
        # Use Datatrove to get cleaned IDs
        logger.info(f"📁 Scanning cleaned directory with Datatrove...")
        try:
            reader = ParquetReader(
                data_folder=str(cleaned_dir),
                glob_pattern="*.parquet",
                limit=-1  # Read all documents
            )
            
            for document in reader():
                if document.id:
                    cleaned_ids.add(document.id)
                        
        except Exception as e:
            logger.warning(f"Datatrove scan error for cleaned: {e}")
        
        # Use Datatrove to get gold IDs
        logger.info(f"📁 Scanning gold directory with Datatrove...")
        try:
            reader = ParquetReader(
                data_folder=str(gold_dir),
                glob_pattern="*.parquet",
                limit=-1  # Read all documents
            )
            
            for document in reader():
                if document.id:
                    gold_ids.add(document.id)
                        
        except Exception as e:
            logger.warning(f"Datatrove scan error for gold: {e}")
        
        # Find intersection
        available_docs = list(cleaned_ids & gold_ids)
        available_docs = sorted(available_docs)[:500]  # Increase limit for UI
        
        config['available_docs'] = available_docs
        config['loaded_count'] = len(available_docs)
        config['loading_status'] = 'complete'
        
        logger.info(f"✅ Datatrove scan found {len(cleaned_ids)} cleaned, {len(gold_ids)} gold, {len(available_docs)} intersecting documents")
        return available_docs
        
    except Exception as e:
        logger.error(f"Datatrove scan error: {e}")
        config['loading_status'] = 'error'
        # Fallback to old method
        return quick_scan_documents_fallback(cleaned_dir, gold_dir, max_docs)


def quick_scan_documents_fallback(cleaned_dir: Path, gold_dir: Path, max_docs: int = 500):
    """
    Fallback document discovery using PyArrow.
    """
    logger.info(f"⚡ Fallback quick scan starting...")
    
    try:
        # Scan only first few files for immediate results
        cleaned_files = list(cleaned_dir.glob("*.parquet"))[:10]
        gold_files = list(gold_dir.glob("*.parquet"))[:10]
        
        cleaned_ids = set()
        gold_ids = set()
        
        # Fast ID collection from cleaned files
        for file_path in cleaned_files:
            try:
                if file_path.stat().st_size == 0:
                    continue
                    
                # Quick PyArrow read
                table = pq.read_table(file_path, columns=['id'])
                file_ids = [id for id in table['id'].to_pylist()[:100] if id]  # First 100 per file
                cleaned_ids.update(file_ids)
                
                if len(cleaned_ids) > max_docs:
                    break
                    
            except Exception as e:
                logger.warning(f"Skip {file_path.name}: {e}")
                continue
        
        # Fast ID collection from gold files
        for file_path in gold_files:
            try:
                if file_path.stat().st_size == 0:
                    continue
                    
                table = pq.read_table(file_path, columns=['id'])
                file_ids = [id for id in table['id'].to_pylist()[:100] if id]
                gold_ids.update(file_ids)
                
                if len(gold_ids) > max_docs:
                    break
                    
            except Exception as e:
                logger.warning(f"Skip gold {file_path.name}: {e}")
                continue
        
        # Find intersection
        available_docs = list(cleaned_ids & gold_ids)
        available_docs = sorted(available_docs)[:200]  # Limit for UI
        
        config['available_docs'] = available_docs
        config['loaded_count'] = len(available_docs)
        config['loading_status'] = 'complete'
        
        logger.info(f"✅ Fallback scan found {len(available_docs)} documents")
        return available_docs
        
    except Exception as e:
        logger.error(f"Fallback scan error: {e}")
        config['loading_status'] = 'error'
        return []


def get_document_by_id_datatrove(directory: Path, doc_id: str):
    """Document retrieval using Datatrove ParquetReader."""
    cache_key = f"{directory}:{doc_id}"
    if cache_key in config['document_cache']:
        return config['document_cache'][cache_key]
    
    logger.info(f"🔍 Datatrove search for document '{doc_id}' in {directory}")
    
    try:
        # Use Datatrove ParquetReader
        reader = ParquetReader(
            data_folder=str(directory),
            glob_pattern="*.parquet",
            limit=-1  # No limit
        )
        
        # Iterate through all documents
        for document in reader():
            if document.id == doc_id:
                logger.info(f"✅ Found document '{doc_id}' via Datatrove")
                config['document_cache'][cache_key] = document.text
                return document.text
        
        # Document not found
        logger.info(f"❌ Document '{doc_id}' not found via Datatrove in {directory}")
        config['document_cache'][cache_key] = None
        return None
        
    except Exception as e:
        logger.error(f"Error using Datatrove to search for {doc_id}: {e}")
        # Fallback to pandas method
        return get_document_by_id_simple(directory, doc_id)


def get_document_by_id_simple(directory: Path, doc_id: str):
    """Simple document retrieval without heavy caching (fallback method)."""
    cache_key = f"{directory}:{doc_id}"
    if cache_key in config['document_cache']:
        return config['document_cache'][cache_key]
    
    # Log the search attempt
    logger.info(f"🔍 Fallback search for document with ID '{doc_id}' in {directory}")
    
    # Search in all parquet files
    parquet_files = glob.glob(str(directory / "*.parquet"))
    
    for file_path in parquet_files:
        try:
            # Read parquet file with pandas - more reliable for complex IDs
            df = pd.read_parquet(file_path)
            
            # Check if document exists in this file
            matches = df[df['id'] == doc_id]
            
            if not matches.empty:
                text = matches.iloc[0]['text']
                logger.info(f"✅ Found document '{doc_id}' in {os.path.basename(file_path)}")
                
                # Cache result
                config['document_cache'][cache_key] = text
                return text
                
        except Exception as e:
            logger.debug(f"Error reading {os.path.basename(file_path)} for doc {doc_id}: {e}")
            continue
    
    # Document not found in any file
    logger.info(f"❌ Document '{doc_id}' not found in {directory}")
    config['document_cache'][cache_key] = None  # Cache negative result
    return None


def create_html_diff(text1: str, text2: str, name1: str = "Gold", name2: str = "Cleaned"):
    """Create HTML diff."""
    differ = difflib.HtmlDiff(wrapcolumn=100, tabsize=2)
    html_diff = differ.make_file(
        text1.splitlines(),
        text2.splitlines(),
        fromdesc=name1,
        todesc=name2,
        context=True,
        numlines=5
    )
    return html_diff


def search_documents_with_datatrove(directory: Path, search_term: str, max_results: int = 100):
    """
    Enhanced search using Datatrove ParquetReader for better handling of complex IDs.
    This properly handles patterns like part-0.parquet/1.
    """
    logger.info(f"🔍 DATATROVE search for '{search_term}' in {directory}")
    
    found_docs = []
    search_term_lower = search_term.lower()
    docs_checked = 0
    
    try:
        # Use Datatrove ParquetReader
        reader = ParquetReader(
            data_folder=str(directory),
            glob_pattern="*.parquet",
            limit=-1  # No limit to get all documents
        )
        
        logger.info(f"Using Datatrove to search all documents in {directory}")
        
        # Iterate through all documents
        for document in reader():
            docs_checked += 1
            
            # Log progress every 1000 documents
            if docs_checked % 1000 == 0:
                logger.info(f"Datatrove search: Searched {docs_checked} documents in {directory}, found {len(found_docs)} matches for '{search_term_lower}'")
            
            # Check if document ID contains search term
            doc_id_str = str(document.id).lower() if document.id else ""
            if search_term_lower in doc_id_str:
                found_docs.append(document.id)
                logger.info(f"✅ Datatrove search: Match found in {directory}: '{document.id}' (term: '{search_term_lower}')")
                
                if len(found_docs) >= max_results:
                    logger.info(f"Datatrove search: Max results ({max_results}) reached for '{search_term_lower}' in {directory}.")
                    break
        
        logger.info(f"Datatrove search completed for {directory}: {len(found_docs)} docs found after checking {docs_checked} documents for term '{search_term_lower}'")
    
    except Exception as e:
        logger.error(f"Datatrove search error: {e}")
        # Fallback to pandas method
        logger.info("Falling back to pandas-based search")
        return search_documents_in_files_thorough(directory, search_term, max_results)
    
    return found_docs[:max_results]


def search_documents_in_files_thorough(directory: Path, search_term: str, max_results: int = 100):
    """
    More thorough search through all parquet files for document IDs containing search_term.
    This handles complex patterns like part-0.parquet/1 by searching in the actual document IDs.
    (Fallback method)
    """
    logger.info(f"🔍 FALLBACK search for '{search_term}' in {directory}")
    
    found_docs = []
    search_term_lower = search_term.lower()
    files_checked = 0
    files_with_hits = 0
    
    try:
        # Get ALL parquet files without limiting
        parquet_files = glob.glob(str(directory / "*.parquet"))
        logger.info(f"Found {len(parquet_files)} parquet files to search in {directory}")
        
        for file_path in parquet_files:
            if len(found_docs) >= max_results:
                break
                
            try:
                files_checked += 1
                
                # Read all document IDs from this file
                df = pd.read_parquet(file_path, columns=['id'])
                if df.empty:
                    continue
                
                # Check each ID for matches
                matches_in_file = 0
                for doc_id in df['id'].values:
                    if doc_id and search_term_lower in str(doc_id).lower():
                        found_docs.append(doc_id)
                        matches_in_file += 1
                        
                        if len(found_docs) >= max_results:
                            break
                
                if matches_in_file > 0:
                    files_with_hits += 1
                    logger.info(f"✅ Found {matches_in_file} matching docs in {os.path.basename(file_path)}")
                
            except Exception as e:
                logger.warning(f"Error searching {os.path.basename(file_path)}: {e}")
                continue
        
        logger.info(f"Search completed: {len(found_docs)} docs found in {files_with_hits}/{files_checked} files")
    
    except Exception as e:
        logger.error(f"Thorough search error: {e}")
    
    return found_docs[:max_results]


def get_intersecting_documents(cleaned_dir: Path, gold_dir: Path, doc_ids: list):
    """Find documents that exist in both cleaned and gold directories."""
    logger.info(f"🔍 Finding intersection for {len(doc_ids)} documents")
    
    # Check which documents exist in both directories
    intersecting_docs = []
    
    # Cache for document existence checks - once we've checked a directory,
    # we can avoid future filesystem lookups
    existence_cache = {
        'cleaned': {},
        'gold': {}
    }
    
    for idx, doc_id in enumerate(doc_ids):
        logger.debug(f"Intersection check for doc_id: '{doc_id}' ({idx+1}/{len(doc_ids)})")
        # First check existence cache
        if doc_id in existence_cache['cleaned']:
            cleaned_exists = existence_cache['cleaned'][doc_id]
            logger.debug(f"'{doc_id}' in cleaned_dir (from cache): {cleaned_exists}")
        else:
            cleaned_exists = get_document_by_id_datatrove(cleaned_dir, doc_id) is not None
            existence_cache['cleaned'][doc_id] = cleaned_exists
            logger.debug(f"'{doc_id}' in cleaned_dir (live check): {cleaned_exists}")
        
        # Only check gold if cleaned exists (optimization)
        if cleaned_exists:
            if doc_id in existence_cache['gold']:
                gold_exists = existence_cache['gold'][doc_id]
                logger.debug(f"'{doc_id}' in gold_dir (from cache): {gold_exists}")
            else:
                gold_exists = get_document_by_id_datatrove(gold_dir, doc_id) is not None
                existence_cache['gold'][doc_id] = gold_exists
                logger.debug(f"'{doc_id}' in gold_dir (live check): {gold_exists}")
                
            if gold_exists:
                logger.info(f"✅ Intersection found for doc_id: '{doc_id}' (exists in both)")
                intersecting_docs.append(doc_id)
            else:
                logger.debug(f"'{doc_id}' exists in cleaned but NOT in gold.")
        else:
            logger.debug(f"'{doc_id}' does NOT exist in cleaned.")
        
        # Log progress every 10 documents
        if (idx + 1) % 10 == 0 or idx == len(doc_ids) - 1:
            logger.info(f"Intersection progress: {idx + 1}/{len(doc_ids)} docs checked, {len(intersecting_docs)} intersecting matches found so far.")
    
    logger.info(f"Intersection complete: Found {len(intersecting_docs)}/{len(doc_ids)} documents present in both directories.")
    return intersecting_docs


@app.route('/api/search')
def api_search():
    """API endpoint for dynamic document search."""
    try:
        search_term = request.args.get('q', '').strip()
        clear_cache = request.args.get('clear_cache', 'false').lower() == 'true'
        force_search = request.args.get('force', 'false').lower() == 'true'
        
        if clear_cache or force_search:
            logger.info("🧹 Clearing document cache before search")
            config['document_cache'] = {}
            config['diff_cache'] = {}
        
        if not search_term:
            # Always refresh available docs when no search term
            logger.info("📋 No search term - refreshing available docs with Datatrove")
            config['available_docs'] = []  # Clear cache
            config['loading_status'] = 'ready'
            
            # Trigger background refresh
            def background_refresh():
                quick_scan_documents_datatrove(
                    Path(config['cleaned_dir']),
                    Path(config['gold_dir'])
                )
            
            thread = threading.Thread(target=background_refresh)
            thread.daemon = True
            thread.start()
            
            return jsonify({
                'success': True,
                'documents': config['available_docs'],
                'count': len(config['available_docs']),
                'searched': False,
                'message': 'Refreshing document list...'
            })
        
        logger.info(f"🔍 Datatrove search request: '{search_term}'")
        start_time = time.time()
        
        # Use Datatrove-based search method with higher limits
        cleaned_results = search_documents_with_datatrove(Path(config['cleaned_dir']), search_term, max_results=500)
        logger.info(f"API Search: Found {len(cleaned_results)} candidates in cleaned_dir for '{search_term}'")
        
        gold_results = search_documents_with_datatrove(Path(config['gold_dir']), search_term, max_results=500)
        logger.info(f"API Search: Found {len(gold_results)} candidates in gold_dir for '{search_term}'")
        
        # Find intersection
        all_candidates = sorted(list(set(cleaned_results + gold_results))) # Sort for consistent processing
        logger.info(f"API Search: Found {len(all_candidates)} unique candidates from both dirs for '{search_term}'. Candidates: {all_candidates[:20]}...") # Log first few
        
        # Find documents that exist in both directories
        intersecting_docs = get_intersecting_documents(
            Path(config['cleaned_dir']),
            Path(config['gold_dir']),
            all_candidates
        )
        
        # Sort results
        intersecting_docs = sorted(intersecting_docs)
        
        elapsed_time = time.time() - start_time
        logger.info(f"🏁 Search completed in {elapsed_time:.2f} seconds, found {len(intersecting_docs)} documents")
        
        return jsonify({
            'success': True,
            'documents': intersecting_docs,
            'count': len(intersecting_docs),
            'searched': True,
            'search_term': search_term,
            'elapsed_time': f"{elapsed_time:.2f}s"
        })
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/')
def index():
    """Main page with document list."""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>FAST Document Diff Viewer (Datatrove)</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        .stats { background: #e3f2fd; padding: 15px; border-radius: 4px; margin-bottom: 20px; }
        .search-box { margin: 20px 0; }
        .search-info { margin: 10px 0; padding: 10px; background: #fffde7; border-radius: 4px; font-size: 0.9em; }
        .search-box input { padding: 10px; width: 300px; border: 1px solid #ddd; border-radius: 4px; }
        .doc-item { 
            display: flex; justify-content: space-between; align-items: center;
            padding: 12px; margin: 5px 0; background: #f8f9fa; border-radius: 4px;
            border-left: 4px solid #007bff;
        }
        .doc-item:hover { background: #e9ecef; }
        .doc-id { font-family: monospace; font-weight: bold; }
        .btn { 
            padding: 6px 12px; margin: 0 2px; border: none; border-radius: 4px; 
            cursor: pointer; text-decoration: none; display: inline-block;
        }
        .btn-primary { background: #007bff; color: white; }
        .btn-secondary { background: #6c757d; color: white; }
        .btn-danger { background: #dc3545; color: white; }
        .btn-warning { background: #ffc107; color: #212529; }
        .btn:hover { opacity: 0.8; }
        .loading { text-align: center; padding: 20px; color: #666; }
        .status-ready { color: #28a745; }
        .status-scanning { color: #ffc107; }
        .status-error { color: #dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ FAST Document Diff Viewer (Datatrove)</h1>
        
        <div class="stats">
            <strong>Status:</strong> <span id="status">Starting...</span><br>
            <strong>Available Documents:</strong> <span id="doc-count">0</span><br>
            <strong>Mode:</strong> <span id="mode">Datatrove-powered search through all parquet files</span>
        </div>
        
        <div class="search-box">
            <input type="text" id="searchInput" placeholder="Search document IDs..." onkeyup="filterDocuments()">
            <button class="btn btn-success" onclick="performActiveSearch()">Suchen</button>
            <button class="btn btn-secondary" onclick="clearSearch()">Clear</button>
            <button class="btn btn-primary" onclick="forceSearch()">Force Search</button>
            <button class="btn btn-danger" onclick="clearCache()">Clear Cache</button>
            <button class="btn btn-warning" onclick="loadRandomDocuments()">🎲 Random 100</button>
        </div>
        
        <div class="search-info">
            <strong>Search Tips (Datatrove-powered):</strong>
            <ul>
                <li>🔍 Search for exact document IDs like <code>part-0.parquet/1</code></li>
                <li>⚡ Powered by Datatrove for better ID handling</li>
                <li>🔄 Search updates automatically as you type</li>
                <li>🧹 Use "Force Search" to search again with cache clearing</li>
                <li>📋 Document must exist in both gold and cleaned directories to appear</li>
            </ul>
        </div>
        
        <div id="documentList" class="document-list">
            <div class="loading">⚡ Loading documents...</div>
        </div>
    </div>

    <script>
        let allDocuments = [];
        let searchTimeout = null;
        
        window.onload = function() {
            loadDocuments();
            monitorStatus();
        };
        
        function loadDocuments() {
            fetch('/api/documents')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        allDocuments = data.documents;
                        document.getElementById('doc-count').textContent = data.count;
                        
                        // Update mode display based on current state
                        if (data.count > 0) {
                            document.getElementById('mode').textContent = 'Datatrove-powered search through all parquet files';
                        }
                        
                        displayDocuments(allDocuments);
                    } else {
                        document.getElementById('documentList').innerHTML = 
                            '<div class="error">Error: ' + data.error + '</div>';
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('documentList').innerHTML = 
                        '<div class="error">Error loading documents</div>';
                });
        }
        
        function displayDocuments(docs) {
            const container = document.getElementById('documentList');
            
            if (docs.length === 0) {
                container.innerHTML = '<div class="loading">No documents found</div>';
                return;
            }
            
            container.innerHTML = docs.map(docId => `
                <div class="doc-item">
                    <div class="doc-id">${docId}</div>
                    <div>
                        <a href="/diff/${encodeURIComponent(docId)}" class="btn btn-primary" target="_blank">View Diff</a>
                        <a href="/raw/${encodeURIComponent(docId)}/gold" class="btn btn-secondary" target="_blank">Gold</a>
                        <a href="/raw/${encodeURIComponent(docId)}/cleaned" class="btn btn-secondary" target="_blank">Cleaned</a>
                    </div>
                </div>
            `).join('');
        }
        
        function filterDocuments() {
            console.log("Frontend: filterDocuments() called."); // Logging
            const searchTerm = document.getElementById('searchInput').value.toLowerCase();
            
            // Clear previous timeout
            if (searchTimeout) {
                clearTimeout(searchTimeout);
            }
            
            if (searchTerm.length === 0) {
                // If search is empty, show all pre-loaded documents
                displayDocuments(allDocuments);
                document.getElementById('doc-count').textContent = allDocuments.length;
                return;
            }
            
            // Debounce search - wait 500ms after user stops typing
            searchTimeout = setTimeout(() => {
                executeSearch(searchTerm, false);
            }, 500);
        }
        
        function forceSearch() {
            const searchTerm = document.getElementById('searchInput').value.toLowerCase();
            if (searchTerm.length === 0) {
                alert("Please enter a search term first");
                return;
            }
            
            executeSearch(searchTerm, true);
        }
        
        function clearCache() {
            // Show loading indicator
            document.getElementById('documentList').innerHTML = 
                '<div class="loading">🧹 Clearing cache...</div>';
            
            // Call API to clear cache
            fetch('/api/clear_cache')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        const searchTerm = document.getElementById('searchInput').value.toLowerCase();
                        if (searchTerm.length > 0) {
                            executeSearch(searchTerm, false);
                        } else {
                            loadDocuments();
                        }
                    } else {
                        document.getElementById('documentList').innerHTML = 
                            '<div class="error">Error: ' + data.error + '</div>';
                    }
                })
                .catch(error => {
                    console.error('Cache clear error:', error);
                    document.getElementById('documentList').innerHTML = 
                        '<div class="error">Error clearing cache</div>';
                });
        }
        
        function performActiveSearch() {
            console.log("Frontend: performActiveSearch() called."); // Logging
            const searchTerm = document.getElementById('searchInput').value.toLowerCase();
            if (searchTerm.length === 0) {
                // Wenn Suchbegriff leer ist, alle initial geladenen Dokumente anzeigen
                console.log("Frontend: performActiveSearch - search term is empty, calling loadDocuments()."); // Logging
                loadDocuments();
                return;
            }
            executeSearch(searchTerm, false); // false, um Cache nicht standardmäßig zu löschen
        }
        
        function executeSearch(searchTerm, clearCache) {
            console.log(`Frontend: Starting search for term: "${searchTerm}", clearCache: ${clearCache}`); // Logging
            // Show loading indicator
            document.getElementById('documentList').innerHTML =
                '<div class="loading">🔍 Datatrove search for "' + searchTerm + '"...</div>';
            
            // Use the search API to search through all parquet files
            fetch(`/api/search?q=${encodeURIComponent(searchTerm)}&clear_cache=${clearCache}`)
                .then(response => response.json())
                .then(data => {
                    console.log('Frontend: Search API response received:', data); // Logging
                    if (data.success) {
                        displayDocuments(data.documents);
                        document.getElementById('doc-count').textContent =
                            `${data.count} (searched in ${data.elapsed_time})`;
                    } else {
                        console.error('Frontend: Search API error:', data.error); // Logging
                        document.getElementById('documentList').innerHTML =
                            '<div class="error">Error: ' + data.error + '</div>';
                    }
                })
                .catch(error => {
                    console.error('Frontend: Fetch error during search:', error); // Logging
                    document.getElementById('documentList').innerHTML =
                        '<div class="error">Error during search</div>';
                });
        }
        
        function clearSearch() {
            document.getElementById('searchInput').value = '';
            displayDocuments(allDocuments);
            document.getElementById('doc-count').textContent = allDocuments.length;
        }
        
        function loadRandomDocuments() {
            console.log("Frontend: loadRandomDocuments() called.");
            
            // Show loading indicator
            document.getElementById('documentList').innerHTML = 
                '<div class="loading">🎲 Loading 100 random documents with Datatrove...</div>';
            
            // Update mode display
            document.getElementById('mode').textContent = '🎲 Random 100 mode - loading...';
            
            // Call API to get random documents
            fetch('/api/random_documents?size=100')
                .then(response => response.json())
                .then(data => {
                    console.log('Frontend: Random documents API response:', data);
                    if (data.success) {
                        // Start monitoring for completion
                        monitorRandomLoading();
                    } else {
                        console.error('Frontend: Random documents API error:', data.error);
                        document.getElementById('documentList').innerHTML = 
                            '<div class="error">Error: ' + data.error + '</div>';
                        document.getElementById('mode').textContent = 'Datatrove-powered search through all parquet files';
                    }
                })
                .catch(error => {
                    console.error('Frontend: Fetch error during random load:', error);
                    document.getElementById('documentList').innerHTML = 
                        '<div class="error">Error loading random documents</div>';
                    document.getElementById('mode').textContent = 'Datatrove-powered search through all parquet files';
                });
        }
        
        function monitorRandomLoading() {
            // Check status every 500ms
            const checkStatus = () => {
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            const status = data.status;
                            
                            if (status === 'complete') {
                                // Loading complete, fetch the new documents
                                fetch('/api/documents')
                                    .then(response => response.json())
                                    .then(data => {
                                        if (data.success) {
                                            allDocuments = data.documents;
                                            document.getElementById('doc-count').textContent = data.count;
                                            document.getElementById('mode').textContent = '🎲 Random 100 mode - ' + data.count + ' documents';
                                            displayDocuments(allDocuments);
                                        }
                                    });
                            } else if (status === 'scanning') {
                                // Still loading, check again
                                setTimeout(checkStatus, 500);
                            } else if (status === 'error') {
                                document.getElementById('documentList').innerHTML = 
                                    '<div class="error">Error loading random documents</div>';
                                document.getElementById('mode').textContent = 'Datatrove-powered search through all parquet files';
                            }
                        }
                    })
                    .catch(error => {
                        console.error('Status check error:', error);
                        setTimeout(checkStatus, 1000);
                    });
            };
            
            checkStatus();
        }
        
        function monitorStatus() {
            function updateStatus() {
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            const statusEl = document.getElementById('status');
                            const status = data.status;
                            
                            statusEl.className = 'status-' + status;
                            
                            if (status === 'ready') {
                                statusEl.textContent = '✅ Ready';
                            } else if (status === 'scanning') {
                                statusEl.textContent = '⚡ Scanning...';
                                setTimeout(updateStatus, 1000);
                            } else if (status === 'complete') {
                                statusEl.textContent = '✅ Complete';
                            } else if (status === 'error') {
                                statusEl.textContent = '❌ Error';
                            }
                        }
                    })
                    .catch(() => {
                        setTimeout(updateStatus, 2000);
                    });
            }
            
            updateStatus();
        }
    </script>
</body>
</html>
    '''


@app.route('/api/documents')
def api_documents():
    """API endpoint to get available documents - IMMEDIATE RESPONSE."""
    try:
        # If no documents loaded yet, trigger quick scan
        if not config['available_docs'] and config['loading_status'] == 'ready':
            # Start background scan
            def background_scan():
                quick_scan_documents_datatrove(
                    Path(config['cleaned_dir']),
                    Path(config['gold_dir'])
                )
            
            thread = threading.Thread(target=background_scan)
            thread.daemon = True
            thread.start()
        
        return jsonify({
            'success': True,
            'documents': config['available_docs'],
            'count': len(config['available_docs']),
            'status': config['loading_status']
        })
    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/status')
def api_status():
    """API endpoint for loading status."""
    return jsonify({
        'success': True,
        'status': config['loading_status'],
        'loaded_count': config['loaded_count']
    })


@app.route('/api/clear_cache')
def api_clear_cache():
    """API endpoint to clear all document caches."""
    try:
        # Clear document and diff caches
        config['document_cache'] = {}
        config['diff_cache'] = {}
        
        logger.info("🧹 Cleared all document and diff caches")
        
        return jsonify({
            'success': True,
            'message': 'All caches cleared successfully'
        })
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/random_documents')
def api_random_documents():
    """API endpoint to get random sample of documents."""
    try:
        sample_size = request.args.get('size', 100, type=int)
        
        logger.info(f"🎲 Random documents request: {sample_size} documents")
        
        # Start background random scan
        def background_random_scan():
            get_random_documents_datatrove(
                Path(config['cleaned_dir']),
                Path(config['gold_dir']),
                sample_size
            )
        
        thread = threading.Thread(target=background_random_scan)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': f'Loading {sample_size} random documents...',
            'status': 'scanning',
            'random_mode': True
        })
        
    except Exception as e:
        logger.error(f"Error getting random documents: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/diff/<path:doc_id>')
def show_diff(doc_id):
    """Show HTML diff for a document."""
    try:
        logger.info(f"📄 Creating diff for: {doc_id}")
        
        # Check cache first
        if doc_id in config['diff_cache']:
            cached = config['diff_cache'][doc_id]
            if cached.get('status') == 'ready':
                response = make_response(cached['html_diff'])
                response.headers['Content-Type'] = 'text/html; charset=utf-8'
                return response
        
        # Load documents using Datatrove
        gold_text = get_document_by_id_datatrove(Path(config['gold_dir']), doc_id)
        cleaned_text = get_document_by_id_datatrove(Path(config['cleaned_dir']), doc_id)
        
        if not gold_text:
            return f"<h1>Error</h1><p>Document '{doc_id}' not found in gold dataset</p>", 404
        
        if not cleaned_text:
            return f"<h1>Error</h1><p>Document '{doc_id}' not found in cleaned dataset</p>", 404
        
        # Check if identical
        if gold_text == cleaned_text:
            return f"""
            <html>
            <head><title>Identical Documents</title></head>
            <body>
                <h1>✅ Documents are Identical</h1>
                <p><strong>Document ID:</strong> {doc_id}</p>
                <p>Gold and cleaned versions are exactly the same.</p>
                <p><a href="javascript:history.back()">← Back</a></p>
            </body>
            </html>
            """
        
        # Create diff
        html_diff = create_html_diff(gold_text, cleaned_text, "Gold Data", "Cleaned Data")
        
        # Cache the result
        config['diff_cache'][doc_id] = {
            'html_diff': html_diff,
            'status': 'ready'
        }
        
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
            text = get_document_by_id_datatrove(Path(config['gold_dir']), doc_id)
        elif doc_type == 'cleaned':
            text = get_document_by_id_datatrove(Path(config['cleaned_dir']), doc_id)
        else:
            return "Invalid document type. Use 'gold' or 'cleaned'.", 400
        
        if not text:
            return f"Document '{doc_id}' not found in {doc_type} dataset", 404
        
        html = f"""
        <html>
        <head>
            <title>{doc_type.title()} Document: {doc_id}</title>
            <style>
                body {{ font-family: monospace; margin: 20px; background: #f5f5f5; }}
                .container {{ background: white; padding: 20px; border-radius: 8px; }}
                pre {{ white-space: pre-wrap; word-wrap: break-word; }}
                .header {{ background: #e3f2fd; padding: 10px; margin-bottom: 20px; border-radius: 4px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>{doc_type.title()} Document</h2>
                    <p><strong>ID:</strong> {doc_id}</p>
                    <p><strong>Length:</strong> {len(text):,} characters</p>
                    <p><a href="javascript:history.back()">← Back</a></p>
                </div>
                <pre>{html.escape(text)}</pre>
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
    parser = argparse.ArgumentParser(description="FAST Web-based document diff viewer")
    
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
        "--port", "-p",
        type=int,
        default=5000,
        help="Port to run the web server on"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to"
    )
    parser.add_argument(
        "--ngrok",
        action="store_true",
        help="Start ngrok tunnel"
    )
    parser.add_argument(
        "--auth-token",
        default=os.environ.get('NGROK_AUTH_TOKEN'),
        help="ngrok auth token"
    )
    
    args = parser.parse_args()
    
    # Set configuration
    config['gold_dir'] = args.gold_dir
    config['cleaned_dir'] = args.cleaned_dir
    
    # Check directories exist
    if not Path(config['gold_dir']).exists():
        print(f"❌ Gold directory not found: {config['gold_dir']}")
        exit(1)
    
    if not Path(config['cleaned_dir']).exists():
        print(f"❌ Cleaned directory not found: {config['cleaned_dir']}")
        exit(1)
    
    print(f"⚡ Starting FAST Document Diff Viewer")
    print(f"Gold data: {config['gold_dir']}")
    print(f"Cleaned data: {config['cleaned_dir']}")
    print(f"Local server: http://{args.host}:{args.port}")
    
    # Start ngrok if requested
    if args.ngrok:
        print(f"\n🚀 Starting ngrok tunnel...")
        
        def start_tunnel():
            time.sleep(2)
            start_ngrok_tunnel(args.port, args.auth_token)
        
        tunnel_thread = threading.Thread(target=start_tunnel)
        tunnel_thread.daemon = True
        tunnel_thread.start()
    
    print(f"\n🚀 Starting server...")
    
    # Run the Flask app
    try:
        app.run(host=args.host, port=args.port, debug=False)
    except KeyboardInterrupt:
        print(f"\n👋 Shutting down...")