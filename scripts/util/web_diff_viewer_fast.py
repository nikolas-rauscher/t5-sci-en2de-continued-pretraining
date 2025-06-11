#!/usr/bin/env python3
"""
FAST Web-based document diff viewer - optimized for quick startup.
Provides immediate UI response while loading documents in background.
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
    'loaded_count': 0
}


def quick_scan_documents(cleaned_dir: Path, gold_dir: Path, max_files: int = 10, max_docs: int = 500):
    """
    Super fast document discovery - returns immediately with partial results.
    """
    logger.info(f"⚡ Quick scan starting...")
    config['loading_status'] = 'scanning'
    
    try:
        # Scan only first few files for immediate results
        cleaned_files = list(cleaned_dir.glob("*.parquet"))[:max_files]
        gold_files = list(gold_dir.glob("*.parquet"))[:max_files]
        
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
        
        logger.info(f"✅ Quick scan found {len(available_docs)} documents")
        return available_docs
        
    except Exception as e:
        logger.error(f"Quick scan error: {e}")
        config['loading_status'] = 'error'
        return []


def get_document_by_id_simple(directory: Path, doc_id: str):
    """Simple document retrieval without heavy caching."""
    cache_key = f"{directory}:{doc_id}"
    if cache_key in config['document_cache']:
        return config['document_cache'][cache_key]
    
    # Search in parquet files
    parquet_files = list(directory.glob("*.parquet"))
    
    for file_path in parquet_files:
        try:
            if file_path.stat().st_size == 0:
                continue
                
            # Use PyArrow filter for direct lookup
            table = pq.read_table(file_path, filters=[('id', '=', doc_id)])
            
            if len(table) > 0:
                df = table.to_pandas()
                text = df.iloc[0]['text']
                
                # Cache result
                config['document_cache'][cache_key] = text
                return text
                
        except Exception as e:
            continue
    
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


@app.route('/')
def index():
    """Main page with document list."""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>FAST Document Diff Viewer</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        .stats { background: #e3f2fd; padding: 15px; border-radius: 4px; margin-bottom: 20px; }
        .search-box { margin: 20px 0; }
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
        .btn:hover { opacity: 0.8; }
        .loading { text-align: center; padding: 20px; color: #666; }
        .status-ready { color: #28a745; }
        .status-scanning { color: #ffc107; }
        .status-error { color: #dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ FAST Document Diff Viewer</h1>
        
        <div class="stats">
            <strong>Status:</strong> <span id="status">Starting...</span><br>
            <strong>Available Documents:</strong> <span id="doc-count">0</span><br>
            <strong>Mode:</strong> Fast startup with limited document set for immediate response
        </div>
        
        <div class="search-box">
            <input type="text" id="searchInput" placeholder="Search document IDs..." onkeyup="filterDocuments()">
            <button class="btn btn-secondary" onclick="clearSearch()">Clear</button>
            <button class="btn btn-primary" onclick="loadDocuments()">Refresh</button>
        </div>
        
        <div id="documentList" class="document-list">
            <div class="loading">⚡ Loading documents...</div>
        </div>
    </div>

    <script>
        let allDocuments = [];
        
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
            const searchTerm = document.getElementById('searchInput').value.toLowerCase();
            const filteredDocs = allDocuments.filter(docId => 
                docId.toLowerCase().includes(searchTerm)
            );
            displayDocuments(filteredDocs);
        }
        
        function clearSearch() {
            document.getElementById('searchInput').value = '';
            displayDocuments(allDocuments);
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
                quick_scan_documents(
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
        
        # Load documents
        gold_text = get_document_by_id_simple(Path(config['gold_dir']), doc_id)
        cleaned_text = get_document_by_id_simple(Path(config['cleaned_dir']), doc_id)
        
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
            text = get_document_by_id_simple(Path(config['gold_dir']), doc_id)
        elif doc_type == 'cleaned':
            text = get_document_by_id_simple(Path(config['cleaned_dir']), doc_id)
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