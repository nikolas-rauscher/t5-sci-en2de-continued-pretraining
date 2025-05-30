#!/usr/bin/env python3
"""
DataTrove Statistik-Merger
===========================

Merged ungemergte DataTrove-Statistiken von mehreren Tasks in einzelne Dateien.

Usage:
    python scripts/merge_datatrove_stats.py                          # Merged data/statistics 
    python scripts/merge_datatrove_stats.py --input-dir PATH         # Merged spezifisches Verzeichnis
    python scripts/merge_datatrove_stats.py --output-dir PATH        # Output in anderes Verzeichnis
"""

import os
import sys
import argparse
from pathlib import Path
import re

# Projekt-Root und DataTrove ins PYTHONPATH
script_dir = os.path.dirname(__file__)
proj_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, proj_root)
sys.path.insert(0, os.path.join(proj_root, "external", "datatrove", "src"))

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.stats.merger import StatsMerger

def main():
    parser = argparse.ArgumentParser(description="Merged ungemergte DataTrove-Statistiken")
    parser.add_argument("--input-dir", help="Input-Verzeichnis mit ungemergten Stats (default: data/statistics)")
    parser.add_argument("--output-dir", help="Output-Verzeichnis für gemergte Stats (default: data/statistics_merged)")
    parser.add_argument("--remove-input", action="store_true", help="Entferne Input-Dateien nach dem Mergen")
    parser.add_argument("--tasks", type=int, default=4, help="Anzahl paralleler Tasks")
    parser.add_argument("--workers", type=int, default=2, help="Anzahl paralleler Worker")
    
    args = parser.parse_args()
    
    # Default-Pfade
    if args.input_dir:
        input_dir = args.input_dir
    else:
        input_dir = os.path.join(proj_root, "data", "statistics")
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(proj_root, "data", "statistics_merged")
    
    print("DataTrove Statistik-Merger")
    print("="*50)
    print(f"📂 Input:  {input_dir}")
    print(f"📁 Output: {output_dir}")
    
    # Prüfe Input-Verzeichnis
    if not os.path.exists(input_dir):
        print(f"❌ Input-Verzeichnis nicht gefunden: {input_dir}")
        return
    
    # Prüfe auf ungemergte JSON-Dateien
    input_path = Path(input_dir)
    json_files = list(input_path.rglob("*.json"))
    
    if not json_files:
        print(f"❌ Keine JSON-Dateien gefunden in: {input_dir}")
        return
    
    print(f"✅ Gefunden: {len(json_files)} JSON-Dateien")
    
    # Prüfe auf gemergte vs ungemergte Stats
    # Ungemergte haben meist viele nummerierte JSON-Dateien (00000.json, 00001.json, etc.)
    numbered_files = [f for f in json_files if re.match(r'^\d{5}\.json$', f.name)]
    metric_files = [f for f in json_files if f.name == 'metric.json']
    
    print(f"📊 Nummerierte Dateien (ungemergt): {len(numbered_files)}")
    print(f"📋 Metric.json Dateien (gemergt): {len(metric_files)}")
    
    if numbered_files == 0:
        print("✅ Statistiken scheinen bereits gemergt zu sein!")
        print("💡 Wenn Probleme auftreten, nutze die Original-DataTrove-Ordner")
        return
    
    print(f"\n🔗 Starte Merge-Prozess...")
    
    # Erstelle Output-Verzeichnis
    os.makedirs(output_dir, exist_ok=True)
    
    # StatsMerger Pipeline
    pipeline = [
        StatsMerger(
            input_folder=input_dir,
            output_folder=output_dir,
            remove_input=args.remove_input
        )
    ]
    
    # Führe Merge aus
    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=args.tasks,
        workers=args.workers,
        logging_dir=f"logs/merge_stats"
    )
    
    try:
        executor.run()
        print("✅ Merge erfolgreich abgeschlossen!")
        print(f"📁 Gemergte Statistiken in: {output_dir}")
        
        # Prüfe Ergebnis
        output_path = Path(output_dir)
        merged_metric_files = list(output_path.rglob("metric.json"))
        print(f"📋 Erstellte metric.json Dateien: {len(merged_metric_files)}")
        
        if args.remove_input:
            print("🗑️  Original-Dateien wurden entfernt")
        else:
            print("💾 Original-Dateien wurden beibehalten")
            
    except Exception as e:
        print(f"❌ Fehler beim Mergen: {e}")
        return

if __name__ == "__main__":
    main() 