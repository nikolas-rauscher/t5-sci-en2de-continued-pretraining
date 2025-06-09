#!/usr/bin/env python3
"""
Analysiert die Ergebnisse einer DataTrove-Bereinigungspipeline, indem es
die von wandb generierten Tabellen auswertet.

Dieses Skript extrahiert Dokument-IDs aus allen .table.json-Dateien eines Laufs,
vergleicht diese Dokumente mit ihren Originalversionen und speichert detaillierte
Diff-Berichte für die weitere Analyse.
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Füge Projekt-Root zum sys.path hinzu, um Importe aus scripts.util zu ermöglichen
script_dir = os.path.dirname(__file__)
proj_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, proj_root)

try:
    from scripts.util.compare_documents import search_in_directory, create_text_diff, save_comparison_results
except ImportError:
    print("Fehler: Konnte Hilfsfunktionen aus 'scripts/util/compare_documents.py' nicht importieren.")
    print("Stelle sicher, dass das Skript im 'scripts'-Verzeichnis ausgeführt wird.")
    sys.exit(1)


def extract_doc_ids_from_tables(tables_dir: Path) -> set:
    """
    Extrahiert alle eindeutigen Dokument-IDs aus den wandb JSON-Tabellendateien.

    Args:
        tables_dir: Der Pfad zum Verzeichnis, das die .table.json-Dateien enthält.

    Returns:
        Ein Set von eindeutigen Dokument-IDs.
    """
    doc_ids = set()
    table_files = list(tables_dir.glob("*.table.json"))

    if not table_files:
        print(f"Warnung: Keine .table.json Dateien in {tables_dir} gefunden.")
        return doc_ids

    print(f"Lese {len(table_files)} Tabellen-Dateien...")
    for table_file in table_files:
        try:
            with open(table_file, 'r', encoding='utf-8') as f:
                table_data = json.load(f)
            
            if not isinstance(table_data, dict) or "columns" not in table_data or "data" not in table_data:
                print(f"Warnung: {table_file.name} hat kein gültiges Tabellenformat.")
                continue

            columns = table_data["columns"]
            if "Doc ID" not in columns:
                print(f"Warnung: Spalte 'Doc ID' nicht in {table_file.name} gefunden.")
                continue

            doc_id_index = columns.index("Doc ID")
            
            count = 0
            for row in table_data["data"]:
                if isinstance(row, list) and len(row) > doc_id_index:
                    doc_ids.add(row[doc_id_index])
                    count += 1
            print(f"  - {table_file.name}: {count} Dokument-IDs extrahiert.")

        except (json.JSONDecodeError, IndexError) as e:
            print(f"Warnung: Konnte Tabelle {table_file.name} nicht verarbeiten: {e}")
            continue

    return doc_ids


def main():
    parser = argparse.ArgumentParser(
        description="Analysiert Cleaning-Ergebnisse und erstellt Diff-Berichte für Dokumente aus wandb-Tabellen.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("run_dir", help="Pfad zum `wandb`-Laufverzeichnis (z.B. wandb/run-xxx-yyy)")
    parser.add_argument(
        "--gold-dir",
        default="data/statistics_data_gold/enriched_documents_statistics_v2",
        help="Verzeichnis mit den originalen Parquet-Dateien"
    )
    parser.add_argument(
        "--cleaned-dir", 
        default="data/cleaned",
        help="Verzeichnis mit den bereinigten Parquet-Dateien"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="cleaning_analysis",
        help="Basis-Ausgabeverzeichnis für die Vergleichsdateien"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Zeigt detaillierten Suchfortschritt an"
    )
    
    args = parser.parse_args()

    run_path = Path(args.run_dir)
    if not run_path.is_dir():
        print(f"Fehler: Das angegebene Lauf-Verzeichnis '{run_path}' existiert nicht.")
        sys.exit(1)

    # Finde die Tabellen-Dateien
    tables_dir = run_path / "files" / "media" / "table" / "tables"
    if not tables_dir.is_dir():
        print(f"Fehler: Das Tabellen-Verzeichnis '{tables_dir}' wurde im Lauf-Verzeichnis nicht gefunden.")
        sys.exit(1)

    # Extrahiere Dokument-IDs aus allen Tabellen
    unique_doc_ids = extract_doc_ids_from_tables(tables_dir)

    if not unique_doc_ids:
        print("Keine Dokument-IDs in den Tabellendateien gefunden. Beende.")
        return

    print(f"\nInsgesamt {len(unique_doc_ids)} eindeutige Dokumente werden für die Analyse vorbereitet.")

    # Bereite Verzeichnisse vor
    gold_dir = Path(args.gold_dir)
    cleaned_dir = Path(args.cleaned_dir)
    run_id = run_path.name
    output_path = Path(args.output_dir) / "diffs" / run_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nVergleiche Dokumente und speichere Diffs nach '{output_path}'...")
    print("-" * 60)

    # Vergleiche jedes Dokument und speichere die Ergebnisse
    processed_count = 0
    for i, doc_id in enumerate(sorted(list(unique_doc_ids)), 1):
        print(f"[{i}/{len(unique_doc_ids)}] Verarbeite Dokument-ID: '{doc_id}'")
        
        gold_doc = search_in_directory(gold_dir, doc_id, args.verbose)
        cleaned_doc = search_in_directory(cleaned_dir, doc_id, args.verbose)

        if not gold_doc:
            print(f"  ✗ Warnung: Dokument '{doc_id}' nicht im Gold-Verzeichnis '{gold_dir}' gefunden.")
            continue
        if not cleaned_doc:
            print(f"  ✗ Warnung: Dokument '{doc_id}' nicht im Cleaned-Verzeichnis '{cleaned_dir}' gefunden.")
            continue
            
        print("  ✓ Dokument in beiden Verzeichnissen gefunden. Erstelle Diff...")
        
        diff_text, stats = create_text_diff(
            gold_doc['text'], 
            cleaned_doc['text'],
            f"Gold ({Path(gold_doc['found_in_file']).name})",
            f"Cleaned ({Path(cleaned_doc['found_in_file']).name})"
        )
        
        save_comparison_results(gold_doc, cleaned_doc, diff_text, stats, output_path, doc_id)
        processed_count += 1
        print("-" * 60)

    print("\nAnalyse abgeschlossen!")
    print(f"{processed_count}/{len(unique_doc_ids)} Dokumente erfolgreich verglichen.")
    print(f"Die Diff-Berichte wurden in '{output_path}' gespeichert.")

if __name__ == "__main__":
    main() 