#!/usr/bin/env python3
"""
DataTrove Statistik-Analyse
===========================

Einfaches aber umfassendes Script zur Analyse von DataTrove-Statistiken.
Automatische Erkennung, Plots und CSV-Export.

Usage:
    python scripts/analyze_stats.py                    # Analysiert neueste Statistiken
    python scripts/analyze_stats.py --stats-dir PATH   # Analysiert spezifisches Verzeichnis
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Projekt-Root ins PYTHONPATH
script_dir = os.path.dirname(__file__)
proj_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, proj_root)

# Optional: Plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('default')
    sns.set_palette("husl")
    PLOTTING_AVAILABLE = True
    print("Matplotlib/Seaborn verfügbar")
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Matplotlib/Seaborn nicht verfügbar. Nur CSV-Export.")

def find_stats_directories():
    """Findet alle verfügbaren Statistik-Verzeichnisse"""
    candidates = []
    
    # 1. Zentrale Statistiken
    central_stats = proj_root + "/data/statistics"
    if os.path.exists(central_stats):
        candidates.append(("Central Stats", central_stats))
    
    # 2. Neueste Hydra-Outputs
    outputs_dir = Path(proj_root) / "outputs"
    if outputs_dir.exists():
        for year_dir in outputs_dir.iterdir():
            if year_dir.is_dir():
                for run_dir in year_dir.iterdir():
                    if run_dir.is_dir():
                        stats_dir = run_dir / "stats"
                        if stats_dir.exists():
                            # Check if it has metric files (DataTrove uses numbered JSON files)
                            has_metrics = any(stats_dir.rglob("*.json"))
                            if has_metrics:
                                candidates.append((f"Hydra Run {year_dir.name}/{run_dir.name}", str(stats_dir)))
    
    # Sortiere nach neuester Änderung
    candidates_with_time = []
    for name, path in candidates:
        try:
            latest_file = max(Path(path).rglob("*.json"), key=lambda f: f.stat().st_mtime)
            mtime = latest_file.stat().st_mtime
            candidates_with_time.append((name, path, mtime))
        except ValueError:
            pass
    
    candidates_with_time.sort(key=lambda x: x[2], reverse=True)
    return [(name, path) for name, path, _ in candidates_with_time]

def load_all_metrics(stats_dir):
    """Lädt alle Metriken aus einem Stats-Verzeichnis"""
    stats_path = Path(stats_dir)
    all_data = defaultdict(lambda: defaultdict(dict))
    
    print(f"Lade Statistiken aus: {stats_dir}")
    
    # DataTrove-Struktur durchsuchen
    for stat_type_dir in stats_path.iterdir():
        if not stat_type_dir.is_dir():
            continue
            
        stat_type = stat_type_dir.name
        
        for grouping_dir in stat_type_dir.iterdir():
            if not grouping_dir.is_dir():
                continue
                
            grouping = grouping_dir.name
            
            for metric_dir in grouping_dir.iterdir():
                if not metric_dir.is_dir():
                    continue
                    
                # DataTrove uses numbered JSON files (00000.json, 00001.json, etc.)
                json_files = list(metric_dir.glob("*.json"))
                if json_files:
                    metric_name = metric_dir.name
                    
                    # If multiple files, merge them; otherwise use the single file
                    if len(json_files) == 1:
                        try:
                            with open(json_files[0], 'r') as f:
                                data = json.load(f)
                            all_data[stat_type][f"{metric_name}_{grouping}"] = data
                        except Exception as e:
                            print(f"Fehler beim Laden {json_files[0]}: {e}")
                    else:
                        # Multiple files - PROPERLY aggregate them!
                        print(f"Aggregiere {len(json_files)} JSON-Dateien in {metric_dir}")
                        try:
                            # Load all files and aggregate
                            total_sum = 0
                            total_count = 0
                            min_val = float('inf')
                            max_val = float('-inf')
                            all_totals = []
                            
                            for json_file in json_files:
                                with open(json_file, 'r') as f:
                                    data = json.load(f)
                                if 'summary' in data:
                                    stats = data['summary']
                                    total_sum += stats.get('total', 0)
                                    total_count += stats.get('n', 0)
                                    min_val = min(min_val, stats.get('min', float('inf')))
                                    max_val = max(max_val, stats.get('max', float('-inf')))
                                    all_totals.append(stats.get('total', 0))
                            
                            # Calculate aggregated stats
                            if total_count > 0:
                                aggregated_data = {
                                    "summary": {
                                        "total": total_sum,
                                        "n": total_count,
                                        "mean": total_sum / total_count,
                                        "min": min_val if min_val != float('inf') else 0,
                                        "max": max_val if max_val != float('-inf') else 0,
                                        "unit": data.get('summary', {}).get('unit', 'task')
                                    }
                                }
                                all_data[stat_type][f"{metric_name}_{grouping}"] = aggregated_data
                            
                        except Exception as e:
                            print(f"Fehler beim Aggregieren von {metric_dir}: {e}")
                            # Fallback: use first file
                            try:
                                with open(json_files[0], 'r') as f:
                                    data = json.load(f)
                                all_data[stat_type][f"{metric_name}_{grouping}"] = data
                            except Exception as e2:
                                print(f"Fallback-Fehler: {e2}")
    
    return all_data

def extract_summary_stats(all_data):
    """Extrahiert Summary-Statistiken für einfache Analyse"""
    summary_data = []
    
    for stat_type, metrics in all_data.items():
        for metric_key, data in metrics.items():
            metric_name, grouping = metric_key.rsplit('_', 1)
            
            if grouping == "summary" and "summary" in data and isinstance(data["summary"], dict):
                stats = data["summary"]
                summary_data.append({
                    'stat_type': stat_type,
                    'metric': metric_name,
                    'count': stats.get('n', 0),  # DataTrove uses 'n' for count
                    'mean': stats.get('mean', 0.0),
                    'std': stats.get('std_dev', 0.0),  # DataTrove uses 'std_dev'
                    'min': stats.get('min', 0.0),
                    'max': stats.get('max', 0.0),
                    'p25': stats.get('p25', stats.get('mean', 0.0)),  # Fallback to mean if no percentiles
                    'p50': stats.get('p50', stats.get('mean', 0.0)),
                    'p75': stats.get('p75', stats.get('mean', 0.0)),
                    'p95': stats.get('p95', stats.get('mean', 0.0)),
                    'p99': stats.get('p99', stats.get('mean', 0.0)),
                    'total': stats.get('total', 0.0),
                    'variance': stats.get('variance', 0.0)
                })
    
    return pd.DataFrame(summary_data)

def print_overview(df):
    """Druckt eine Übersicht der Statistiken"""
    print("\n" + "="*80)
    print("DATATROVE STATISTIK-ÜBERSICHT")
    print("="*80)
    
    if df.empty:
        print("Keine Summary-Statistiken gefunden!")
        return
    
    print(f"\nGesamt: {len(df)} Metriken in {df['stat_type'].nunique()} Kategorien")
    print(f"Dokumente verarbeitet: {df['count'].iloc[0]:,} (bei allen Metriken)")
    
    for stat_type in df['stat_type'].unique():
        subset = df[df['stat_type'] == stat_type]
        print(f"\n{stat_type.upper()} ({len(subset)} Metriken)")
        print("-" * 60)
        
        for _, row in subset.iterrows():
            print(f"  {row['metric']:20} | "
                  f"Mean: {row['mean']:8.2f} | "
                  f"Std: {row['std']:8.2f} | "
                  f"P50: {row['p50']:8.2f} | "
                  f"P95: {row['p95']:8.2f}")

def create_plots(df, output_dir="data/statistics/analysis/basic_plots"):
    """Erstellt umfassende Plots der Statistiken"""
    if not PLOTTING_AVAILABLE:
        print("Plotting nicht verfügbar")
        return
    
    if df.empty:
        print("Keine Daten für Plots")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nErstelle Plots in: {output_dir}/")
    
    # 1. Übersichts-Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('DataTrove Statistics Overview', fontsize=16, fontweight='bold')
    
    # Plot 1: Dokument-Counts (sollten alle gleich sein)
    unique_counts = df['count'].nunique()
    if unique_counts == 1:
        axes[0, 0].bar(['All Metrics'], [df['count'].iloc[0]])
        axes[0, 0].set_title('Processed Documents')
        axes[0, 0].set_ylabel('Document Count')
        axes[0, 0].ticklabel_format(style='plain', axis='y')
    
    # Plot 2: Mittelwerte pro Statistik-Typ
    if df['stat_type'].nunique() > 1:
        mean_by_type = df.groupby('stat_type')['mean'].mean()
        axes[0, 1].bar(mean_by_type.index, mean_by_type.values)
        axes[0, 1].set_title('Average Values by Stat Type')
        axes[0, 1].set_ylabel('Mean Value')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Verteilung der Mittelwerte
    axes[1, 0].hist(df['mean'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Distribution of Mean Values')
    axes[1, 0].set_xlabel('Mean Value')
    axes[1, 0].set_ylabel('Frequency')
    
    # Plot 4: Mean vs Std Scatter
    colors = plt.cm.tab10(np.linspace(0, 1, df['stat_type'].nunique()))
    for i, stat_type in enumerate(df['stat_type'].unique()):
        subset = df[df['stat_type'] == stat_type]
        axes[1, 1].scatter(subset['mean'], subset['std'], 
                          c=[colors[i]], label=stat_type, alpha=0.7, s=60)
    
    axes[1, 1].set_title('Mean vs Standard Deviation')
    axes[1, 1].set_xlabel('Mean')
    axes[1, 1].set_ylabel('Standard Deviation')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/overview_dashboard.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detaillierte Plots pro Statistik-Typ
    for stat_type in df['stat_type'].unique():
        subset = df[df['stat_type'] == stat_type]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{stat_type.upper()} - Detailed Analysis', fontsize=14, fontweight='bold')
        
        # Mittelwerte
        axes[0, 0].bar(subset['metric'], subset['mean'])
        axes[0, 0].set_title('Mean Values')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylabel('Mean')
        
        # Standardabweichungen
        axes[0, 1].bar(subset['metric'], subset['std'])
        axes[0, 1].set_title('Standard Deviations')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_ylabel('Std Dev')
        
        # Perzentile
        x = np.arange(len(subset))
        width = 0.2
        axes[1, 0].bar(x - width, subset['p25'], width, label='P25', alpha=0.8)
        axes[1, 0].bar(x, subset['p50'], width, label='P50', alpha=0.8)
        axes[1, 0].bar(x + width, subset['p95'], width, label='P95', alpha=0.8)
        axes[1, 0].set_title('Percentiles')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(subset['metric'], rotation=45)
        axes[1, 0].legend()
        axes[1, 0].set_ylabel('Value')
        
        # Min/Max
        axes[1, 1].bar(x - width/2, subset['min'], width, label='Min', alpha=0.8)
        axes[1, 1].bar(x + width/2, subset['max'], width, label='Max', alpha=0.8)
        axes[1, 1].set_title('Min/Max Values')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(subset['metric'], rotation=45)
        axes[1, 1].legend()
        axes[1, 1].set_ylabel('Value')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{stat_type}_detailed.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Plots erstellt:")
    print(f"  {output_dir}/overview_dashboard.png")
    for stat_type in df['stat_type'].unique():
        print(f"  {output_dir}/{stat_type}_detailed.png")

def export_csv(df, all_data, output_file="data/statistics/analysis/datatrove_analysis.csv"):
    """Exportiert detaillierte CSV-Analyse"""
    if df.empty:
        print("Keine Daten für CSV-Export")
        return
    
    # Erweitere DataFrame mit zusätzlichen Infos
    df_export = df.copy()
    df_export['analysis_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Berechne zusätzliche Metriken
    df_export['range'] = df_export['max'] - df_export['min']
    df_export['cv'] = df_export['std'] / df_export['mean']  # Coefficient of Variation
    df_export['iqr'] = df_export['p75'] - df_export['p25']  # Interquartile Range
    
    # Sortiere für bessere Lesbarkeit
    df_export = df_export.sort_values(['stat_type', 'metric'])
    
    # Erstelle Ordner falls nicht vorhanden
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Exportiere
    df_export.to_csv(output_file, index=False, float_format='%.6f')
    print(f"\nCSV exportiert: {output_file}")
    print(f"   Spalten: {len(df_export.columns)}")
    print(f"   Zeilen: {len(df_export)}")
    
    return df_export

def main():
    parser = argparse.ArgumentParser(description="DataTrove Statistik-Analyse")
    parser.add_argument("--stats-dir", help="Spezifisches Statistik-Verzeichnis")
    parser.add_argument("--no-plots", action="store_true", help="Keine Plots erstellen")
    parser.add_argument("--no-csv", action="store_true", help="Kein CSV-Export")
    
    args = parser.parse_args()
    
    print("DataTrove Statistik-Analyse")
    print("="*50)
    
    # Finde Statistik-Verzeichnis
    if args.stats_dir:
        if not os.path.exists(args.stats_dir):
            print(f"Verzeichnis nicht gefunden: {args.stats_dir}")
            return
        stats_dir = args.stats_dir
        print(f"Analysiere: {stats_dir}")
    else:
        print("Suche nach Statistik-Verzeichnissen...")
        candidates = find_stats_directories()
        
        if not candidates:
            print("Keine Statistiken gefunden!")
            print("   Versuche: python scripts/analyze_stats.py --stats-dir YOUR_PATH")
            return
        
        stats_dir = candidates[0][1]  # Nehme neuestes
        print(f"Neuestes gefunden: {candidates[0][0]} -> {stats_dir}")
    
    # Lade alle Metriken
    all_data = load_all_metrics(stats_dir)
    
    if not all_data:
        print("Keine Metriken gefunden!")
        return
    
    # Extrahiere Summary-Stats
    df = extract_summary_stats(all_data)
    
    if df.empty:
        print("Keine Summary-Statistiken gefunden!")
        return
    
    # Zeige Übersicht
    print_overview(df)
    
    # Erstelle Plots
    if not args.no_plots:
        create_plots(df)
    
    # CSV-Export
    if not args.no_csv:
        export_csv(df, all_data)
    
    print(f"\nAnalyse abgeschlossen!")

if __name__ == "__main__":
    main() 