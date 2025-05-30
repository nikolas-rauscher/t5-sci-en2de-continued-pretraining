#!/usr/bin/env python3
"""
DataTrove Bucket-basierte Histogram-Plots
=========================================

Erstellt echte Histogramme aus DataTrove Bucket-Daten mit intelligentem Sampling.

Usage:
    python scripts/create_datatrove_plots.py                    # Analysiert data/statistics
    python scripts/create_datatrove_plots.py --stats-dir PATH   # Analysiert spezifisches Verzeichnis
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Projekt-Root ins PYTHONPATH
script_dir = os.path.dirname(__file__)
proj_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, proj_root)

# Plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('default')
    sns.set_palette("husl")
    PLOTTING_AVAILABLE = True
    print("✅ Matplotlib/Seaborn verfügbar")
except ImportError:
    PLOTTING_AVAILABLE = False
    print("❌ Matplotlib/Seaborn nicht verfügbar!")
    sys.exit(1)

def create_datatrove_plots(stats_dir="data/statistics", output_dir=None):
    """Erstellt echte Histogramme aus den DataTrove Bucket-Daten mit intelligentem Sampling"""
    
    print("  📊 Erstelle echte Histogramme aus DataTrove Bucket-Daten...")
    
    # Output-Verzeichnis bestimmen
    if output_dir is None:
        output_dir = Path(stats_dir) / "analysis" / "datatrove_plots"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  📁 Output-Verzeichnis: {output_dir}")
    
    # Sammle alle verfügbaren Bucket-Daten direkt aus den Histogram-Dateien
    bucket_data = {}
    
    # Suche nach DataTrove Statistik-Struktur: stats_dir/{stat_type}/histogram/{metric}/
    stats_path = Path(stats_dir)
    
    if not stats_path.exists():
        print(f"  ❌ Statistik-Verzeichnis nicht gefunden: {stats_dir}")
        return
    
    # Finde alle *_stats Ordner
    stats_folders = [d for d in stats_path.iterdir() if d.is_dir() and d.name.endswith('_stats')]
    
    if not stats_folders:
        # Alternative: alle Ordner mit histogram/ Unterordner
        stats_folders = [d for d in stats_path.iterdir() if d.is_dir() and (d / "histogram").exists()]
    
    print(f"  🔍 Gefundene Stats-Ordner: {[f.name for f in stats_folders]}")
    
    for stat_folder in stats_folders:
        histogram_dir = stat_folder / "histogram"
        if not histogram_dir.exists():
            continue
            
        print(f"    📂 Durchsuche {stat_folder.name}/histogram/")
        
        for metric_dir in histogram_dir.iterdir():
            if not metric_dir.is_dir():
                continue
                
            metric_name = metric_dir.name
            
            # Suche nach metric.json oder mehreren JSON-Dateien
            metric_files = []
            
            # Erst nach metric.json suchen
            main_metric_file = metric_dir / "metric.json"
            if main_metric_file.exists():
                metric_files = [main_metric_file]
            else:
                # Falls keine metric.json, suche nach nummerierten JSON-Dateien
                json_files = list(metric_dir.glob("*.json"))
                if json_files:
                    metric_files = sorted(json_files)  # Sortiere für konsistente Reihenfolge
                    print(f"      🔍 Gefunden: {len(json_files)} JSON-Dateien in {metric_name}")
            
            if not metric_files:
                continue
            
            try:
                # Lade und kombiniere alle JSON-Dateien für diese Metrik
                combined_histogram = {}
                total_file_size = 0
                
                for metric_file in metric_files:
                    file_size = metric_file.stat().st_size
                    total_file_size += file_size
                    
                    # Sehr große einzelne Dateien überspringen
                    if file_size > 500 * 1024 * 1024:  # 500MB pro Datei
                        file_size_mb = file_size / (1024 * 1024)
                        print(f"      ⚠️  Überspringe {metric_file.name} (zu groß: {file_size_mb:.1f}MB)")
                        continue
                
                # Gesamtdateigröße prüfen
                total_size_mb = total_file_size / (1024 * 1024)
                if total_size_mb > 1000:  # 1GB Limit für alle Dateien zusammen
                    print(f"      ⚠️  Überspringe {metric_name} (Gesamtgröße zu groß: {total_size_mb:.1f}MB)")
                    continue
                
                print(f"      🔍 Lade {stat_folder.name}/{metric_name} ({len(metric_files)} Dateien, {total_size_mb:.1f}MB)")
                
                # Lade alle JSON-Dateien und kombiniere die Histogramme
                for i, metric_file in enumerate(metric_files):
                    with open(metric_file, 'r') as f:
                        histogram_data = json.load(f)
                    
                    if isinstance(histogram_data, dict):
                        # Kombiniere Histogramme (addiere Counts für gleiche Buckets)
                        for bucket_value, count_data in histogram_data.items():
                            if bucket_value in combined_histogram:
                                # Addiere zu existierendem Bucket
                                if isinstance(count_data, dict):
                                    existing_count = combined_histogram[bucket_value].get('n', combined_histogram[bucket_value].get('count', 0))
                                    new_count = count_data.get('n', count_data.get('count', 0))
                                    combined_histogram[bucket_value] = {'n': existing_count + new_count}
                                else:
                                    combined_histogram[bucket_value] += int(count_data)
                            else:
                                # Neuer Bucket
                                combined_histogram[bucket_value] = count_data
                
                if not combined_histogram:
                    print(f"      ⚠️  Keine gültigen Daten in {metric_name}")
                    continue
                
                # Konvertiere zu numerischen Buckets mit intelligentem Sampling
                buckets = {}
                total_docs = 0
                valid_entries = 0
                
                # Intelligentes Sampling basierend auf Gesamtdateigröße
                all_items = list(combined_histogram.items())
                
                if total_size_mb > 200:  # Random Sampling für >200MB
                    import random
                    sample_size = min(5000, len(all_items) // 10)  # Max 5000 Buckets oder 10%
                    sampled_items = random.sample(all_items, sample_size) if len(all_items) > sample_size else all_items
                    sample_info = f"Random Sample {len(sampled_items)}/{len(all_items)}"
                elif total_size_mb > 100:  # Systematic Sampling für >100MB
                    step = max(1, len(all_items) // 8000)  # Max 8000 Buckets
                    sampled_items = all_items[::step]
                    sample_info = f"Systematic Sample {len(sampled_items)}/{len(all_items)}"
                elif total_size_mb > 50:  # Leichtes Sampling für >50MB
                    step = max(1, len(all_items) // 15000)  # Max 15000 Buckets
                    sampled_items = all_items[::step]
                    sample_info = f"Light Sample {len(sampled_items)}/{len(all_items)}"
                else:
                    sampled_items = all_items
                    sample_info = "Vollständig"
                
                # Verarbeite sampled items
                for bucket_value, count_data in sampled_items:
                    try:
                        value = float(bucket_value)
                        
                        # Handle verschiedene DataTrove Formate
                        if isinstance(count_data, dict):
                            count = count_data.get('n', count_data.get('count', 0))
                        else:
                            count = int(count_data)
                        
                        if count > 0:
                            buckets[value] = count
                            total_docs += count
                            valid_entries += 1
                    
                    except (ValueError, TypeError, KeyError):
                        continue
                
                if buckets and valid_entries > 10:  # Mindestens 10 gültige Buckets
                    key = f"{stat_folder.name}_{metric_name}"
                    bucket_data[key] = {
                        'buckets': buckets,
                        'total_docs': total_docs,
                        'sample_info': sample_info,
                        'file_size_mb': total_size_mb,
                        'num_files': len(metric_files)
                    }
                    print(f"      ✅ {key}: {valid_entries} Buckets, {total_docs:,} Docs ({sample_info})")
                else:
                    print(f"      ⚠️  Unzureichende Daten: {valid_entries} Buckets")
                        
            except Exception as e:
                print(f"      ❌ Fehler beim Laden {metric_name}: {e}")
    
    if not bucket_data:
        print("  ⚠️  Keine Bucket-Daten gefunden für Plots")
        return
    
    print(f"  📊 Erstelle Histogramme für {len(bucket_data)} Metriken...")
    
    # Wähle wichtige Metriken für Übersichtsplot
    important_metrics = [
        'doc_stats_length__chars',
        'token_stats_token_count__chars', 
        'line_stats_n_lines__chars',
        'paragraph_stats_n_paragraphs__chars',
        'word_stats_n_words__chars',
        'sentence_stats_n_sentences__chars'
    ]
    
    # Filtere verfügbare wichtige Metriken
    available_important = [m for m in important_metrics if m in bucket_data]
    if not available_important:
        # Falls keine wichtigen Metriken, nimm die ersten verfügbaren
        available_important = list(bucket_data.keys())[:6]
    
    # 1. Übersichtsplot mit den wichtigsten Metriken
    n_metrics = min(6, len(available_important))
    if n_metrics >= 4:
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        axes = axes.flatten()
    elif n_metrics >= 2:
        fig, axes = plt.subplots(1, n_metrics, figsize=(8*n_metrics, 8))
        if n_metrics == 1:
            axes = [axes]
    else:
        # Einzelplot
        fig, axes = plt.subplots(1, 1, figsize=(12, 8))
        axes = [axes]
    
    fig.suptitle('DataTrove - Echte Verteilungen aus Histogram-Buckets', fontsize=20, fontweight='bold')
    
    for i, metric_key in enumerate(available_important[:n_metrics]):
        ax = axes[i] if i < len(axes) else axes[-1]
        
        data = bucket_data[metric_key]
        buckets = data['buckets']
        total_docs = data['total_docs']
        sample_info = data['sample_info']
        
        # Sortiere Buckets nach Werten
        sorted_items = sorted(buckets.items())
        values = np.array([x[0] for x in sorted_items])
        counts = np.array([x[1] for x in sorted_items])
        
        # Erstelle Histogram mit besserer Visualisierung
        if len(values) > 100:
            # Für viele Buckets: Line Plot
            ax.plot(values, counts, linewidth=2, alpha=0.8, color='steelblue')
            ax.fill_between(values, counts, alpha=0.3, color='steelblue')
        else:
            # Für wenige Buckets: Bar Plot
            ax.bar(values, counts, alpha=0.7, edgecolor='black', linewidth=0.5, color='steelblue')
        
        # Formatierung
        metric_name = metric_key.replace('_stats_', ': ').replace('__', ' (').replace('_', ' ').title()
        if '(' in metric_name and not metric_name.endswith(')'):
            metric_name += ')'
        
        ax.set_title(f'{metric_name}\n{total_docs:,} Dokumente • {sample_info}', fontweight='bold')
        ax.set_xlabel('Wert')
        ax.set_ylabel('Anzahl Dokumente')
        ax.grid(alpha=0.3)
        
        # Intelligente Achsen-Formatierung
        if max(counts) > 1000:
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        if max(values) > 10000:
            ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        
        # Log-Scale für bessere Sichtbarkeit bei extremen Unterschieden
        if len(counts) > 0 and max(counts) / min([c for c in counts if c > 0]) > 1000:
            ax.set_yscale('log')
            ax.set_ylabel('Anzahl Dokumente (log)')
        
        # Statistiken einblenden
        mean_val = sum(v * c for v, c in zip(values, counts)) / total_docs
        median_idx = total_docs // 2
        cumsum = np.cumsum(counts)
        median_val = values[np.searchsorted(cumsum, median_idx)]
        
        stats_text = f'Mean: {mean_val:.1f}\nMedian: {median_val:.1f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Verstecke ungenutzte Subplots
    for j in range(n_metrics, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    overview_file = output_dir / "datatrove_histograms_overview.png"
    plt.savefig(overview_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✅ Übersichtsplot: {overview_file.name}")
    
    # 2. Detaillierte Einzelplots für alle Metriken (Top 10)
    top_metrics = sorted(bucket_data.items(), key=lambda x: x[1]['total_docs'], reverse=True)[:10]
    
    for metric_key, data in top_metrics:
        buckets = data['buckets']
        total_docs = data['total_docs']
        sample_info = data['sample_info']
        file_size_mb = data['file_size_mb']
        num_files = data['num_files']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(f'{metric_key.replace("_", " ").title()}\n{total_docs:,} Dokumente • {sample_info} • {file_size_mb:.1f}MB ({num_files} Dateien)', 
                     fontsize=14, fontweight='bold')
        
        # Sortiere Buckets
        sorted_items = sorted(buckets.items())
        values = np.array([x[0] for x in sorted_items])
        counts = np.array([x[1] for x in sorted_items])
        
        # Plot 1: Verteilung (Linear Scale)
        if len(values) > 200:
            ax1.plot(values, counts, linewidth=1.5, alpha=0.8, color='steelblue')
            ax1.fill_between(values, counts, alpha=0.3, color='steelblue')
        else:
            ax1.bar(values, counts, alpha=0.7, edgecolor='black', linewidth=0.3, color='steelblue')
        
        ax1.set_title('Verteilung (Linear)', fontweight='bold')
        ax1.set_xlabel('Wert')
        ax1.set_ylabel('Anzahl Dokumente')
        ax1.grid(alpha=0.3)
        
        # Plot 2: Verteilung (Log Scale) - nur wenn sinnvoll
        if len([c for c in counts if c > 0]) > 10 and max(counts) / min([c for c in counts if c > 0]) > 10:
            if len(values) > 200:
                ax2.plot(values, counts, linewidth=1.5, alpha=0.8, color='orange')
                ax2.fill_between(values, counts, alpha=0.3, color='orange')
            else:
                ax2.bar(values, counts, alpha=0.7, edgecolor='black', linewidth=0.3, color='orange')
            
            ax2.set_yscale('log')
            ax2.set_title('Verteilung (Log Scale)', fontweight='bold')
            ax2.set_xlabel('Wert')
            ax2.set_ylabel('Anzahl Dokumente (log)')
            ax2.grid(alpha=0.3)
        else:
            # Cumulative Distribution Plot stattdessen
            cumsum_norm = np.cumsum(counts) / total_docs
            ax2.plot(values, cumsum_norm, linewidth=2, color='green', marker='o', markersize=2)
            ax2.set_title('Kumulative Verteilung', fontweight='bold')
            ax2.set_xlabel('Wert')
            ax2.set_ylabel('Kumulative Wahrscheinlichkeit')
            ax2.grid(alpha=0.3)
            ax2.set_ylim(0, 1)
        
        # Statistiken berechnen und anzeigen
        mean_val = sum(v * c for v, c in zip(values, counts)) / total_docs
        
        # Perzentile berechnen
        cumsum = np.cumsum(counts)
        percentiles = {}
        for p in [25, 50, 75, 95, 99]:
            target = (p / 100.0) * total_docs
            idx = np.searchsorted(cumsum, target)
            if idx < len(values):
                percentiles[p] = values[idx]
        
        stats_text = f"""Statistiken:
Mean: {mean_val:.1f}
Min: {values[0]:.1f}
P25: {percentiles.get(25, 'N/A')}
P50: {percentiles.get(50, 'N/A')}
P75: {percentiles.get(75, 'N/A')}
P95: {percentiles.get(95, 'N/A')}
Max: {values[-1]:.1f}
Buckets: {len(values)}"""
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Formatiere Achsen
        for ax in [ax1, ax2]:
            if max(values) > 10000:
                ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
            if max(counts) > 1000:
                ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        
        # Sicherer Dateiname
        safe_filename = metric_key.replace('__', '_').replace(' ', '_').lower()
        plot_file = output_dir / f"histogram_{safe_filename}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  ✅ {len(top_metrics)} detaillierte Histogramme erstellt")
    
    # 3. Zusammenfassungsstatistiken in JSON speichern
    summary_stats = {}
    for metric_key, data in bucket_data.items():
        buckets = data['buckets']
        total_docs = data['total_docs']
        
        sorted_items = sorted(buckets.items())
        values = np.array([x[0] for x in sorted_items])
        counts = np.array([x[1] for x in sorted_items])
        
        mean_val = sum(v * c for v, c in zip(values, counts)) / total_docs
        
        # Berechne Perzentile
        cumsum = np.cumsum(counts)
        percentiles = {}
        for p in [25, 50, 75, 95, 99]:
            target = (p / 100.0) * total_docs
            idx = np.searchsorted(cumsum, target)
            if idx < len(values):
                percentiles[f'p{p}'] = float(values[idx])
        
        summary_stats[metric_key] = {
            'total_documents': int(total_docs),
            'mean': float(mean_val),
            'min': float(values[0]),
            'max': float(values[-1]),
            'num_buckets': len(values),
            'sample_info': data['sample_info'],
            'file_size_mb': data['file_size_mb'],
            'num_files': data['num_files'],
            **percentiles
        }
    
    # Speichere Zusammenfassung
    summary_file = output_dir / "datatrove_summary_stats.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"  📊 Zusammenfassungsstatistiken gespeichert: {summary_file.name}")
    print(f"  📁 Alle Dateien gespeichert in: {output_dir}")
    
    return len(bucket_data)

def main():
    parser = argparse.ArgumentParser(description="DataTrove Bucket-basierte Histogram-Plots")
    parser.add_argument("--stats-dir", type=str, default="data/statistics", help="Pfad zum Statistik-Verzeichnis")
    parser.add_argument("--output-dir", type=str, help="Output-Verzeichnis für Plots")
    
    args = parser.parse_args()
    
    print("DataTrove Bucket-basierte Histogram-Plots")
    print("=" * 50)
    
    # Prüfe Verzeichnis
    if not os.path.exists(args.stats_dir):
        print(f"❌ Statistik-Verzeichnis nicht gefunden: {args.stats_dir}")
        return
    
    print(f"📂 Analysiere Verzeichnis: {args.stats_dir}")
    
    # Erstelle Plots
    num_metrics = create_datatrove_plots(args.stats_dir, args.output_dir)
    
    if num_metrics > 0:
        print()
        print("✅ DataTrove Histogram-Plots erfolgreich erstellt!")
        print(f"📊 {num_metrics} Metriken verarbeitet")
    else:
        print("❌ Keine verwertbaren Metriken gefunden!")

if __name__ == "__main__":
    main() 