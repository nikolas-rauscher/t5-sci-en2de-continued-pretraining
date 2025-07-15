#!/usr/bin/env python3
"""
DataTrove Statistik-Analyse (KORRIGIERT)
=========================================

Verwendet die echten Histogram-Daten zur Berechnung korrekter Percentile.

Usage:
    python scripts/analyze_stats.py                    # Analysiert data/statistics
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
    print("✅ Matplotlib/Seaborn verfügbar")
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️  Matplotlib/Seaborn nicht verfügbar. Nur CSV-Export.")

def calculate_percentiles_from_histogram(histogram_data):
    """
    Berechnet echte Percentile aus Histogram-Buckets.
    
    Args:
        histogram_data: Dict mit Buckets wie {'30392': {'total': 5379384, 'n': 177, 'mean': 30392.0}}
    
    Returns:
        Dict mit berechneten Statistiken
    """
    if not histogram_data:
        return None
    
    # Erstelle gewichtete Samples aus Histogram-Buckets
    values = []
    weights = []
    
    for bucket_value, bucket_data in histogram_data.items():
        try:
            # Bucket-Wert (center of bucket)
            value = float(bucket_value)
            # Anzahl der Dokumente in diesem Bucket
            count = bucket_data.get('n', 0)
            
            if count > 0:
                values.append(value)
                weights.append(count)
        except (ValueError, KeyError):
            continue
    
    if not values:
        return None
    
    # Konvertiere zu numpy arrays
    values = np.array(values)
    weights = np.array(weights)
    
    # Sortiere nach Werten
    sort_idx = np.argsort(values)
    values_sorted = values[sort_idx]
    weights_sorted = weights[sort_idx]
    
    # Berechne kumulative Gewichte
    cumsum = np.cumsum(weights_sorted)
    total_weight = cumsum[-1]
    
    # Berechne Percentile
    percentiles = [25, 50, 75, 95, 99]
    result = {}
    
    for p in percentiles:
        target = (p / 100.0) * total_weight
        # Finde den Bucket, der das target percentile enthält
        idx = np.searchsorted(cumsum, target, side='right')
        if idx >= len(values_sorted):
            idx = len(values_sorted) - 1
        result[f'p{p}'] = values_sorted[idx]
    
    # Berechne gewichteten Durchschnitt
    mean = np.average(values_sorted, weights=weights_sorted)
    
    # Berechne gewichtete Standardabweichung
    variance = np.average((values_sorted - mean)**2, weights=weights_sorted)
    std = np.sqrt(variance)
    
    result.update({
        'mean': mean,
        'std': std,
        'variance': variance,
        'min': values_sorted[0],
        'max': values_sorted[-1],
        'n_buckets': len(values),
        'total_samples': int(total_weight)
    })
    
    return result

def get_default_stats_dir():
    """Gibt das Standard-Statistik-Verzeichnis zurück"""
    return os.path.join(proj_root, "data", "statistics")

def count_json_files(stats_dir):
    """Zählt JSON-Dateien im Statistik-Verzeichnis"""
    try:
        stats_path = Path(stats_dir)
        if not stats_path.exists():
            return 0
        return len(list(stats_path.rglob("*.json")))
    except Exception:
        return 0

def load_all_metrics(stats_dir):
    """Lädt alle Metriken aus einem Stats-Verzeichnis (KORRIGIERT für DataTrove-Struktur)"""
    stats_path = Path(stats_dir)
    all_data = defaultdict(lambda: defaultdict(dict))
    
    print(f"Lade Statistiken aus: {stats_dir}")
    
    if not stats_path.exists():
        print(f"Statistik-Verzeichnis nicht gefunden: {stats_dir}")
        return all_data
    
    # DataTrove-Struktur durchsuchen: {stats_dir}/{stat_type}/[summary|histogram]/{metric}/[metric.json|00000.json,...]
    stats_folders = [d for d in stats_path.iterdir() if d.is_dir() and ('_stats' in d.name)]
    
    if not stats_folders:
        # Alternative: alle Ordner mit histogram/ oder summary/ Unterordner
        stats_folders = [d for d in stats_path.iterdir() if d.is_dir() and 
                        ((d / "histogram").exists() or (d / "summary").exists())]
    
    print(f"Gefundene Stats-Ordner: {[f.name for f in stats_folders]}")
    
    for stat_type_dir in stats_folders:
        if not stat_type_dir.is_dir():
            continue
            
        stat_type = stat_type_dir.name
        
        # Durchsuche summary und histogram Unterverzeichnisse
        for group_dir in stat_type_dir.iterdir():
            if not group_dir.is_dir():
                continue
                
            group_type = group_dir.name  # "summary" oder "histogram"
            if group_type not in ["summary", "histogram"]:
                continue
            
            print(f"  Verarbeite {stat_type}/{group_type}/")
            
            for metric_dir in group_dir.iterdir():
                if not metric_dir.is_dir():
                    continue
                    
                metric_name = metric_dir.name
                
                # Suche nach JSON-Dateien (metric.json oder nummerierte Dateien)
                json_files = []
                
                # Erst nach metric.json suchen
                metric_json = metric_dir / "metric.json"
                if metric_json.exists():
                    json_files = [metric_json]
                else:
                    # Suche nach nummerierten JSON-Dateien
                    numbered_files = list(metric_dir.glob("*.json"))
                    if numbered_files:
                        json_files = sorted(numbered_files)  # Sortiere für konsistente Reihenfolge
                
                if not json_files:
                    continue
                
                print(f"    📄 {metric_name}: {len(json_files)} JSON-Dateien")
                
                try:
                    # Lade und kombiniere alle JSON-Dateien für diese Metrik
                    combined_data = None
                    total_samples = 0
                    
                    # Verarbeite ALLE Dateien für vollständige Analyse (keine Sampling)
                    sampled_files = json_files
                    print(f"      📊 Verarbeite alle {len(json_files)} Dateien")
                    
                    if group_type == "summary":
                        # Für Summary: Kombiniere n_samples aber nehme nur einen Satz Stats
                        for json_file in sampled_files:
                            with open(json_file, 'r') as f:
                                data = json.load(f)
                            
                            if 'summary' in data and isinstance(data['summary'], dict):
                                summary_stats = data['summary']
                                if combined_data is None:
                                    combined_data = {
                                        'mean': summary_stats.get('mean', 0),
                                        'std': summary_stats.get('std_dev', summary_stats.get('std', 0)),
                                        'variance': summary_stats.get('variance', 0),
                                        'min': summary_stats.get('min', 0),
                                        'max': summary_stats.get('max', 0),
                                        'n_samples': summary_stats.get('n', 0),
                                        'p25': summary_stats.get('p25'),
                                        'p50': summary_stats.get('p50', summary_stats.get('median')),
                                        'p75': summary_stats.get('p75'),
                                        'p95': summary_stats.get('p95'),
                                        'p99': summary_stats.get('p99'),
                                        'data_type': 'summary'
                                    }
                                    total_samples = summary_stats.get('n', 0)
                                else:
                                    # Akkumuliere nur n_samples
                                    total_samples += summary_stats.get('n', 0)
                        
                        if combined_data:
                            combined_data['n_samples'] = total_samples
                    
                    elif group_type == "histogram":
                        # Für Histogram: Kombiniere alle Buckets über alle Dateien
                        combined_histogram = {}
                        
                        for json_file in sampled_files:
                            file_size_mb = json_file.stat().st_size / (1024 * 1024)
                            if file_size_mb > 100:
                                print(f"        📊 Große Datei: {json_file.name} ({file_size_mb:.1f}MB)")
                            
                            with open(json_file, 'r') as f:
                                data = json.load(f)
                            
                            if isinstance(data, dict) and len(data) > 0:
                                # Kombiniere Histogramme - addiere Counts für gleiche Buckets
                                for bucket_value, count_data in data.items():
                                    try:
                                        # Normalisiere bucket_value zu float für konsistente Keys
                                        value = float(bucket_value)
                                        if isinstance(count_data, dict):
                                            count = count_data.get('n', count_data.get('count', 0))
                                        else:
                                            count = int(count_data)
                                        
                                        if count > 0:
                                            # Verwende normalisierten float-Wert als Key
                                            if value in combined_histogram:
                                                combined_histogram[value] += count
                                            else:
                                                combined_histogram[value] = count
                                    except (ValueError, TypeError):
                                        continue
                        
                        # Berechne Statistiken aus kombiniertem Histogram
                        if combined_histogram:
                            values = []
                            counts = []
                            total_count = 0
                            
                            for value, count in combined_histogram.items():
                                # value ist bereits float von oben
                                values.append(value)
                                counts.append(count)
                                total_count += count
                            
                            if values and counts:
                                values_arr = np.array(values)
                                counts_arr = np.array(counts)
                                
                                # Sortiere nach Werten
                                sorted_idx = np.argsort(values_arr)
                                sorted_values = values_arr[sorted_idx]
                                sorted_counts = counts_arr[sorted_idx]
                                
                                # Gewichteter Durchschnitt
                                mean_val = np.average(sorted_values, weights=sorted_counts)
                                
                                # Gewichtete Standardabweichung
                                variance_val = np.average((sorted_values - mean_val)**2, weights=sorted_counts)
                                std_val = np.sqrt(variance_val)
                                
                                # Perzentile (korrigierte Berechnung)
                                cumsum = np.cumsum(sorted_counts)
                                percentiles = {}
                                for p in [25, 50, 75, 95, 99]:
                                    target = (p / 100.0) * total_count
                                    # Finde den Index wo cumsum >= target
                                    idx = np.searchsorted(cumsum, target, side='left')
                                    if idx >= len(sorted_values):
                                        idx = len(sorted_values) - 1
                                    percentiles[f'p{p}'] = float(sorted_values[idx])
                                
                                combined_data = {
                                    'mean': mean_val,
                                    'std': std_val,
                                    'variance': variance_val,
                                    'min': float(np.min(sorted_values)),
                                    'max': float(np.max(sorted_values)),
                                    'n_samples': total_count,
                                    'p25': percentiles.get('p25'),
                                    'p50': percentiles.get('p50'),
                                    'p75': percentiles.get('p75'),
                                    'p95': percentiles.get('p95'),
                                    'p99': percentiles.get('p99'),
                                    'data_type': 'histogram'
                                }
                                total_samples = total_count
                    
                    if combined_data:
                        # Speichere die verarbeiteten Daten
                        key = f"{stat_type},{metric_name}"
                        all_data[group_type][key] = combined_data
                        print(f"      ✅ {key}: {total_samples:,} Samples")
                    
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"      ❌ Fehler beim Laden von {metric_name}: {e}")
                    continue
    
    # Statistik über geladene Daten
    total_metrics = sum(len(group_data) for group_data in all_data.values())
    print(f"\n✅ Gesamt: {total_metrics} Metriken geladen")
    for group_type, group_data in all_data.items():
        print(f"   {group_type}: {len(group_data)} Metriken")
    
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

def create_comprehensive_plots(df, plot_folder):
    """Erstellt echte Histogramme und Verteilungsplots aus Bucket-Daten"""
    if not PLOTTING_AVAILABLE:
        print("  ⚠️  Matplotlib nicht verfügbar, überspringe Plots")
        return
    
    # Filtere nur Histogram-Daten für echte Verteilungen
    df_hist = df[df['data_source'] == 'histogram'].copy()
    
    if df_hist.empty:
        print("  ⚠️  Keine Histogram-Daten für Plots verfügbar")
        return
    
    print(f"  📊 Erstelle Verteilungsplots mit {len(df_hist)} Histogram-Metriken...")
    
    try:
        # 1. Box Plot Übersicht - zeigt Quartile und Outliers
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('DataTrove Verteilungsanalyse - Box Plots', fontsize=16, fontweight='bold')
        
        # Box Plot 1: Dokument-Level Metriken
        doc_metrics = df_hist[df_hist['category'] == 'DOC_STATS'].copy()
        if not doc_metrics.empty:
            # Normalisiere für bessere Vergleichbarkeit
            doc_data = []
            doc_labels = []
            for _, row in doc_metrics.iterrows():
                if row['max'] > row['min']:  # Nur wenn Variation vorhanden
                    # Simuliere Daten aus Perzentilen für Box Plot
                    data_points = [row['min'], row['p25'], row['p50'], row['p75'], row['p95'], row['max']]
                    doc_data.append(data_points)
                    doc_labels.append(row['metric'].replace('__chars', ''))
            
            if doc_data:
                axes[0, 0].boxplot(doc_data, labels=doc_labels)
                axes[0, 0].set_title('Document-Level Statistiken', fontweight='bold')
                axes[0, 0].tick_params(axis='x', rotation=45)
                axes[0, 0].set_ylabel('Normalisierte Werte')
                axes[0, 0].grid(alpha=0.3)
        
        # Box Plot 2: Text-Qualität Metriken (Ratios)
        ratio_metrics = df_hist[df_hist['metric'].str.contains('ratio', case=False)].copy()
        if not ratio_metrics.empty:
            ratio_data = []
            ratio_labels = []
            for _, row in ratio_metrics.iterrows():
                data_points = [row['min'], row['p25'], row['p50'], row['p75'], row['p95'], row['max']]
                ratio_data.append(data_points)
                ratio_labels.append(f"{row['category']}.{row['metric'][:15]}")
            
            if ratio_data:
                axes[0, 1].boxplot(ratio_data, labels=ratio_labels)
                axes[0, 1].set_title('Text-Qualität Ratios', fontweight='bold')
                axes[0, 1].tick_params(axis='x', rotation=45)
                axes[0, 1].set_ylabel('Ratio Werte')
                axes[0, 1].grid(alpha=0.3)
        
        # Scatter Plot 3: Mean vs Variabilität
        axes[1, 0].scatter(df_hist['mean'], df_hist['cv'], 
                          c=pd.Categorical(df_hist['category']).codes, 
                          cmap='Set1', alpha=0.7, s=100)
        axes[1, 0].set_xlabel('Mean Value')
        axes[1, 0].set_ylabel('Coefficient of Variation')
        axes[1, 0].set_title('Variabilität vs Durchschnitt', fontweight='bold')
        axes[1, 0].grid(alpha=0.3)
        
        # Histogram 4: CV Verteilung
        axes[1, 1].hist(df_hist['cv'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_xlabel('Coefficient of Variation')
        axes[1, 1].set_ylabel('Anzahl Metriken')
        axes[1, 1].set_title('Verteilung der Variabilität', fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plot_file = plot_folder / "distribution_overview.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✅ Verteilungs-Übersicht: {plot_file.name}")
        
        # 2. Detaillierte Verteilungsplots pro Kategorie
        for category in sorted(df_hist['category'].unique()):
            cat_data = df_hist[df_hist['category'] == category].copy()
            
            if len(cat_data) < 2:
                continue
            
            # Anzahl Subplots basierend auf Anzahl Metriken
            n_metrics = len(cat_data)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1 or n_cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            fig.suptitle(f'{category} - Verteilungsanalyse ({n_metrics} Metriken)', 
                        fontsize=14, fontweight='bold')
            
            for i, (_, row) in enumerate(cat_data.iterrows()):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                
                # Erstelle "Histogram" aus Perzentilen
                # Simuliere Verteilung mit verschiedenen Bereichen
                bins = ['Min-P25', 'P25-P50', 'P50-P75', 'P75-P95', 'P95-Max']
                heights = [
                    row['p25'] - row['min'],
                    row['p50'] - row['p25'], 
                    row['p75'] - row['p50'],
                    row['p95'] - row['p75'],
                    row['max'] - row['p95']
                ]
                
                # Normalisiere Heights für bessere Visualisierung
                heights = [max(h, 0.1) for h in heights]  # Mindesthöhe für Sichtbarkeit
                
                colors = ['lightcoral', 'gold', 'lightgreen', 'lightblue', 'plum']
                bars = ax.bar(bins, heights, color=colors, alpha=0.7, edgecolor='black')
                
                # Füge Werte hinzu
                for bar, height in zip(bars, heights):
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                               f'{height:.2f}', ha='center', va='center', fontweight='bold')
                
                ax.set_title(f'{row["metric"][:20]}...', fontweight='bold')
                ax.set_ylabel('Bereichsgröße')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(alpha=0.3)
                
                # Zusätzliche Statistiken als Text
                stats_text = f"Mean: {row['mean']:.2f}\nStd: {row['std']:.2f}\nCV: {row['cv']:.2f}"
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Verstecke ungenutzte Subplots
            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            plot_file = plot_folder / f"{category.lower()}_distributions.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    ✅ {category}: {plot_file.name}")
        
        # 3. Perzentile-Vergleichsplot (echte Verteilungsformen)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Perzentile-Vergleich zwischen Kategorien', fontsize=16, fontweight='bold')
        
        # Plot 1: P50 (Median) Vergleich
        for category in df_hist['category'].unique():
            cat_data = df_hist[df_hist['category'] == category]
            axes[0, 0].scatter(range(len(cat_data)), cat_data['p50'], 
                             label=category, alpha=0.7, s=60)
        axes[0, 0].set_title('Mediane (P50) nach Kategorie')
        axes[0, 0].set_xlabel('Metrik Index')
        axes[0, 0].set_ylabel('Median Wert')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Plot 2: IQR (P75-P25) Vergleich
        df_iqr = df_hist.dropna(subset=['iqr'])
        for category in df_iqr['category'].unique():
            cat_data = df_iqr[df_iqr['category'] == category]
            axes[0, 1].scatter(range(len(cat_data)), cat_data['iqr'], 
                             label=category, alpha=0.7, s=60)
        axes[0, 1].set_title('Interquartile Range (IQR)')
        axes[0, 1].set_xlabel('Metrik Index')
        axes[0, 1].set_ylabel('IQR Wert')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Plot 3: Skewness Approximation (P95-P50 vs P50-P25)
        p95_p50 = df_hist['p95'] - df_hist['p50']
        p50_p25 = df_hist['p50'] - df_hist['p25']
        skew_approx = (p95_p50 - p50_p25) / (p95_p50 + p50_p25)
        
        axes[1, 0].scatter(df_hist['mean'], skew_approx, 
                          c=pd.Categorical(df_hist['category']).codes, 
                          cmap='Set1', alpha=0.7, s=60)
        axes[1, 0].set_xlabel('Mean Value')
        axes[1, 0].set_ylabel('Skewness Approximation')
        axes[1, 0].set_title('Verteilungsasymmetrie')
        axes[1, 0].grid(alpha=0.3)
        
        # Plot 4: Tail Heaviness (P99-P95 vs P95-P75)
        tail_ratio = (df_hist['p99'] - df_hist['p95']) / (df_hist['p95'] - df_hist['p75'])
        # Filter out inf and nan values
        tail_ratio_clean = tail_ratio[np.isfinite(tail_ratio)]
        if len(tail_ratio_clean) > 0:
            axes[1, 1].hist(tail_ratio_clean, bins=20, alpha=0.7, color='purple', edgecolor='black')
            axes[1, 1].set_xlabel('Tail Heaviness Ratio')
            axes[1, 1].set_ylabel('Anzahl Metriken')
            axes[1, 1].set_title('Verteilung der Tail-Schwere')
            axes[1, 1].grid(alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Keine gültigen\nTail-Ratio Daten', 
                           ha='center', va='center', transform=axes[1, 1].transAxes,
                           fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
        
        plt.tight_layout()
        plot_file = plot_folder / "percentile_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✅ Perzentile-Analyse: {plot_file.name}")
        
        print(f"  📊 Alle Verteilungsplots erfolgreich erstellt in: {plot_folder}")
        
    except Exception as e:
        print(f"  ❌ Fehler beim Plot-Erstellen: {e}")
        import traceback
        traceback.print_exc()

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

def create_analysis_dataframe(all_data):
    """Erstellt DataFrame für Analyse (BEVORZUGT HISTOGRAM-DATEN)"""
    rows = []
    
    # Priorisiere Histogram-Daten gegenüber Summary-Daten
    for group_type in ['histogram', 'summary']:
        if group_type not in all_data:
            continue
        
        for metric_key, stats in all_data[group_type].items():
            # Überspringe wenn wir bereits Histogram-Daten für diese Metrik haben
            if group_type == 'summary':
                # Prüfe ob Histogram-Version existiert
                if 'histogram' in all_data and metric_key in all_data['histogram']:
                    continue  # Überspringe Summary, da Histogram verfügbar
            
            stat_type, metric_name = metric_key.split(',', 1)
            
            row = {
                'category': stat_type.upper(),
                'metric': metric_name,
                'mean': stats.get('mean', 0),
                'std': stats.get('std', 0),
                'variance': stats.get('variance', 0),
                'cv': stats.get('std', 0) / stats.get('mean', 1) if stats.get('mean', 0) != 0 else 0,
                'iqr': (stats.get('p75', 0) - stats.get('p25', 0)) if stats.get('p75') is not None and stats.get('p25') is not None else None,
                'min': stats.get('min', 0),
                'max': stats.get('max', 0),
                'p25': stats.get('p25'),
                'p50': stats.get('p50'),
                'p75': stats.get('p75'),
                'p95': stats.get('p95'),
                'p99': stats.get('p99'),
                'n_samples': stats.get('n_samples', stats.get('total_samples', 0)),
                'data_source': stats.get('data_type', 'unknown')
            }
            rows.append(row)
    
    return pd.DataFrame(rows)

def create_bucket_based_plots(all_data, plot_folder, stats_dir="data/statistics"):
    """Erstellt echte Histogramme aus den Bucket-Daten der DataTrove Statistiken mit intelligentem Sampling"""
    if not PLOTTING_AVAILABLE:
        print("  ⚠️  Matplotlib nicht verfügbar, überspringe Bucket-Plots")
        return
    
    print("  📊 Erstelle echte Histogramme aus Bucket-Daten...")
    
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
    
    # Metriken, die typischerweise gute Histogram-Daten haben (mit __chars Suffix)
    priority_metrics = [
        'length__chars', 'token_count__chars', 'n_words__chars', 'n_lines__chars', 
        'n_paragraphs__chars', 'n_sentences__chars', 'avg_word_length__chars',
        'avg_line_length__chars', 'avg_sentence_length__chars', 'avg_paragraph_length__chars',
        'fasttext_en__chars'
    ]
    
    # Ratio-Metriken, die auch brauchbar sind
    ratio_metrics = [
        'digit_ratio__chars', 'punctuation_ratio__chars', 'uppercase_ratio__chars',
        'white_space_ratio__chars', 'non_alpha_digit_ratio__chars', 'elipsis_ratio__chars',
        'stop_word_ratio__chars', 'short_word_ratio_3__chars', 'long_word_ratio_7__chars',
        'capitalized_word_ratio__chars', 'uppercase_word_ratio__chars', 'type_token_ratio__chars'
    ]
    
    useful_metrics = set(priority_metrics + ratio_metrics)
    
    for stat_folder in stats_folders:
        histogram_dir = stat_folder / "histogram"
        if not histogram_dir.exists():
            continue
            
        print(f"    📂 Durchsuche {stat_folder.name}/histogram/")
        
        # Sammle alle verfügbaren Metriken
        available_metrics = [d.name for d in histogram_dir.iterdir() if d.is_dir()]
        
        # Priorisiere nützliche Metriken
        useful_available = [m for m in available_metrics if m in useful_metrics]
        other_available = [m for m in available_metrics if m not in useful_metrics]
        
        # Verarbeite erst die nützlichen Metriken
        metrics_to_process = useful_available + other_available[:5]  # Max 5 zusätzliche
        
        print(f"      🎯 Verarbeite {len(metrics_to_process)} von {len(available_metrics)} Metriken")
        
        for metric_name in metrics_to_process:
            metric_dir = histogram_dir / metric_name
            if not metric_dir.is_dir():
                continue
            
            # Schnelle Vorprüfung: Gibt es überhaupt JSON-Dateien?
            json_files = list(metric_dir.glob("*.json"))
            if not json_files:
                continue
            
            # Schätze Dateigröße
            total_size = sum(f.stat().st_size for f in json_files[:3])  # Nur erste 3 prüfen
            estimated_total_mb = (total_size * len(json_files) / 3) / (1024 * 1024)
            
            # Überspringe wenn zu groß und nicht in Prioritätsliste
            if estimated_total_mb > 500 and metric_name not in useful_metrics:
                print(f"      ⏭️  Überspringe {metric_name}: ~{estimated_total_mb:.0f}MB, nicht prioritär")
                continue
            
            try:
                # Sortiere für konsistente Reihenfolge
                json_files = sorted(json_files)
                
                # Intelligente Dateiauswahl basierend auf Größe
                if estimated_total_mb > 1000:  # >1GB: Nur erste 3 Dateien
                    sampled_files = json_files[:3]
                    sample_info = f"Erste 3/{len(json_files)} Dateien"
                elif estimated_total_mb > 500:  # >500MB: Sample
                    step = max(1, len(json_files) // 10)
                    sampled_files = json_files[::step]
                    sample_info = f"Sample {len(sampled_files)}/{len(json_files)} Dateien"
                elif len(json_files) > 20:  # Viele Dateien: Sample
                    step = max(1, len(json_files) // 15)
                    sampled_files = json_files[::step]
                    sample_info = f"Sample {len(sampled_files)}/{len(json_files)} Dateien"
                else:
                    sampled_files = json_files
                    sample_info = "Alle Dateien"
                
                print(f"      🔍 {metric_name}: {len(sampled_files)}/{len(json_files)} Dateien, ~{estimated_total_mb:.0f}MB")
                
                # Lade und kombiniere JSON-Dateien
                combined_histogram = {}
                total_file_size = 0
                
                for json_file in sampled_files:
                    file_size = json_file.stat().st_size
                    total_file_size += file_size
                    
                    # Überspringe sehr große einzelne Dateien
                    if file_size > 100 * 1024 * 1024:  # 100MB pro Datei
                        continue
                    
                    with open(json_file, 'r') as f:
                        histogram_data = json.load(f)
                    
                    if isinstance(histogram_data, dict):
                        # Kombiniere Histogramme
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
                                combined_histogram[bucket_value] = count_data
                
                if not combined_histogram:
                    continue
                
                # Konvertiere zu numerischen Buckets mit Sampling
                buckets = {}
                total_docs = 0
                valid_entries = 0
                
                all_items = list(combined_histogram.items())
                total_size_mb = total_file_size / (1024 * 1024)
                
                # Intelligentes Sampling
                if total_size_mb > 200:
                    import random
                    sample_size = min(3000, len(all_items) // 10)
                    sampled_items = random.sample(all_items, sample_size) if len(all_items) > sample_size else all_items
                    sampling_info = f"Random Sample {len(sampled_items)}/{len(all_items)}"
                elif total_size_mb > 100:
                    step = max(1, len(all_items) // 5000)
                    sampled_items = all_items[::step]
                    sampling_info = f"Systematic Sample {len(sampled_items)}/{len(all_items)}"
                else:
                    sampled_items = all_items
                    sampling_info = "Vollständig"
                
                # Verarbeite Items
                for bucket_value, count_data in sampled_items:
                    try:
                        value = float(bucket_value)
                        
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
                
                if buckets and valid_entries > 20:  # Mindestens 20 gültige Buckets
                    key = f"{stat_folder.name}_{metric_name}"
                    bucket_data[key] = {
                        'buckets': buckets,
                        'total_docs': total_docs,
                        'sample_info': f"{sample_info}, {sampling_info}",
                        'file_size_mb': total_size_mb,
                        'num_files': len(sampled_files),
                        'priority': metric_name in useful_metrics
                    }
                    print(f"        ✅ {valid_entries} Buckets, {total_docs:,} Docs")
                        
            except Exception as e:
                print(f"        ❌ Fehler: {e}")
                continue
    
    if not bucket_data:
        print("  ⚠️  Keine Bucket-Daten gefunden für Plots")
        return
    
    print(f"  📊 Erstelle Histogramme für {len(bucket_data)} Metriken...")
    
    # Sortiere nach Priorität und wähle die besten für Übersicht
    priority_metrics_available = {k: v for k, v in bucket_data.items() if v['priority']}
    other_metrics_available = {k: v for k, v in bucket_data.items() if not v['priority']}
    
    # Nimm die wichtigsten Metriken für Übersichtsplot
    overview_metrics = list(priority_metrics_available.keys())[:6]
    if len(overview_metrics) < 6:
        overview_metrics.extend(list(other_metrics_available.keys())[:6-len(overview_metrics)])
    
    # 1. Übersichtsplot mit den wichtigsten Metriken
    if overview_metrics:
        n_metrics = len(overview_metrics)
        if n_metrics >= 4:
            fig, axes = plt.subplots(2, 3, figsize=(24, 16))
            axes = axes.flatten()
        elif n_metrics >= 2:
            fig, axes = plt.subplots(1, n_metrics, figsize=(8*n_metrics, 8))
            if n_metrics == 1:
                axes = [axes]
        else:
            return
        
        fig.suptitle('DataTrove - Wichtigste Histogram-Verteilungen', fontsize=20, fontweight='bold')
        
        for i, metric_key in enumerate(overview_metrics):
            ax = axes[i] if i < len(axes) else axes[-1]
            
            data = bucket_data[metric_key]
            buckets = data['buckets']
            total_docs = data['total_docs']
            
            # Sortiere Buckets nach Werten
            sorted_items = sorted(buckets.items())
            values = np.array([x[0] for x in sorted_items])
            counts = np.array([x[1] for x in sorted_items])
            
            # Erstelle Plot
            if len(values) > 100:
                ax.plot(values, counts, linewidth=2, alpha=0.8, color='steelblue')
                ax.fill_between(values, counts, alpha=0.3, color='steelblue')
            else:
                ax.bar(values, counts, alpha=0.7, edgecolor='black', linewidth=0.5, color='steelblue')
            
            # Formatierung
            metric_name = metric_key.replace('_stats_', ': ').replace('__', ' (').replace('_', ' ').title()
            if '(' in metric_name and not metric_name.endswith(')'):
                metric_name += ')'
            
            ax.set_title(f'{metric_name}\n{total_docs:,} Dokumente', fontweight='bold')
            ax.set_xlabel('Wert')
            ax.set_ylabel('Anzahl Dokumente')
            ax.grid(alpha=0.3)
            
            # Formatierung
            if max(counts) > 1000:
                ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            if max(values) > 10000:
                ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
            
            # Log-Scale bei extremen Unterschieden
            if len(counts) > 0 and max(counts) / min([c for c in counts if c > 0]) > 1000:
                ax.set_yscale('log')
                ax.set_ylabel('Anzahl Dokumente (log)')
            
            # Statistiken
            mean_val = sum(v * c for v, c in zip(values, counts)) / total_docs
            median_idx = total_docs // 2
            cumsum = np.cumsum(counts)
            median_val = values[np.searchsorted(cumsum, median_idx)]
            
            stats_text = f'Mean: {mean_val:.1f}\nMedian: {median_val:.1f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Verstecke ungenutzte Subplots
        for j in range(len(overview_metrics), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(plot_folder / "bucket_histograms_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✅ Übersichtsplot: bucket_histograms_overview.png")
    
    # 2. Erstelle nur für die wichtigsten Metriken detaillierte Plots
    important_for_details = list(priority_metrics_available.keys())[:12]  # Max 12 detaillierte Plots
    
    for metric_key in important_for_details:
        data = bucket_data[metric_key]
        buckets = data['buckets']
        total_docs = data['total_docs']
        sample_info = data['sample_info']
        file_size_mb = data['file_size_mb']
        num_files = data['num_files']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(f'{metric_key.replace("_", " ").title()}\n{total_docs:,} Dokumente • {file_size_mb:.1f}MB ({num_files} Dateien)', 
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
        
        # Plot 2: Log Scale oder Cumulative
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
            # Cumulative Distribution
            cumsum_norm = np.cumsum(counts) / total_docs
            ax2.plot(values, cumsum_norm, linewidth=2, color='green', marker='o', markersize=2)
            ax2.set_title('Kumulative Verteilung', fontweight='bold')
            ax2.set_xlabel('Wert')
            ax2.set_ylabel('Kumulative Wahrscheinlichkeit')
            ax2.grid(alpha=0.3)
            ax2.set_ylim(0, 1)
        
        # Statistiken
        mean_val = sum(v * c for v, c in zip(values, counts)) / total_docs
        
        # Perzentile
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
        plt.savefig(plot_folder / f"histogram_{safe_filename}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  ✅ {len(important_for_details)} wichtige Histogramme erstellt")
    print(f"  📁 Gespeichert in: {plot_folder}")
    
    # 3. Zusammenfassung speichern
    summary_stats = {}
    for metric_key, data in bucket_data.items():
        buckets = data['buckets']
        total_docs = data['total_docs']
        
        sorted_items = sorted(buckets.items())
        values = np.array([x[0] for x in sorted_items])
        counts = np.array([x[1] for x in sorted_items])
        
        mean_val = sum(v * c for v, c in zip(values, counts)) / total_docs
        
        # Perzentile
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
            'priority': data['priority'],
            **percentiles
        }
    
    # Speichere Zusammenfassung
    with open(plot_folder / "bucket_summary_stats.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"  📊 Zusammenfassungsstatistiken gespeichert: bucket_summary_stats.json")

def main():
    parser = argparse.ArgumentParser(description="DataTrove Statistik-Analyse")
    parser.add_argument("--stats-dir", type=str, help="Pfad zum Statistik-Verzeichnis")
    parser.add_argument("--no-plots", action="store_true", help="Deaktiviere Plot-Generierung")
    
    args = parser.parse_args()
    
    print("DataTrove Statistik-Analyse")
    print("=" * 50)
    
    # Bestimme Verzeichnis
    if args.stats_dir:
        stats_dir = args.stats_dir
        print(f"Analysiere benutzerdefiniertes Verzeichnis: {stats_dir}")
    else:
        stats_dir = get_default_stats_dir()
        print(f"Analysiere Standard-Verzeichnis: {stats_dir}")
    
    # Prüfe JSON-Dateien
    json_count = count_json_files(stats_dir)
    if json_count == 0:
        print("❌ Keine JSON-Dateien gefunden!")
        return
    
    print(f"✅ Gefunden: {json_count} JSON-Dateien")
    
    # Lade alle Metriken (KORRIGIERT)
    all_data = load_all_metrics(stats_dir)
    
    if not all_data:
        print("❌ Keine verwertbaren Statistiken gefunden!")
        return
    
    # Erstelle DataFrame (BEVORZUGT HISTOGRAM-DATEN)
    df = create_analysis_dataframe(all_data)
    
    if df.empty:
        print("❌ Keine Daten für Analyse verfügbar!")
        return
    
    # Zeige Übersicht
    print("\n" + "=" * 80)
    print("DATATROVE STATISTIK-ÜBERSICHT (KORRIGIERT)")
    print("=" * 80)
    print()
    
    # Anzahl der Dokumente (aus erstem verfügbaren Eintrag)
    first_entry = df.iloc[0]
    n_docs = first_entry['n_samples']
    print(f"Gesamt: {len(df)} Metriken in {df['category'].nunique()} Kategorien")
    print(f"Dokumente verarbeitet: {n_docs:,} (bei den meisten Metriken)")
    print()
    
    # Gruppiere nach Kategorien
    for category in sorted(df['category'].unique()):
        cat_df = df[df['category'] == category]
        print(f"{category} ({len(cat_df)} Metriken)")
        print("-" * 60)
        
        for _, row in cat_df.iterrows():
            metric = row['metric']
            mean = row['mean']
            std = row['std']
            p50 = row['p50']
            p95 = row['p95']
            data_source = row['data_source']
            
            # Formatierung je nach Wertbereich
            if mean < 1:
                mean_str = f"{mean:.2f}"
                std_str = f"{std:.2f}"
                p50_str = f"{p50:.2f}"
                p95_str = f"{p95:.2f}"
            elif mean < 100:
                mean_str = f"{mean:.1f}"
                std_str = f"{std:.1f}"
                p50_str = f"{p50:.1f}"
                p95_str = f"{p95:.1f}"
            else:
                mean_str = f"{mean:.0f}"
                std_str = f"{std:.0f}"
                p50_str = f"{p50:.0f}"
                p95_str = f"{p95:.0f}"
            
            # Zeige Datenquelle (Histogram vs Summary)
            source_indicator = "📊" if data_source == "histogram" else "📋"
            
            print(f"  {metric:20s} {source_indicator} | Mean: {mean_str:>8s} | Std: {std_str:>8s} | P50: {p50_str:>8s} | P95: {p95_str:>8s}")
        
        print()
    
    # CSV Export
    output_folder = Path(stats_dir) / "analysis"
    output_folder.mkdir(exist_ok=True)
    csv_file = output_folder / "datatrove_analysis.csv"
    
    df.to_csv(csv_file, index=False, float_format='%.6f')
    
    print(f"CSV exportiert: {csv_file}")
    print(f"   Spalten: {len(df.columns)}")
    print(f"   Zeilen: {len(df)}")
    
    # Plots (falls verfügbar und gewünscht)
    if PLOTTING_AVAILABLE and not args.no_plots:
        plot_folder = output_folder / "basic_plots"
        plot_folder.mkdir(exist_ok=True)
        
        try:
            # Erstelle aggregierte Plots (Perzentile)
            create_comprehensive_plots(df, plot_folder)
            print(f"📊 Aggregierte Plots gespeichert in: {plot_folder}")
            
            # Erstelle echte Histogramme aus Bucket-Daten
            create_bucket_based_plots(all_data, plot_folder, stats_dir)
            print(f"📊 Bucket-basierte Histogramme gespeichert in: {plot_folder}")
            
        except Exception as e:
            print(f"⚠️  Fehler beim Erstellen der Plots: {e}")
    
    print()
    print("✅ Analyse abgeschlossen!")
    print(f"📁 Datenquelle: {stats_dir}")
    print(f"📊 Histogram-basierte Metriken: {len(df[df['data_source'] == 'histogram'])}")
    print(f"📋 Summary-basierte Metriken: {len(df[df['data_source'] == 'summary'])}")

if __name__ == "__main__":
    main() 