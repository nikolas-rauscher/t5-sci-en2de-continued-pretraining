"""
Text Normalization Pipeline Module für DataTrove

Normalisiert Whitespace, Newlines ohne andere Textänderungen.
Kann separat oder nach anderen Cleaning-Stages verwendet werden.

Usage:
```python
from src.dataprep.text_normalizer import TextNormalizer

normalizer = TextNormalizer(
    normalize_spaces=True,
    normalize_newlines=True,
    strip_whitespace=True,
    log_to_wandb=True,
    wandb_project="BA-DataTrove"
)
```
"""

import re
import logging
from typing import Dict, Any
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.hashing import create_hash_func, HashConfig

log = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    log.warning("wandb not available - text normalization stats will not be logged")


class TextNormalizer(BaseFilter):
    """
    Text-Normalisierung Pipeline Module.
    Normalisiert Whitespace, Newlines ohne andere Textänderungen.
    """
    
    name = "🧽 Text Normalizer"
    
    def __init__(
        self,
        normalize_spaces: bool = True,
        normalize_newlines: bool = True,
        strip_whitespace: bool = True,
        fix_sentence_spacing: bool = True,
        track_changes: bool = True,
        wandb_project: str = "BA-DataTrove",
        wandb_group: str = "text-normalization",
        log_to_wandb: bool = True,
        exclusion_writer: DiskWriter = None
    ):
        """
        Args:
            normalize_spaces: Multiple Spaces/Tabs → Single Space
            normalize_newlines: Multiple Newlines → Double Newline  
            strip_whitespace: Whitespace am Anfang/Ende entfernen
            fix_sentence_spacing: Leading spaces vor Sätzen entfernen
            track_changes: Ob Änderungen getrackt werden sollen
            wandb_project: W&B Projekt Name
            wandb_group: W&B Gruppe für Normalization
            log_to_wandb: Ob Stats zu W&B geloggt werden sollen
            exclusion_writer: Optional writer für leere Dokumente
        """
        super().__init__(exclusion_writer)
        self.normalize_spaces = normalize_spaces
        self.normalize_newlines = normalize_newlines
        self.strip_whitespace = strip_whitespace
        self.fix_sentence_spacing = fix_sentence_spacing
        self.track_changes = track_changes
        self.log_to_wandb = log_to_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        self.wandb_group = wandb_group
        
        # DataTrove's Hash-Funktion für Change-Detection
        if self.track_changes:
            hash_config = HashConfig(precision=64, hash_fc="sha1")
            self.hash_func = create_hash_func(hash_config)
        
        # Stats für W&B
        self.normalization_stats = {
            "docs_processed": 0,
            "docs_normalized": 0,
            "total_length_reduction": 0,
            "total_word_reduction": 0,
            "space_normalizations": 0,
            "newline_normalizations": 0,
            "strip_normalizations": 0,
            "sentence_spacing_fixes": 0
        }
        
        # Top-K Dokumente mit meisten Normalisierungs-Issues tracken
        self.top_normalization_docs = []
        self.max_tracked_docs = 20  # Top 20 Ausreißer
        
        # W&B Initialisierung
        self.wandb_initialized = False
        if self.log_to_wandb:
            self._init_wandb()
    
    def _init_wandb(self):
        """Prüft ob bereits eine W&B Session läuft - nutzt shared session"""
        try:
            # Prüfe ob bereits eine wandb session läuft
            if wandb.run is not None:
                self.wandb_initialized = True
                log.info(f"📊 Using existing W&B session for Text Normalizer")
            else:
                # Fallback: eigene Session (sollte normalerweise nicht passieren)
                wandb.init(
                    project=self.wandb_project,
                    group=self.wandb_group,
                    tags=["text-normalization", "datatrove", "whitespace-cleaning"],
                    job_type="text-normalization",
                    notes="Text normalization with whitespace/newline cleaning analytics",
                    config={
                        "normalize_spaces": self.normalize_spaces,
                        "normalize_newlines": self.normalize_newlines,
                        "strip_whitespace": self.strip_whitespace,
                        "track_changes": self.track_changes
                    }
                )
                self.wandb_initialized = True
                log.info(f"📊 W&B initialized for Text Normalizer - project: {self.wandb_project}")
        except Exception as e:
            log.warning(f"Failed to initialize W&B for text normalization: {e}")
            self.log_to_wandb = False
    
    def filter(self, doc: Document) -> bool:
        """Text-Normalisierung mit detailliertem Tracking"""
        original_text = doc.text
        self.normalization_stats["docs_processed"] += 1
        
        # Pre-normalization Metriken
        pre_metrics = None
        if self.track_changes:
            original_hash = self.hash_func(original_text)
            pre_word_count = len(original_text.split()) if original_text.strip() else 0
            pre_metrics = {
                "length": len(original_text),
                "n_words": pre_word_count,
                "hash": original_hash
            }
        
        # Schritt-für-Schritt Normalisierung mit Tracking
        normalized_text = original_text
        changes_made = {
            "spaces": False,
            "newlines": False,
            "sentence_spacing_fixes": False,
            "strip": False
        }
        
        if self.normalize_spaces:
            before_spaces = normalized_text
            normalized_text = re.sub(r'[ \t]+', ' ', normalized_text)
            if normalized_text != before_spaces:
                changes_made["spaces"] = True
                self.normalization_stats["space_normalizations"] += 1
        
        if self.normalize_newlines:
            before_newlines = normalized_text
            normalized_text = re.sub(r'\n{3,}', '\n\n', normalized_text)
            if normalized_text != before_newlines:
                changes_made["newlines"] = True
                self.normalization_stats["newline_normalizations"] += 1
        
        if self.fix_sentence_spacing:
            # Leading spaces vor Sätzen entfernen
            before_sentence_fix = normalized_text
            # Pattern: Nach Punkt/Start + mehrere Spaces + Großbuchstabe
            normalized_text = re.sub(r'(^|\.)[ ]{2,}([A-Z])', r'\1 \2', normalized_text)
            # Auch Start-of-line leading spaces
            normalized_text = re.sub(r'^[ ]+', '', normalized_text, flags=re.MULTILINE)
            if normalized_text != before_sentence_fix:
                changes_made["sentence_spacing_fixes"] = True
                self.normalization_stats["sentence_spacing_fixes"] += 1
        
        if self.strip_whitespace:
            before_strip = normalized_text
            normalized_text = normalized_text.strip()
            if normalized_text != before_strip:
                changes_made["strip"] = True
                self.normalization_stats["strip_normalizations"] += 1
        
        # Text updaten
        doc.text = normalized_text
        
        # Post-normalization Metriken und Change-Detection
        text_changed = original_text != normalized_text
        post_metrics = None
        
        if self.track_changes:
            post_hash = self.hash_func(normalized_text)
            post_word_count = len(normalized_text.split()) if normalized_text.strip() else 0
            post_metrics = {
                "length": len(normalized_text),
                "n_words": post_word_count,
                "hash": post_hash
            }
            
            # Change Metriken
            length_reduction = pre_metrics["length"] - post_metrics["length"]
            word_reduction = pre_metrics["n_words"] - post_metrics["n_words"]
            
            # Metadaten updaten
            doc.metadata["norm_text_changed"] = text_changed
            doc.metadata["norm_length_reduction"] = length_reduction
            doc.metadata["norm_word_reduction"] = word_reduction
            doc.metadata["norm_changes_made"] = changes_made
        
        # Stats updaten
        if text_changed:
            self.normalization_stats["docs_normalized"] += 1
            if self.track_changes:
                self.normalization_stats["total_length_reduction"] += length_reduction
                self.normalization_stats["total_word_reduction"] += word_reduction
                
                # Top-K Normalization Documents tracken
                total_fixes = sum(changes_made.values())
                if total_fixes > 0:
                    doc_info = {
                        "doc_id": str(doc.id) if doc.id else f"unknown_{self.normalization_stats['docs_processed']}",
                        "total_fixes": total_fixes,
                        "length_reduction": length_reduction,
                        "spaces_fixed": changes_made["spaces"],
                        "newlines_fixed": changes_made["newlines"],
                        "sentence_spacing_fixed": changes_made["sentence_spacing_fixes"],
                        "strip_fixed": changes_made["strip"],
                        "original_length": pre_metrics["length"],
                        "reduction_ratio": length_reduction / pre_metrics["length"] if pre_metrics["length"] > 0 else 0.0
                    }
                    
                    # Sortiert einfügen (Top-K List) - nach Length Reduction
                    self.top_normalization_docs.append(doc_info)
                    self.top_normalization_docs.sort(key=lambda x: x["length_reduction"], reverse=True)
                    
                    # Nur Top-K behalten
                    if len(self.top_normalization_docs) > self.max_tracked_docs:
                        self.top_normalization_docs = self.top_normalization_docs[:self.max_tracked_docs]
        
        # Pipeline Stats
        self.stat_update("docs_processed")
        if text_changed:
            self.stat_update("docs_normalized")
            
        # W&B Logging
        if self.wandb_initialized:
            self._log_document_metrics(text_changed, changes_made, pre_metrics, post_metrics)
            
            # Aggregierte Stats alle 50 Dokumente
            if self.normalization_stats["docs_processed"] % 50 == 0:
                self._log_aggregated_stats()
        
        return bool(normalized_text.strip())
    
    def _log_document_metrics(self, text_changed: bool, changes_made: Dict[str, bool], 
                            pre_metrics: Dict[str, Any], post_metrics: Dict[str, Any]):
        """Loggt per-document Normalization Metriken zu W&B"""
        # Basic sinnvolle Metriken
        doc_metrics = {
            "norm/text_changed": 1 if text_changed else 0,
            "norm/spaces_normalized": 1 if changes_made["spaces"] else 0,
            "norm/newlines_normalized": 1 if changes_made["newlines"] else 0,
            "norm/sentence_spacing_fixes": 1 if changes_made["sentence_spacing_fixes"] else 0,
            "norm/total_changes": sum(changes_made.values())
        }
        
        # Length reduction (sinnvoll für Whitespace-Analyse)
        if self.track_changes and pre_metrics and post_metrics and text_changed:
            length_reduction = pre_metrics["length"] - post_metrics["length"]
            
            doc_metrics.update({
                "norm/length_reduction": length_reduction,
                "norm/length_reduction_ratio": (
                    length_reduction / pre_metrics["length"] if pre_metrics["length"] > 0 else 0.0
                )
            })
        
        # Log zu W&B
        wandb.log(doc_metrics)
    
    def _log_aggregated_stats(self):
        """Loggt aggregierte Normalization Stats zu W&B"""
        if not self.wandb_initialized:
            return
            
        try:
            stats = self.normalization_stats.copy()
            
            # Aggregierte Stats
            agg_stats = {
                "norm_agg/docs_processed": stats["docs_processed"],
                "norm_agg/docs_normalized": stats["docs_normalized"],
                "norm_agg/normalization_rate": (
                    stats["docs_normalized"] / stats["docs_processed"] 
                    if stats["docs_processed"] > 0 else 0
                ),
                "norm_agg/space_normalizations": stats["space_normalizations"],
                "norm_agg/newline_normalizations": stats["newline_normalizations"],
                "norm_agg/avg_length_reduction": (
                    stats["total_length_reduction"] / stats["docs_normalized"]
                    if stats["docs_normalized"] > 0 else 0
                ),
                "norm_agg/sentence_spacing_fixes": stats["sentence_spacing_fixes"]
            }
            
            wandb.log(agg_stats)
            
            log.info(f"📊 Logged normalization stats to W&B: {stats['docs_processed']} docs, "
                    f"{stats['docs_normalized']} normalized "
                    f"({stats['docs_normalized'] / stats['docs_processed']:.1%} rate)")
            
        except Exception as e:
            log.warning(f"Failed to log aggregated normalization stats to W&B: {e}")
    
    def run(self, data, rank: int = 0, world_size: int = 1):
        """Override run method für finale W&B Stats"""
        try:
            # Normal pipeline run
            yield from super().run(data, rank, world_size)
        finally:
            # Final logging (ohne wandb.finish() - managed by main script)
            if self.wandb_initialized and self.normalization_stats["docs_processed"] > 0:
                self._log_final_summary()
                
                log.info(f"✅ Text normalization completed: {self.normalization_stats}")
    
    def _log_final_summary(self):
        """Loggt finale Normalization Summary"""
        # Final aggregated stats
        self._log_aggregated_stats()
        
        # Summary Metriken
        stats = self.normalization_stats
        final_stats = {
            "norm_summary/total_docs_processed": stats["docs_processed"],
            "norm_summary/total_docs_normalized": stats["docs_normalized"],
            "norm_summary/final_normalization_rate": (
                stats["docs_normalized"] / stats["docs_processed"]
                if stats["docs_processed"] > 0 else 0
            ),
            "norm_summary/total_length_reduction": stats["total_length_reduction"],
            "norm_summary/total_space_fixes": stats["space_normalizations"],
            "norm_summary/total_newline_fixes": stats["newline_normalizations"],
            "norm_summary/sentence_spacing_fixes": stats["sentence_spacing_fixes"]
        }
        
        wandb.log(final_stats)
        
        # Top Normalization Documents Tabelle
        if self.top_normalization_docs:
            # Zwei verschiedene Sortierungen erstellen
            docs_by_absolute = sorted(self.top_normalization_docs, key=lambda x: x["length_reduction"], reverse=True)
            docs_by_percentage = sorted(self.top_normalization_docs, key=lambda x: x["reduction_ratio"], reverse=True)
            
            # Tabelle 1: Absolute Length Reduction
            table_data_abs = []
            for i, doc_info in enumerate(docs_by_absolute[:10], 1):  # Top 10
                table_data_abs.append([
                    i,  # Rank
                    doc_info["doc_id"],
                    doc_info["length_reduction"],
                    f"{doc_info['reduction_ratio']:.3%}",
                    doc_info["total_fixes"],
                    "✓" if doc_info["spaces_fixed"] else "",
                    "✓" if doc_info["newlines_fixed"] else "",
                    "✓" if doc_info["sentence_spacing_fixed"] else "",
                    "✓" if doc_info["strip_fixed"] else "",
                    doc_info["original_length"]
                ])
            
            abs_table = wandb.Table(
                columns=[
                    "Rank", "Document ID", "Char Reduction", "Reduction %", "Total Fixes", 
                    "Spaces", "Newlines", "Sentence Spacing", "Strip", "Original Length"
                ],
                data=table_data_abs
            )
            
            # Tabelle 2: Percentage Reduction  
            table_data_pct = []
            for i, doc_info in enumerate(docs_by_percentage[:10], 1):  # Top 10
                table_data_pct.append([
                    i,  # Rank
                    doc_info["doc_id"],
                    f"{doc_info['reduction_ratio']:.3%}",
                    doc_info["length_reduction"],
                    doc_info["total_fixes"],
                    "✓" if doc_info["spaces_fixed"] else "",
                    "✓" if doc_info["newlines_fixed"] else "",
                    "✓" if doc_info["sentence_spacing_fixed"] else "",
                    "✓" if doc_info["strip_fixed"] else "",
                    doc_info["original_length"]
                ])
            
            pct_table = wandb.Table(
                columns=[
                    "Rank", "Document ID", "Reduction %", "Char Reduction", "Total Fixes",
                    "Spaces", "Newlines", "Sentence Spacing", "Strip", "Original Length"
                ],
                data=table_data_pct
            )
            
            # Beide Tabellen loggen
            wandb.log({
                "norm_summary/top_documents_absolute_reduction": abs_table,
                "norm_summary/top_documents_percentage_reduction": pct_table
            })
            
            log.info(f"📊 Top normalization outliers: "
                    f"Max absolute: {docs_by_absolute[0]['length_reduction']} chars, "
                    f"Max percentage: {docs_by_percentage[0]['reduction_ratio']:.2%}")
        
        # Histogram: Length Reduction Distribution
        if self.top_normalization_docs:
            length_reductions = [doc["length_reduction"] for doc in self.top_normalization_docs if doc["length_reduction"] > 0]
            if length_reductions:
                wandb.log({
                    "norm_summary/length_reduction_distribution": wandb.Histogram(length_reductions),
                    "norm_summary/max_length_reduction": max(length_reductions),
                    "norm_summary/avg_length_reduction_outliers": sum(length_reductions) / len(length_reductions)
                })
        
        log.info(f"📋 Normalization Summary: {stats['docs_normalized']}/{stats['docs_processed']} docs normalized, "
                f"{stats['total_length_reduction']} chars reduced, "
                f"{stats['space_normalizations']} space fixes, {stats['newline_normalizations']} newline fixes, "
                f"{stats['sentence_spacing_fixes']} sentence spacing fixes") 