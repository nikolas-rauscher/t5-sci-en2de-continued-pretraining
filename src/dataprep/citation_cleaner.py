"""
Multi-Citation Cleaning Pipeline Module für DataTrove

Entfernt verschiedene Citation-Typen mit konfigurierbaren Regex-Patterns und bietet
detaillierte W&B Analytics für jeden Citation-Typ separat.

Citation Types:
- semicolon_blocks: "Author1 ; Author2 ; Author3" 
- intext_citations: "[12]", "[3, 7]", "(Smith, 2020)"

Usage:
```python
from src.dataprep.citation_cleaner import CitationCleaner

cleaner = CitationCleaner(
    citation_patterns={
        "semicolon_blocks": r'\s(?:[\w\-]+(?:\d+)?\s+;\s+)+(?:[\w\-]+(?:\d+)?)(?:\s*;)?\s+',
        "intext_citations": r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]|\(.*?\d{4}[a-z]?\)'
    },
    log_to_wandb=True
)
```

Pipeline Integration:
```python
from datatrove.pipeline import Pipeline
from src.dataprep.citation_cleaner import CitationCleaner
from src.dataprep.text_normalizer import TextNormalizer

pipeline = Pipeline([
    CitationCleaner(log_to_wandb=True),    # 1. Citations entfernen
    TextNormalizer(),                      # 2. Text normalisieren
])
```
"""

import re
import logging
from typing import Dict, Any, Optional, List
from collections import defaultdict
import heapq

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.hashing import create_hash_func, HashConfig
from datatrove.utils.typeshelper import Languages

log = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    log.warning("wandb not available - citation cleaning stats will not be logged")


class CitationCleaner(BaseFilter):
    """
    Multi-Type Citation Cleaner der verschiedene Referenz-Arten mit Regex entfernt.
    Trackt jeden Citation-Typ separat für detaillierte Analytics.
    """
    
    name = "🧹 Multi-Citation Cleaner"
    
    def __init__(
        self, 
        citation_patterns: Dict[str, str] = None,
        replacement: str = ' ',
        track_changes: bool = True,
        wandb_project: str = "BA-DataTrove",
        wandb_group: str = "citation-cleaning",
        log_to_wandb: bool = True,
        exclusion_writer: DiskWriter = None
    ):
        """
        Args:
            citation_patterns: Dict mit Citation-Pattern {"type_name": "regex_pattern"}
            replacement: Womit Citations ersetzt werden 
            track_changes: Ob Änderungen getrackt werden sollen
            wandb_project: W&B Projekt Name
            wandb_group: W&B Gruppe 
            log_to_wandb: Ob Stats zu W&B geloggt werden sollen
            exclusion_writer: Optional writer für leere Dokumente
        """
        super().__init__(exclusion_writer)
        
        # Default Citation Patterns wenn keine angegeben
        if citation_patterns is None:
            citation_patterns = {
                "semicolon_blocks": r'\s(?:[\w\-]+(?:\d+)?\s+;\s+)+(?:[\w\-]+(?:\d+)?)(?:\s*;)?\s+',
                "intext_citations": r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]|\(.*?\d{4}[a-z]?\)'
            }
        
        self.citation_patterns = citation_patterns
        self.citation_regexes = {
            name: re.compile(pattern) 
            for name, pattern in citation_patterns.items()
        }
        self.replacement = replacement
        self.track_changes = track_changes
        self.log_to_wandb = log_to_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        self.wandb_group = wandb_group
        
        # DataTrove's Hash-Funktion
        hash_config = HashConfig(precision=64, hash_fc="sha1")
        self.hash_func = create_hash_func(hash_config)
        
        # Stats für jeden Citation-Typ separat
        self.citation_stats = {}
        for citation_type in self.citation_patterns.keys():
            self.citation_stats[citation_type] = {
                "docs_with_citations": 0,
                "total_citations_removed": 0,
                "total_length_reduction": 0,
                "citation_distribution": defaultdict(int),
                "top_citation_docs": []
            }
        
        # Gesamt-Stats
        self.cleaning_stats = {
            "docs_processed": 0,
            "docs_with_any_citations": 0,
            "total_citations_all_types": 0,
            "total_length_reduction": 0,
            "total_word_reduction": 0
        }
        
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
                log.info(f"📊 Using existing W&B session for Citation Cleaner")
            else:
                # Fallback: eigene Session (sollte normalerweise nicht passieren)
                wandb.init(
                    project=self.wandb_project,
                    group=self.wandb_group,
                    tags=["citation-cleaning", "datatrove", "text-preprocessing"],
                    job_type="data-cleaning",
                    notes="Citation cleaning with detailed analytics",
                    config={
                        "citation_patterns": self.citation_patterns,
                        "replacement": self.replacement,
                        "track_changes": self.track_changes
                    }
                )
                self.wandb_initialized = True
                log.info(f"📊 W&B initialized for Citation Cleaner - project: {self.wandb_project}")
        except Exception as e:
            log.warning(f"Failed to initialize W&B for citation cleaner: {e}")
            self.log_to_wandb = False
    
    def _extract_key_metrics(self, doc: Document) -> Dict[str, Any]:
        """Extrahiert die wichtigsten Metriken aus den vorhandenen DataTrove Stats"""
        metadata = doc.metadata
        return {
            "length": int(metadata.get("length", len(doc.text))),
            "n_words": int(metadata.get("n_words", 0)),
            "n_sentences": int(metadata.get("n_sentences", 0)),
            "n_lines": int(metadata.get("n_lines", 0)),
            "n_paragraphs": int(metadata.get("n_paragraphs", 0))
        }
    
    def _get_doc_info(self, doc: Document) -> Dict[str, str]:
        """Extrahiert Document-Identifikation und Metadata"""
        metadata = doc.metadata
        return {
            "doc_id": str(doc.id) if doc.id else "unknown",  # Direkt aus doc.id
            "title": str(metadata.get("title", ""))[:100],  # Truncate title
            "publication_year": str(metadata.get("publication_year", "unknown")),
            "authors": str(metadata.get("authors", ""))[:100]
        }
    
    def _categorize_citations(self, count: int) -> str:
        """Kategorisiert Citations nach Anzahl"""
        if count == 0:
            return "none"
        elif count <= 2:
            return "low"
        elif count <= 5:
            return "medium"
        elif count <= 10:
            return "high"
        else:
            return "very_high"
    
    def filter(self, doc: Document) -> bool:
        """Citation Cleaning mit detailliertem Tracking"""
        original_text = doc.text
        self.cleaning_stats["docs_processed"] += 1
        
        # Document Info extrahieren
        doc_info = self._get_doc_info(doc)
        
        # Pre-cleaning Metriken (bevor text geändert wird!)
        pre_metrics = None
        if self.track_changes:
            original_hash = self.hash_func(original_text)
            # Konsistente Word-Berechnung: immer len(text.split())
            pre_word_count = len(original_text.split()) if original_text.strip() else 0
            pre_metrics = {
                "length": len(original_text),
                "n_words": pre_word_count,  # Eigene Berechnung statt DataTrove Metadaten
                "n_sentences": int(doc.metadata.get("n_sentences", 0)),
                "n_lines": int(doc.metadata.get("n_lines", 0)),
                "n_paragraphs": int(doc.metadata.get("n_paragraphs", 0))
            }
            
            doc.metadata["citation_pre_hash"] = str(original_hash)
            doc.metadata["citation_pre_length"] = pre_metrics["length"]
            doc.metadata["citation_pre_words"] = pre_metrics["n_words"]
        
        # Citations finden und entfernen
        citations_found = {}
        for citation_type, citation_regex in self.citation_regexes.items():
            citations_found[citation_type] = citation_regex.findall(original_text)
        
        # Citation-Längen berechnen
        citation_lengths = {}
        total_citation_length = 0
        for citation_type, citations in citations_found.items():
            citation_lengths[citation_type] = [len(citation) for citation in citations]
            total_citation_length += sum(citation_lengths[citation_type])
            avg_citation_length = total_citation_length / len(citations) if len(citations) > 0 else 0
        
        cleaned_text = original_text
        for citation_type, citations in citations_found.items():
            cleaned_text = self.citation_regexes[citation_type].sub(self.replacement, cleaned_text)
        
        # Text updaten (ohne Normalisierung - das macht TextNormalizer)
        doc.text = cleaned_text
        
        # Post-cleaning Metriken (nach text update!)
        post_metrics = None
        if self.track_changes:
            cleaned_hash = self.hash_func(cleaned_text)
            # Neue word count für cleaned text
            post_word_count = len(cleaned_text.split()) if cleaned_text.strip() else 0
            post_metrics = {
                "length": len(cleaned_text),
                "n_words": post_word_count,
                "n_sentences": pre_metrics["n_sentences"],  # Bleibt gleich
                "n_lines": pre_metrics["n_lines"],  # Bleibt gleich  
                "n_paragraphs": pre_metrics["n_paragraphs"]  # Bleibt gleich
            }
            
            doc.metadata["citation_post_hash"] = str(cleaned_hash)
            doc.metadata["citation_post_length"] = post_metrics["length"]
            doc.metadata["citation_post_words"] = post_metrics["n_words"]
            
            # Änderungs-Metriken
            text_changed = original_hash != cleaned_hash
            length_reduction = pre_metrics["length"] - post_metrics["length"]
            word_reduction = pre_metrics["n_words"] - post_metrics["n_words"]  # Konsistent berechnet
            
            doc.metadata["citation_text_changed"] = text_changed
            doc.metadata["citation_length_reduction"] = length_reduction
            doc.metadata["citation_word_reduction"] = word_reduction
            doc.metadata["citation_length_reduction_ratio"] = (
                length_reduction / pre_metrics["length"] if pre_metrics["length"] > 0 else 0.0
            )
        
        # Citation-spezifische Metadaten
        for citation_type, citations in citations_found.items():
            citations_count = len(citations)
            doc.metadata[f"{citation_type}_citations_removed"] = citations_count
            doc.metadata[f"had_{citation_type}_citations"] = citations_count > 0
            
            # Detaillierte Analytics updaten
            self._update_analytics(citation_type, citations_count, doc_info, pre_metrics)
            
            # Stats für W&B updaten
            if citations_count > 0:
                self.citation_stats[citation_type]["docs_with_citations"] += 1
                self.citation_stats[citation_type]["total_citations_removed"] += citations_count
                
            if self.track_changes and doc.metadata.get(f"{citation_type}_text_changed", False):
                self.citation_stats[citation_type]["total_length_reduction"] += doc.metadata[f"{citation_type}_length_reduction"]
                self.citation_stats[citation_type]["total_word_reduction"] += doc.metadata[f"{citation_type}_word_reduction"]
            
            # Pipeline Stats
            self.stat_update("docs_processed")
            if citations_count > 0:
                self.stat_update(f"{citation_type}_citations_removed", citations_count)
                self.stat_update(f"{citation_type}_docs_with_citations")
            
            # W&B Logging - detailliert per document
            if self.wandb_initialized:
                self._log_document_metrics(citation_type, citations_count, doc_info, pre_metrics, post_metrics, 
                                         total_citation_length, avg_citation_length)
                
                # Aggregierte Stats alle 50 Dokumente
                if self.cleaning_stats["docs_processed"] % 50 == 0:
                    self._log_aggregated_stats()
        
        # Gesamt-Stats updaten
        self.cleaning_stats["docs_with_any_citations"] += 1
        self.cleaning_stats["total_citations_all_types"] += sum(len(c) for c in citations_found.values())
        
        if self.track_changes and any(doc.metadata.get(f"{citation_type}_text_changed", False) for citation_type in self.citation_regexes):
            self.cleaning_stats["total_length_reduction"] += sum(doc.metadata[f"{citation_type}_length_reduction"] for citation_type in self.citation_regexes)
            self.cleaning_stats["total_word_reduction"] += sum(doc.metadata[f"{citation_type}_word_reduction"] for citation_type in self.citation_regexes)
        
        # W&B Logging - detailliert per document
        if self.wandb_initialized:
            self._log_final_summary()
            
        return bool(cleaned_text.strip())
    
    def _update_analytics(self, citation_type: str, citations_count: int, doc_info: Dict[str, str], pre_metrics: Dict[str, Any]):
        """Updatet detaillierte Analytics"""
        # Citation Distribution
        self.citation_stats[citation_type]["citation_distribution"][citations_count] += 1
        
        # Top Citation Documents (Top 20) - ohne journal
        if citations_count > 0:
            doc_entry = (citations_count, doc_info["doc_id"], doc_info["title"], "")
            if len(self.citation_stats[citation_type]["top_citation_docs"]) < 20:
                heapq.heappush(self.citation_stats[citation_type]["top_citation_docs"], doc_entry)
            elif citations_count > self.citation_stats[citation_type]["top_citation_docs"][0][0]:
                heapq.heapreplace(self.citation_stats[citation_type]["top_citation_docs"], doc_entry)
        
        # Length vs Citations Correlation
        if pre_metrics:
            self.citation_stats[citation_type]["length_vs_citations"].append((pre_metrics["length"], citations_count))
    
    def _log_document_metrics(self, citation_type: str, citations_count: int, doc_info: Dict[str, str], 
                            pre_metrics: Dict[str, Any], post_metrics: Dict[str, Any],
                            total_citation_length: int, avg_citation_length: float):
        """Loggt detaillierte per-document Metriken"""
        citation_category = self._categorize_citations(citations_count)
        
        # Basic per-document Metriken
        doc_metrics = {
            "doc/citations_found": citations_count,
            "doc/had_citations": 1 if citations_count > 0 else 0,
            "doc/citation_category": citation_category,
            "doc/doc_id": doc_info["doc_id"][:8],  # Kurze ID für bessere Anzeige
        }
        
        # Citation-Längen Metriken
        if citations_count > 0:
            doc_metrics.update({
                "doc/total_citation_length": total_citation_length,
                "doc/avg_citation_length": avg_citation_length,
                "doc/citation_length_ratio": (
                    total_citation_length / pre_metrics["length"] 
                    if pre_metrics and pre_metrics["length"] > 0 else 0.0
                )
            })
        
        # Text Änderungs-Metriken
        if self.track_changes and pre_metrics and post_metrics:
            length_reduction = pre_metrics["length"] - post_metrics["length"]
            word_reduction = pre_metrics["n_words"] - post_metrics["n_words"]  # Konsistent berechnet
            
            doc_metrics.update({
                "doc/pre_length": pre_metrics["length"],
                "doc/post_length": post_metrics["length"],
                "doc/length_reduction": length_reduction,
                "doc/length_reduction_ratio": (
                    length_reduction / pre_metrics["length"] if pre_metrics["length"] > 0 else 0.0
                ),
                "doc/pre_words": pre_metrics["n_words"],
                "doc/post_words": post_metrics["n_words"],
                "doc/word_reduction": word_reduction,
                "doc/word_reduction_ratio": (
                    word_reduction / pre_metrics["n_words"] if pre_metrics["n_words"] > 0 else 0.0
                ),
            })
        
        # Log zu W&B
        wandb.log({f"{citation_type}/{k}": v for k, v in doc_metrics.items()})
        
        # Spezielle Logs für interessante Dokumente
        if citations_count >= 10:
            log.info(f"🔥 High citation document: {doc_info['doc_id']} "
                    f"({citations_count} citations) - {doc_info['title'][:60]}...")
        
        # Debug Log für word reduction
        if citations_count > 0 and self.track_changes and pre_metrics and post_metrics:
            word_reduction = pre_metrics["n_words"] - post_metrics["n_words"]
            if word_reduction > 0:
                log.info(f"📝 Word reduction in {doc_info['doc_id']}: {pre_metrics['n_words']} → {post_metrics['n_words']} (-{word_reduction})")
    
    def _log_aggregated_stats(self):
        """Loggt aggregierte und analytische Stats zu W&B"""
        if not self.wandb_initialized:
            return
            
        try:
            stats = self.cleaning_stats.copy()
            
            # Basic aggregierte Stats
            agg_stats = {
                "agg/docs_processed": stats["docs_processed"],
                "agg/docs_with_citations": stats["docs_with_any_citations"],
                "agg/total_citations_removed": stats["total_citations_all_types"],
                "agg/citation_rate": (
                    stats["docs_with_any_citations"] / stats["docs_processed"] 
                    if stats["docs_processed"] > 0 else 0
                ),
                "agg/avg_citations_per_doc": (
                    stats["total_citations_all_types"] / stats["docs_with_any_citations"]
                    if stats["docs_with_any_citations"] > 0 else 0
                ),
                "agg/avg_length_reduction": (
                    stats["total_length_reduction"] / stats["docs_with_any_citations"]
                    if stats["docs_with_any_citations"] > 0 else 0
                ),
                "agg/avg_word_reduction": (
                    stats["total_word_reduction"] / stats["docs_with_any_citations"]
                    if stats["docs_with_any_citations"] > 0 else 0
                )
            }
            
            # Citation Distribution Analytics
            distribution_stats = {}
            for category in ["none", "low", "medium", "high", "very_high"]:
                count = sum(
                    self.citation_stats[citation_type]["citation_distribution"][c] 
                    for citation_type in self.citation_regexes 
                    for c in self.citation_stats[citation_type]["citation_distribution"] 
                    if self._categorize_citations(c) == category
                )
                distribution_stats[f"dist/{citation_type}_{category}_citation_docs"] = count
                distribution_stats[f"dist/{citation_type}_{category}_citation_ratio"] = (
                    count / stats["docs_processed"] if stats["docs_processed"] > 0 else 0
                )
            
            # Combine und log
            all_stats = {**agg_stats, **distribution_stats}
            wandb.log(all_stats)
            
            log.info(f"📊 Logged detailed stats to W&B: {stats['docs_processed']} docs, "
                    f"{stats['docs_with_any_citations']} with citations, "
                    f"{len([c for citation_type in self.citation_regexes for c in self.citation_stats[citation_type]['citation_distribution'] if c >= 10])} high-citation docs")
            
        except Exception as e:
            log.warning(f"Failed to log aggregated stats to W&B: {e}")
    
    def run(self, data, rank: int = 0, world_size: int = 1):
        """Override run method um finale detaillierte W&B Stats zu loggen"""
        try:
            # Normal pipeline run
            yield from super().run(data, rank, world_size)
        finally:
            # Final comprehensive logging (ohne wandb.finish() - managed by main script)
            if self.wandb_initialized and self.cleaning_stats["docs_processed"] > 0:
                self._log_final_summary()
                
                log.info(f"✅ Citation cleaning completed: {self.cleaning_stats}")
    
    def _log_final_summary(self):
        """Loggt finale Summary mit Top-Documents und Insights"""
        # Final aggregated stats
        self._log_aggregated_stats()
        
        # Summary Metriken
        final_stats = {
            "summary/total_docs_processed": self.cleaning_stats["docs_processed"],
            "summary/total_docs_with_citations": self.cleaning_stats["docs_with_any_citations"],
            "summary/total_citations_removed": self.cleaning_stats["total_citations_all_types"],
            "summary/final_citation_rate": (
                self.cleaning_stats["docs_with_any_citations"] / self.cleaning_stats["docs_processed"]
                if self.cleaning_stats["docs_processed"] > 0 else 0
            ),
            "summary/total_length_reduction": self.cleaning_stats["total_length_reduction"],
            "summary/total_word_reduction": self.cleaning_stats["total_word_reduction"],
            "summary/unique_citation_counts": len(self.citation_stats),
            "summary/max_citations_in_doc": max(max(c for citation_type in self.citation_regexes for c in self.citation_stats[citation_type]["citation_distribution"]) for citation_type in self.citation_regexes) if self.citation_stats else 0
        }
        
        wandb.log(final_stats)
        
        # Log Top Citation Documents als Table (ohne Journal)
        for citation_type, top_docs in self.citation_stats.items():
            if top_docs["top_citation_docs"]:
                top_docs_data = []
                for citations, doc_id, title, _ in sorted(top_docs["top_citation_docs"], reverse=True):
                    top_docs_data.append([citations, doc_id, title[:80]])
                
                table = wandb.Table(
                    columns=["Citations", "Doc ID", "Title"],
                    data=top_docs_data
                )
                wandb.log({f"top_{citation_type}_citation_documents": table})
                
                log.info(f"📋 Top {citation_type} citation document: {top_docs_data[0][0]} citations - {top_docs_data[0][2]}")
        
        # Citation Distribution als Histogram
        for citation_type, citation_stats in self.citation_stats.items():
            citation_counts = []
            for count, freq in citation_stats["citation_distribution"].items():
                citation_counts.extend([count] * freq)
            
            wandb.log({f"citation_distribution_histogram_{citation_type}": wandb.Histogram(citation_counts)}) 