"""
Multi-Citation Cleaning Pipeline Module für DataTrove

Entfernt verschiedene Citation-Typen mit konfigurierbaren Regex-Patterns und bietet
detaillierte W&B Analytics für jeden Citation-Typ separat.

Enhanced with SMART VALIDATION to prevent false positives, especially for semicolon_blocks.

Citation Types:
- semicolon_blocks: "Author1 ; Author2 ; Author3" (with smart author validation)
- eckige_klammern_numerisch: "[12]", "[3, 7]", "[1-5]"
- consecutive_numeric_citations: "(1)(2)(3)(4)" - consecutive citation chains
- isolated_numeric_citations: "(1)", "(23)" - isolated numbers (but keeps "(25°C)", "(100%)")
- autor_jahr_multi_klammer: "(Smith, 2020; Jones, 2021)"
- autor_jahr_klammer_einzel: "(Smith, 2020)"
- ref_nummer: "ref 1", "refs 1-3", "Ref. 12" (case-insensitive)
- page_references: "p. 123", "pp. 45-67", "pages 1,3,5" (case-insensitive)
- figure_table_refs: "Fig. 1", "Table 2a", "figure 3,4" (case-insensitive)

REMOVED for T5 pretraining (important for text flow):
- autor_jahr_text: "Smith (2020)", "Smith et al. (2020)" 
- chapter_section: "Chapter 3", "section 2.1"
- runde_klammern_numerisch: "(1)", "(1a)" - useful for math/data

Smart Validation Features:
- Semicolon blocks: Simple rule-based validation with stopwords and biological terms
- Context analysis: Academic citation patterns vs random text
- Clear rejection reasons for debugging

Usage:
```python
from src.dataprep.multi_citation_cleaner import MultiCitationCleaner

# With smart validation (default)
cleaner = MultiCitationCleaner(
    enable_smart_validation=True,
    semicolon_max_authors=12,
    log_to_wandb=True
)

# Legacy mode (no validation)
cleaner = MultiCitationCleaner(
    enable_smart_validation=False,
    log_to_wandb=True
)
```
"""

import re
import logging
from typing import Dict, Any, Optional, List, Set, Tuple
from collections import defaultdict, Counter
import heapq

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.hashing import create_hash_func, HashConfig

# Import simple validator
from src.dataprep.semicolon_validator import SemicolonCitationValidator
from src.dataprep.context_validator import ContextValidator
from src.dataprep.citation_cleaner_logger import CitationCleanerLogger
from src.dataprep.citation_cleaner_worker_stats import CitationCleanerWorkerStats
from src.dataprep.text_cleaners import ComprehensiveLineCleaner

log = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    log.warning("wandb not available - multi-citation cleaning stats will not be logged")


class MultiCitationCleaner(BaseFilter):
    """
    Enhanced Multi-Type Citation Cleaner mit einfacher Validierung.
    Trackt jeden Citation-Typ separat und verhindert False Positives.
    """
    
    name = "🧹 Smart Multi-Citation Cleaner"
    
    def __init__(
        self, 
        citation_patterns: Dict[str, str] = None,
        replacement: str = ' ',
        track_changes: bool = True,
        
        # Debug Mode
        debug_mode: bool = False,
        
        # Smart Validation Parameters
        enable_smart_validation: bool = True,
        semicolon_max_authors: int = 12,
        semicolon_min_authors: int = 2,
        
        # Logging Parameters
        max_false_positive_samples: int = 100, 
        max_top_citation_docs: int = 50,        
        
        # W&B Parameters
        wandb_project: str = "BA-DataTrove",
        wandb_group: str = "smart-multi-citation-cleaning",
        log_to_wandb: bool = True,
        exclusion_writer: DiskWriter = None
    ):
        """
        Args:
            citation_patterns: Dict mit Citation-Pattern {"type_name": "regex_pattern"}
            replacement: Womit Citations ersetzt werden 
            track_changes: Ob Änderungen getrackt werden sollen
            
            # Debug Mode
            debug_mode: Statt Text zu entfernen, Debug-Tags einfügen die zeigen welche Methode entfernt hätte
            
            # Smart Validation
            enable_smart_validation: Intelligente Validierung aktivieren
            semicolon_max_authors: Max. Anzahl Autoren in Semicolon-Listen
            semicolon_min_authors: Min. Anzahl Autoren in Semicolon-Listen  
            
            # Logging
            max_false_positive_samples: Max Anzahl False Positive Samples zu loggen
            max_top_citation_docs: Max Anzahl Top Citation Documents zu tracken
            
            wandb_project: W&B Projekt Name
            wandb_group: W&B Gruppe 
            log_to_wandb: Ob Stats zu W&B geloggt werden sollen
            exclusion_writer: Optional writer für leere Dokumente
        """
        super().__init__(exclusion_writer)
        
        # Debug Mode Settings
        self.debug_mode = debug_mode
        if self.debug_mode:
            log.info("🐛 DEBUG MODE ENABLED - Text wird mit Debug-Tags markiert statt entfernt")
        
        # Smart Validation Settings
        self.enable_smart_validation = enable_smart_validation
        
        # Initialize validators
        if self.enable_smart_validation:
            self.semicolon_validator = SemicolonCitationValidator(
                max_authors=semicolon_max_authors,
                min_authors=semicolon_min_authors
            )
            self.context_validator = ContextValidator()
        
        # Default Citation Patterns wenn keine angegeben
        if citation_patterns is None:
            citation_patterns = {
                # Enhanced Patterns mit besserer Precision
                "semicolon_blocks": r'\s(?:[\w\-]+(?:\d+)?\s+;\s+)+(?:[\w\-]+(?:\d+)?)(?:\s*;)?\s+',
                
                # Verbesserte numerische Citation-Patterns
                "eckige_klammern_numerisch": r"\[\s*\d+(?:,\s*\d+)*\s*(?:-\s*\d+)?\s*\]", # [1,2,3] or [1-3] 
                "consecutive_numeric_citations": r"(?:\(\s*\d+\s*\)){2,}",  # (1)(2)(3) patterns
                "isolated_numeric_citations": r"\(\s*\d{1,3}\s*\)(?!\s*[A-Za-z°%])",  # (1), (23) but not (25°C) or (100%)
                
                # Autor-Jahr Citation-Patterns - REMOVED autor_jahr_text (important for text flow)
                "autor_jahr_multi_klammer": r"\((?:[A-Z][A-Za-z'-]+(?:\s+(?:and|et)\s+[A-Z][A-Za-z'-]+)?(?:,\s*Jr\.?)?(?: et al\.)?,\s*\d{4}[a-z]?\s*;\s*)+[A-Z][A-Za-z'-]+(?:\s+(?:and|et)\s+[A-Z][A-Za-z'-]+)?(?:,\s*Jr\.?)?(?: et al\.)?,\s*\d{4}[a-z]?\)",
                "autor_jahr_klammer_einzel": r"\([A-Z][A-Za-z'-]+(?:\s+(?:and|et)\s+[A-Z][A-Za-z'-]+)?(?:,\s*Jr\.?)?(?: et al\.)?,\s*\d{4}[a-z]?\)",
                
                # Author-Year Citations in Square Brackets (NEW) - Universal Unicode support
                "autor_jahr_eckig_einzel": r"\[[\w\u00C0-\u017F\u0400-\u04FF\u4E00-\u9FFF\u0590-\u05FF\u0600-\u06FF]+(?:\s+(?:and|et|und|y|et)\s+[\w\u00C0-\u017F\u0400-\u04FF\u4E00-\u9FFF\u0590-\u05FF\u0600-\u06FF]+)*(?:,\s*Jr\.?)?(?: et al\.)?,\s*\d{4}[a-z]?\]",
                "autor_jahr_eckig_multi": r"\[(?:[\w\u00C0-\u017F\u0400-\u04FF\u4E00-\u9FFF\u0590-\u05FF\u0600-\u06FF]+(?:\s+(?:and|et|und|y|et)\s+[\w\u00C0-\u017F\u0400-\u04FF\u4E00-\u9FFF\u0590-\u05FF\u0600-\u06FF]+)*(?:,\s*Jr\.?)?(?: et al\.)?,\s*\d{4}[a-z]?\s*;\s*)+[\w\u00C0-\u017F\u0400-\u04FF\u4E00-\u9FFF\u0590-\u05FF\u0600-\u06FF]+(?:\s+(?:and|et|und|y|et)\s+[\w\u00C0-\u017F\u0400-\u04FF\u4E00-\u9FFF\u0590-\u05FF\u0600-\u06FF]+)*(?:,\s*Jr\.?)?(?: et al\.)?,\s*\d{4}[a-z]?\]",
                
                # Referenz-Nummern
                "ref_nummer": r"\b(?:ref|refs)\.?\s*\d+(?:,\s*\d+)*(?:-\s*\d+)?\b",
                
                # Zusätzliche häufige Citation-Patterns - REMOVED chapter_section (important for structure)
                "page_references": r"(?:^|\s)(?:p|pp|page|pages)\.?\s*\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*(?=\s|$)", # Only standalone page refs, not in citations
                "figure_table_refs": r"(?:\(\s*)?(?:fig|figure|tab|table|tbl)\.?\s*\d+(?:[a-z])?(?:,\s*\d+(?:[a-z])?)*(?:\s*[;:]\s*[^)]+)?(?:\s*\))?", # Capture complete figure citations with additional content
            }
        
        self.citation_patterns = citation_patterns
        self.citation_regexes = {}
        for name, pattern in citation_patterns.items():
            # Case-insensitive für bestimmte patterns
            if name in ["ref_nummer", "page_references", "figure_table_refs"]:
                self.citation_regexes[name] = re.compile(pattern, re.IGNORECASE)
            else:
                self.citation_regexes[name] = re.compile(pattern)
        self.replacement = replacement
        self.track_changes = track_changes
        self.log_to_wandb = log_to_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        self.wandb_group = wandb_group
        
        # DataTrove's Hash-Funktion
        if self.track_changes:
            hash_config = HashConfig(precision=64, hash_fc="sha1")
            self.hash_func = create_hash_func(hash_config)
        
        # Stats für jeden Citation-Typ separat
        self.citation_stats = {}
        for citation_type in self.citation_patterns.keys():
            self.citation_stats[citation_type] = {
                "docs_with_citations": 0,
                "total_citations_removed": 0,
                "total_citations_found": 0,
                "total_citations_rejected": 0,  # Smart validation rejects
                "total_length_reduction": 0,
                "citation_distribution": defaultdict(int),
                "top_citation_docs": [],
                "false_positive_samples": []  # Store rejected citations for analysis
            }
        
        # Gesamt-Stats
        self.cleaning_stats = {
            "docs_processed": 0,
            "docs_with_any_citations": 0,
            "total_citations_all_types": 0,
            "total_citations_rejected": 0,
            "total_length_reduction": 0,
            "total_word_reduction": 0,
            "smart_validation_enabled": self.enable_smart_validation,
            "debug_mode_enabled": self.debug_mode,
            
            # Figure-only line removal stats
            "docs_with_figure_lines_removed": 0,
            "total_figure_lines_removed": 0,
            "total_figure_line_length_reduction": 0,
            "figure_line_removal_samples": [],
            
            # Appendix section removal stats
            "docs_with_appendix_sections_removed": 0,
            "total_appendix_sections_removed": 0,
            "total_appendix_lines_removed": 0,
            "total_appendix_length_reduction": 0,
            "appendix_removal_samples": [],
            
            # Short line removal stats
            "docs_with_short_lines_removed": 0,
            "total_short_lines_removed": 0,
            "total_short_line_length_reduction": 0,
            "short_line_removal_samples": [],
            
            # Top Documents tracking
            "top_figure_line_removal_docs": [],  # (lines_removed, doc_id, title, length_reduction)
            "top_appendix_removal_docs": [],     # (sections_removed, doc_id, title, length_reduction)
            "top_short_line_removal_docs": [],   # (lines_removed, doc_id, title, length_reduction)
            "top_combined_reduction_docs": [],   # (total_reduction, doc_id, title, citations+figure_lines+appendix+short_lines)
            
        }
        
        # Logger Initialisierung - W&B nur für main process (rank 0)
        self.wandb_enabled = log_to_wandb and WANDB_AVAILABLE
        self.logger = CitationCleanerLogger(
            wandb_project=wandb_project,
            wandb_group=wandb_group,
            log_to_wandb=False,  # Will be enabled in run() for rank 0 only
            max_false_positive_samples=max_false_positive_samples,
            max_top_citation_docs=max_top_citation_docs
        )
        
        # Worker Stats Handler
        self.worker_stats = CitationCleanerWorkerStats(
            citation_patterns=self.citation_patterns,
            max_false_positive_samples=max_false_positive_samples,
            max_top_citation_docs=max_top_citation_docs,
            enable_smart_validation=enable_smart_validation
        )
        
        # Comprehensive Line Cleaner with all specialized cleaners - pass debug mode
        self.line_cleaner = ComprehensiveLineCleaner(debug_mode=self.debug_mode)
        
        # Logging Settings
        self.max_false_positive_samples = max_false_positive_samples
        self.max_top_citation_docs = max_top_citation_docs
        
        # Top-K Dokumente mit meisten Normalisierungs-Issues tracken
        self.top_normalization_docs = []
        self.max_tracked_docs = self.max_top_citation_docs  # Use configurable parameter
    
    def _validate_citation(self, citation_type: str, match_text: str) -> Tuple[bool, str]:
        """Simple validation for different citation types"""
        if not self.enable_smart_validation:
            return True, "validation_disabled"
        
        # Special validation for semicolon blocks
        if citation_type == "semicolon_blocks":
            is_valid, reason = self.semicolon_validator.validate(match_text)
            return is_valid, reason
        
        # For other citation types: basic validation
        if citation_type == "autor_jahr_text":
            # Check for unrealistic years
            year_match = re.search(r'\((\d{4})[a-z]?\)', match_text)
            if year_match:
                year = int(year_match.group(1))
                if year < 1800 or year > 2030:
                    return False, f"unrealistic_year ({year})"
        
        # Default: accept if no issues
        return True, "valid_citation"
    
    def _validate_citation_with_context(self, citation_type: str, match_text: str, 
                                      start_pos: int, end_pos: int, full_text: str) -> Tuple[bool, str]:
        """Enhanced validation with context analysis for structural vs citation references"""
        if not self.enable_smart_validation:
            return True, "validation_disabled"
        
        # Special validation for semicolon blocks
        if citation_type == "semicolon_blocks":
            is_valid, validation_info = self.semicolon_validator.validate(match_text, start_pos, end_pos, full_text)
            # Extract simple reason from validation_info dict
            reason = validation_info.get("validation_steps", ["unknown"])[-1] if isinstance(validation_info, dict) else str(validation_info)
            return is_valid, reason
        
        # Enhanced validation for author-year citations with sentence flow analysis
        if citation_type in ["autor_jahr_text", "autor_jahr_klammer_einzel", "autor_jahr_multi_klammer"]:
            return self.context_validator.validate_author_year_citation(
                match_text, start_pos, end_pos, full_text
            )
        
        # Context analysis for figure/table/section references
        if citation_type in ["figure_table_refs", "page_references"]:
            return self.context_validator.validate_structural_reference(
                match_text, start_pos, end_pos, full_text
            )
        
        # Special validation for isolated numeric citations to avoid list items
        if citation_type == "isolated_numeric_citations":
            # Get text before the match to check for list item patterns
            context_start = max(0, start_pos - 100)  # Look back 100 chars
            text_before = full_text[context_start:start_pos]
            
            # Check if this looks like a list item
            if self.context_validator._is_likely_list_item(text_before, match_text):
                return False, "likely_list_item"
        
        # Simple validation for ref_nummer: check if it's in a sentence or isolated
        if citation_type == "ref_nummer":
            # Get context around the match (30 chars before/after)
            context_start = max(0, start_pos - 30)
            context_end = min(len(full_text), end_pos + 30)
            before_text = full_text[context_start:start_pos]
            after_text = full_text[end_pos:context_end]
            
            # Check if ref is surrounded by sentence text (letters/words)
            has_text_before = bool(re.search(r'[a-zA-Z]', before_text[-10:] if len(before_text) >= 10 else before_text))
            has_text_after = bool(re.search(r'[a-zA-Z]', after_text[:10] if len(after_text) >= 10 else after_text))
            
            # If ref is in middle of sentence → keep (probably part of text flow)
            if has_text_before and has_text_after:
                return False, "ref_in_sentence_flow"
            
            # If ref is isolated or at sentence boundaries → remove (probably citation)
            return True, "isolated_ref_citation"
        
        
        # Default: accept if no issues
        return True, "valid_citation"
    
    def filter(self, doc: Document) -> bool:
        """Enhanced Multi-Type Citation Cleaning mit Simple Validation"""
        original_text = doc.text
        self.cleaning_stats["docs_processed"] += 1
        
        # Citations für jeden Typ finden und validieren
        citations_found = {}
        citations_removed = {}
        citations_rejected = {}
        cleaned_text = original_text
        
        for citation_type, citation_regex in self.citation_regexes.items():
            # Finde alle Matches mit Positionen
            matches = list(citation_regex.finditer(cleaned_text))
            citations_found[citation_type] = [match.group() for match in matches]
            
            validated_matches = []
            rejected_matches = []
            
            # Smart Validation für alle Citation-Typen
            if self.enable_smart_validation:
                for match in matches:
                    match_text = match.group()
                    
                    # Context-aware validation for citations that need position analysis
                    if citation_type in ["semicolon_blocks", "figure_table_refs", "page_references", "isolated_numeric_citations", "ref_nummer"]:
                        is_valid, reason = self._validate_citation_with_context(
                            citation_type, match_text, match.start(), match.end(), cleaned_text
                        )
                    else:
                        # Simple validation for other types
                        is_valid, reason = self._validate_citation(citation_type, match_text)
                    
                    if is_valid:
                        validated_matches.append(match)
                    else:
                        rejected_matches.append({
                            "match": match_text,
                            "reason": reason,
                            "position": (match.start(), match.end())
                        })
                        
                        # Store sample für Analyse (mit konfigurierbarem Limit)
                        if len(self.citation_stats[citation_type]["false_positive_samples"]) < self.max_false_positive_samples:
                            # Extract context around the rejection (larger window)
                            context_size = 400  # Increased from 200 to 400
                            start_pos = match.start()
                            end_pos = match.end()
                            context_start = max(0, start_pos - context_size)
                            context_end = min(len(cleaned_text), end_pos + context_size)
                            
                            before_context = cleaned_text[context_start:start_pos]
                            after_context = cleaned_text[end_pos:context_end]
                            
                            # Get proper doc ID - same as top_documents
                            doc_id = str(doc.id) if doc.id else "unknown"
                            
                            self.citation_stats[citation_type]["false_positive_samples"].append({
                                "match": match_text,
                                "doc_id": doc_id,
                                "reason": reason,
                                "confidence": 0.0,  # Simple validator doesn't provide confidence
                                "before_context": before_context[-150:] if len(before_context) > 150 else before_context,  # Show more context
                                "after_context": after_context[:150] if len(after_context) > 150 else after_context,    # Show more context
                                "position": f"{start_pos}-{end_pos}",
                                "full_context_available": len(before_context) + len(after_context)  # Debug info
                            })
            else:
                # Ohne Smart Validation: Akzeptiere alle
                validated_matches = matches
            
            citations_removed[citation_type] = [match.group() for match in validated_matches]
            citations_rejected[citation_type] = rejected_matches
            
            # Text cleanen nur für validierte Matches
            for match in reversed(validated_matches):  # Rückwärts für korrekte Positionen
                start, end = match.span()
                if self.debug_mode:
                    # Debug-Tag statt Entfernung
                    debug_tag = f"[DEBUG:citation:{citation_type}]"
                    cleaned_text = cleaned_text[:start] + debug_tag + cleaned_text[end:]
                else:
                    cleaned_text = cleaned_text[:start] + self.replacement + cleaned_text[end:]
            
            # POST-PROCESSING: Remove figure/table-only lines
            cleaned_text, figure_removal_stats = self._remove_figure_only_lines(cleaned_text, str(doc.id))
            
            # POST-PROCESSING: Remove appendix/reference sections
            cleaned_text, appendix_removal_stats = self._detect_appendix_sections(cleaned_text, str(doc.id))
            
            # POST-PROCESSING: Remove isolated short lines
            cleaned_text, short_line_removal_stats = self.line_cleaner.clean_lines(cleaned_text, str(doc.id))
            
            # Update figure line removal stats
            if figure_removal_stats["lines_removed"] > 0:
                self.cleaning_stats["docs_with_figure_lines_removed"] += 1
                self.cleaning_stats["total_figure_lines_removed"] += figure_removal_stats["lines_removed"]
                self.cleaning_stats["total_figure_line_length_reduction"] += figure_removal_stats["length_reduction"]
                
                # Store samples for W&B (with limit)
                if len(self.cleaning_stats["figure_line_removal_samples"]) < self.max_false_positive_samples:
                    for removed_line in figure_removal_stats["removed_lines"]:
                        if len(self.cleaning_stats["figure_line_removal_samples"]) < self.max_false_positive_samples:
                            sample = {
                                "doc_id": figure_removal_stats["doc_id"],
                                "line_content": removed_line["line_content"],
                                "line_number": removed_line["line_number"],
                                "reason": removed_line["reason"],
                                "length": removed_line["length"]
                            }
                            self.cleaning_stats["figure_line_removal_samples"].append(sample)
                
                # Track top figure line removal documents
                doc_title = str(doc.metadata.get("title", ""))[:50] if doc.metadata else ""
                figure_doc_entry = (
                    figure_removal_stats["lines_removed"], 
                    figure_removal_stats["doc_id"], 
                    doc_title,
                    figure_removal_stats["length_reduction"]
                )
                
                if len(self.cleaning_stats["top_figure_line_removal_docs"]) < self.max_top_citation_docs:
                    heapq.heappush(self.cleaning_stats["top_figure_line_removal_docs"], figure_doc_entry)
                elif figure_removal_stats["lines_removed"] > self.cleaning_stats["top_figure_line_removal_docs"][0][0]:
                    heapq.heapreplace(self.cleaning_stats["top_figure_line_removal_docs"], figure_doc_entry)
                
                # Add metadata to document
                doc.metadata["figure_lines_removed"] = figure_removal_stats["lines_removed"]
                doc.metadata["figure_line_length_reduction"] = figure_removal_stats["length_reduction"]
            else:
                doc.metadata["figure_lines_removed"] = 0
                doc.metadata["figure_line_length_reduction"] = 0
            
            # Update appendix section removal stats
            if appendix_removal_stats["sections_removed"] > 0:
                self.cleaning_stats["docs_with_appendix_sections_removed"] += 1
                self.cleaning_stats["total_appendix_sections_removed"] += appendix_removal_stats["sections_removed"]
                self.cleaning_stats["total_appendix_lines_removed"] += appendix_removal_stats["lines_removed"]
                self.cleaning_stats["total_appendix_length_reduction"] += appendix_removal_stats["length_reduction"]
                
                # Store samples for W&B (with limit)
                if len(self.cleaning_stats["appendix_removal_samples"]) < self.max_false_positive_samples:
                    for removed_section in appendix_removal_stats["removed_sections"]:
                        if len(self.cleaning_stats["appendix_removal_samples"]) < self.max_false_positive_samples:
                            sample = {
                                "doc_id": appendix_removal_stats["doc_id"],
                                "section_type": removed_section["section_type"],
                                "lines_count": removed_section["lines_count"],
                                "start_line": removed_section["start_line"],
                                "confidence": removed_section["confidence"],
                                "sample_lines": removed_section["sample_lines"]
                            }
                            self.cleaning_stats["appendix_removal_samples"].append(sample)
                
                # Track top appendix removal documents
                doc_title = str(doc.metadata.get("title", ""))[:50] if doc.metadata else ""
                appendix_doc_entry = (
                    appendix_removal_stats["sections_removed"], 
                    appendix_removal_stats["doc_id"], 
                    doc_title,
                    appendix_removal_stats["length_reduction"]
                )
                
                if len(self.cleaning_stats["top_appendix_removal_docs"]) < self.max_top_citation_docs:
                    heapq.heappush(self.cleaning_stats["top_appendix_removal_docs"], appendix_doc_entry)
                elif appendix_removal_stats["sections_removed"] > self.cleaning_stats["top_appendix_removal_docs"][0][0]:
                    heapq.heapreplace(self.cleaning_stats["top_appendix_removal_docs"], appendix_doc_entry)
                
                # Add metadata to document
                doc.metadata["appendix_sections_removed"] = appendix_removal_stats["sections_removed"]
                doc.metadata["appendix_lines_removed"] = appendix_removal_stats["lines_removed"]
                doc.metadata["appendix_length_reduction"] = appendix_removal_stats["length_reduction"]
            else:
                doc.metadata["appendix_sections_removed"] = 0
                doc.metadata["appendix_lines_removed"] = 0
                doc.metadata["appendix_length_reduction"] = 0
            
            # Update short line removal stats
            if short_line_removal_stats["total_lines_removed"] > 0:
                self.cleaning_stats["docs_with_short_lines_removed"] += 1
                self.cleaning_stats["total_short_lines_removed"] += short_line_removal_stats["total_lines_removed"]
                self.cleaning_stats["total_short_line_length_reduction"] += short_line_removal_stats["total_length_reduction"]
                
                # Store samples for W&B (with limit)
                if len(self.cleaning_stats["short_line_removal_samples"]) < self.max_false_positive_samples:
                    for removed_line in short_line_removal_stats["removed_lines"]:
                        if len(self.cleaning_stats["short_line_removal_samples"]) < self.max_false_positive_samples:
                            sample = {
                                "doc_id": short_line_removal_stats["doc_id"],
                                "line_content": removed_line["line_content"],
                                "line_number": removed_line["line_number"],
                                "word_count": removed_line.get("word_count", 0),  # Not all cleaners track word count
                                "reason": removed_line["reason"],
                                "length": removed_line["length"]
                            }
                            self.cleaning_stats["short_line_removal_samples"].append(sample)
                
                # Track top short line removal documents
                doc_title = str(doc.metadata.get("title", ""))[:50] if doc.metadata else ""
                short_line_doc_entry = (
                    short_line_removal_stats["total_lines_removed"], 
                    short_line_removal_stats["doc_id"], 
                    doc_title,
                    short_line_removal_stats["total_length_reduction"]
                )
                
                if len(self.cleaning_stats["top_short_line_removal_docs"]) < self.max_top_citation_docs:
                    heapq.heappush(self.cleaning_stats["top_short_line_removal_docs"], short_line_doc_entry)
                elif short_line_removal_stats["total_lines_removed"] > self.cleaning_stats["top_short_line_removal_docs"][0][0]:
                    heapq.heapreplace(self.cleaning_stats["top_short_line_removal_docs"], short_line_doc_entry)
                
                # Add metadata to document
                doc.metadata["short_lines_removed"] = short_line_removal_stats["total_lines_removed"]
                doc.metadata["short_line_length_reduction"] = short_line_removal_stats["total_length_reduction"]
            else:
                doc.metadata["short_lines_removed"] = 0
                doc.metadata["short_line_length_reduction"] = 0
            
            # Stats updaten
            total_found = len(citations_found[citation_type])
            total_removed = len(citations_removed[citation_type])
            total_rejected = len(citations_rejected[citation_type])
            
            self.citation_stats[citation_type]["total_citations_found"] += total_found
            self.citation_stats[citation_type]["total_citations_removed"] += total_removed
            self.citation_stats[citation_type]["total_citations_rejected"] += total_rejected
            
            # Metadaten für diesen Citation-Typ
            doc.metadata[f"{citation_type}_citations_found"] = total_found
            doc.metadata[f"{citation_type}_citations_removed"] = total_removed
            doc.metadata[f"{citation_type}_citations_rejected"] = total_rejected
            doc.metadata[f"had_{citation_type}_citations"] = total_removed > 0
            
            # Analytics für diesen Typ updaten
            if total_removed > 0:
                self.citation_stats[citation_type]["docs_with_citations"] += 1
                self.citation_stats[citation_type]["citation_distribution"][total_removed] += 1
                
                # Top Documents für diesen Typ
                doc_entry = (total_removed, str(doc.id) if doc.id else "unknown", 
                           str(doc.metadata.get("title", ""))[:50], "")
                if len(self.citation_stats[citation_type]["top_citation_docs"]) < self.max_top_citation_docs:
                    heapq.heappush(self.citation_stats[citation_type]["top_citation_docs"], doc_entry)
                elif total_removed > self.citation_stats[citation_type]["top_citation_docs"][0][0]:
                    heapq.heapreplace(self.citation_stats[citation_type]["top_citation_docs"], doc_entry)
                
                # Pipeline Stats für diesen Typ
                self.stat_update(f"{citation_type}_citations_removed", total_removed)
                self.stat_update(f"{citation_type}_docs_with_citations")
            
            if total_rejected > 0:
                self.stat_update(f"{citation_type}_citations_rejected", total_rejected)
        
        # Text updaten
        doc.text = cleaned_text
        
        # Post-cleaning Metriken
        if self.track_changes:
            original_hash = self.hash_func(original_text)
            post_word_count = len(cleaned_text.split()) if cleaned_text.strip() else 0
            
            text_changed = original_hash != self.hash_func(cleaned_text)
            length_reduction = len(original_text) - len(cleaned_text)
            word_reduction = len(original_text.split()) - post_word_count
            
            doc.metadata["citation_text_changed"] = text_changed
            doc.metadata["citation_length_reduction"] = length_reduction
            doc.metadata["citation_word_reduction"] = word_reduction
            doc.metadata["smart_validation_enabled"] = self.enable_smart_validation
            
            if text_changed:
                self.cleaning_stats["total_length_reduction"] += length_reduction
                self.cleaning_stats["total_word_reduction"] += word_reduction
        
        # Gesamt-Stats updaten
        total_citations_removed = sum(len(citations) for citations in citations_removed.values())
        total_citations_rejected = sum(len(rejected) for rejected in citations_rejected.values())
        
        if total_citations_removed > 0:
            self.cleaning_stats["docs_with_any_citations"] += 1
            self.cleaning_stats["total_citations_all_types"] += total_citations_removed
        
        if total_citations_rejected > 0:
            self.cleaning_stats["total_citations_rejected"] += total_citations_rejected
        
        # Track combined top documents (citations + figure lines + appendix + short lines)
        total_combined_reduction = 0
        if self.track_changes:
            total_combined_reduction = (length_reduction + 
                                      figure_removal_stats.get("length_reduction", 0) +
                                      appendix_removal_stats.get("length_reduction", 0) +
                                      short_line_removal_stats.get("total_length_reduction", 0))
        
        if (total_combined_reduction > 0 or total_citations_removed > 0 or 
            figure_removal_stats.get("lines_removed", 0) > 0 or
            appendix_removal_stats.get("sections_removed", 0) > 0 or
            short_line_removal_stats.get("total_lines_removed", 0) > 0):
            doc_title = str(doc.metadata.get("title", ""))[:50] if doc.metadata else ""
            combined_info = (f"C:{total_citations_removed},"
                           f"F:{figure_removal_stats.get('lines_removed', 0)},"
                           f"A:{appendix_removal_stats.get('sections_removed', 0)},"
                           f"S:{short_line_removal_stats.get('total_lines_removed', 0)}")
            
            combined_doc_entry = (
                total_combined_reduction,  # Sort by total text reduction
                str(doc.id) if doc.id else "unknown",
                doc_title,
                combined_info
            )
            
            if len(self.cleaning_stats["top_combined_reduction_docs"]) < self.max_top_citation_docs:
                heapq.heappush(self.cleaning_stats["top_combined_reduction_docs"], combined_doc_entry)
            elif total_combined_reduction > self.cleaning_stats["top_combined_reduction_docs"][0][0]:
                heapq.heapreplace(self.cleaning_stats["top_combined_reduction_docs"], combined_doc_entry)
        
        # Pipeline Stats
        self.stat_update("docs_processed")
        if total_citations_removed > 0:
            self.stat_update("docs_with_any_citations")
        if total_citations_rejected > 0:
            self.stat_update("citations_rejected_by_validation", total_citations_rejected)
        
        # W&B Logging - nur für rank 0 
        if hasattr(self, 'current_rank') and self.current_rank == 0 and self.logger.wandb_initialized:
            self.logger.log_document_metrics(citations_found, citations_removed, citations_rejected, doc.metadata, figure_removal_stats, self.enable_smart_validation)
            
            # Aggregierte Stats alle 50 Dokumente
            if self.cleaning_stats["docs_processed"] % 50 == 0:
                self.logger.log_aggregated_stats(self.citation_stats, self.cleaning_stats, self.enable_smart_validation)
        
        return bool(cleaned_text.strip())
    
    def _remove_figure_only_lines(self, text: str, doc_id: str = None) -> tuple[str, dict]:
        """Remove lines containing figure/table references and captions aggressively with detailed logging"""
        lines = text.splitlines()
        cleaned_lines = []
        removed_lines = []
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            # Skip empty lines
            if not stripped_line:
                cleaned_lines.append(line)
                continue
            
            # AGGRESSIVE Figure/Table line detection
            should_remove = False
            removal_reason = ""
            
            # Pattern 1: Lines starting with numbers + Figure/Table (e.g., "41 Figure 4.1: Goals...")
            if re.match(r'^\d+\s+(?:fig|figure|tab|table|tbl)\.?\s*\d+', stripped_line, re.IGNORECASE):
                should_remove = True
                removal_reason = "numbered_figure_caption"
            
            # Pattern 2: Lines starting with Figure/Table + decimal numbers (e.g., "Figure 5 . 2 :")
            elif re.match(r'^(?:fig|figure|tab|table|tbl)\.?\s*\d+(?:\s*\.\s*\d+)*\s*[:.]?\s*', stripped_line, re.IGNORECASE):
                should_remove = True
                removal_reason = "figure_header"
            
            # Pattern 3: Just numbers and punctuation (leftover captions like "5", "41")
            elif re.match(r'^\s*\d+\s*[:.]?\s*$', stripped_line):
                should_remove = True
                removal_reason = "orphaned_number"
            
            # Pattern 4: Figure references with descriptive text but short enough to be captions
            # (Less than 80 chars and contains Figure/Table)
            elif (len(stripped_line) < 80 and 
                  re.search(r'(?:fig|figure|tab|table|tbl)\.?\s*\d+', stripped_line, re.IGNORECASE)):
                should_remove = True
                removal_reason = "short_figure_caption"
            
            # Pattern 5: Lines that are mostly numbers and punctuation with minimal text
            elif re.match(r'^\s*\d+(?:\s*\.\s*\d+)*\s*[:.]?\s*\w{0,20}\s*[:.]?\s*$', stripped_line):
                should_remove = True
                removal_reason = "numeric_header"
            
            # Pattern 6: Table column headers that got separated
            elif (re.match(r'^(?:variable|coefficient|std\.?\s*err|p-?value|confidence|interval|estimate|statistic)', stripped_line, re.IGNORECASE) and
                  len(stripped_line) < 60):
                should_remove = True
                removal_reason = "table_column_header"
            
            # Pattern 7: Statistical notation lines (stars, crosses, etc.)
            elif re.match(r'^\s*[\*\+†‡§¶#]+\s*[^a-zA-Z]*$', stripped_line):
                should_remove = True  
                removal_reason = "statistical_notation"
            
            # Pattern 8: Standalone section numbers without content
            elif re.match(r'^\s*(?:\d+\.\d+|\d+\s*$|[IVX]+\.\s*$)', stripped_line):
                should_remove = True
                removal_reason = "section_number"
            
            # Pattern 9: Page headers/footers (common OCR artifacts)
            elif (re.match(r'^\s*(?:page\s+\d+|chapter\s+\d+|\d+\s*$)', stripped_line, re.IGNORECASE) and
                  len(stripped_line) < 20):
                should_remove = True
                removal_reason = "page_header"
            
            # Pattern 10: Mathematical notation lines (often OCR artifacts)
            elif re.match(r'^\s*[=<>≤≥≠±∓×÷∑∏∫∂∇∆]+\s*$', stripped_line):
                should_remove = True
                removal_reason = "math_notation"
            
            # Pattern 11: Units and measurements lines (often separated from content)
            elif (re.match(r'^\s*(?:mm|cm|kg|mg|ml|μl|°c|°f|hz|khz|mhz|ghz|v|mv|ma|μa)\s*$', stripped_line, re.IGNORECASE) and
                  len(stripped_line) < 15):
                should_remove = True
                removal_reason = "units_line"
            
            # Pattern 12: Standalone abbreviations (common in scientific texts)
            elif (re.match(r'^\s*(?:fig|tab|eq|ref|sec|ch|app|supp|vs|cf|ibid|loc|cit|al|inc|ltd|corp)\s*\.?\s*$', stripped_line, re.IGNORECASE) and
                  len(stripped_line) < 10):
                should_remove = True
                removal_reason = "standalone_abbreviation"
            
            # Pattern 13: Zeichen-zu-Wort-Ratio Check für kurze Lines (unter 100 Zeichen)
            # BUT: Skip mathematical content
            elif len(stripped_line) < 100 and not self._is_mathematical_content(stripped_line):  
                words = stripped_line.split()
                if len(words) > 0:
                    # Zähle Sonderzeichen (Nicht-Alphanumerisch, außer Leerzeichen)
                    special_chars = len(re.findall(r'[^\w\s]', stripped_line))
                    total_chars = len(stripped_line.replace(' ', ''))  # Ohne Leerzeichen
                    
                    if total_chars > 0:
                        special_char_ratio = special_chars / total_chars
                        # Wenn >50% Sonderzeichen in kurzer Line -> wahrscheinlich Artifact
                        if special_char_ratio > 0.5 and len(words) <= 3:
                            should_remove = True
                            removal_reason = f"high_special_char_ratio_{special_char_ratio:.2f}"
            
            if should_remove:
                # Store removed line for analytics
                removed_lines.append({
                    "line_content": stripped_line[:100],  # Limit length for W&B
                    "line_number": i + 1,
                    "length": len(stripped_line),
                    "reason": removal_reason
                })
                
                if self.debug_mode:
                    # Debug-Tag statt Entfernung
                    debug_tag = f"[DEBUG:figure_line:{removal_reason}]"
                    cleaned_lines.append(debug_tag)
                else:
                    # Line wird nicht hinzugefügt (entfernt)
                    continue
            else:
                # Keep the line
                cleaned_lines.append(line)
        
        # Statistics
        stats = {
            "lines_removed": len(removed_lines),
            "length_reduction": sum(item["length"] for item in removed_lines) if not self.debug_mode else 0,
            "removed_lines": removed_lines[:10],  # Store max 10 samples per document
            "doc_id": doc_id or "unknown"
        }
        
        return '\n'.join(cleaned_lines), stats
    
    def _is_mathematical_content(self, line: str) -> bool:
        """Check if line contains mathematical formulas that should be preserved"""
        # Mathematical patterns that indicate scientific content
        math_patterns = [
            r'[αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]',  # Greek letters
            r'[∂∇∑∏∫≡≠≤≥±∓×÷∈∉⊂⊃∪∩]',  # Math operators
            r'⟨[^⟩]*⟩',                    # Bra-ket notation
            r'\|[^|]*⟩',                   # Ket notation  
            r'⟨[^|]*\|',                   # Bra notation
            r'[a-zA-Z]+_[a-zA-Z0-9]+',     # Subscripts like γA, γB
            r'[a-zA-Z]+\^[a-zA-Z0-9]+',    # Superscripts
            r'\\[a-z]+\{[^}]*\}',          # LaTeX commands
            r'[a-zA-Z]+=.*[a-zA-Z]',       # Equations like tl=t0[...]
            r'ℋ|ℰ|ℱ|ℊ|ℋ|ℌ|ℍ|ℎ|ℏ|ℐ',      # Script letters (Hamiltonian etc.)
            r'Π†|Π|†|‡|°|µ|σ|ρ|τ|φ|χ|ψ|ω', # Special math symbols
        ]
        
        return any(re.search(pattern, line) for pattern in math_patterns)
    
    def _detect_appendix_sections(self, text: str, doc_id: str = None) -> tuple[str, dict]:
        """
        Detect and remove appendix/reference sections using character ratio analysis.
        
        Targets scientific tables, gene lists, acknowledgments, and reference sections
        that are harmful for T5 pretraining.
        """
        lines = text.splitlines()
        cleaned_lines = []
        removed_sections = []
        current_section_lines = []
        in_appendix_section = False
        section_start_line = 0
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            # Skip empty lines but don't reset section detection
            if not stripped_line:
                if in_appendix_section:
                    current_section_lines.append(line)
                else:
                    cleaned_lines.append(line)
                continue
            
            # Detect start of appendix sections
            section_indicators = [
                # Table indicators
                r'^table\s+\d+', r'^tab\s+\d+', r'^figure\s+\d+', r'^fig\s+\d+',
                
                # Scientific sections
                r'^acknowledgments?$', r'^acknowledgements?$', r'^references?$', 
                r'^bibliography$', r'^appendix', r'^supplementary',
                
                # ENTFERNT: Method sections - zu viele False Positives bei normalem Content
                # r'^materials?\s+and\s+methods?$', r'^methods?$', r'^procedures?$',
                
                # Gene/protein lists
                r'^gene\s+symbol', r'^protein\s+name', r'^antibody\s+supplier',
                
                # Table content patterns
                r'^[A-Z][A-Z0-9]+\s+[A-Z][A-Z0-9]+\s+[A-Z][A-Z0-9]+',  # Column headers
            ]
            
            is_section_start = any(re.search(pattern, stripped_line, re.IGNORECASE) 
                                 for pattern in section_indicators)
            
            if is_section_start and not in_appendix_section:
                in_appendix_section = True
                section_start_line = i + 1
                current_section_lines = [line]
                continue
            
            if in_appendix_section:
                current_section_lines.append(line)
                
                # Check if we should end the section (normal paragraph text returns)
                if len(current_section_lines) > 5:  # Only after minimum section length
                    # Calculate ratios for this line
                    line_stats = self._calculate_line_stats(stripped_line)
                    
                    # End section if we hit normal prose text
                    if (line_stats['word_count'] > 15 and 
                        line_stats['special_char_ratio'] < 0.15 and
                        line_stats['uppercase_ratio'] < 0.3 and
                        not re.search(r'^[A-Z]{2,}', stripped_line) and  # Not gene names
                        not re.search(r'\d+\.\d+', stripped_line) and    # Not version numbers
                        not re.search(r'[:\t]', stripped_line)):         # Not tabular data
                        
                        # This looks like normal text, end the section
                        section_stats = self._analyze_section_stats(current_section_lines[:-1])  # Exclude current line
                        
                        if self._is_likely_appendix_section(section_stats):
                            removed_sections.append({
                                "start_line": section_start_line,
                                "end_line": i,
                                "lines_count": len(current_section_lines) - 1,
                                "total_length": sum(len(l) for l in current_section_lines[:-1]),
                                "section_type": section_stats.get('section_type', 'unknown'),
                                "confidence": section_stats.get('confidence', 0.0),
                                "sample_lines": current_section_lines[:3]  # First 3 lines as sample
                            })
                            
                            if self.debug_mode:
                                # Debug-Tag statt Entfernung
                                debug_tag = f"[DEBUG:appendix:{section_stats.get('section_type', 'unknown')}]"
                                cleaned_lines.append(debug_tag)
                            # Else: Section wird nicht hinzugefügt (entfernt)
                        else:
                            # Not an appendix, add back to cleaned text
                            cleaned_lines.extend(current_section_lines[:-1])
                        
                        # Reset and add current line as normal
                        in_appendix_section = False
                        current_section_lines = []
                        cleaned_lines.append(line)
                        continue
                
                # Continue collecting section lines
                continue
            
            # Normal line - add to cleaned text
            cleaned_lines.append(line)
        
        # Handle any remaining section at end of document
        if in_appendix_section and current_section_lines:
            section_stats = self._analyze_section_stats(current_section_lines)
            if self._is_likely_appendix_section(section_stats):
                removed_sections.append({
                    "start_line": section_start_line,
                    "end_line": len(lines),
                    "lines_count": len(current_section_lines),
                    "total_length": sum(len(l) for l in current_section_lines),
                    "section_type": section_stats.get('section_type', 'unknown'),
                    "confidence": section_stats.get('confidence', 0.0),
                    "sample_lines": current_section_lines[:3]
                })
                
                if self.debug_mode:
                    # Debug-Tag statt Entfernung
                    debug_tag = f"[DEBUG:appendix_end:{section_stats.get('section_type', 'unknown')}]"
                    cleaned_lines.append(debug_tag)
                # Else: Section wird nicht hinzugefügt (entfernt)
            else:
                cleaned_lines.extend(current_section_lines)
        
        # Statistics
        total_removed_lines = sum(section['lines_count'] for section in removed_sections)
        total_removed_length = sum(section['total_length'] for section in removed_sections)
        
        stats = {
            "sections_removed": len(removed_sections),
            "lines_removed": total_removed_lines,
            "length_reduction": total_removed_length if not self.debug_mode else 0,
            "removed_sections": removed_sections[:5],  # Store max 5 samples per document
            "doc_id": doc_id or "unknown"
        }
        
        return '\n'.join(cleaned_lines), stats
    
    def _calculate_line_stats(self, line: str) -> dict:
        """Calculate various statistics for a single line"""
        if not line:
            return {'word_count': 0, 'special_char_ratio': 0, 'uppercase_ratio': 0}
        
        words = line.split()
        word_count = len(words)
        
        # Special characters (non-alphanumeric except spaces)
        special_chars = len(re.findall(r'[^\w\s]', line))
        total_chars = len(line.replace(' ', ''))
        special_char_ratio = special_chars / total_chars if total_chars > 0 else 0
        
        # Uppercase letters
        uppercase_chars = sum(1 for c in line if c.isupper())
        letter_chars = sum(1 for c in line if c.isalpha())
        uppercase_ratio = uppercase_chars / letter_chars if letter_chars > 0 else 0
        
        return {
            'word_count': word_count,
            'special_char_ratio': special_char_ratio,
            'uppercase_ratio': uppercase_ratio,
            'total_length': len(line)
        }
    
    def _analyze_section_stats(self, section_lines: List[str]) -> dict:
        """Analyze statistics for an entire section to determine if it's an appendix"""
        if not section_lines:
            return {'confidence': 0.0, 'section_type': 'empty'}
        
        # Calculate aggregate statistics
        total_lines = len(section_lines)
        non_empty_lines = [line.strip() for line in section_lines if line.strip()]
        
        if not non_empty_lines:
            return {'confidence': 0.0, 'section_type': 'empty'}
        
        # Line-by-line analysis
        line_stats = [self._calculate_line_stats(line) for line in non_empty_lines]
        
        # Aggregate metrics
        avg_special_char_ratio = sum(s['special_char_ratio'] for s in line_stats) / len(line_stats)
        avg_uppercase_ratio = sum(s['uppercase_ratio'] for s in line_stats) / len(line_stats)
        avg_word_count = sum(s['word_count'] for s in line_stats) / len(line_stats)
        
        # Count specific patterns
        tabular_lines = sum(1 for line in non_empty_lines if '\t' in line or re.search(r'\s{3,}', line))
        gene_lines = sum(1 for line in non_empty_lines if re.search(r'^[A-Z0-9]{2,}\s+[A-Z0-9]{2,}', line))
        short_lines = sum(1 for s in line_stats if s['word_count'] < 5)
        numeric_lines = sum(1 for line in non_empty_lines if re.search(r'\d+\.\d+|\d+%', line))
        
        # Calculate confidence score
        confidence = 0.0
        section_type = "unknown"
        
        # CONSERVATIVE: Nur bei SEHR hohen Ratios als appendix klassifizieren
        
        # High special character ratio (tables, technical data) - erhöht von 0.2 auf 0.35
        if avg_special_char_ratio > 0.35:
            confidence += 0.3
            section_type = "technical_table"
        
        # High uppercase ratio (gene names, abbreviations) - bleibt bei 0.4
        if avg_uppercase_ratio > 0.4:
            confidence += 0.25
            section_type = "gene_list"
        
        # Many short lines (tabular data) - erhöht von 0.5 auf 0.7
        if short_lines / len(non_empty_lines) > 0.7:
            confidence += 0.2
            section_type = "tabular_data"
        
        # Tabular formatting - bleibt bei 0.3
        if tabular_lines / len(non_empty_lines) > 0.3:
            confidence += 0.3
            section_type = "formatted_table"
        
        # Gene name patterns - bleibt bei 0.3
        if gene_lines / len(non_empty_lines) > 0.3:
            confidence += 0.35
            section_type = "gene_table"
        
        # Many numeric values - erhöht von 0.5 auf 0.8 (Citations haben oft Jahre)
        if numeric_lines / len(non_empty_lines) > 0.8:
            confidence += 0.25
            section_type = "numeric_data"
        
        # Low average word count (not prose) - reduziert von 8 auf 5 (Citations haben längere Wörter)
        if avg_word_count < 5:
            confidence += 0.15
        
        return {
            'confidence': min(confidence, 1.0),
            'section_type': section_type,
            'total_lines': total_lines,
            'avg_special_char_ratio': avg_special_char_ratio,
            'avg_uppercase_ratio': avg_uppercase_ratio,
            'avg_word_count': avg_word_count,
            'tabular_lines_ratio': tabular_lines / len(non_empty_lines),
            'gene_lines_ratio': gene_lines / len(non_empty_lines)
        }
    
    def _is_likely_appendix_section(self, section_stats: dict) -> bool:
        """Determine if a section should be removed based on statistics"""
        confidence = section_stats.get('confidence', 0.0)
        
        # VERY Conservative threshold - erhöht von 0.6 auf 0.8 um wissenschaftlichen Text zu bewahren
        return confidence > 0.8
    

    
    def run(self, data, rank: int = 0, world_size: int = 1):
        """Override run method - alle sammeln, rank 0 aggregiert für W&B"""
        # Store rank für W&B check
        self.current_rank = rank
        self.world_size = world_size
        
        # W&B nur für main process (rank 0) aktivieren
        if rank == 0 and self.wandb_enabled:
            self.logger.log_to_wandb = True
            self.logger._init_wandb()
        
        try:
            yield from super().run(data, rank, world_size)
        finally:
            # Alle Worker: Lokale Stats speichern für Aggregation
            self.worker_stats.save_worker_stats(rank, self.citation_stats, self.cleaning_stats)
            
            # Alle Worker: Lokale Stats loggen
            log.info(f"✅ Citation cleaning rank {rank} completed: {self.cleaning_stats['docs_processed']} docs, "
                    f"{self.cleaning_stats['docs_with_any_citations']} with citations, "
                    f"{self.cleaning_stats['total_citations_all_types']} removed")
            
            # W&B Logging mit aggregierten Daten (nur rank 0)
            if rank == 0 and self.logger.wandb_initialized:
                # Warte auf andere Worker
                self.worker_stats.wait_for_workers(world_size)
                
                # Aggregiere Stats von allen Workern
                aggregated_citation_stats, aggregated_cleaning_stats = self.worker_stats.aggregate_all_worker_stats(world_size)
                
                if aggregated_cleaning_stats["docs_processed"] > 0:
                    self.logger.log_final_summary(aggregated_citation_stats, aggregated_cleaning_stats, self.enable_smart_validation)
                    
                    summary_msg = (f"📋 W&B Summary (ALL {world_size} workers): {aggregated_cleaning_stats['docs_with_any_citations']}/"
                                  f"{aggregated_cleaning_stats['docs_processed']} docs with citations, "
                                  f"{aggregated_cleaning_stats['total_citations_all_types']} removed")
                    
                    if aggregated_cleaning_stats["total_citations_rejected"] > 0:
                        summary_msg += f", {aggregated_cleaning_stats['total_citations_rejected']} rejected by validation"
                    
                    if aggregated_cleaning_stats["total_figure_lines_removed"] > 0:
                        summary_msg += f", {aggregated_cleaning_stats['total_figure_lines_removed']} figure lines removed"
                    
                    log.info(summary_msg)
                    log.info("✅ W&B shows complete aggregated dataset stats from all workers!") 