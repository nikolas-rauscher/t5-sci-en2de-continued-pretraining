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
# Removed unused imports: defaultdict, Counter, heapq (now handled by CitationStatsManager)

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.hashing import create_hash_func, HashConfig

# Import simple validator
from src.dataprep.semicolon_validator import SemicolonCitationValidator
from src.dataprep.context_validator import ContextValidator
from src.dataprep.citation_cleaner_logger import CitationCleanerLogger
from src.dataprep.citation_cleaner_worker_stats import CitationCleanerWorkerStats
from src.dataprep.citation_stats_manager import CitationStatsManager
from src.dataprep.text_cleaners import ComprehensiveLineCleaner
from src.dataprep.figure_table_cleaner import FigureTableCleaner
from src.dataprep.appendix_section_cleaner import AppendixSectionCleaner

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
        
        # Citation Statistics Manager
        self.stats_manager = CitationStatsManager(
            citation_patterns=self.citation_patterns,
            max_false_positive_samples=max_false_positive_samples,
            max_top_citation_docs=max_top_citation_docs,
            enable_smart_validation=self.enable_smart_validation
        )
        
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
        
        # Figure/Table Line Cleaner
        self.figure_table_cleaner = FigureTableCleaner(debug_mode=self.debug_mode)
        
        # Appendix Section Cleaner
        self.appendix_section_cleaner = AppendixSectionCleaner(debug_mode=self.debug_mode)
        
        # Logging Settings (needed for other components)
        self.max_false_positive_samples = max_false_positive_samples
        self.max_top_citation_docs = max_top_citation_docs
    
    # Citation types that need context-aware validation
    CONTEXT_AWARE_TYPES = {"semicolon_blocks", "figure_table_refs", "page_references", 
                          "isolated_numeric_citations", "ref_nummer"}

    # Validation constants
    CONTEXT_SIZE_SMALL = 15
    CONTEXT_SIZE_MEDIUM = 30  
    CONTEXT_SIZE_LARGE = 100

    def _process_citations(self, text: str) -> Tuple[Dict, Dict, Dict, str]:
        """Process all citation types and return results"""
        citations_found = {}
        citations_removed = {}
        citations_rejected = {}
        cleaned_text = text
        
        for citation_type, citation_regex in self.citation_regexes.items():
            matches = list(citation_regex.finditer(cleaned_text))
            citations_found[citation_type] = [match.group() for match in matches]
            
            validated_matches, rejected_matches = self._validate_matches(
                matches, citation_type, cleaned_text
            )
            
            citations_removed[citation_type] = [match.group() for match in validated_matches]
            citations_rejected[citation_type] = rejected_matches
            
            cleaned_text = self._apply_cleaning(cleaned_text, validated_matches, citation_type)
            
        return citations_found, citations_removed, citations_rejected, cleaned_text

    def _validate_matches(self, matches: List, citation_type: str, text: str) -> Tuple[List, List]:
        """Validate citation matches and return validated/rejected lists"""
        if not self.enable_smart_validation:
            return matches, []
            
        validated_matches = []
        rejected_matches = []
        
        for match in matches:
            match_text = match.group()
            
            if citation_type in self.CONTEXT_AWARE_TYPES:
                is_valid, reason = self._validate_citation_with_context(
                    citation_type, match_text, match.start(), match.end(), text
                )
            else:
                is_valid, reason = self._validate_citation(citation_type, match_text)
            
            if is_valid:
                validated_matches.append(match)
            else:
                rejected_matches.append({
                    "match": match_text,
                    "reason": reason,
                    "position": (match.start(), match.end())
                })
                
        return validated_matches, rejected_matches

    def _apply_cleaning(self, text: str, matches: List, citation_type: str) -> str:
        """Apply text cleaning for validated matches"""
        for match in reversed(matches):
            start, end = match.span()
            if self.debug_mode:
                debug_tag = f"[DEBUG:citation:{citation_type}]"
                text = text[:start] + debug_tag + text[end:]
            else:
                text = text[:start] + self.replacement + text[end:]
        return text

    def _update_document_metadata(self, doc: Document, figure_stats: Dict, 
                                appendix_stats: Dict, short_line_stats: Dict,
                                citations_found: Dict, citations_removed: Dict, 
                                citations_rejected: Dict) -> None:
        """Update document metadata with all cleaning results"""
        # Figure lines metadata
        doc.metadata["figure_lines_removed"] = figure_stats.get("lines_removed", 0)
        doc.metadata["figure_line_length_reduction"] = figure_stats.get("length_reduction", 0)
        
        # Appendix metadata  
        doc.metadata["appendix_sections_removed"] = appendix_stats.get("sections_removed", 0)
        doc.metadata["appendix_lines_removed"] = appendix_stats.get("lines_removed", 0)
        doc.metadata["appendix_length_reduction"] = appendix_stats.get("length_reduction", 0)
        
        # Short lines metadata
        doc.metadata["short_lines_removed"] = short_line_stats.get("total_lines_removed", 0)
        doc.metadata["short_line_length_reduction"] = short_line_stats.get("total_length_reduction", 0)
        
        # Citation metadata
        for citation_type in self.citation_patterns.keys():
            total_found = len(citations_found[citation_type])
            total_removed = len(citations_removed[citation_type])
            total_rejected = len(citations_rejected[citation_type])
            
            doc.metadata[f"{citation_type}_citations_found"] = total_found
            doc.metadata[f"{citation_type}_citations_removed"] = total_removed
            doc.metadata[f"{citation_type}_citations_rejected"] = total_rejected
            doc.metadata[f"had_{citation_type}_citations"] = total_removed > 0

    def filter(self, doc: Document) -> bool:
        """Enhanced Multi-Type Citation Cleaning with modular validation"""
        original_text = doc.text
        
        # Process all citations
        citations_found, citations_removed, citations_rejected, cleaned_text = self._process_citations(original_text)
        
        # Post-processing cleaners
        cleaned_text, figure_removal_stats = self.figure_table_cleaner.clean(cleaned_text, str(doc.id))
        cleaned_text, appendix_removal_stats = self.appendix_section_cleaner.clean(cleaned_text, str(doc.id))
        cleaned_text, short_line_removal_stats = self.line_cleaner.clean_lines(cleaned_text, str(doc.id))
        
        # Track all stats
        self.stats_manager.track_figure_line_removal(doc, figure_removal_stats)
        self.stats_manager.track_appendix_section_removal(doc, appendix_removal_stats)
        self.stats_manager.track_short_line_removal(doc, short_line_removal_stats)
        
        # Update document metadata
        self._update_document_metadata(doc, figure_removal_stats, appendix_removal_stats, 
                                     short_line_removal_stats, citations_found, 
                                     citations_removed, citations_rejected)
        
        # Track citation results
        for citation_type in self.citation_patterns.keys():
            self.stats_manager.track_citation_results(
                citation_type, citations_found[citation_type],
                citations_removed[citation_type], citations_rejected[citation_type], doc
            )
            
            # Pipeline stats
            total_removed = len(citations_removed[citation_type])
            total_rejected = len(citations_rejected[citation_type])
            
            if total_removed > 0:
                self.stat_update(f"{citation_type}_citations_removed", total_removed)
                self.stat_update(f"{citation_type}_docs_with_citations")
            if total_rejected > 0:
                self.stat_update(f"{citation_type}_citations_rejected", total_rejected)
        
        # Update document text
        doc.text = cleaned_text
        
        # Post-cleaning metrics
        if self.track_changes:
            self._update_text_change_metrics(doc, original_text, cleaned_text, 
                                           figure_removal_stats, appendix_removal_stats, 
                                           short_line_removal_stats)
        
        # Overall tracking
        total_citations_removed = sum(len(citations) for citations in citations_removed.values())
        total_citations_rejected = sum(len(rejected) for rejected in citations_rejected.values())
        
        self.stats_manager.track_document_processing(doc, total_citations_removed, total_citations_rejected,
                                                   doc.metadata.get("citation_length_reduction", 0),
                                                   doc.metadata.get("citation_word_reduction", 0))
        
        # W&B logging
        if hasattr(self, 'current_rank') and self.current_rank == 0 and self.logger.wandb_initialized:
            citation_stats, cleaning_stats = self.stats_manager.get_stats()
            self.logger.log_document_metrics(citations_found, citations_removed, citations_rejected, 
                                           doc.metadata, figure_removal_stats, self.enable_smart_validation)
            
            if cleaning_stats["docs_processed"] % 50 == 0:
                self.logger.log_aggregated_stats(citation_stats, cleaning_stats, self.enable_smart_validation)
        
        # Pipeline stats
        self.stat_update("docs_processed")
        if total_citations_removed > 0:
            self.stat_update("docs_with_any_citations")
        if total_citations_rejected > 0:
            self.stat_update("citations_rejected_by_validation", total_citations_rejected)
        
        return bool(cleaned_text.strip())

    def _update_text_change_metrics(self, doc: Document, original_text: str, cleaned_text: str,
                                   figure_stats: Dict, appendix_stats: Dict, short_line_stats: Dict) -> None:
        """Calculate and update text change metrics"""
        original_hash = self.hash_func(original_text)
        post_word_count = len(cleaned_text.split()) if cleaned_text.strip() else 0
        
        text_changed = original_hash != self.hash_func(cleaned_text)
        length_reduction = len(original_text) - len(cleaned_text)
        word_reduction = len(original_text.split()) - post_word_count
        
        doc.metadata["citation_text_changed"] = text_changed
        doc.metadata["citation_length_reduction"] = length_reduction
        doc.metadata["citation_word_reduction"] = word_reduction
        doc.metadata["smart_validation_enabled"] = self.enable_smart_validation
        
        # Track combined reduction
        total_combined_reduction = (length_reduction + 
                                  figure_stats.get("length_reduction", 0) +
                                  appendix_stats.get("length_reduction", 0) +
                                  short_line_stats.get("total_length_reduction", 0))
        
        self.stats_manager.track_combined_reduction(doc, 
                                                  sum(len(citations) for citations in doc.metadata.get("citations_removed", {}).values() if isinstance(citations, list)),
                                                  figure_stats, appendix_stats, short_line_stats, total_combined_reduction)

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
    
    def _validate_semicolon_citations(self, match_text: str, start_pos: int, end_pos: int, full_text: str) -> Tuple[bool, str]:
        """Validate semicolon block citations"""
        is_valid, validation_info = self.semicolon_validator.validate(match_text, start_pos, end_pos, full_text)
        reason = validation_info.get("validation_steps", ["unknown"])[-1] if isinstance(validation_info, dict) else str(validation_info)
        return is_valid, reason

    def _validate_author_year_citations(self, citation_type: str, match_text: str, start_pos: int, end_pos: int, full_text: str) -> Tuple[bool, str]:
        """Validate author-year citations with sentence flow analysis"""
        return self.context_validator.validate_author_year_citation(match_text, start_pos, end_pos, full_text)

    def _validate_page_references(self, match_text: str, start_pos: int, end_pos: int, full_text: str) -> Tuple[bool, str]:
        """Validate page references with context analysis"""
        simple_p_pattern = re.search(r"p\.?\s*\d+", match_text.strip(), re.IGNORECASE)
        
        if simple_p_pattern:
            return self._validate_simple_page_ref(match_text, start_pos, end_pos, full_text, simple_p_pattern)
        else:
            # Complex page patterns (pages 12-15, pp. 1,3,5)
            return self.context_validator.validate_structural_reference(match_text, start_pos, end_pos, full_text)

    def _validate_simple_page_ref(self, match_text: str, start_pos: int, end_pos: int, full_text: str, p_pattern) -> Tuple[bool, str]:
        """Validate simple 'p53' style page references"""
        actual_p_match = p_pattern.group()
        match_offset_in_full = match_text.find(actual_p_match)
        actual_start_pos = start_pos + match_offset_in_full
        actual_end_pos = actual_start_pos + len(actual_p_match)
        
        # Check sentence flow embedding using ContextValidator
        if self.context_validator.is_embedded_in_sentence_flow(actual_start_pos, actual_end_pos, full_text, self.CONTEXT_SIZE_SMALL):
            return False, "page_ref_embedded_in_flow"
        
        # Check if standalone in line using ContextValidator
        if self.context_validator.is_alone_in_line(start_pos, end_pos, match_text, full_text):
            return True, "page_ref_alone_in_line"
        else:
            return False, "page_ref_not_alone_in_line"

    def _validate_figure_table_refs(self, match_text: str, start_pos: int, end_pos: int, full_text: str) -> Tuple[bool, str]:
        """Validate figure/table references"""
        return self.context_validator.validate_structural_reference(match_text, start_pos, end_pos, full_text)

    def _validate_numeric_citations(self, match_text: str, start_pos: int, full_text: str) -> Tuple[bool, str]:
        """Validate isolated numeric citations to avoid list items"""
        context_start = max(0, start_pos - self.CONTEXT_SIZE_LARGE)
        text_before = full_text[context_start:start_pos]
        
        if self.context_validator._is_likely_list_item(text_before, match_text):
            return False, "likely_list_item"
        return True, "valid_numeric_citation"

    def _validate_ref_nummer(self, match_text: str, start_pos: int, end_pos: int, full_text: str) -> Tuple[bool, str]:
        """Validate ref number citations"""
        context_start = max(0, start_pos - self.CONTEXT_SIZE_MEDIUM)
        context_end = min(len(full_text), end_pos + self.CONTEXT_SIZE_MEDIUM)
        before_text = full_text[context_start:start_pos]
        after_text = full_text[end_pos:context_end]
        
        # Check if surrounded by text
        has_text_before = bool(re.search(r'[a-zA-Z]', before_text[-10:] if len(before_text) >= 10 else before_text))
        has_text_after = bool(re.search(r'[a-zA-Z]', after_text[:10] if len(after_text) >= 10 else after_text))
        
        if has_text_before and has_text_after:
            return False, "ref_in_sentence_flow"
        return True, "isolated_ref_citation"

    def _validate_citation_with_context(self, citation_type: str, match_text: str, 
                                      start_pos: int, end_pos: int, full_text: str) -> Tuple[bool, str]:
        """Enhanced validation with context analysis - now modular"""
        if not self.enable_smart_validation:
            return True, "validation_disabled"
        
        # Dispatch to specific validators
        validators = {
            "semicolon_blocks": lambda: self._validate_semicolon_citations(match_text, start_pos, end_pos, full_text),
            "page_references": lambda: self._validate_page_references(match_text, start_pos, end_pos, full_text),
            "figure_table_refs": lambda: self._validate_figure_table_refs(match_text, start_pos, end_pos, full_text),
            "isolated_numeric_citations": lambda: self._validate_numeric_citations(match_text, start_pos, full_text),
            "ref_nummer": lambda: self._validate_ref_nummer(match_text, start_pos, end_pos, full_text)
        }
        
        # Author-year types share same validator
        author_year_types = {"autor_jahr_text", "autor_jahr_klammer_einzel", "autor_jahr_multi_klammer"}
        if citation_type in author_year_types:
            return self._validate_author_year_citations(citation_type, match_text, start_pos, end_pos, full_text)
        
        # Use specific validator or default to valid
        validator = validators.get(citation_type)
        if validator:
            return validator()
        
        return True, "valid_citation"

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
            # Get current stats from stats manager
            citation_stats, cleaning_stats = self.stats_manager.get_stats()
            
            # Alle Worker: Lokale Stats speichern für Aggregation
            self.worker_stats.save_worker_stats(rank, citation_stats, cleaning_stats)
            
            # Alle Worker: Lokale Stats loggen
            log.info(f"✅ Citation cleaning rank {rank} completed: {cleaning_stats['docs_processed']} docs, "
                    f"{cleaning_stats['docs_with_any_citations']} with citations, "
                    f"{cleaning_stats['total_citations_all_types']} removed")
            
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