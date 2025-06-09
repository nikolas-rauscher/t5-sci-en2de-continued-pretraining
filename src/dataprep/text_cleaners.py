"""
Specialized Text Cleaners für DataTrove

Verschiedene spezialisierte Cleaner für unterschiedliche Text-Cleaning-Aufgaben.
Jeder Cleaner hat eine klare, spezifische Verantwortung.
"""

import re
from typing import Dict, List, Tuple, Set
from abc import ABC, abstractmethod


class BaseTextCleaner(ABC):
    """Base class for all text cleaners"""
    
    @abstractmethod
    def clean(self, text: str, doc_id: str = None) -> Tuple[str, Dict]:
        """Clean text and return cleaned text with stats"""
        pass


class RepetitionCleaner(BaseTextCleaner):
    """
    Cleaner für wiederholte Zeilen (OCR-Artefakte)
    """
    
    def __init__(self, max_recent_lines: int = 5):
        """
        Args:
            max_recent_lines: Anzahl der letzten Zeilen für Wiederholungserkennung
        """
        self.max_recent_lines = max_recent_lines
    
    def clean(self, text: str, doc_id: str = None) -> Tuple[str, Dict]:
        """Remove repeated lines that are likely OCR artifacts"""
        lines = text.splitlines()
        cleaned_lines = []
        removed_lines = []
        recent_lines = []
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            # Keep empty lines
            if not stripped_line:
                cleaned_lines.append(line)
                continue
            
            # Check for repetition
            if stripped_line in recent_lines:
                removed_lines.append({
                    "line_content": stripped_line[:100],
                    "line_number": i + 1,
                    "length": len(stripped_line),
                    "reason": "repeated_line"
                })
                continue
            
            # Update tracking and keep line
            recent_lines.append(stripped_line)
            if len(recent_lines) > self.max_recent_lines:
                recent_lines.pop(0)
            
            cleaned_lines.append(line)
        
        stats = {
            "lines_removed": len(removed_lines),
            "length_reduction": sum(item["length"] for item in removed_lines),
            "removed_lines": removed_lines[:10],
            "doc_id": doc_id or "unknown"
        }
        
        return '\n'.join(cleaned_lines), stats


class TabularDataCleaner(BaseTextCleaner):
    """
    Cleaner für Tabellendaten mit übermäßigen Leerzeichen
    """
    
    def __init__(self, min_space_sequences: int = 2, space_ratio_threshold: float = 0.4):
        """
        Args:
            min_space_sequences: Mindestanzahl von 3+-Leerzeichen-Sequenzen
            space_ratio_threshold: Schwellenwert für Leerzeichen-Anteil
        """
        self.min_space_sequences = min_space_sequences
        self.space_ratio_threshold = space_ratio_threshold
    
    def clean(self, text: str, doc_id: str = None) -> Tuple[str, Dict]:
        """Remove lines with excessive spacing indicating tabular data"""
        lines = text.splitlines()
        cleaned_lines = []
        removed_lines = []
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            # Keep empty lines
            if not stripped_line:
                cleaned_lines.append(line)
                continue
            
            should_remove, reason = self._is_tabular_data(stripped_line)
            
            if should_remove:
                removed_lines.append({
                    "line_content": stripped_line[:100],
                    "line_number": i + 1,
                    "length": len(stripped_line),
                    "reason": reason
                })
                continue
            
            cleaned_lines.append(line)
        
        stats = {
            "lines_removed": len(removed_lines),
            "length_reduction": sum(item["length"] for item in removed_lines),
            "removed_lines": removed_lines[:10],
            "doc_id": doc_id or "unknown"
        }
        
        return '\n'.join(cleaned_lines), stats
    
    def _is_tabular_data(self, line: str) -> Tuple[bool, str]:
        """Check if line represents tabular data"""
        # Method 1: Multiple space sequences
        multiple_space_sequences = len(re.findall(r'\s{3,}', line))
        if multiple_space_sequences >= self.min_space_sequences:
            return True, "tabular_data_spacing"
        
        # Method 2: Space ratio for longer lines
        if len(line) > 20:
            space_count = line.count(' ')
            space_ratio = space_count / len(line)
            if space_ratio > self.space_ratio_threshold:
                return True, "excessive_spacing_ratio"
        
        return False, ""


class ShortLineCleaner(BaseTextCleaner):
    """
    Cleaner für isolierte kurze Zeilen (1-3 Wörter)
    """
    
    def __init__(self):
        # Important single words that should be kept
        self.important_single_words = {
            'INTRODUCTION', 'CONCLUSION', 'RESULTS', 'DISCUSSION', 
            'METHODS', 'BACKGROUND', 'SUMMARY', 'OVERVIEW', 'ANALYSIS',
            'EXPERIMENT', 'PROCEDURE', 'MATERIALS', 'PROTOCOL', 'DESIGN',
            'IMPLEMENTATION', 'EVALUATION', 'LIMITATIONS', 'FUTURE',
            'APPLICATIONS', 'IMPLICATIONS', 'SIGNIFICANCE', 'ABSTRACT',
            'OBJECTIVES', 'HYPOTHESIS', 'APPROACH', 'FRAMEWORK', 'MODEL',
            'ALGORITHM', 'BASELINE', 'COMPARISON', 'VALIDATION', 'TESTING'
        }
        
        # Important two-word patterns
        self.important_two_word_patterns = {
            'RELATED WORK', 'FUTURE WORK', 'CASE STUDY', 'USER STUDY',
            'DATA ANALYSIS', 'STATISTICAL ANALYSIS', 'EXPERIMENTAL DESIGN',
            'LITERATURE REVIEW', 'SYSTEMATIC REVIEW', 'META ANALYSIS',
            'RESEARCH QUESTIONS', 'RESEARCH OBJECTIVES', 'ETHICAL CONSIDERATIONS',
            'PERFORMANCE EVALUATION', 'MODEL VALIDATION', 'EXPERIMENTAL SETUP',
            'BASELINE COMPARISON', 'ABLATION STUDY', 'ERROR ANALYSIS',
            'THEORETICAL BACKGROUND', 'CONCEPTUAL FRAMEWORK', 'PROBLEM STATEMENT'
        }
        
        # Important three-word patterns
        self.important_three_word_patterns = {
            'MATERIALS AND METHODS', 'RESULTS AND DISCUSSION', 'CONCLUSIONS AND IMPLICATIONS',
            'EXPERIMENTAL VALIDATION RESULTS', 'STATISTICAL SIGNIFICANCE TESTING', 'MODEL PERFORMANCE EVALUATION',
            'LITERATURE REVIEW METHODOLOGY', 'DATA COLLECTION PROCEDURES', 'ETHICAL APPROVAL CONSIDERATIONS',
            'LIMITATIONS AND ASSUMPTIONS', 'FUTURE RESEARCH DIRECTIONS', 'PRACTICAL IMPLEMENTATION CONSIDERATIONS',
            'THEORETICAL FRAMEWORK DEVELOPMENT', 'EMPIRICAL RESULTS ANALYSIS', 'COMPARATIVE PERFORMANCE ANALYSIS'
        }
    
    def clean(self, text: str, doc_id: str = None) -> Tuple[str, Dict]:
        """Remove isolated short lines that break text flow"""
        lines = text.splitlines()
        cleaned_lines = []
        removed_lines = []
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            # Keep empty lines
            if not stripped_line:
                cleaned_lines.append(line)
                continue
            
            words = stripped_line.split()
            word_count = len(words)
            
            should_remove, reason = self._should_remove_short_line(stripped_line, words, word_count)
            
            if should_remove:
                removed_lines.append({
                    "line_content": stripped_line[:100],
                    "line_number": i + 1,
                    "word_count": word_count,
                    "length": len(stripped_line),
                    "reason": reason
                })
                continue
            
            cleaned_lines.append(line)
        
        stats = {
            "lines_removed": len(removed_lines),
            "length_reduction": sum(item["length"] for item in removed_lines),
            "removed_lines": removed_lines[:10],
            "doc_id": doc_id or "unknown"
        }
        
        return '\n'.join(cleaned_lines), stats
    
    def _should_remove_short_line(self, line: str, words: List[str], word_count: int) -> Tuple[bool, str]:
        """Determine if a short line should be removed"""
        if word_count == 1:
            return self._check_single_word(words[0])
        elif word_count == 2:
            return self._check_two_words(line)
        elif word_count == 3:
            return self._check_three_words(line)
        
        return False, ""
    
    def _check_single_word(self, word: str) -> Tuple[bool, str]:
        """Check single word lines"""
        word_upper = word.upper()
        
        if word_upper not in self.important_single_words:
            if (len(word_upper) < 10 and
                not re.search(r'\d', word_upper) and
                not word_upper.endswith('ING') and
                not re.match(r'^[A-Z]{2,}S$', word_upper) and
                word_upper not in ['PROBLEM', 'SOLUTION', 'APPROACH', 'THEORY', 'PRACTICE', 'REVIEW']):
                return True, "isolated_meaningless_word"
        
        return False, ""
    
    def _check_two_words(self, line: str) -> Tuple[bool, str]:
        """Check two word lines"""
        line_upper = line.upper()
        
        # Keep numbered section headers
        if re.match(r'^\d+[\.\)]\s+[A-Z]', line) or re.match(r'^[IVX]+[\.\)]\s+[A-Z]', line):
            return False, ""
        
        # Remove structural headers
        if (line_upper not in self.important_two_word_patterns and
            (re.match(r'^TABLE\s+\d+', line_upper) or
             re.match(r'^FIGURE\s+\d+', line_upper) or
             re.match(r'^CHAPTER\s+\d+', line_upper) or
             re.match(r'^SECTION\s+\d+', line_upper) or
             line_upper in ['NAME SUPPLIER', 'ANTIBODY SUPPLIER', 'GENE NAME'])):
            return True, "structural_two_word_header"
        
        return False, ""
    
    def _check_three_words(self, line: str) -> Tuple[bool, str]:
        """Check three word lines"""
        line_upper = line.upper()
        
        # Keep numbered section headers
        if re.match(r'^\d+[\.\)]\s+[A-Z]', line) or re.match(r'^[IVX]+[\.\)]\s+[A-Z]', line):
            return False, ""
        
        # Remove metadata headers
        if (line_upper not in self.important_three_word_patterns and
            (re.match(r'^KEYWORDS?:\s+', line_upper) or
             re.match(r'^TABLE\s+\d+\s+', line_upper) or
             re.match(r'^FIGURE\s+\d+\s+', line_upper) or
             line_upper in ['TABLE OF CONTENTS', 'LIST OF TABLES', 'LIST OF FIGURES', 'LIST OF ABBREVIATIONS'])):
            return True, "metadata_or_structural_header"
        
        return False, ""


class MetadataCleaner(BaseTextCleaner):
    """
    Cleaner für Metadata-Zeilen (Keywords, technische Codes)
    """
    
    def clean(self, text: str, doc_id: str = None) -> Tuple[str, Dict]:
        """Remove metadata lines like keywords and technical codes"""
        lines = text.splitlines()
        cleaned_lines = []
        removed_lines = []
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            # Keep empty lines
            if not stripped_line:
                cleaned_lines.append(line)
                continue
            
            should_remove, reason = self._is_metadata_line(stripped_line)
            
            if should_remove:
                removed_lines.append({
                    "line_content": stripped_line[:100],
                    "line_number": i + 1,
                    "length": len(stripped_line),
                    "reason": reason
                })
                continue
            
            cleaned_lines.append(line)
        
        stats = {
            "lines_removed": len(removed_lines),
            "length_reduction": sum(item["length"] for item in removed_lines),
            "removed_lines": removed_lines[:10],
            "doc_id": doc_id or "unknown"
        }
        
        return '\n'.join(cleaned_lines), stats
    
    def _is_metadata_line(self, line: str) -> Tuple[bool, str]:
        """Check if line is metadata that should be removed"""
        # Keywords lines
        if re.match(r'^[Kk]eywords?:\s+', line):
            return True, "keywords_metadata"
        
        # Technical codes (only short ones, not drug names like GEFITINIB)
        if (len(line.split()) <= 8 and
            re.search(r'^[A-Z0-9\-]{2,}\s+[A-Z0-9\-]{2,}', line) or
            (re.match(r'^[A-Z0-9\s\-\.]+$', line) and
             not re.search(r'[a-z]', line) and
             len(line.replace(' ', '')) < 8)):  # Only short codes
            return True, "technical_code_line"
        
        return False, ""


class ComprehensiveLineCleaner:
    """
    Orchestrator der alle spezialisierten Cleaner kombiniert
    """
    
    def __init__(self):
        self.repetition_cleaner = RepetitionCleaner(max_recent_lines=5)
        self.tabular_cleaner = TabularDataCleaner(min_space_sequences=2, space_ratio_threshold=0.4)
        self.short_line_cleaner = ShortLineCleaner()
        self.metadata_cleaner = MetadataCleaner()
    
    def clean_lines(self, text: str, doc_id: str = None) -> Tuple[str, Dict]:
        """
        Apply all line cleaning steps in sequence
        
        Returns:
            Tuple of (cleaned_text, combined_stats)
        """
        current_text = text
        combined_stats = {
            "total_lines_removed": 0,
            "total_length_reduction": 0,
            "removed_lines": [],
            "doc_id": doc_id or "unknown",
            "cleaning_steps": {}
        }
        
        # Step 1: Remove repeated lines
        current_text, rep_stats = self.repetition_cleaner.clean(current_text, doc_id)
        combined_stats["cleaning_steps"]["repetition"] = rep_stats
        
        # Step 2: Remove tabular data
        current_text, tab_stats = self.tabular_cleaner.clean(current_text, doc_id)
        combined_stats["cleaning_steps"]["tabular"] = tab_stats
        
        # Step 3: Remove short lines
        current_text, short_stats = self.short_line_cleaner.clean(current_text, doc_id)
        combined_stats["cleaning_steps"]["short_lines"] = short_stats
        
        # Step 4: Remove metadata
        current_text, meta_stats = self.metadata_cleaner.clean(current_text, doc_id)
        combined_stats["cleaning_steps"]["metadata"] = meta_stats
        
        # Combine stats
        for step_stats in combined_stats["cleaning_steps"].values():
            combined_stats["total_lines_removed"] += step_stats["lines_removed"]
            combined_stats["total_length_reduction"] += step_stats["length_reduction"]
            combined_stats["removed_lines"].extend(step_stats["removed_lines"])
        
        # Limit samples
        combined_stats["removed_lines"] = combined_stats["removed_lines"][:10]
        
        return current_text, combined_stats 