"""
Figure and Table Line Cleaner Module

Specialized cleaner for removing figure/table references and captions that are
harmful for T5 pretraining, while preserving mathematical content.

Extracted from MultiCitationCleaner to improve modularity and maintainability.
"""

import re
from typing import Dict, List, Tuple
from src.dataprep.text_cleaners import BaseTextCleaner


class FigureTableCleaner(BaseTextCleaner):
    """
    Cleaner for figure/table references and captions with mathematical content preservation
    """
    
    def __init__(self, debug_mode: bool = False):
        """
        Args:
            debug_mode: If True, add debug tags instead of removing lines
        """
        super().__init__(debug_mode)
    
    def clean(self, text: str, doc_id: str = None) -> Tuple[str, Dict]:
        """
        Remove lines containing figure/table references and captions aggressively with detailed logging
        
        Args:
            text: Input text to clean
            doc_id: Document ID for logging (optional)
            
        Returns:
            Tuple of (cleaned_text, stats_dict)
        """
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