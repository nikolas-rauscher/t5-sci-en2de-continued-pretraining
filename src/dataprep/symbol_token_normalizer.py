"""
Symbol Token Normalizer for T5

Normalizes scientific symbols for better T5 tokenization.
"""

import re
import logging
from typing import Dict, Tuple
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter

log = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class SymbolTokenNormalizer(BaseFilter):
    
    name = "Symbol Token Normalizer"
    
    def __init__(
        self,
        debug_mode: bool = False,
        log_to_wandb: bool = True,
        exclusion_writer: DiskWriter = None
    ):
        super().__init__(exclusion_writer)
        self.debug_mode = debug_mode
        self.log_to_wandb = log_to_wandb and WANDB_AVAILABLE
        
        # Math operators
        self.math_operators = {
            r'≤': '<=', r'≥': '>=', r'≠': '!=', r'±': '+/-',
            r'×': '*', r'÷': '/', r'≈': '~=', r'∞': 'infinity',
            r'∂': 'partial', r'∇': 'nabla', r'∆': 'DELTA',
            r'≅': 'congruent', r'≃': 'simeq', r'∝': 'proportional',
            r'≡': 'equivalent', r'≢': 'not_equivalent', r'≅': 'congruent'
        }
        
        # Math symbols
        self.math_symbols = {
            r'∑': 'SUM', r'∏': 'PRODUCT', r'∫': 'integral',
            r'√': 'sqrt', r'∛': 'cube_root', r'∜': 'fourth_root',
            r'⌈': 'ceil', r'⌉': 'ceil', r'⌊': 'floor', r'⌋': 'floor',
            r'|': 'absolute', r'∥': 'parallel', r'⊥': 'perpendicular'
        }
        
        # Logic and set symbols
        self.logic_set_symbols = {
            r'∧': 'AND', r'∨': 'OR', r'¬': 'NOT',
            r'∀': 'forall', r'∃': 'exists', r'∄': 'not_exists',
            r'∈': 'in', r'∉': 'not_in', r'⊂': 'subset', r'⊃': 'superset',
            r'∪': 'union', r'∩': 'intersection', r'∅': 'emptyset',
            r'⊆': 'subset_equal', r'⊇': 'superset_equal', r'⊊': 'proper_subset',
            r'⊋': 'proper_superset', r'⊕': 'xor', r'⊗': 'tensor_product'
        }
        
        # Fractions
        self.fractions = {
            r'½': '1/2', r'⅓': '1/3', r'¼': '1/4', r'¾': '3/4',
            r'⅛': '1/8', r'⅜': '3/8', r'⅝': '5/8', r'⅞': '7/8',
            r'⅕': '1/5', r'⅖': '2/5', r'⅗': '3/5', r'⅘': '4/5',
            r'⅙': '1/6', r'⅚': '5/6', r'⅐': '1/7', r'⅑': '1/9', r'⅒': '1/10'
        }
        
        # Superscripts (common ones)
        self.superscripts = {
            r'¹': '^1', r'²': '^2', r'³': '^3', r'⁴': '^4', r'⁵': '^5',
            r'⁶': '^6', r'⁷': '^7', r'⁸': '^8', r'⁹': '^9', r'¹⁰': '^10',
            r'¹¹': '^11', r'¹²': '^12', r'¹³': '^13', r'¹⁴': '^14', r'¹⁵': '^15',
            r'¹⁶': '^16', r'¹⁷': '^17', r'¹⁸': '^18', r'¹⁹': '^19', r'²⁰': '^20',
            r'²¹': '^21', r'²²': '^22', r'²³': '^23', r'²⁴': '^24', r'²⁵': '^25',
            r'²⁶': '^26', r'²⁷': '^27', r'²⁸': '^28', r'²⁹': '^29', r'³⁰': '^30'
        }
        
        # Subscripts
        self.subscripts = {
            r'₁': '_1', r'₂': '_2', r'₃': '_3', r'₄': '_4', r'₅': '_5',
            r'₆': '_6', r'₇': '_7', r'₈': '_8', r'₉': '_9', r'₀': '_0',
            r'₊': '_+', r'₋': '_-', r'₌': '_=', r'₍': '_(', r'₎': '_)'
        }
        
        # Quotation marks and guillemets
        self.quotation_marks = {
            r'"': '"', r'"': '"', r''': "'", r''': "'",
            r'‚': ',', r'„': '"', r'‹': '<', r'›': '>',
            r'«': '<<', r'»': '>>', r'‹': '<', r'›': '>',
            r'❛': "'", r'❜': "'", r'❝': '"', r'❞': '"'
        }
        
        # Legal and currency symbols
        self.legal_currency = {
            r'©': '(c)', r'®': '(R)', r'™': '(TM)',
            r'§': 'section', r'¶': 'paragraph',
            r'€': 'EUR', r'£': 'GBP', r'¥': 'JPY', r'¢': 'cents',
            r'$': 'USD', r'₹': 'INR', r'₽': 'RUB', r'₩': 'KRW',
            r'₪': 'ILS', r'₦': 'NGN', r'₨': 'PKR', r'₫': 'VND'
        }
        
        # Greek letters
        self.greek_letters = {
            r'α': 'alpha', r'β': 'beta', r'γ': 'gamma', r'δ': 'delta',
            r'ε': 'epsilon', r'ζ': 'zeta', r'η': 'eta', r'θ': 'theta',
            r'ι': 'iota', r'κ': 'kappa', r'λ': 'lambda', r'μ': 'mu',
            r'ν': 'nu', r'ξ': 'xi', r'π': 'pi', r'ρ': 'rho',
            r'σ': 'sigma', r'τ': 'tau', r'υ': 'upsilon', r'φ': 'phi',
            r'χ': 'chi', r'ψ': 'psi', r'ω': 'omega',
            r'Α': 'Alpha', r'Β': 'Beta', r'Γ': 'Gamma', r'Δ': 'Delta',
            r'Θ': 'Theta', r'Λ': 'Lambda', r'Π': 'Pi', r'Σ': 'Sigma',
            r'Φ': 'Phi', r'Ω': 'Omega', r'µ': 'mu',
            r'ϑ': 'vartheta', r'ϒ': 'Upsilon', r'ϕ': 'varphi',
            r'ϖ': 'varpi', r'ϱ': 'varrho', r'ς': 'varsigma'
        }
        
        # Scientific notation
        self.scientific_notation = {
            r'\d+\.\d+e[+-]?\d+': '[SCI_NUM]',
            r'\d+e[+-]?\d+': '[SCI_NUM]'
        }
        
        # Bullet points
        self.bullet_points = {
            r'•': '*', r'‣': '*', r'‧': '*', r'⁃': '*',
            r'◦': '*', r'‣': '*', r'⁌': '*', r'⁍': '*',
            r'•': '*', r'‣': '*', r'‧': '*', r'⁃': '*'
        }
        
        
        # Arrows
        self.arrows = {
            r'→': '->', r'←': '<-', r'↑': '^', r'↓': 'v',
            r'↔': '<->', r'⇒': '=>', r'⇐': '<=', r'⇔': '<=>'
        }
        
        # Special minus and dashes
        self.special_minus = {
            r'−': '-', r'‐': '-', r'–': '-', r'—': '-'
        }
        
        # Invisible mathematical characters
        self.invisible_chars = {
            r'\u2062': '', r'\u2063': '', r'\u2061': ''
        }
        
        # Cyrillic to Latin (common OCR errors)
        self.cyrillic_to_latin = {
            r'а': 'a', r'е': 'e', r'о': 'o', r'р': 'p', r'с': 'c',
            r'у': 'y', r'х': 'x', r'А': 'A', r'В': 'B', r'Е': 'E',
            r'К': 'K', r'М': 'M', r'Н': 'H', r'О': 'O', r'Р': 'P',
            r'С': 'C', r'Т': 'T', r'У': 'Y', r'Х': 'X', r'и': 'u',
            r'н': 'h', r'т': 't', r'ь': 'b'
        }
        
        # Table of Contents and structural patterns (REMOVE completely)
        self.toc_patterns = {
            r'\w+[.\s]{5,}\d+': '',                    # "Introduction ... 5" -> remove
            r'^[.\t\s]{10,}$': '',                     # Lines of only dots/tabs -> remove
            r'\w+[.\t]{5,}\w+': '',                    # "Chapter\t\t\tTitle" -> remove
        }
        
        # Tabular data patterns (REMOVE completely)  
        self.tabular_patterns = {
            r'^[\d\t\s\.]{20,}$': '',                  # Lines mostly numbers/tabs -> remove
            r'^.*\t.*\t.*\t.*$': '',                   # 4+ tab-separated values -> remove
        }
        
        # Excessive dots (SHORTEN to ellipsis)
        self.dot_shortening = {
            r'\.{4,}': '...',                          # 4+ dots -> ellipsis
            r'(\. ){3,}': ' ... ',                     # Spaced dots -> ellipsis
        }
        
        self.custom_stats = {
            "docs_processed": 0,
            "docs_normalized": 0,
            "total_normalizations": 0
        }
        
        self.wandb_initialized = False
        self.current_rank = 0
    
    def _normalize_symbols(self, text: str) -> Tuple[str, int]:
        normalized_text = text
        normalizations = 0
        
        if self.debug_mode:
            all_patterns = {**self.math_operators, **self.math_symbols, **self.logic_set_symbols,
                           **self.fractions, **self.superscripts, **self.quotation_marks,
                           **self.legal_currency, **self.greek_letters, **self.bullet_points, **self.arrows}
            for pattern, replacement in all_patterns.items():
                if re.search(pattern, normalized_text):
                    normalized_text = re.sub(pattern, f'[DEBUG:symbol:{replacement}]', normalized_text)
                    normalizations += 1
            return normalized_text, normalizations
        
        # Apply all normalizations in order
        pattern_groups = [
            self.invisible_chars,  # Remove invisible chars first
            self.toc_patterns,  # Remove table of contents
            self.tabular_patterns,  # Remove tabular data
            self.dot_shortening,  # Shorten excessive dots
            self.cyrillic_to_latin,  # Fix OCR errors
            self.special_minus,  # Normalize dashes
            self.quotation_marks,  # Normalize quotes
            self.bullet_points,  # Normalize bullets
            self.arrows,  # Normalize arrows
            self.fractions,  # Normalize fractions
            self.superscripts,  # Normalize superscripts
            self.legal_currency,  # Normalize legal/currency symbols
            self.logic_set_symbols,  # Logic and set symbols
            self.math_operators,  # Math operators
            self.math_symbols,  # Math symbols
            self.greek_letters,  # Greek letters
            self.scientific_notation  # Scientific notation
        ]
        
        for patterns in pattern_groups:
            for pattern, replacement in patterns.items():
                count = len(re.findall(pattern, normalized_text))
                if count > 0:
                    normalized_text = re.sub(pattern, replacement, normalized_text)
                    normalizations += count
        
        return normalized_text, normalizations
    
    def filter(self, doc: Document) -> bool:
        original_text = doc.text
        self.custom_stats["docs_processed"] += 1
        
        normalized_text, normalizations_count = self._normalize_symbols(original_text)
        doc.text = normalized_text
        
        text_changed = original_text != normalized_text
        if text_changed:
            self.custom_stats["docs_normalized"] += 1
            self.custom_stats["total_normalizations"] += normalizations_count
            
            doc.metadata["symbol_text_changed"] = True
            doc.metadata["symbol_normalizations"] = normalizations_count
        else:
            doc.metadata["symbol_text_changed"] = False
            doc.metadata["symbol_normalizations"] = 0
        
        self.stat_update("docs_processed")
        if text_changed:
            self.stat_update("docs_with_symbol_normalizations")
            self.stat_update("total_symbol_normalizations", normalizations_count)
        
        if self.current_rank == 0 and self.wandb_initialized and self.custom_stats["docs_processed"] % 100 == 0:
            self._log_stats()
        
        return bool(normalized_text.strip())
    
    def _log_stats(self):
        if not self.wandb_initialized:
            return
            
        try:
            wandb.log({
                "symbol_norm/docs_processed": self.custom_stats["docs_processed"],
                "symbol_norm/docs_normalized": self.custom_stats["docs_normalized"],
                "symbol_norm/normalization_rate": (
                    self.custom_stats["docs_normalized"] / self.custom_stats["docs_processed"] 
                    if self.custom_stats["docs_processed"] > 0 else 0
                ),
                "symbol_norm/total_normalizations": self.custom_stats["total_normalizations"]
            })
        except Exception as e:
            log.warning(f"Failed to log symbol normalization stats: {e}")
    
    def run(self, data, rank: int = 0, world_size: int = 1):
        self.current_rank = rank
        
        if rank == 0 and self.log_to_wandb:
            self._init_wandb()
        
        try:
            yield from super().run(data, rank, world_size)
        finally:
            log.info(f"Symbol normalization rank {rank}: {self.custom_stats['docs_processed']} docs, "
                    f"{self.custom_stats['docs_normalized']} normalized, "
                    f"{self.custom_stats['total_normalizations']} symbols")
    
    def _init_wandb(self):
        try:
            if wandb.run is not None:
                self.wandb_initialized = True
                log.info("Using existing W&B session for Symbol Normalizer")
            else:
                wandb.init(
                    project="BA-DataTrove",
                    group="symbol-normalization",
                    tags=["symbol-normalization", "t5-optimization"],
                    job_type="symbol-normalization",
                    config={"debug_mode": self.debug_mode}
                )
                self.wandb_initialized = True
                log.info("W&B initialized for Symbol Normalizer")
        except Exception as e:
            log.warning(f"Failed to initialize W&B: {e}")
            self.log_to_wandb = False