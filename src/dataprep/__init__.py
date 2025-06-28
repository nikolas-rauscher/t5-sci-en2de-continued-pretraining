"""
DataTrove-based Data Preprocessing Components

Contains modular text cleaning components and pipeline steps for data preprocessing:

Components:
- MultiCitationCleaner: Removes academic citations with smart validation
- TextNormalizer: Normalizes whitespace and newlines  
- SlidingWindowProcessor: Precomputes sliding windows for T5 training

"""

from .multi_citation_cleaner import MultiCitationCleaner
from .text_normalizer import TextNormalizer
from .sliding_window_processor import SlidingWindowProcessor

__all__ = [
    "MultiCitationCleaner",
    "TextNormalizer",
    "SlidingWindowProcessor"
] 