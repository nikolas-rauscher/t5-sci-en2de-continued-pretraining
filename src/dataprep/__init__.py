"""
DataTrove Text Preprocessing Modules

Modular text cleaning components für DataTrove Pipeline:
- CitationCleaner: Einfacher Citation Cleaner (legacy)
- MultiCitationCleaner: Multi-Type Citation Cleaner
- TextNormalizer: Whitespace/Newline Normalisierung

Usage:
```python
from src.dataprep import CitationCleaner, TextNormalizer

# Or individual imports:
from src.dataprep.citation_cleaner import CitationCleaner
from src.dataprep.text_normalizer import TextNormalizer
```
"""

from .citation_cleaner import CitationCleaner
from .multi_citation_cleaner import MultiCitationCleaner
from .text_normalizer import TextNormalizer

__all__ = [
    "CitationCleaner",
    "MultiCitationCleaner",
    "TextNormalizer",
] 