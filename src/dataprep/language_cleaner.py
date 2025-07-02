"""Conservative paragraph-level language cleaner for T5 pretraining"""

import re
from typing import Dict, Tuple
from src.dataprep.text_cleaners import BaseTextCleaner


class LanguageQualityCleaner(BaseTextCleaner):
    """Conservative paragraph-level language cleaner"""
    
    def __init__(self, english_threshold: float = 0.7, debug_mode: bool = False):
        super().__init__(debug_mode)
        self.english_threshold = english_threshold
    
    def clean(self, text: str, doc_id: str = None) -> Tuple[str, Dict]:
        paragraphs = re.split(r'\n\s*\n', text)
        cleaned_paragraphs = []
        removed_paragraphs = []
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            english_ratio = self._estimate_english_ratio(paragraph)
            
            if english_ratio < self.english_threshold:
                removed_paragraphs.append({
                    "paragraph_number": i + 1,
                    "english_ratio": english_ratio,
                    "length": len(paragraph),
                    "preview": paragraph[:200].replace('\n', '\\n'),
                    "issue_type": self._classify_language_issue(paragraph)
                })
                
                if self.debug_mode:
                    debug_tag = f"[DEBUG:language:english_ratio_{english_ratio:.2f}]"
                    cleaned_paragraphs.append(debug_tag)
                else:
                    continue
            else:
                cleaned_paragraphs.append(paragraph)
        
        cleaned_text = '\n\n'.join(cleaned_paragraphs).strip()
        
        if len(cleaned_text) < 100:
            cleaned_text = ""
        
        stats = {
            "paragraphs_removed": len(removed_paragraphs),
            "length_reduction": sum(item["length"] for item in removed_paragraphs) if not self.debug_mode else 0,
            "removed_paragraphs": removed_paragraphs[:5],
            "document_too_short": len(cleaned_text) == 0,
            "doc_id": doc_id or "unknown"
        }
        
        return cleaned_text, stats
    
    def _estimate_english_ratio(self, text: str) -> float:
        if not text:
            return 1.0
        
        text_no_spaces = text.replace(' ', '').replace('\n', '').replace('\t', '')
        if not text_no_spaces:
            return 1.0
        
        total_chars = len(text_no_spaces)
        
        japanese_chars = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text))
        korean_chars = len(re.findall(r'[\uAC00-\uD7AF]', text))
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        cyrillic_chars = len(re.findall(r'[\u0400-\u04FF]', text))
        
        if len(re.findall(r'\[DEBUG:', text)) > 0:
            return 0.0
        
        non_english_penalty = 0.0
        
        if japanese_chars > 0:
            non_english_penalty += (japanese_chars / total_chars) * 2.0
        if korean_chars > 0:
            non_english_penalty += (korean_chars / total_chars) * 2.0
        if arabic_chars > 0:
            non_english_penalty += (arabic_chars / total_chars) * 2.0
        if cyrillic_chars > 0:
            non_english_penalty += (cyrillic_chars / total_chars) * 1.5
        
        non_ascii_chars = sum(1 for c in text_no_spaces if ord(c) > 127)
        other_non_ascii = non_ascii_chars - japanese_chars - korean_chars - arabic_chars - cyrillic_chars
        if other_non_ascii > 0:
            non_english_penalty += (other_non_ascii / total_chars) * 0.5
        
        return max(0.0, 1.0 - non_english_penalty)
    
    def _classify_language_issue(self, text: str) -> str:
        if re.search(r'\[DEBUG:', text):
            return "debug_tags"
        elif re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text):
            return "japanese_chinese"
        elif re.search(r'[\uAC00-\uD7AF]', text):
            return "korean"
        elif re.search(r'[\u0600-\u06FF]', text):
            return "arabic"
        elif re.search(r'[\u0400-\u04FF]', text):
            return "cyrillic"
        elif sum(1 for c in text if ord(c) > 127) / len(text.replace(' ', '')) > 0.2:
            return "high_non_ascii"
        else:
            return "unknown_language_issue"


class SmartLanguageCleaner(BaseTextCleaner):
    """Smart language cleaner using precomputed fasttext_en scores"""
    
    def __init__(self, fasttext_threshold: float = 0.8, paragraph_threshold: float = 0.7, debug_mode: bool = False):
        super().__init__(debug_mode)
        self.fasttext_threshold = fasttext_threshold
        self.paragraph_cleaner = LanguageQualityCleaner(english_threshold=paragraph_threshold, debug_mode=debug_mode)
    
    def clean(self, text: str, doc_id: str = None, fasttext_en: float = 1.0) -> Tuple[str, Dict]:
        if fasttext_en >= self.fasttext_threshold:
            return text, {
                "skipped": True,
                "reason": f"good_fasttext_en_score_{fasttext_en:.3f}",
                "paragraphs_removed": 0,
                "length_reduction": 0,
                "doc_id": doc_id or "unknown"
            }
        
        return self.paragraph_cleaner.clean(text, doc_id)