"""
Data preprocessing pipeline for tokenizer training.

This module provides comprehensive text preprocessing capabilities including
cleaning, normalization, filtering, and data augmentation for tokenizer training.
"""

import re
import json
import html
import ftfy
import unicodedata
from typing import List, Dict, Tuple, Iterator, Optional, Union, Set, Callable
from collections import Counter, defaultdict
import logging
import random
import hashlib
from dataclasses import dataclass
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing pipeline."""
    # Unicode and encoding
    fix_encoding: bool = True
    normalize_unicode: str = 'NFC'  # NFC, NFD, NFKC, NFKD, or None
    
    # Text cleaning
    remove_control_chars: bool = True
    remove_zero_width_chars: bool = True
    fix_html_entities: bool = True
    remove_urls: bool = False
    remove_emails: bool = False
    remove_phone_numbers: bool = False
    
    # Whitespace handling
    normalize_whitespace: bool = True
    preserve_paragraph_breaks: bool = True
    min_line_length: int = 10
    
    # Language filtering
    allowed_languages: Optional[Set[str]] = None  # ISO language codes
    detect_language: bool = False
    
    # Content filtering
    min_text_length: int = 50
    max_text_length: int = 1000000
    remove_duplicates: bool = True
    duplicate_threshold: float = 0.85  # Jaccard similarity threshold
    
    # Quality filtering
    min_word_count: int = 10
    max_repetition_ratio: float = 0.3
    min_unique_words_ratio: float = 0.3
    max_special_char_ratio: float = 0.2
    
    # Case handling
    preserve_case: bool = True
    title_case_threshold: float = 0.8  # Ratio for detecting title case
    
    # Sentence handling
    split_sentences: bool = False
    min_sentence_length: int = 20
    max_sentence_length: int = 1000


class TextCleaner:
    """
    Advanced text cleaning utilities with configurable options.
    """
    
    def __init__(self, config: PreprocessingConfig = None):
        """Initialize with preprocessing configuration."""
        self.config = config or PreprocessingConfig()
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient text processing."""
        # Control characters (excluding newlines and tabs if preserving)
        control_chars = r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]'
        if self.config.preserve_paragraph_breaks:
            self.control_char_re = re.compile(control_chars)
        else:
            self.control_char_re = re.compile(r'[\x00-\x1F\x7F-\x9F]')
        
        # Zero-width characters
        self.zero_width_re = re.compile(r'[\u200B-\u200D\u2060\uFEFF]')
        
        # URL patterns
        self.url_re = re.compile(
            r'https?://(?:[-\w.])+(?::[0-9]+)?(?:/(?:[\w/_.])*)?(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?',
            re.IGNORECASE
        )
        
        # Email patterns
        self.email_re = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        # Phone number patterns (basic)
        self.phone_re = re.compile(
            r'(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\d{3}[-.\s]\d{3}[-.\s]\d{4}'
        )
        
        # Whitespace normalization
        self.whitespace_re = re.compile(r'\s+')
        self.paragraph_break_re = re.compile(r'\n\s*\n')
        
        # Repetition detection
        self.repetition_re = re.compile(r'(.{1,50}?)\1{2,}', re.DOTALL)
    
    def clean_text(self, text: str) -> str:
        """
        Apply comprehensive text cleaning.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Fix encoding issues
        if self.config.fix_encoding:
            text = ftfy.fix_text(text)
        
        # Fix HTML entities
        if self.config.fix_html_entities:
            text = html.unescape(text)
        
        # Unicode normalization
        if self.config.normalize_unicode:
            text = unicodedata.normalize(self.config.normalize_unicode, text)
        
        # Remove control characters
        if self.config.remove_control_chars:
            text = self.control_char_re.sub('', text)
        
        # Remove zero-width characters
        if self.config.remove_zero_width_chars:
            text = self.zero_width_re.sub('', text)
        
        # Remove URLs
        if self.config.remove_urls:
            text = self.url_re.sub(' [URL] ', text)
        
        # Remove emails
        if self.config.remove_emails:
            text = self.email_re.sub(' [EMAIL] ', text)
        
        # Remove phone numbers
        if self.config.remove_phone_numbers:
            text = self.phone_re.sub(' [PHONE] ', text)
        
        # Normalize whitespace
        if self.config.normalize_whitespace:
            if self.config.preserve_paragraph_breaks:
                # Preserve paragraph breaks but normalize other whitespace
                paragraphs = self.paragraph_break_re.split(text)
                paragraphs = [self.whitespace_re.sub(' ', p).strip() for p in paragraphs]
                text = '\n\n'.join(p for p in paragraphs if len(p) >= self.config.min_line_length)
            else:
                text = self.whitespace_re.sub(' ', text).strip()
        
        return text


class QualityFilter:
    """
    Text quality assessment and filtering.
    """
    
    def __init__(self, config: PreprocessingConfig = None):
        """Initialize with preprocessing configuration."""
        self.config = config or PreprocessingConfig()
    
    def calculate_metrics(self, text: str) -> Dict[str, float]:
        """
        Calculate text quality metrics.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary of quality metrics
        """
        if not text:
            return {'length': 0, 'word_count': 0, 'quality_score': 0.0}
        
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        
        # Basic metrics
        metrics = {
            'length': char_count,
            'word_count': word_count,
            'avg_word_length': sum(len(w) for w in words) / word_count if words else 0,
        }
        
        # Unique words ratio
        unique_words = set(words)
        metrics['unique_words_ratio'] = len(unique_words) / word_count if word_count > 0 else 0
        
        # Special character ratio
        special_chars = sum(1 for c in text if not (c.isalnum() or c.isspace()))
        metrics['special_char_ratio'] = special_chars / char_count if char_count > 0 else 0
        
        # Repetition detection
        repetition_matches = re.findall(r'(.{3,}?)\1{1,}', text)
        repetition_chars = sum(len(match[0]) * 2 for match in repetition_matches)  # *2 for original + repeat
        metrics['repetition_ratio'] = repetition_chars / char_count if char_count > 0 else 0
        
        # Digit ratio
        digits = sum(1 for c in text if c.isdigit())
        metrics['digit_ratio'] = digits / char_count if char_count > 0 else 0
        
        # Uppercase ratio
        uppercase = sum(1 for c in text if c.isupper())
        metrics['uppercase_ratio'] = uppercase / char_count if char_count > 0 else 0
        
        # Line count and average line length
        lines = text.split('\n')
        metrics['line_count'] = len(lines)
        metrics['avg_line_length'] = sum(len(line) for line in lines) / len(lines) if lines else 0
        
        # Calculate overall quality score (0-1)
        quality_score = self._calculate_quality_score(metrics)
        metrics['quality_score'] = quality_score
        
        return metrics
    
    def _calculate_quality_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score based on metrics."""
        score = 1.0
        
        # Penalize excessive repetition
        if metrics['repetition_ratio'] > self.config.max_repetition_ratio:
            score *= 0.5
        
        # Penalize low unique word ratio
        if metrics['unique_words_ratio'] < self.config.min_unique_words_ratio:
            score *= 0.7
        
        # Penalize excessive special characters
        if metrics['special_char_ratio'] > self.config.max_special_char_ratio:
            score *= 0.8
        
        # Penalize excessive uppercase (likely spam/low quality)
        if metrics['uppercase_ratio'] > 0.5:
            score *= 0.6
        
        # Bonus for reasonable average word length
        if 3 <= metrics['avg_word_length'] <= 8:
            score *= 1.1
        
        return min(score, 1.0)
    
    def is_high_quality(self, text: str, threshold: float = 0.5) -> bool:
        """
        Determine if text meets quality standards.
        
        Args:
            text: Text to evaluate
            threshold: Minimum quality score threshold
            
        Returns:
            True if text is high quality
        """
        metrics = self.calculate_metrics(text)
        
        # Basic length checks
        if (metrics['length'] < self.config.min_text_length or 
            metrics['length'] > self.config.max_text_length or
            metrics['word_count'] < self.config.min_word_count):
            return False
        
        # Quality score check
        return metrics['quality_score'] >= threshold


class DuplicateDetector:
    """
    Efficient duplicate detection using various similarity measures.
    """
    
    def __init__(self, threshold: float = 0.85):
        """
        Initialize duplicate detector.
        
        Args:
            threshold: Similarity threshold for considering texts as duplicates
        """
        self.threshold = threshold
        self.seen_hashes = set()
        self.seen_shingles = defaultdict(list)
    
    def compute_hash(self, text: str) -> str:
        """Compute hash of normalized text."""
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def compute_shingles(self, text: str, k: int = 5) -> Set[str]:
        """Compute k-shingles (k-grams) of text."""
        text = re.sub(r'\s+', ' ', text.lower().strip())
        words = text.split()
        if len(words) < k:
            return {text}
        return {' '.join(words[i:i+k]) for i in range(len(words) - k + 1)}
    
    def jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
    def is_duplicate(self, text: str) -> bool:
        """
        Check if text is a duplicate of previously seen text.
        
        Args:
            text: Text to check for duplicates
            
        Returns:
            True if text is likely a duplicate
        """
        # Fast hash-based exact duplicate detection
        text_hash = self.compute_hash(text)
        if text_hash in self.seen_hashes:
            return True
        
        # Shingle-based near-duplicate detection
        shingles = self.compute_shingles(text)
        shingle_hash = hash(frozenset(shingles))
        
        for seen_shingle_hash, seen_shingles in self.seen_shingles.items():
            if self.jaccard_similarity(shingles, seen_shingles) >= self.threshold:
                return True
        
        # Store for future comparisons
        self.seen_hashes.add(text_hash)
        self.seen_shingles[shingle_hash] = shingles
        
        return False
    
    def reset(self):
        """Clear all stored hashes and shingles."""
        self.seen_hashes.clear()
        self.seen_shingles.clear()


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline combining all preprocessing steps.
    """
    
    def __init__(self, config: PreprocessingConfig = None):
        """Initialize preprocessing pipeline."""
        self.config = config or PreprocessingConfig()
        self.cleaner = TextCleaner(self.config)
        self.quality_filter = QualityFilter(self.config)
        self.duplicate_detector = DuplicateDetector(self.config.duplicate_threshold) if self.config.remove_duplicates else None
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'after_cleaning': 0,
            'after_quality_filter': 0,
            'after_deduplication': 0,
            'final_count': 0,
        }
    
    def process_single_text(self, text: str) -> Optional[str]:
        """
        Process a single text through the complete pipeline.
        
        Args:
            text: Input text to process
            
        Returns:
            Processed text or None if filtered out
        """
        if not text or not isinstance(text, str):
            return None
        
        self.stats['total_processed'] += 1
        
        # Step 1: Text cleaning
        cleaned_text = self.cleaner.clean_text(text)
        if not cleaned_text:
            return None
        
        self.stats['after_cleaning'] += 1
        
        # Step 2: Quality filtering
        if not self.quality_filter.is_high_quality(cleaned_text):
            return None
        
        self.stats['after_quality_filter'] += 1
        
        # Step 3: Duplicate detection
        if self.duplicate_detector and self.duplicate_detector.is_duplicate(cleaned_text):
            return None
        
        self.stats['after_deduplication'] += 1
        
        # Step 4: Sentence splitting (if enabled)
        if self.config.split_sentences:
            sentences = self._split_into_sentences(cleaned_text)
            filtered_sentences = [
                s for s in sentences 
                if self.config.min_sentence_length <= len(s) <= self.config.max_sentence_length
            ]
            if filtered_sentences:
                cleaned_text = ' '.join(filtered_sentences)
            else:
                return None
        
        self.stats['final_count'] += 1
        return cleaned_text
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple rules."""
        # Simple sentence boundary detection
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def process_batch(self, texts: List[str], max_workers: int = None) -> List[str]:
        """
        Process a batch of texts in parallel.
        
        Args:
            texts: List of input texts
            max_workers: Number of parallel workers
            
        Returns:
            List of processed texts (filtered)
        """
        if max_workers is None:
            max_workers = mp.cpu_count()
        
        if max_workers == 1:
            # Single-threaded processing
            results = []
            for text in texts:
                processed = self.process_single_text(text)
                if processed is not None:
                    results.append(processed)
            return results
        
        # Multi-threaded processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.process_single_text, texts))
        
        # Filter out None results
        return [result for result in results if result is not None]
    
    def process_iterator(self, text_iterator: Iterator[str]) -> Iterator[str]:
        """
        Process texts from an iterator (memory-efficient for large datasets).
        
        Args:
            text_iterator: Iterator yielding input texts
            
        Yields:
            Processed texts
        """
        for text in text_iterator:
            processed = self.process_single_text(text)
            if processed is not None:
                yield processed
    
    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """Get preprocessing statistics."""
        total = self.stats['total_processed']
        if total == 0:
            return self.stats.copy()
        
        stats = self.stats.copy()
        stats['cleaning_retention_rate'] = self.stats['after_cleaning'] / total
        stats['quality_retention_rate'] = self.stats['after_quality_filter'] / total
        stats['dedup_retention_rate'] = self.stats['after_deduplication'] / total
        stats['overall_retention_rate'] = self.stats['final_count'] / total
        
        return stats
    
    def reset_statistics(self):
        """Reset preprocessing statistics."""
        for key in self.stats:
            self.stats[key] = 0
        if self.duplicate_detector:
            self.duplicate_detector.reset()


def create_preprocessing_pipeline(
    min_text_length: int = 50,
    max_text_length: int = 1000000,
    remove_duplicates: bool = True,
    quality_threshold: float = 0.5,
    **kwargs
) -> PreprocessingPipeline:
    """
    Create a preprocessing pipeline with common settings.
    
    Args:
        min_text_length: Minimum text length
        max_text_length: Maximum text length
        remove_duplicates: Whether to remove duplicates
        quality_threshold: Quality score threshold
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured preprocessing pipeline
    """
    config = PreprocessingConfig(
        min_text_length=min_text_length,
        max_text_length=max_text_length,
        remove_duplicates=remove_duplicates,
        **kwargs
    )
    
    return PreprocessingPipeline(config)