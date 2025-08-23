"""
Utility functions for tokenizer operations and text processing.

This module provides helper functions for data preprocessing, text cleaning,
vocabulary analysis, and other common tokenizer operations.
"""

import re
import json
import gzip
import os
import sys
from typing import List, Dict, Tuple, Iterator, Optional, Union, Set
from collections import Counter, defaultdict
import unicodedata
import logging
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Text preprocessing utilities for tokenizer training and inference.
    """
    
    def __init__(self, 
                 normalize_unicode: bool = True,
                 strip_accents: bool = False,
                 lowercase: bool = False,
                 remove_control_chars: bool = True):
        """
        Initialize text processor with configuration.
        
        Args:
            normalize_unicode: Whether to apply Unicode normalization
            strip_accents: Whether to remove accent marks
            lowercase: Whether to convert to lowercase
            remove_control_chars: Whether to remove control characters
        """
        self.normalize_unicode = normalize_unicode
        self.strip_accents = strip_accents
        self.lowercase = lowercase
        self.remove_control_chars = remove_control_chars
        
        # Compile regex patterns
        self.control_char_re = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]')
        self.whitespace_re = re.compile(r'\s+')
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text according to processor configuration.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Unicode normalization
        if self.normalize_unicode:
            text = unicodedata.normalize('NFC', text)
        
        # Remove control characters
        if self.remove_control_chars:
            text = self.control_char_re.sub('', text)
        
        # Strip accents
        if self.strip_accents:
            text = ''.join(c for c in unicodedata.normalize('NFD', text)
                          if unicodedata.category(c) != 'Mn')
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Normalize whitespace
        text = self.whitespace_re.sub(' ', text).strip()
        
        return text
    
    def split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using simple heuristics.
        
        Args:
            text: Input text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be improved with spacy/nltk)
        sentence_endings = re.compile(r'[.!?]+\s+')
        sentences = sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def filter_by_length(self, texts: List[str], 
                        min_length: int = 10, 
                        max_length: int = 10000) -> List[str]:
        """
        Filter texts by length criteria.
        
        Args:
            texts: List of texts to filter
            min_length: Minimum text length
            max_length: Maximum text length
            
        Returns:
            Filtered list of texts
        """
        return [text for text in texts 
                if min_length <= len(text) <= max_length]
    
    def deduplicate(self, texts: List[str]) -> List[str]:
        """
        Remove duplicate texts while preserving order.
        
        Args:
            texts: List of texts to deduplicate
            
        Returns:
            Deduplicated list of texts
        """
        seen = set()
        result = []
        for text in texts:
            if text not in seen:
                seen.add(text)
                result.append(text)
        return result


class DataLoader:
    """
    Utilities for loading training data from various sources.
    """
    
    @staticmethod
    def load_text_file(filepath: str, encoding: str = 'utf-8') -> str:
        """Load text from a single file."""
        try:
            if filepath.endswith('.gz'):
                with gzip.open(filepath, 'rt', encoding=encoding) as f:
                    return f.read()
            else:
                with open(filepath, 'r', encoding=encoding) as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return ""
    
    @staticmethod
    def load_text_files(filepaths: List[str], 
                       encoding: str = 'utf-8',
                       max_workers: int = 4) -> List[str]:
        """
        Load text from multiple files in parallel.
        
        Args:
            filepaths: List of file paths to load
            encoding: Text encoding
            max_workers: Number of parallel workers
            
        Returns:
            List of loaded texts
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            texts = list(executor.map(
                lambda fp: DataLoader.load_text_file(fp, encoding), 
                filepaths
            ))
        return [text for text in texts if text]  # Filter empty texts
    
    @staticmethod
    def load_directory(directory: str, 
                      pattern: str = "*.txt",
                      recursive: bool = True,
                      encoding: str = 'utf-8') -> List[str]:
        """
        Load all text files from a directory.
        
        Args:
            directory: Directory path
            pattern: File pattern to match
            recursive: Whether to search recursively
            encoding: Text encoding
            
        Returns:
            List of loaded texts
        """
        path = Path(directory)
        if recursive:
            filepaths = list(path.rglob(pattern))
        else:
            filepaths = list(path.glob(pattern))
        
        filepaths = [str(fp) for fp in filepaths]
        logger.info(f"Found {len(filepaths)} files in {directory}")
        
        return DataLoader.load_text_files(filepaths, encoding)
    
    @staticmethod
    def load_jsonl(filepath: str, 
                   text_field: str = 'text',
                   encoding: str = 'utf-8') -> List[str]:
        """
        Load texts from a JSONL file.
        
        Args:
            filepath: Path to JSONL file
            text_field: Field name containing text
            encoding: Text encoding
            
        Returns:
            List of extracted texts
        """
        texts = []
        try:
            open_fn = gzip.open if filepath.endswith('.gz') else open
            with open_fn(filepath, 'rt', encoding=encoding) as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if text_field in data:
                            texts.append(data[text_field])
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error loading JSONL {filepath}: {e}")
        
        return texts


class VocabAnalyzer:
    """
    Utilities for analyzing vocabulary and token statistics.
    """
    
    def __init__(self, tokenizer):
        """Initialize with a trained tokenizer."""
        self.tokenizer = tokenizer
    
    def analyze_text(self, text: str) -> Dict[str, Union[int, float, List]]:
        """
        Analyze tokenization statistics for a text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        tokens = self.tokenizer.encode(text)
        
        # Basic statistics
        stats = {
            'text_length': len(text),
            'num_tokens': len(tokens),
            'compression_ratio': len(text) / len(tokens) if tokens else 0,
            'unique_tokens': len(set(tokens)),
            'token_ids': tokens[:100],  # First 100 tokens
        }
        
        # Token frequency analysis
        token_counts = Counter(tokens)
        stats['most_common_tokens'] = [
            (self.tokenizer.id_to_token(tid), count) 
            for tid, count in token_counts.most_common(10)
        ]
        
        return stats
    
    def analyze_vocab_coverage(self, texts: List[str]) -> Dict[str, Union[int, float]]:
        """
        Analyze vocabulary coverage over a set of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Coverage statistics
        """
        all_tokens = []
        unk_count = 0
        unk_token_id = self.tokenizer.token_to_id("<|unk|>")
        
        for text in texts:
            tokens = self.tokenizer.encode(text)
            all_tokens.extend(tokens)
            if unk_token_id is not None:
                unk_count += tokens.count(unk_token_id)
        
        total_tokens = len(all_tokens)
        unique_tokens = len(set(all_tokens))
        
        return {
            'total_tokens': total_tokens,
            'unique_tokens_used': unique_tokens,
            'vocab_utilization': unique_tokens / self.tokenizer.get_vocab_size(),
            'unknown_rate': unk_count / total_tokens if total_tokens > 0 else 0,
            'coverage_rate': 1 - (unk_count / total_tokens) if total_tokens > 0 else 1,
        }
    
    def find_rare_tokens(self, texts: List[str], threshold: int = 5) -> List[Tuple[str, int]]:
        """
        Find tokens that appear rarely in the given texts.
        
        Args:
            texts: List of texts to analyze
            threshold: Minimum frequency threshold
            
        Returns:
            List of (token, count) pairs for rare tokens
        """
        all_tokens = []
        for text in texts:
            all_tokens.extend(self.tokenizer.encode(text))
        
        token_counts = Counter(all_tokens)
        rare_tokens = [
            (self.tokenizer.id_to_token(tid), count)
            for tid, count in token_counts.items()
            if count < threshold and self.tokenizer.id_to_token(tid) is not None
        ]
        
        return sorted(rare_tokens, key=lambda x: x[1])


class BatchProcessor:
    """
    Utilities for processing large datasets in batches.
    """
    
    def __init__(self, batch_size: int = 1000, max_workers: int = None):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Size of each processing batch
            max_workers: Number of parallel workers
        """
        self.batch_size = batch_size
        self.max_workers = max_workers or mp.cpu_count()
    
    def process_texts_parallel(self, 
                             texts: List[str], 
                             process_fn: callable,
                             use_processes: bool = False) -> List:
        """
        Process texts in parallel batches.
        
        Args:
            texts: List of texts to process
            process_fn: Function to apply to each text
            use_processes: Whether to use processes (True) or threads (False)
            
        Returns:
            List of processed results
        """
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            results = list(executor.map(process_fn, texts))
        
        return results
    
    def batch_generator(self, data: List, batch_size: int = None) -> Iterator[List]:
        """
        Generate batches from data.
        
        Args:
            data: Input data list
            batch_size: Size of each batch (defaults to instance batch_size)
            
        Yields:
            Batches of data
        """
        batch_size = batch_size or self.batch_size
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]


def compute_token_stats(texts: List[str], tokenizer) -> Dict[str, float]:
    """
    Compute comprehensive tokenization statistics.
    
    Args:
        texts: List of texts to analyze
        tokenizer: Trained tokenizer instance
        
    Returns:
        Dictionary of statistics
    """
    total_chars = sum(len(text) for text in texts)
    all_tokens = []
    
    for text in texts:
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
    
    total_tokens = len(all_tokens)
    unique_tokens = len(set(all_tokens))
    
    # Compute token length distribution
    token_lengths = [len(tokenizer.id_to_token(tid) or "") for tid in all_tokens]
    avg_token_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0
    
    return {
        'total_texts': len(texts),
        'total_characters': total_chars,
        'total_tokens': total_tokens,
        'unique_tokens': unique_tokens,
        'avg_chars_per_text': total_chars / len(texts) if texts else 0,
        'avg_tokens_per_text': total_tokens / len(texts) if texts else 0,
        'compression_ratio': total_chars / total_tokens if total_tokens > 0 else 0,
        'vocab_utilization': unique_tokens / tokenizer.get_vocab_size(),
        'avg_token_length': avg_token_length,
    }


def save_vocab_to_file(tokenizer, filepath: str, format: str = 'json'):
    """
    Save tokenizer vocabulary to a file.
    
    Args:
        tokenizer: Trained tokenizer instance
        filepath: Output file path
        format: Output format ('json' or 'txt')
    """
    vocab = tokenizer.get_vocab()
    
    if format == 'json':
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
    elif format == 'txt':
        with open(filepath, 'w', encoding='utf-8') as f:
            for token, token_id in sorted(vocab.items(), key=lambda x: x[1]):
                f.write(f"{token_id}\t{repr(token)}\n")
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Vocabulary saved to {filepath}")


def merge_tokenizers(tokenizers: List, vocab_size: int = None) -> 'BPETokenizer':
    """
    Merge multiple trained tokenizers into a single tokenizer.
    
    Args:
        tokenizers: List of trained tokenizer instances
        vocab_size: Target vocabulary size for merged tokenizer
        
    Returns:
        New merged tokenizer instance
    """
    from .core import BPETokenizer, TokenizerConfig
    
    if not tokenizers:
        raise ValueError("Cannot merge empty list of tokenizers")
    
    # Collect all vocabularies and merge rules
    all_vocab = {}
    all_merges = []
    
    for tokenizer in tokenizers:
        all_vocab.update(tokenizer.get_vocab())
        all_merges.extend(tokenizer.merges)
    
    # Create new tokenizer configuration
    config = TokenizerConfig(vocab_size=vocab_size or len(all_vocab))
    merged_tokenizer = BPETokenizer(config)
    
    # Set combined vocabulary and merges
    merged_tokenizer.vocab = all_vocab
    merged_tokenizer.inverse_vocab = {v: k for k, v in all_vocab.items()}
    merged_tokenizer.merges = all_merges
    merged_tokenizer.merge_ranks = {pair: i for i, pair in enumerate(all_merges)}
    merged_tokenizer._compiled = True
    
    logger.info(f"Merged {len(tokenizers)} tokenizers into vocabulary of size {len(all_vocab)}")
    
    return merged_tokenizer


def validate_tokenizer(tokenizer, test_texts: List[str] = None) -> Dict[str, bool]:
    """
    Validate a trained tokenizer for correctness.
    
    Args:
        tokenizer: Trained tokenizer instance
        test_texts: Optional list of test texts
        
    Returns:
        Dictionary of validation results
    """
    results = {
        'has_vocab': bool(tokenizer.vocab),
        'has_inverse_vocab': bool(tokenizer.inverse_vocab),
        'has_merges': bool(tokenizer.merges),
        'vocab_consistency': False,
        'encode_decode_consistency': False,
        'special_tokens_present': False,
    }
    
    # Check vocabulary consistency
    if tokenizer.vocab and tokenizer.inverse_vocab:
        vocab_consistent = (
            len(tokenizer.vocab) == len(tokenizer.inverse_vocab) and
            all(tokenizer.inverse_vocab.get(v) == k for k, v in tokenizer.vocab.items())
        )
        results['vocab_consistency'] = vocab_consistent
    
    # Check special tokens
    special_tokens = ["<|endoftext|>", "<|startoftext|>", "<|pad|>", "<|unk|>"]
    results['special_tokens_present'] = all(
        token in tokenizer.vocab for token in special_tokens
    )
    
    # Test encode/decode consistency
    if test_texts is None:
        test_texts = [
            "Hello, world!",
            "This is a test sentence with numbers 123 and symbols @#$.",
            "Unicode test: héllo wörld 🌍",
            "",  # Empty string
            "a",  # Single character
        ]
    
    encode_decode_consistent = True
    for text in test_texts:
        try:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            # Allow for some normalization differences
            if not (decoded == text or decoded.strip() == text.strip()):
                encode_decode_consistent = False
                logger.warning(f"Encode/decode mismatch: '{text}' -> '{decoded}'")
        except Exception as e:
            logger.error(f"Error in encode/decode for '{text}': {e}")
            encode_decode_consistent = False
    
    results['encode_decode_consistency'] = encode_decode_consistent
    
    return results