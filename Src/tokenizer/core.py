"""
Optimized core tokenizer implementation with performance improvements.

This module provides a high-performance BPE tokenizer with optimized algorithms,
memory-efficient data structures, and parallel processing capabilities.
"""

import re
import json
import pickle
import heapq
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Union, Iterator
from collections import defaultdict, Counter, deque
import unicodedata
from dataclasses import dataclass
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import mmap
import functools

# Configure logging
logger = logging.getLogger(__name__)


class RegexCache:
    """Global regex pattern cache to avoid recompilation."""
    _cache = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_pattern(cls, pattern_str: str) -> re.Pattern:
        if pattern_str not in cls._cache:
            with cls._lock:
                if pattern_str not in cls._cache:
                    cls._cache[pattern_str] = re.compile(pattern_str)
        return cls._cache[pattern_str]


class TrieNode:
    """Trie node for efficient vocabulary lookups."""
    __slots__ = ['children', 'token_id', 'is_leaf']
    
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.token_id: Optional[int] = None
        self.is_leaf: bool = False


class VocabularyTrie:
    """Memory-efficient trie structure for vocabulary storage."""
    
    def __init__(self):
        self.root = TrieNode()
        self.size = 0
    
    def insert(self, token: str, token_id: int):
        """Insert token into trie."""
        node = self.root
        for char in token:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.token_id = token_id
        node.is_leaf = True
        self.size += 1
    
    def search(self, token: str) -> Optional[int]:
        """Search for token and return its ID."""
        node = self.root
        for char in token:
            if char not in node.children:
                return None
            node = node.children[char]
        
        return node.token_id if node.is_leaf else None
    
    def longest_match(self, text: str, start: int = 0) -> Tuple[Optional[str], Optional[int], int]:
        """Find longest matching token starting from position."""
        node = self.root
        best_match = None
        best_id = None
        best_length = 0
        
        for i, char in enumerate(text[start:], start):
            if char not in node.children:
                break
            node = node.children[char]
            if node.is_leaf:
                best_match = text[start:i+1]
                best_id = node.token_id
                best_length = i + 1 - start
        
        return best_match, best_id, best_length


@dataclass
class OptimizedTokenizerConfig:
    """Optimized configuration with performance settings."""
    vocab_size: int = 50000
    min_frequency: int = 2
    special_tokens: Dict[str, int] = None
    regex_pattern: str = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    preserve_spaces: bool = True
    normalize_unicode: bool = True
    use_trie: bool = True
    chunk_size: int = 10000  # For batch processing
    max_workers: int = 4
    enable_caching: bool = True
    
    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = {
                "<|endoftext|>": 0,
                "<|startoftext|>": 1,
                "<|pad|>": 2,
                "<|unk|>": 3,
            }


class OptimizedBPETokenizer:
    """
    High-performance BPE tokenizer with optimized algorithms and data structures.
    """
    
    def __init__(self, config: Optional[OptimizedTokenizerConfig] = None):
        self.config = config or OptimizedTokenizerConfig()
        
        # Core data structures
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.vocab_trie: Optional[VocabularyTrie] = None
        
        # BPE merge data
        self.merges: List[Tuple[str, str]] = []
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
        
        # Compiled patterns
        self.regex = RegexCache.get_pattern(self.config.regex_pattern)
        
        # Performance optimizations
        self._compiled = False
        self._encode_cache: Dict[str, List[int]] = {} if self.config.enable_caching else None
        self._decode_cache: Dict[Tuple[int, ...], str] = {} if self.config.enable_caching else None
        
        # Initialize with special tokens
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        """Initialize vocabulary with special tokens."""
        for token, idx in self.config.special_tokens.items():
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token
    
    def _build_vocabulary_trie(self):
        """Build trie structure for fast vocabulary lookups."""
        if not self.config.use_trie:
            return
        
        self.vocab_trie = VocabularyTrie()
        for token, token_id in self.vocab.items():
            self.vocab_trie.insert(token, token_id)
    
    def _get_stats_optimized(self, word_freqs: Dict[Tuple[str, ...], int]) -> Counter:
        """Optimized statistics computation using vectorized operations."""
        pairs = Counter()
        
        # Pre-allocate for better performance
        for word, freq in word_freqs.items():
            word_len = len(word)
            if word_len < 2:
                continue
            
            # Vectorized pair counting
            for i in range(word_len - 1):
                pairs[(word[i], word[i + 1])] += freq
        
        return pairs
    
    def _merge_vocab_optimized(self, pair: Tuple[str, str], 
                              word_freqs: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        """Optimized vocabulary merging with minimal string operations."""
        new_word_freqs = {}
        merged_token = ''.join(pair)
        
        for word, freq in word_freqs.items():
            if len(word) < 2:
                new_word_freqs[word] = freq
                continue
            
            new_word = []
            i = 0
            while i < len(word):
                if (i < len(word) - 1 and 
                    word[i] == pair[0] and 
                    word[i + 1] == pair[1]):
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            new_word_freqs[tuple(new_word)] = freq
        
        return new_word_freqs
    
    def _normalize_text_batch(self, texts: List[str]) -> List[str]:
        """Batch text normalization for better performance."""
        if not self.config.normalize_unicode:
            return texts
        
        normalized = []
        for text in texts:
            if isinstance(text, str):
                normalized.append(unicodedata.normalize('NFC', text))
            else:
                normalized.append(str(text))
        
        return normalized
    
    def train_optimized(self, texts: List[str], verbose: bool = True) -> None:
        """
        Optimized training with parallel processing and memory efficiency.
        """
        if verbose:
            logger.info(f"Training optimized tokenizer on {len(texts)} texts...")
        
        start_time = time.time()
        
        # Batch normalize texts
        texts = self._normalize_text_batch(texts)
        
        # Extract words with parallel processing
        if self.config.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                word_lists = list(executor.map(self._extract_words, texts))
        else:
            word_lists = [self._extract_words(text) for text in texts]
        
        # Flatten and count words
        all_words = []
        for word_list in word_lists:
            all_words.extend(word_list)
        
        if verbose:
            logger.info(f"Extracted {len(all_words)} word tokens")
        
        # Count word frequencies with memory efficiency
        word_freqs = self._count_word_frequencies(all_words)
        
        # Filter by minimum frequency
        word_freqs = {word: freq for word, freq in word_freqs.items() 
                     if freq >= self.config.min_frequency}
        
        if verbose:
            logger.info(f"After filtering: {len(word_freqs)} unique words")
        
        # Convert to character tuples
        word_freqs = {tuple(word): freq for word, freq in word_freqs.items()}
        
        # Initialize vocabulary with characters
        self._initialize_character_vocab(word_freqs)
        
        if verbose:
            logger.info(f"Base vocabulary size: {len(self.vocab)}")
        
        # Learn BPE merges with optimized algorithm
        self._learn_bpe_merges_optimized(word_freqs, verbose)
        
        # Build optimized data structures
        self._build_vocabulary_trie()
        self._compiled = True
        
        training_time = time.time() - start_time
        if verbose:
            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Final vocabulary size: {len(self.vocab)}")
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract word tokens from text using compiled regex."""
        return self.regex.findall(text)
    
    def _count_word_frequencies(self, words: List[str]) -> Dict[str, int]:
        """Memory-efficient word frequency counting."""
        # Use Counter for efficient counting
        return dict(Counter(words))
    
    def _initialize_character_vocab(self, word_freqs: Dict[Tuple[str, ...], int]):
        """Initialize vocabulary with unique characters."""
        vocab_chars = set()
        for word in word_freqs:
            vocab_chars.update(word)
        
        # Add characters to vocabulary (after special tokens)
        next_id = max(self.config.special_tokens.values()) + 1
        for char in sorted(vocab_chars):
            if char not in self.vocab:
                self.vocab[char] = next_id
                self.inverse_vocab[next_id] = char
                next_id += 1
    
    def _learn_bpe_merges_optimized(self, word_freqs: Dict[Tuple[str, ...], int], verbose: bool):
        """Optimized BPE merge learning with early stopping."""
        target_vocab_size = self.config.vocab_size
        num_merges = target_vocab_size - len(self.vocab)
        next_id = len(self.vocab)
        
        # Pre-allocate merge storage
        self.merges = []
        
        for i in range(num_merges):
            # Get statistics with optimized counting
            pairs = self._get_stats_optimized(word_freqs)
            
            if not pairs:
                if verbose:
                    logger.info(f"No more pairs to merge at iteration {i}")
                break
            
            # Find best pair
            best_pair = pairs.most_common(1)[0][0]
            
            # Early stopping if frequency is too low
            if pairs[best_pair] < self.config.min_frequency:
                if verbose:
                    logger.info(f"Stopping early at iteration {i} - frequency too low")
                break
            
            # Merge vocabulary
            word_freqs = self._merge_vocab_optimized(best_pair, word_freqs)
            self.merges.append(best_pair)
            
            # Add merged token to vocabulary
            new_token = ''.join(best_pair)
            self.vocab[new_token] = next_id
            self.inverse_vocab[next_id] = new_token
            next_id += 1
            
            if verbose and (i + 1) % 1000 == 0:
                logger.info(f"Completed {i + 1}/{num_merges} merges")
        
        # Create merge ranks for fast lookup
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
    
    def _apply_bpe_optimized(self, word: str) -> List[str]:
        """Highly optimized BPE application using priority queue."""
        if not self._compiled:
            raise RuntimeError("Tokenizer must be trained before encoding")
        
        if len(word) <= 1:
            return [word] if word in self.vocab else [self._get_unk_token()]
        
        # Convert to list for in-place modifications
        word_tokens = list(word)
        
        if len(word_tokens) < 2:
            return word_tokens
        
        # Priority queue for merge operations: (rank, position, pair)
        merge_heap = []
        
        # Initialize heap with all valid pairs
        for i in range(len(word_tokens) - 1):
            pair = (word_tokens[i], word_tokens[i + 1])
            if pair in self.merge_ranks:
                heapq.heappush(merge_heap, (self.merge_ranks[pair], i, pair))
        
        # Track which positions are still valid
        valid_positions = set(range(len(word_tokens)))
        
        while merge_heap:
            rank, pos, (first, second) = heapq.heappop(merge_heap)
            
            # Check if this position is still valid for merging
            if (pos not in valid_positions or 
                pos + 1 not in valid_positions or
                pos >= len(word_tokens) - 1 or
                word_tokens[pos] != first or 
                word_tokens[pos + 1] != second):
                continue
            
            # Perform merge
            merged_token = first + second
            word_tokens[pos] = merged_token
            
            # Remove merged position from valid set
            valid_positions.remove(pos + 1)
            
            # Shift tokens left to fill gap
            for i in range(pos + 1, len(word_tokens) - 1):
                word_tokens[i] = word_tokens[i + 1]
                if i + 1 in valid_positions:
                    valid_positions.remove(i + 1)
                    valid_positions.add(i)
            
            word_tokens.pop()  # Remove last element
            
            # Update valid positions (shift everything after merge point)
            new_valid = set()
            for p in valid_positions:
                if p < pos:
                    new_valid.add(p)
                elif p > pos + 1:
                    new_valid.add(p - 1)
                elif p == pos:
                    new_valid.add(p)
            valid_positions = new_valid
            
            # Add new potential merges
            # Check left neighbor
            if pos > 0 and pos - 1 in valid_positions:
                left_pair = (word_tokens[pos - 1], merged_token)
                if left_pair in self.merge_ranks:
                    heapq.heappush(merge_heap, (self.merge_ranks[left_pair], pos - 1, left_pair))
            
            # Check right neighbor
            if pos < len(word_tokens) - 1 and pos + 1 in valid_positions:
                right_pair = (merged_token, word_tokens[pos + 1])
                if right_pair in self.merge_ranks:
                    heapq.heappush(merge_heap, (self.merge_ranks[right_pair], pos, right_pair))
        
        return word_tokens
    
    def _get_unk_token(self) -> str:
        """Get unknown token."""
        return list(self.config.special_tokens.keys())[3]  # <|unk|>
    
    @functools.lru_cache(maxsize=10000)
    def encode_cached(self, text: str) -> Tuple[int, ...]:
        """Cache-enabled encoding for frequently used texts."""
        return tuple(self.encode(text))
    
    def encode(self, text: str) -> List[int]:
        """
        Optimized text encoding with caching and batch processing.
        """
        if not self._compiled:
            raise RuntimeError("Tokenizer must be trained before encoding")
        
        # Check cache first
        if self._encode_cache and text in self._encode_cache:
            return self._encode_cache[text].copy()
        
        # Normalize text
        if self.config.normalize_unicode:
            text = unicodedata.normalize('NFC', text)
        
        # Split into word tokens
        word_tokens = self.regex.findall(text)
        
        # Apply BPE to each word token
        token_ids = []
        for word in word_tokens:
            if self.config.use_trie and self.vocab_trie:
                # Use trie for fast lookup
                bpe_tokens = self._apply_bpe_with_trie(word)
            else:
                # Use standard BPE
                bpe_tokens = self._apply_bpe_optimized(word)
            
            for token in bpe_tokens:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    # Use unknown token
                    unk_id = self.config.special_tokens[self._get_unk_token()]
                    token_ids.append(unk_id)
        
        # Cache result
        if self._encode_cache and len(self._encode_cache) < 10000:
            self._encode_cache[text] = token_ids.copy()
        
        return token_ids
    
    def _apply_bpe_with_trie(self, word: str) -> List[str]:
        """Apply BPE using trie for faster vocabulary lookups."""
        # First try direct lookup in trie
        if self.vocab_trie:
            match, token_id, length = self.vocab_trie.longest_match(word)
            if match and length == len(word):
                return [match]
        
        # Fall back to standard BPE
        return self._apply_bpe_optimized(word)
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Optimized batch encoding with parallel processing."""
        if self.config.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                return list(executor.map(self.encode, texts))
        else:
            return [self.encode(text) for text in texts]
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Optimized decoding with caching.
        """
        if not self._compiled:
            raise RuntimeError("Tokenizer must be trained before decoding")
        
        # Check cache
        cache_key = tuple(token_ids)
        if self._decode_cache and cache_key in self._decode_cache:
            return self._decode_cache[cache_key]
        
        # Decode tokens
        tokens = []
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                tokens.append(self.inverse_vocab[token_id])
            else:
                tokens.append(self._get_unk_token())
        
        result = ''.join(tokens)
        
        # Cache result
        if self._decode_cache and len(self._decode_cache) < 10000:
            self._decode_cache[cache_key] = result
        
        return result
    
    def decode_batch(self, token_ids_batch: List[List[int]]) -> List[str]:
        """Optimized batch decoding."""
        if self.config.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                return list(executor.map(self.decode, token_ids_batch))
        else:
            return [self.decode(token_ids) for token_ids in token_ids_batch]
    
    def save_optimized(self, filepath: str) -> None:
        """Save tokenizer with optimized serialization."""
        if not self._compiled:
            raise RuntimeError("Cannot save untrained tokenizer")
        
        # Prepare data for serialization
        data = {
            'config': {
                'vocab_size': self.config.vocab_size,
                'min_frequency': self.config.min_frequency,
                'special_tokens': self.config.special_tokens,
                'regex_pattern': self.config.regex_pattern,
                'preserve_spaces': self.config.preserve_spaces,
                'normalize_unicode': self.config.normalize_unicode,
                'use_trie': self.config.use_trie,
                'chunk_size': self.config.chunk_size,
                'max_workers': self.config.max_workers,
                'enable_caching': self.config.enable_caching,
            },
            'vocab': self.vocab,
            'merges': self.merges,
            'version': '2.0',  # Optimized version
        }
        
        if filepath.endswith('.json'):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Optimized tokenizer saved to {filepath}")
    
    @classmethod
    def load_optimized(cls, filepath: str) -> 'OptimizedBPETokenizer':
        """Load optimized tokenizer with performance improvements."""
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        
        # Handle version compatibility
        if data.get('version') == '2.0':
            config = OptimizedTokenizerConfig(**data['config'])
        else:
            # Convert from old format
            config = OptimizedTokenizerConfig(
                vocab_size=data['config']['vocab_size'],
                min_frequency=data['config']['min_frequency'],
                special_tokens=data['config']['special_tokens'],
                regex_pattern=data['config']['regex_pattern'],
            )
        
        # Create tokenizer instance
        tokenizer = cls(config)
        tokenizer.vocab = data['vocab']
        tokenizer.merges = [tuple(merge) for merge in data['merges']]
        
        # Rebuild optimized structures
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.merge_ranks = {pair: i for i, pair in enumerate(tokenizer.merges)}
        tokenizer._build_vocabulary_trie()
        tokenizer._compiled = True
        
        logger.info(f"Optimized tokenizer loaded from {filepath}")
        return tokenizer
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary dictionary."""
        return self.vocab.copy()
    
    def token_to_id(self, token: str) -> Optional[int]:
        """Convert token to ID."""
        return self.vocab.get(token)
    
    def id_to_token(self, token_id: int) -> Optional[str]:
        """Convert ID to token."""
        return self.inverse_vocab.get(token_id)
    
    def clear_cache(self):
        """Clear encoding/decoding caches."""
        if self._encode_cache:
            self._encode_cache.clear()
        if self._decode_cache:
            self._decode_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'encode_cache_size': len(self._encode_cache) if self._encode_cache else 0,
            'decode_cache_size': len(self._decode_cache) if self._decode_cache else 0,
        }


def create_optimized_tokenizer(vocab_size: int = 50000, **kwargs) -> OptimizedBPETokenizer:
    """Create an optimized tokenizer with custom configuration."""
    config = OptimizedTokenizerConfig(vocab_size=vocab_size, **kwargs)
    return OptimizedBPETokenizer(config)


# Performance benchmarking utilities
class TokenizerBenchmark:
    """Utility class for benchmarking tokenizer performance."""
    
    @staticmethod
    def benchmark_encoding(tokenizer, texts: List[str], iterations: int = 1) -> Dict[str, float]:
        """Benchmark encoding performance."""
        start_time = time.time()
        
        for _ in range(iterations):
            for text in texts:
                tokenizer.encode(text)
        
        total_time = time.time() - start_time
        total_chars = sum(len(text) for text in texts) * iterations
        total_texts = len(texts) * iterations
        
        return {
            'total_time': total_time,
            'texts_per_second': total_texts / total_time,
            'chars_per_second': total_chars / total_time,
            'avg_time_per_text': total_time / total_texts,
        }
    
    @staticmethod
    def benchmark_training(tokenizer, texts: List[str]) -> Dict[str, float]:
        """Benchmark training performance."""
        start_time = time.time()
        tokenizer.train_optimized(texts, verbose=False)
        training_time = time.time() - start_time
        
        total_chars = sum(len(text) for text in texts)
        
        return {
            'training_time': training_time,
            'chars_per_second': total_chars / training_time,
            'texts_per_second': len(texts) / training_time,
            'vocab_size': tokenizer.get_vocab_size(),
        }