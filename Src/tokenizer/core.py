"""
Core tokenizer implementation with BPE (Byte Pair Encoding) support.

This module provides the main Tokenizer class that handles encoding and decoding
of text using learned BPE merges and a vocabulary.
"""

import re
import json
import pickle
from typing import Dict, List, Tuple, Optional, Set, Union, Iterator
from collections import defaultdict, Counter
import unicodedata
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TokenizerConfig:
    """Configuration class for tokenizer settings."""
    vocab_size: int = 50000
    min_frequency: int = 2
    special_tokens: Dict[str, int] = None
    regex_pattern: str = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    preserve_spaces: bool = True
    normalize_unicode: bool = True
    
    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = {
                "<|endoftext|>": 0,
                "<|startoftext|>": 1,
                "<|pad|>": 2,
                "<|unk|>": 3,
            }


class BPETokenizer:
    """
    Byte Pair Encoding (BPE) Tokenizer implementation.
    
    This tokenizer learns merge rules from training data and can encode/decode
    text efficiently using the learned vocabulary.
    """
    
    def __init__(self, config: Optional[TokenizerConfig] = None):
        """Initialize the tokenizer with given configuration."""
        self.config = config or TokenizerConfig()
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
        self.regex = re.compile(self.config.regex_pattern)
        self._compiled = False
        
        # Initialize with special tokens
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        """Initialize vocabulary with special tokens."""
        for token, idx in self.config.special_tokens.items():
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token
    
    def _get_stats(self, word_freqs: Dict[Tuple[str, ...], int]) -> Counter:
        """Get statistics of adjacent symbol pairs in word frequencies."""
        pairs = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs
    
    def _merge_vocab(self, pair: Tuple[str, str], word_freqs: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        """Merge the most frequent pair in the vocabulary."""
        new_word_freqs = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        for word in word_freqs:
            new_word = p.sub(''.join(pair), ' '.join(word))
            new_word_freqs[tuple(new_word.split())] = word_freqs[word]
        
        return new_word_freqs
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text according to configuration."""
        if self.config.normalize_unicode:
            text = unicodedata.normalize('NFC', text)
        return text
    
    def _get_word_tokens(self, text: str) -> List[str]:
        """Split text into word tokens using regex pattern."""
        return self.regex.findall(text)
    
    def train(self, texts: List[str], verbose: bool = True) -> None:
        """
        Train the tokenizer on a list of texts.
        
        Args:
            texts: List of training texts
            verbose: Whether to print training progress
        """
        if verbose:
            logger.info(f"Training tokenizer on {len(texts)} texts...")
        
        # Normalize and tokenize all texts
        all_words = []
        for text in texts:
            normalized_text = self._normalize_text(text)
            word_tokens = self._get_word_tokens(normalized_text)
            all_words.extend(word_tokens)
        
        if verbose:
            logger.info(f"Found {len(all_words)} word tokens")
        
        # Count word frequencies
        word_freqs = Counter(all_words)
        
        # Filter by minimum frequency
        word_freqs = {word: freq for word, freq in word_freqs.items() 
                     if freq >= self.config.min_frequency}
        
        if verbose:
            logger.info(f"After filtering: {len(word_freqs)} unique words")
        
        # Convert words to character sequences
        word_freqs = {tuple(word): freq for word, freq in word_freqs.items()}
        
        # Initialize vocabulary with characters
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
        
        if verbose:
            logger.info(f"Base vocabulary size: {len(self.vocab)}")
        
        # Learn BPE merges
        target_vocab_size = self.config.vocab_size
        num_merges = target_vocab_size - len(self.vocab)
        
        for i in range(num_merges):
            pairs = self._get_stats(word_freqs)
            if not pairs:
                if verbose:
                    logger.info(f"No more pairs to merge at iteration {i}")
                break
            
            best_pair = pairs.most_common(1)[0][0]
            word_freqs = self._merge_vocab(best_pair, word_freqs)
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
        self._compiled = True
        
        if verbose:
            logger.info(f"Training complete. Final vocabulary size: {len(self.vocab)}")
    
    def _apply_bpe(self, word: str) -> List[str]:
        """Apply BPE merges to a word."""
        if not self._compiled:
            raise RuntimeError("Tokenizer must be trained before encoding")
        
        if len(word) <= 1:
            return [word] if word in self.vocab else [self.config.special_tokens.get("<|unk|>", "<|unk|>")]
        
        # Convert word to list of characters
        word_tokens = list(word)
        
        while len(word_tokens) > 1:
            # Find all possible pairs
            pairs = []
            for i in range(len(word_tokens) - 1):
                pair = (word_tokens[i], word_tokens[i + 1])
                if pair in self.merge_ranks:
                    pairs.append((self.merge_ranks[pair], i, pair))
            
            if not pairs:
                break
            
            # Get the pair with lowest rank (highest priority)
            pairs.sort()
            _, pos, (first, second) = pairs[0]
            
            # Merge the pair
            new_word_tokens = []
            i = 0
            while i < len(word_tokens):
                if i == pos:
                    new_word_tokens.append(first + second)
                    i += 2
                else:
                    new_word_tokens.append(word_tokens[i])
                    i += 1
            
            word_tokens = new_word_tokens
        
        return word_tokens
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: Input text to encode
            
        Returns:
            List of token IDs
        """
        if not self._compiled:
            raise RuntimeError("Tokenizer must be trained before encoding")
        
        # Normalize text
        text = self._normalize_text(text)
        
        # Split into word tokens
        word_tokens = self._get_word_tokens(text)
        
        # Apply BPE to each word token
        token_ids = []
        for word in word_tokens:
            bpe_tokens = self._apply_bpe(word)
            for token in bpe_tokens:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    # Use unknown token
                    unk_token = list(self.config.special_tokens.keys())[3]  # <|unk|>
                    token_ids.append(self.config.special_tokens[unk_token])
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs to decode
            
        Returns:
            Decoded text string
        """
        if not self._compiled:
            raise RuntimeError("Tokenizer must be trained before decoding")
        
        tokens = []
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                tokens.append(self.inverse_vocab[token_id])
            else:
                # Use unknown token
                unk_token = list(self.config.special_tokens.keys())[3]  # <|unk|>
                tokens.append(unk_token)
        
        return ''.join(tokens)
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode a batch of texts."""
        return [self.encode(text) for text in texts]
    
    def decode_batch(self, token_ids_batch: List[List[int]]) -> List[str]:
        """Decode a batch of token ID sequences."""
        return [self.decode(token_ids) for token_ids in token_ids_batch]
    
    def save(self, filepath: str) -> None:
        """Save the trained tokenizer to a file."""
        if not self._compiled:
            raise RuntimeError("Cannot save untrained tokenizer")
        
        data = {
            'config': {
                'vocab_size': self.config.vocab_size,
                'min_frequency': self.config.min_frequency,
                'special_tokens': self.config.special_tokens,
                'regex_pattern': self.config.regex_pattern,
                'preserve_spaces': self.config.preserve_spaces,
                'normalize_unicode': self.config.normalize_unicode,
            },
            'vocab': self.vocab,
            'merges': self.merges,
        }
        
        if filepath.endswith('.json'):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        
        logger.info(f"Tokenizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BPETokenizer':
        """Load a trained tokenizer from a file."""
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        
        # Reconstruct config
        config = TokenizerConfig(**data['config'])
        
        # Create tokenizer instance
        tokenizer = cls(config)
        tokenizer.vocab = data['vocab']
        tokenizer.merges = [tuple(merge) for merge in data['merges']]
        
        # Rebuild inverse vocab and merge ranks
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.merge_ranks = {pair: i for i, pair in enumerate(tokenizer.merges)}
        tokenizer._compiled = True
        
        logger.info(f"Tokenizer loaded from {filepath}")
        return tokenizer
    
    def get_vocab_size(self) -> int:
        """Get the size of the vocabulary."""
        return len(self.vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        """Get a copy of the vocabulary."""
        return self.vocab.copy()
    
    def token_to_id(self, token: str) -> Optional[int]:
        """Convert a token to its ID."""
        return self.vocab.get(token)
    
    def id_to_token(self, token_id: int) -> Optional[str]:
        """Convert a token ID to its token."""
        return self.inverse_vocab.get(token_id)


def create_tokenizer(vocab_size: int = 50000, **kwargs) -> BPETokenizer:
    """
    Convenience function to create a tokenizer with custom vocabulary size.
    
    Args:
        vocab_size: Target vocabulary size
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured BPETokenizer instance
    """
    config = TokenizerConfig(vocab_size=vocab_size, **kwargs)
    return BPETokenizer(config)