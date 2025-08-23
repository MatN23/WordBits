"""
Model registry and pretrained tokenizer management.

This module provides a registry system for managing different tokenizer configurations
and downloading/loading pretrained tokenizers for various models.
"""

import os
import json
import pickle
import hashlib
import urllib.request
import urllib.parse
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
import threading

from .core import BPETokenizer, TokenizerConfig, create_tokenizer

logger = logging.getLogger(__name__)


@dataclass
class TokenizerMetadata:
    """Metadata for a registered tokenizer."""
    name: str
    description: str
    vocab_size: int
    language: str = "en"
    domain: str = "general"
    model_family: Optional[str] = None
    version: str = "1.0"
    file_url: Optional[str] = None
    file_hash: Optional[str] = None
    file_size: Optional[int] = None
    config: Optional[Dict[str, Any]] = None
    created_by: str = "unknown"
    license: str = "unknown"
    tags: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class TokenizerRegistry:
    """
    Registry for managing tokenizer configurations and pretrained models.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the tokenizer registry.
        
        Args:
            cache_dir: Directory for caching downloaded tokenizers
        """
        self.cache_dir = Path(cache_dir or os.path.expanduser("~/.tokenizer_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._registry: Dict[str, TokenizerMetadata] = {}
        self._lock = threading.Lock()
        
        # Initialize with built-in configurations
        self._register_builtin_tokenizers()
    
    def _register_builtin_tokenizers(self):
        """Register built-in tokenizer configurations."""
        
        # GPT-style tokenizers
        self.register(TokenizerMetadata(
            name="gpt2-small",
            description="GPT-2 small model tokenizer (117M parameters)",
            vocab_size=50257,
            model_family="gpt2",
            domain="general",
            config={
                "vocab_size": 50257,
                "special_tokens": {
                    "<|endoftext|>": 50256,
                    "<|startoftext|>": 50255,
                    "<|pad|>": 50254,
                    "<|unk|>": 50253,
                },
                "regex_pattern": r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
            },
            tags=["gpt", "autoregressive", "english"]
        ))
        
        self.register(TokenizerMetadata(
            name="gpt2-medium",
            description="GPT-2 medium model tokenizer (345M parameters)",
            vocab_size=50257,
            model_family="gpt2",
            domain="general",
            config={
                "vocab_size": 50257,
                "special_tokens": {
                    "<|endoftext|>": 50256,
                    "<|startoftext|>": 50255,  
                    "<|pad|>": 50254,
                    "<|unk|>": 50253,
                },
            },
            tags=["gpt", "autoregressive", "english"]
        ))
        
        # Code-specific tokenizers
        self.register(TokenizerMetadata(
            name="code-tokenizer",
            description="Tokenizer optimized for source code",
            vocab_size=32000,
            domain="code",
            config={
                "vocab_size": 32000,
                "special_tokens": {
                    "<|endoftext|>": 0,
                    "<|startoftext|>": 1,
                    "<|pad|>": 2,
                    "<|unk|>": 3,
                    "<|indent|>": 4,
                    "<|dedent|>": 5,
                    "<|newline|>": 6,
                },
                "regex_pattern": r"""[ \t]+|[a-zA-Z_][a-zA-Z0-9_]*|\d+\.?\d*|[^\w\s]|\n"""
            },
            tags=["code", "programming", "software"]
        ))
        
        # Multilingual tokenizers
        self.register(TokenizerMetadata(
            name="multilingual-base",
            description="Base multilingual tokenizer",
            vocab_size=64000,
            language="multilingual",
            domain="general",
            config={
                "vocab_size": 64000,
                "special_tokens": {
                    "<|endoftext|>": 0,
                    "<|startoftext|>": 1,
                    "<|pad|>": 2,
                    "<|unk|>": 3,
                },
                "normalize_unicode": True,
            },
            tags=["multilingual", "general"]
        ))
        
        # Domain-specific tokenizers
        self.register(TokenizerMetadata(
            name="scientific-tokenizer",
            description="Tokenizer optimized for scientific text",
            vocab_size=40000,
            domain="scientific",
            config={
                "vocab_size": 40000,
                "special_tokens": {
                    "<|endoftext|>": 0,
                    "<|startoftext|>": 1,
                    "<|pad|>": 2,
                    "<|unk|>": 3,
                    "<|formula|>": 4,
                    "<|citation|>": 5,
                },
            },
            tags=["scientific", "academic", "research"]
        ))
    
    def register(self, metadata: TokenizerMetadata):
        """
        Register a new tokenizer configuration.
        
        Args:
            metadata: Tokenizer metadata to register
        """
        with self._lock:
            self._registry[metadata.name] = metadata
        logger.info(f"Registered tokenizer: {metadata.name}")
    
    def unregister(self, name: str):
        """
        Unregister a tokenizer configuration.
        
        Args:
            name: Name of tokenizer to unregister
        """
        with self._lock:
            if name in self._registry:
                del self._registry[name]
                logger.info(f"Unregistered tokenizer: {name}")
    
    def list_tokenizers(self, 
                       domain: Optional[str] = None,
                       language: Optional[str] = None,
                       model_family: Optional[str] = None,
                       tags: Optional[List[str]] = None) -> List[str]:
        """
        List available tokenizers with optional filtering.
        
        Args:
            domain: Filter by domain
            language: Filter by language
            model_family: Filter by model family
            tags: Filter by tags (must have all specified tags)
            
        Returns:
            List of matching tokenizer names
        """
        results = []
        
        for name, metadata in self._registry.items():
            # Apply filters
            if domain and metadata.domain != domain:
                continue
            if language and metadata.language != language:
                continue
            if model_family and metadata.model_family != model_family:
                continue
            if tags and not all(tag in metadata.tags for tag in tags):
                continue
            
            results.append(name)
        
        return sorted(results)
    
    def get_metadata(self, name: str) -> Optional[TokenizerMetadata]:
        """
        Get metadata for a registered tokenizer.
        
        Args:
            name: Name of the tokenizer
            
        Returns:
            Tokenizer metadata or None if not found
        """
        return self._registry.get(name)
    
    def create_tokenizer(self, name: str, **override_config) -> BPETokenizer:
        """
        Create a tokenizer instance from registered configuration.
        
        Args:
            name: Name of the registered tokenizer
            **override_config: Configuration parameters to override
            
        Returns:
            Configured tokenizer instance
        """
        metadata = self.get_metadata(name)
        if not metadata:
            raise ValueError(f"Unknown tokenizer: {name}")
        
        # Merge configuration
        config_dict = metadata.config.copy() if metadata.config else {}
        config_dict.update(override_config)
        
        # Create tokenizer config
        config = TokenizerConfig(**config_dict)
        
        return BPETokenizer(config)
    
    def download_tokenizer(self, name: str, force_download: bool = False) -> Path:
        """
        Download a pretrained tokenizer to cache.
        
        Args:
            name: Name of the tokenizer to download
            force_download: Whether to force re-download if cached
            
        Returns:
            Path to the downloaded tokenizer file
        """
        metadata = self.get_metadata(name)
        if not metadata or not metadata.file_url:
            raise ValueError(f"No download URL available for tokenizer: {name}")
        
        # Check cache
        cache_path = self.cache_dir / f"{name}.pkl"
        if cache_path.exists() and not force_download:
            # Verify hash if available
            if metadata.file_hash and not self._verify_file_hash(cache_path, metadata.file_hash):
                logger.warning(f"Hash mismatch for cached {name}, re-downloading...")
            else:
                logger.info(f"Using cached tokenizer: {cache_path}")
                return cache_path
        
        # Download tokenizer
        logger.info(f"Downloading tokenizer {name} from {metadata.file_url}")
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            try:
                urllib.request.urlretrieve(metadata.file_url, tmp_file.name)
                
                # Verify hash
                if metadata.file_hash and not self._verify_file_hash(Path(tmp_file.name), metadata.file_hash):
                    raise RuntimeError(f"Downloaded file hash mismatch for {name}")
                
                # Move to cache
                shutil.move(tmp_file.name, cache_path)
                logger.info(f"Downloaded tokenizer to: {cache_path}")
                
            except Exception as e:
                # Clean up temp file
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
                raise RuntimeError(f"Failed to download tokenizer {name}: {e}")
        
        return cache_path
    
    def load_tokenizer(self, name: str, force_download: bool = False) -> BPETokenizer:
        """
        Load a pretrained tokenizer.
        
        Args:
            name: Name of the tokenizer to load
            force_download: Whether to force re-download if cached
            
        Returns:
            Loaded tokenizer instance
        """
        metadata = self.get_metadata(name)
        if not metadata:
            raise ValueError(f"Unknown tokenizer: {name}")
        
        # If no download URL, create from config
        if not metadata.file_url:
            logger.info(f"Creating tokenizer {name} from configuration (not pretrained)")
            return self.create_tokenizer(name)
        
        # Download and load pretrained tokenizer
        tokenizer_path = self.download_tokenizer(name, force_download)
        tokenizer = BPETokenizer.load(str(tokenizer_path))
        
        logger.info(f"Loaded pretrained tokenizer: {name}")
        return tokenizer
    
    def _verify_file_hash(self, filepath: Path, expected_hash: str) -> bool:
        """Verify file hash matches expected value."""
        try:
            hasher = hashlib.sha256()
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest() == expected_hash
        except Exception as e:
            logger.error(f"Error verifying hash for {filepath}: {e}")
            return False
    
    def clear_cache(self, name: Optional[str] = None):
        """
        Clear cached tokenizer files.
        
        Args:
            name: Specific tokenizer to clear, or None for all
        """
        if name:
            cache_path = self.cache_dir / f"{name}.pkl"
            if cache_path.exists():
                cache_path.unlink()
                logger.info(f"Cleared cache for tokenizer: {name}")
        else:
            # Clear all cached tokenizers
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("Cleared all tokenizer cache")
    
    def export_registry(self, filepath: str):
        """
        Export registry to a JSON file.
        
        Args:
            filepath: Path to save registry JSON
        """
        registry_data = {
            name: asdict(metadata) for name, metadata in self._registry.items()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(registry_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Registry exported to: {filepath}")
    
    def import_registry(self, filepath: str, merge: bool = True):
        """
        Import registry from a JSON file.
        
        Args:
            filepath: Path to registry JSON file
            merge: Whether to merge with existing registry or replace
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            registry_data = json.load(f)
        
        if not merge:
            self._registry.clear()
        
        for name, metadata_dict in registry_data.items():
            metadata = TokenizerMetadata(**metadata_dict)
            self.register(metadata)
        
        logger.info(f"Registry imported from: {filepath}")
    
    def search_tokenizers(self, query: str) -> List[Tuple[str, float]]:
        """
        Search tokenizers by name, description, or tags.
        
        Args:
            query: Search query
            
        Returns:
            List of (tokenizer_name, relevance_score) tuples, sorted by relevance
        """
        query_lower = query.lower()
        results = []
        
        for name, metadata in self._registry.items():
            score = 0.0
            
            # Exact name match gets highest score
            if query_lower == name.lower():
                score = 1.0
            # Partial name match
            elif query_lower in name.lower():
                score = 0.8
            # Description match
            elif query_lower in metadata.description.lower():
                score = 0.6
            # Tag match
            elif any(query_lower in tag.lower() for tag in metadata.tags):
                score = 0.4
            # Domain/language match
            elif query_lower in metadata.domain.lower() or query_lower in metadata.language.lower():
                score = 0.3
            
            if score > 0:
                results.append((name, score))
        
        # Sort by relevance score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results


# Global registry instance
_global_registry = None


def get_registry() -> TokenizerRegistry:
    """Get the global tokenizer registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = TokenizerRegistry()
    return _global_registry


def list_tokenizers(**filters) -> List[str]:
    """List available tokenizers with optional filtering."""
    return get_registry().list_tokenizers(**filters)


def load_tokenizer(name: str, force_download: bool = False) -> BPETokenizer:
    """Load a tokenizer by name from the global registry."""
    return get_registry().load_tokenizer(name, force_download)


def create_tokenizer_from_config(name: str, **config_overrides) -> BPETokenizer:
    """Create a tokenizer from registered configuration."""
    return get_registry().create_tokenizer(name, **config_overrides)


def register_tokenizer(metadata: TokenizerMetadata):
    """Register a new tokenizer in the global registry."""
    get_registry().register(metadata)


def get_tokenizer_info(name: str) -> Optional[TokenizerMetadata]:
    """Get information about a registered tokenizer."""
    return get_registry().get_metadata(name)


def search_tokenizers(query: str) -> List[Tuple[str, float]]:
    """Search for tokenizers by query."""
    return get_registry().search_tokenizers(query)


class TokenizerCollection:
    """
    A collection of tokenizers for ensemble or comparison purposes.
    """
    
    def __init__(self, tokenizer_names: List[str] = None):
        """
        Initialize tokenizer collection.
        
        Args:
            tokenizer_names: List of tokenizer names to load
        """
        self.tokenizers: Dict[str, BPETokenizer] = {}
        self.registry = get_registry()
        
        if tokenizer_names:
            self.load_tokenizers(tokenizer_names)
    
    def load_tokenizers(self, tokenizer_names: List[str]):
        """Load multiple tokenizers into the collection."""
        for name in tokenizer_names:
            try:
                tokenizer = self.registry.load_tokenizer(name)
                self.tokenizers[name] = tokenizer
                logger.info(f"Loaded tokenizer: {name}")
            except Exception as e:
                logger.error(f"Failed to load tokenizer {name}: {e}")
    
    def add_tokenizer(self, name: str, tokenizer: BPETokenizer):
        """Add a tokenizer instance to the collection."""
        self.tokenizers[name] = tokenizer
    
    def compare_tokenization(self, text: str) -> Dict[str, List[int]]:
        """
        Compare how different tokenizers encode the same text.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Dictionary mapping tokenizer names to token IDs
        """
        results = {}
        for name, tokenizer in self.tokenizers.items():
            try:
                tokens = tokenizer.encode(text)
                results[name] = tokens
            except Exception as e:
                logger.error(f"Tokenization failed for {name}: {e}")
                results[name] = []
        
        return results
    
    def analyze_compression(self, texts: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Analyze compression ratios across different tokenizers.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary of tokenizer performance metrics
        """
        results = {}
        
        for name, tokenizer in self.tokenizers.items():
            total_chars = sum(len(text) for text in texts)
            total_tokens = 0
            
            for text in texts:
                try:
                    tokens = tokenizer.encode(text)
                    total_tokens += len(tokens)
                except Exception as e:
                    logger.error(f"Error tokenizing with {name}: {e}")
                    continue
            
            if total_tokens > 0:
                results[name] = {
                    'compression_ratio': total_chars / total_tokens,
                    'tokens_per_char': total_tokens / total_chars,
                    'avg_token_length': total_chars / total_tokens,
                }
        
        return results
    
    def get_tokenizer_names(self) -> List[str]:
        """Get list of loaded tokenizer names."""
        return list(self.tokenizers.keys())
    
    def get_tokenizer(self, name: str) -> Optional[BPETokenizer]:
        """Get a specific tokenizer from the collection."""
        return self.tokenizers.get(name)


def create_tokenizer_collection(domain: str = None, **filters) -> TokenizerCollection:
    """
    Create a tokenizer collection with tokenizers matching given criteria.
    
    Args:
        domain: Domain to filter by
        **filters: Additional filters for tokenizer selection
        
    Returns:
        TokenizerCollection instance
    """
    registry = get_registry()
    tokenizer_names = registry.list_tokenizers(domain=domain, **filters)
    
    return TokenizerCollection(tokenizer_names)


def benchmark_tokenizers(tokenizer_names: List[str], 
                        test_texts: List[str],
                        metrics: List[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Benchmark multiple tokenizers on test texts.
    
    Args:
        tokenizer_names: List of tokenizer names to benchmark
        test_texts: List of test texts
        metrics: List of metrics to compute
        
    Returns:
        Benchmark results dictionary
    """
    if metrics is None:
        metrics = ['compression_ratio', 'vocab_utilization', 'encoding_time']
    
    collection = TokenizerCollection(tokenizer_names)
    results = {}
    
    import time
    
    for name, tokenizer in collection.tokenizers.items():
        tokenizer_results = {}
        
        # Compression ratio
        if 'compression_ratio' in metrics:
            total_chars = sum(len(text) for text in test_texts)
            total_tokens = sum(len(tokenizer.encode(text)) for text in test_texts)
            tokenizer_results['compression_ratio'] = total_chars / total_tokens if total_tokens > 0 else 0
        
        # Vocabulary utilization
        if 'vocab_utilization' in metrics:
            all_tokens = set()
            for text in test_texts:
                all_tokens.update(tokenizer.encode(text))
            tokenizer_results['vocab_utilization'] = len(all_tokens) / tokenizer.get_vocab_size()
        
        # Encoding time
        if 'encoding_time' in metrics:
            start_time = time.time()
            for text in test_texts:
                tokenizer.encode(text)
            end_time = time.time()
            tokenizer_results['encoding_time'] = (end_time - start_time) / len(test_texts)
        
        results[name] = tokenizer_results
    
    return results