"""
Advanced BPE Tokenizer Library

A comprehensive tokenizer library similar to tiktoken, providing BPE (Byte Pair Encoding)
tokenization with extensive preprocessing, analysis, and management capabilities.

This library offers:
- Fast BPE tokenization with configurable vocabulary sizes
- Comprehensive text preprocessing pipeline
- Tokenizer registry system for managing pretrained models
- CLI interface for training, testing, and analysis
- Utilities for performance benchmarking and quality assessment

Example usage:
    >>> from tokenizer import BPETokenizer, create_tokenizer
    >>> 
    >>> # Create and train a tokenizer
    >>> tokenizer = create_tokenizer(vocab_size=32000)
    >>> texts = ["Hello world!", "This is a sample text."]
    >>> tokenizer.train(texts)
    >>> 
    >>> # Encode and decode text
    >>> tokens = tokenizer.encode("Hello world!")
    >>> decoded = tokenizer.decode(tokens)
    >>> 
    >>> # Save and load tokenizer
    >>> tokenizer.save("my_tokenizer.pkl")
    >>> loaded_tokenizer = BPETokenizer.load("my_tokenizer.pkl")

For more advanced usage, see the individual module documentation:
- core: Core tokenizer implementation
- utils: Utility functions and analysis tools
- preprocessing: Data preprocessing pipeline
- registry: Tokenizer management and pretrained models
- cli: Command-line interface
"""

__version__ = "1.0.0"
__author__ = "Tokenizer Library Team"
__license__ = "MIT"

import logging
from typing import List, Optional, Dict, Any

# Core tokenizer classes and functions
from .core import (
    BPETokenizer,
    TokenizerConfig,
    create_tokenizer,
)

# Utility functions
from .utils import (
    TextProcessor,
    DataLoader,
    VocabAnalyzer,
    BatchProcessor,
    compute_token_stats,
    save_vocab_to_file,
    validate_tokenizer,
    merge_tokenizers,
)

# Preprocessing pipeline
from .preprocessing import (
    PreprocessingPipeline,
    PreprocessingConfig,
    TextCleaner,
    QualityFilter,
    DuplicateDetector,
    create_preprocessing_pipeline,
)

# Registry system
from .registry import (
    TokenizerRegistry,
    TokenizerMetadata,
    TokenizerCollection,
    get_registry,
    list_tokenizers,
    load_tokenizer,
    create_tokenizer_from_config,
    register_tokenizer,
    get_tokenizer_info,
    search_tokenizers,
    create_tokenizer_collection,
    benchmark_tokenizers,
)

# Configure default logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Public API
__all__ = [
    # Core classes
    "BPETokenizer",
    "TokenizerConfig",
    
    # Factory functions
    "create_tokenizer",
    "create_tokenizer_from_config",
    "create_preprocessing_pipeline",
    "create_tokenizer_collection",
    
    # Utility classes
    "TextProcessor",
    "DataLoader", 
    "VocabAnalyzer",
    "BatchProcessor",
    
    # Preprocessing classes
    "PreprocessingPipeline",
    "PreprocessingConfig",
    "TextCleaner",
    "QualityFilter",
    "DuplicateDetector",
    
    # Registry classes and functions
    "TokenizerRegistry",
    "TokenizerMetadata", 
    "TokenizerCollection",
    "get_registry",
    "list_tokenizers",
    "load_tokenizer",
    "register_tokenizer",
    "get_tokenizer_info",
    "search_tokenizers",
    
    # Utility functions
    "compute_token_stats",
    "save_vocab_to_file",
    "validate_tokenizer",
    "merge_tokenizers",
    "benchmark_tokenizers",
]


def get_version() -> str:
    """Get the current version of the tokenizer library."""
    return __version__


def get_available_tokenizers() -> List[str]:
    """Get list of all available tokenizers in the registry."""
    return list_tokenizers()


def quick_tokenize(text: str, 
                  tokenizer_name: str = "gpt2-small",
                  return_tokens: bool = False) -> List:
    """
    Quick tokenization using a pretrained tokenizer.
    
    Args:
        text: Text to tokenize
        tokenizer_name: Name of the tokenizer to use
        return_tokens: If True, return token strings; if False, return token IDs
        
    Returns:
        List of token IDs or token strings
        
    Example:
        >>> tokens = quick_tokenize("Hello world!", "gpt2-small")
        >>> print(tokens)  # [15496, 995, 0]
        >>> 
        >>> token_strings = quick_tokenize("Hello world!", "gpt2-small", return_tokens=True)
        >>> print(token_strings)  # ["Hello", " world", "!"]
    """
    try:
        tokenizer = load_tokenizer(tokenizer_name)
    except ValueError:
        # Fallback to creating a basic tokenizer
        logging.warning(f"Tokenizer '{tokenizer_name}' not found, using default configuration")
        tokenizer = create_tokenizer()
        # Would need training data in real scenario
        tokenizer.train([text])  # Minimal training for demo
    
    token_ids = tokenizer.encode(text)
    
    if return_tokens:
        return [tokenizer.id_to_token(tid) or f"<{tid}>" for tid in token_ids]
    else:
        return token_ids


def train_simple_tokenizer(texts: List[str],
                          vocab_size: int = 32000,
                          output_path: Optional[str] = None,
                          preprocessing: bool = True,
                          **config_kwargs) -> BPETokenizer:
    """
    Train a tokenizer with sensible defaults.
    
    Args:
        texts: List of training texts
        vocab_size: Target vocabulary size
        output_path: Optional path to save the trained tokenizer
        preprocessing: Whether to apply text preprocessing
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Trained tokenizer instance
        
    Example:
        >>> texts = ["Hello world!", "This is sample text.", "More training data..."]
        >>> tokenizer = train_simple_tokenizer(texts, vocab_size=1000)
        >>> tokens = tokenizer.encode("Hello world!")
        >>> print(tokens)
    """
    # Apply preprocessing if requested
    if preprocessing:
        pipeline = create_preprocessing_pipeline(
            min_text_length=10,
            max_text_length=100000,
            remove_duplicates=True
        )
        texts = pipeline.process_batch(texts)
        logging.info(f"Preprocessing retained {len(texts)} texts")
    
    # Create and configure tokenizer
    config_dict = {"vocab_size": vocab_size}
    config_dict.update(config_kwargs)
    
    tokenizer = create_tokenizer(**config_dict)
    
    # Train tokenizer
    logging.info(f"Training tokenizer with {len(texts)} texts...")
    tokenizer.train(texts)
    
    # Save if path provided
    if output_path:
        tokenizer.save(output_path)
        logging.info(f"Tokenizer saved to {output_path}")
    
    return tokenizer


def analyze_tokenization_quality(tokenizer: BPETokenizer,
                                test_texts: List[str],
                                detailed: bool = False) -> Dict[str, Any]:
    """
    Analyze the quality of a trained tokenizer.
    
    Args:
        tokenizer: Trained tokenizer to analyze
        test_texts: List of test texts for evaluation
        detailed: Whether to return detailed analysis
        
    Returns:
        Dictionary containing quality metrics
        
    Example:
        >>> tokenizer = load_tokenizer("gpt2-small")
        >>> test_texts = ["Sample text for analysis", "Another test sentence"]
        >>> quality = analyze_tokenization_quality(tokenizer, test_texts)
        >>> print(f"Compression ratio: {quality['compression_ratio']:.2f}")
    """
    analyzer = VocabAnalyzer(tokenizer)
    
    # Basic statistics
    stats = compute_token_stats(test_texts, tokenizer)
    
    # Coverage analysis
    coverage = analyzer.analyze_vocab_coverage(test_texts)
    
    # Validation
    validation = validate_tokenizer(tokenizer, test_texts[:5])
    
    quality_metrics = {
        "compression_ratio": stats["compression_ratio"],
        "vocab_utilization": coverage["vocab_utilization"],
        "coverage_rate": coverage["coverage_rate"],
        "unknown_rate": coverage["unknown_rate"],
        "passes_validation": all(validation.values()),
        "vocab_size": tokenizer.get_vocab_size(),
        "avg_tokens_per_text": stats["avg_tokens_per_text"],
    }
    
    if detailed:
        quality_metrics.update({
            "detailed_stats": stats,
            "detailed_coverage": coverage,
            "validation_results": validation,
        })
    
    return quality_metrics


def setup_logging(level: str = "INFO", 
                 format_string: Optional[str] = None) -> None:
    """
    Set up logging for the tokenizer library.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set library loggers
    for module_name in ['tokenizer.core', 'tokenizer.utils', 'tokenizer.preprocessing', 
                       'tokenizer.registry', 'tokenizer.cli']:
        logger = logging.getLogger(module_name)
        logger.setLevel(getattr(logging, level.upper()))


# Convenience aliases for backward compatibility
Tokenizer = BPETokenizer
Config = TokenizerConfig


# Package metadata
def print_info():
    """Print package information."""
    print(f"Tokenizer Library v{__version__}")
    print(f"Author: {__author__}")
    print(f"License: {__license__}")
    print()
    print("Available tokenizers:")
    for name in get_available_tokenizers():
        metadata = get_tokenizer_info(name)
        if metadata:
            print(f"  - {name}: {metadata.description}")
    print()
    print("For help, see documentation or run: python -m tokenizer --help")


if __name__ == "__main__":
    print_info()