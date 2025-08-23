# Advanced BPE Tokenizer Library

A comprehensive Python tokenizer library implementing Byte Pair Encoding (BPE) with extensive preprocessing, analysis, and management capabilities. This library provides a complete toolkit for training, testing, and deploying tokenizers similar to tiktoken, with additional features for text preprocessing and model registry management.

## Features

### 🚀 Core Functionality
- **Fast BPE Tokenization**: Efficient Byte Pair Encoding implementation with configurable vocabulary sizes
- **Flexible Configuration**: Customizable regex patterns, special tokens, and normalization options
- **Batch Processing**: Parallel encoding/decoding for large datasets
- **Multiple Formats**: JSON and pickle serialization support

### 🧹 Text Preprocessing Pipeline
- **Advanced Cleaning**: Unicode normalization, HTML entity decoding, control character removal
- **Quality Filtering**: Content quality assessment and filtering based on multiple metrics
- **Duplicate Detection**: Efficient near-duplicate detection using Jaccard similarity
- **Language Support**: Multilingual text processing with encoding fixes

### 📚 Model Registry System
- **Pretrained Models**: Built-in configurations for GPT-2, code-specific, and domain-specific tokenizers
- **Easy Management**: Register, search, and load tokenizers with metadata
- **Caching**: Automatic download and caching of pretrained models
- **Collections**: Group and compare multiple tokenizers

### 🔧 CLI Interface
- **Complete Command Line**: Train, encode, decode, analyze, and benchmark tokenizers
- **Batch Operations**: Process multiple files and datasets efficiently
- **Analysis Tools**: Comprehensive tokenization quality analysis and visualization

### 🔍 Analysis & Utilities
- **Vocabulary Analysis**: Coverage analysis, rare token detection, compression metrics
- **Quality Validation**: Automated tokenizer validation and consistency checking
- **Performance Benchmarking**: Compare multiple tokenizers across various metrics

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd tokenizer-library

# Install dependencies
pip install -r requirements.txt

# Install the library
pip install -e .
```

### Dependencies
- `ftfy`: For encoding error fixing
- `unicodedata`: For Unicode normalization (built-in)
- Standard library modules: `re`, `json`, `pickle`, `pathlib`, `multiprocessing`

## Quick Start

### Basic Usage

```python
from tokenizer import BPETokenizer, create_tokenizer

# Create and train a tokenizer
tokenizer = create_tokenizer(vocab_size=32000)
texts = ["Hello world!", "This is a sample text.", "More training data..."]
tokenizer.train(texts)

# Encode and decode text
tokens = tokenizer.encode("Hello world!")
print(f"Tokens: {tokens}")

decoded = tokenizer.decode(tokens)
print(f"Decoded: {decoded}")

# Save and load tokenizer
tokenizer.save("my_tokenizer.pkl")
loaded_tokenizer = BPETokenizer.load("my_tokenizer.pkl")
```

### Using Pretrained Tokenizers

```python
from tokenizer import load_tokenizer, list_tokenizers

# List available tokenizers
print("Available tokenizers:", list_tokenizers())

# Load a pretrained tokenizer
tokenizer = load_tokenizer("gpt2-small")
tokens = tokenizer.encode("Hello world!")
print(f"GPT-2 tokens: {tokens}")
```

### Text Preprocessing

```python
from tokenizer import create_preprocessing_pipeline

# Create preprocessing pipeline
pipeline = create_preprocessing_pipeline(
    min_text_length=50,
    max_text_length=10000,
    remove_duplicates=True
)

# Process texts
raw_texts = ["Some text with issues...", "Duplicate text", "Duplicate text"]
clean_texts = pipeline.process_batch(raw_texts)
print(f"Processed {len(clean_texts)} texts from {len(raw_texts)}")
```

### Advanced Training with Preprocessing

```python
from tokenizer import train_simple_tokenizer

# Train with automatic preprocessing
texts = load_training_data()  # Your training data
tokenizer = train_simple_tokenizer(
    texts,
    vocab_size=50000,
    preprocessing=True,
    output_path="advanced_tokenizer.pkl"
)
```

## Command Line Interface

### Training a Tokenizer

```bash
# Basic training
python -m tokenizer train \
    --input data/*.txt \
    --output my_tokenizer.pkl \
    --vocab-size 32000 \
    --preprocessing

# Advanced training with custom configuration
python -m tokenizer train \
    --input corpus/ \
    --output domain_tokenizer.pkl \
    --vocab-size 50000 \
    --config-file config.json \
    --workers 8 \
    --special-tokens "<|custom|>" "<|special|>"
```

### Encoding and Decoding

```bash
# Encode text
python -m tokenizer encode \
    --tokenizer gpt2-small \
    --text "Hello, world!" \
    --format tokens

# Decode token IDs
python -m tokenizer decode \
    --tokenizer my_tokenizer.pkl \
    --tokens "15496 995 0"

# Batch processing
python -m tokenizer encode \
    --tokenizer my_tokenizer.pkl \
    --input-file texts.txt \
    --output-file tokens.txt \
    --format ids
```

### Analysis and Benchmarking

```bash
# Analyze tokenizer performance
python -m tokenizer analyze \
    --tokenizer gpt2-small \
    --input test_data.txt \
    --metrics compression coverage quality \
    --save-vocab vocab.json

# Benchmark multiple tokenizers
python -m tokenizer benchmark \
    --tokenizers gpt2-small code-tokenizer multilingual-base \
    --input benchmark_data.txt \
    --metrics compression speed coverage
```

### Managing Tokenizers

```bash
# List available tokenizers
python -m tokenizer list

# Search for specific tokenizers
python -m tokenizer search "code"

# Get tokenizer information
python -m tokenizer info gpt2-small

# Preprocess data
python -m tokenizer preprocess \
    --input raw_data/ \
    --output clean_data.txt \
    --remove-duplicates \
    --min-length 100
```

## Configuration

### Tokenizer Configuration

```python
from tokenizer import TokenizerConfig, BPETokenizer

config = TokenizerConfig(
    vocab_size=50000,
    min_frequency=2,
    special_tokens={
        "<|endoftext|>": 0,
        "<|startoftext|>": 1,
        "<|pad|>": 2,
        "<|unk|>": 3,
        "<|custom|>": 4,
    },
    regex_pattern=r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
    preserve_spaces=True,
    normalize_unicode=True
)

tokenizer = BPETokenizer(config)
```

### Preprocessing Configuration

```python
from tokenizer import PreprocessingConfig, PreprocessingPipeline

config = PreprocessingConfig(
    # Unicode handling
    fix_encoding=True,
    normalize_unicode='NFC',
    
    # Text cleaning
    remove_control_chars=True,
    fix_html_entities=True,
    remove_urls=True,
    
    # Quality filtering
    min_text_length=100,
    max_text_length=100000,
    remove_duplicates=True,
    duplicate_threshold=0.85,
    
    # Content quality
    min_word_count=20,
    max_repetition_ratio=0.3,
    min_unique_words_ratio=0.4
)

pipeline = PreprocessingPipeline(config)
```

## Built-in Tokenizers

The library comes with several pretrained tokenizer configurations:

| Name | Description | Vocab Size | Domain |
|------|-------------|------------|---------|
| `gpt2-small` | GPT-2 small model tokenizer | 50,257 | General |
| `gpt2-medium` | GPT-2 medium model tokenizer | 50,257 | General |
| `code-tokenizer` | Optimized for source code | 32,000 | Code |
| `multilingual-base` | Base multilingual tokenizer | 64,000 | Multilingual |
| `scientific-tokenizer` | Optimized for scientific text | 40,000 | Scientific |

## Advanced Usage

### Custom Tokenizer Registry

```python
from tokenizer import TokenizerMetadata, register_tokenizer

# Register a custom tokenizer
metadata = TokenizerMetadata(
    name="custom-domain",
    description="Custom domain-specific tokenizer",
    vocab_size=30000,
    domain="custom",
    language="en",
    config={
        "vocab_size": 30000,
        "special_tokens": {"<|domain|>": 4},
    },
    tags=["custom", "domain-specific"]
)

register_tokenizer(metadata)
```

### Tokenizer Collections

```python
from tokenizer import TokenizerCollection

# Create a collection for comparison
collection = TokenizerCollection([
    "gpt2-small", 
    "code-tokenizer", 
    "multilingual-base"
])

# Compare tokenization across models
text = "Hello, world! This is a test."
results = collection.compare_tokenization(text)

for name, tokens in results.items():
    print(f"{name}: {tokens}")

# Analyze compression ratios
compression_analysis = collection.analyze_compression([text])
print("Compression analysis:", compression_analysis)
```

### Quality Analysis

```python
from tokenizer import analyze_tokenization_quality

# Analyze tokenizer quality
test_texts = ["Sample text for analysis", "Another test sentence"]
quality_metrics = analyze_tokenization_quality(tokenizer, test_texts, detailed=True)

print(f"Compression ratio: {quality_metrics['compression_ratio']:.2f}")
print(f"Vocabulary utilization: {quality_metrics['vocab_utilization']:.2%}")
print(f"Coverage rate: {quality_metrics['coverage_rate']:.2%}")
```

### Parallel Processing

```python
from tokenizer import BatchProcessor

# Process large datasets efficiently
processor = BatchProcessor(batch_size=1000, max_workers=8)

def encode_text(text):
    return tokenizer.encode(text)

# Parallel encoding
large_texts = load_large_dataset()
all_tokens = processor.process_texts_parallel(large_texts, encode_text)
```

## API Reference

### Core Classes

- **`BPETokenizer`**: Main tokenizer class with training, encoding, and decoding methods
- **`TokenizerConfig`**: Configuration class for tokenizer parameters
- **`PreprocessingPipeline`**: Complete text preprocessing pipeline
- **`TokenizerRegistry`**: Registry for managing pretrained tokenizers
- **`TokenizerCollection`**: Collection class for managing multiple tokenizers

### Key Functions

- **`create_tokenizer()`**: Factory function for creating configured tokenizers
- **`train_simple_tokenizer()`**: High-level function for training with sensible defaults
- **`load_tokenizer()`**: Load tokenizer by name from registry
- **`analyze_tokenization_quality()`**: Comprehensive quality analysis
- **`benchmark_tokenizers()`**: Performance benchmarking across multiple tokenizers

## Performance Considerations

### Training Performance
- Use multiprocessing for large datasets (`--workers` parameter)
- Enable preprocessing pipeline for better quality training data
- Consider vocabulary size vs. compression ratio trade-offs

### Memory Usage
- Large vocabularies require more memory
- Use batch processing for encoding/decoding large datasets
- Consider streaming processing for very large corpora

### Speed Optimization
- Compiled tokenizers are much faster than unconfigured ones
- Parallel processing scales well for batch operations
- Caching of pretrained models reduces load times

## Examples and Use Cases

### Domain-Specific Training
```python
# Train a tokenizer for code
code_texts = load_code_files("src/")
code_tokenizer = train_simple_tokenizer(
    code_texts,
    vocab_size=32000,
    regex_pattern=r"""[ \t]+|[a-zA-Z_][a-zA-Z0-9_]*|\d+\.?\d*|[^\w\s]|\n""",
    special_tokens={"<|indent|>": 4, "<|dedent|>": 5}
)
```

### Multilingual Processing
```python
# Process multilingual data
multilingual_pipeline = create_preprocessing_pipeline(
    normalize_unicode='NFC',
    fix_encoding=True,
    remove_duplicates=True
)

clean_texts = multilingual_pipeline.process_batch(multilingual_texts)
multilingual_tokenizer = train_simple_tokenizer(clean_texts, vocab_size=64000)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Changelog

### Version 1.0.0
- Initial release with core BPE tokenization
- Comprehensive preprocessing pipeline
- Model registry system
- CLI interface
- Built-in pretrained tokenizer configurations
- Quality analysis and benchmarking tools

## Support

For issues, feature requests, or questions:
- Open an issue on GitHub
- Check the documentation for detailed API reference
- Review examples in the `examples/` directory

## Acknowledgments

- Inspired by OpenAI's tiktoken library
- Uses Byte Pair Encoding as described in the original BPE paper
- Built with modern Python best practices and extensive testing