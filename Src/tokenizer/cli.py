"""
Command-line interface for the tokenizer library.

Provides a comprehensive CLI for training, testing, and managing tokenizers
with support for various data formats and preprocessing options.
"""

import argparse
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import multiprocessing as mp

from .core import BPETokenizer, TokenizerConfig, create_tokenizer
from .utils import (
    DataLoader, TextProcessor, VocabAnalyzer, BatchProcessor, 
    compute_token_stats, save_vocab_to_file, validate_tokenizer
)
from .preprocessing import PreprocessingPipeline, PreprocessingConfig, create_preprocessing_pipeline
from .registry import (
    get_registry, list_tokenizers, load_tokenizer, register_tokenizer,
    search_tokenizers, TokenizerMetadata, create_tokenizer_collection,
    benchmark_tokenizers
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Tokenizer CLI - Train, test, and manage BPE tokenizers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new tokenizer
  tokenizer train --input data/*.txt --output my_tokenizer.pkl --vocab-size 32000
  
  # Test tokenization on text
  tokenizer encode --tokenizer my_tokenizer.pkl --text "Hello world!"
  
  # List available pretrained tokenizers
  tokenizer list
  
  # Load and test a pretrained tokenizer
  tokenizer encode --tokenizer gpt2-small --text "Hello world!"
  
  # Benchmark multiple tokenizers
  tokenizer benchmark --tokenizers gpt2-small code-tokenizer --input test_data.txt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new tokenizer')
    _add_train_arguments(train_parser)
    
    # Encode command
    encode_parser = subparsers.add_parser('encode', help='Encode text with tokenizer')
    _add_encode_arguments(encode_parser)
    
    # Decode command
    decode_parser = subparsers.add_parser('decode', help='Decode tokens to text')
    _add_decode_arguments(decode_parser)
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze tokenizer or text')
    _add_analyze_arguments(analyze_parser)
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available tokenizers')
    _add_list_arguments(list_parser)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show tokenizer information')
    _add_info_arguments(info_parser)
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for tokenizers')
    _add_search_arguments(search_parser)
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark tokenizers')
    _add_benchmark_arguments(benchmark_parser)
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess text data')
    _add_preprocess_arguments(preprocess_parser)
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert tokenizer formats')
    _add_convert_arguments(convert_parser)
    
    return parser


def _add_train_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the train command."""
    parser.add_argument('--input', '-i', required=True, nargs='+',
                       help='Input files or directories containing training data')
    parser.add_argument('--output', '-o', required=True,
                       help='Output path for trained tokenizer')
    parser.add_argument('--vocab-size', type=int, default=50000,
                       help='Target vocabulary size (default: 50000)')
    parser.add_argument('--min-frequency', type=int, default=2,
                       help='Minimum frequency for BPE merge (default: 2)')
    parser.add_argument('--format', choices=['auto', 'txt', 'jsonl'], default='auto',
                       help='Input file format (default: auto-detect)')
    parser.add_argument('--text-field', default='text',
                       help='Field name for text in JSONL files (default: text)')
    parser.add_argument('--max-files', type=int,
                       help='Maximum number of files to process')
    parser.add_argument('--preprocessing', action='store_true',
                       help='Enable text preprocessing pipeline')
    parser.add_argument('--workers', type=int, default=mp.cpu_count(),
                       help='Number of parallel workers')
    parser.add_argument('--regex-pattern',
                       help='Custom regex pattern for tokenization')
    parser.add_argument('--special-tokens', nargs='+', 
                       help='Additional special tokens to add')
    parser.add_argument('--config-file',
                       help='JSON configuration file for advanced options')


def _add_encode_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the encode command."""
    parser.add_argument('--tokenizer', '-t', required=True,
                       help='Tokenizer file path or registered name')
    parser.add_argument('--text', 
                       help='Text to encode (use stdin if not provided)')
    parser.add_argument('--input-file', 
                       help='File containing text to encode')
    parser.add_argument('--output-file',
                       help='Output file for encoded tokens (default: stdout)')
    parser.add_argument('--format', choices=['ids', 'tokens', 'json'], default='ids',
                       help='Output format (default: ids)')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for processing multiple texts')


def _add_decode_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the decode command."""
    parser.add_argument('--tokenizer', '-t', required=True,
                       help='Tokenizer file path or registered name')
    parser.add_argument('--tokens',
                       help='Token IDs to decode (space-separated)')
    parser.add_argument('--input-file',
                       help='File containing token IDs to decode')
    parser.add_argument('--output-file',
                       help='Output file for decoded text (default: stdout)')
    parser.add_argument('--format', choices=['text', 'json'], default='text',
                       help='Output format (default: text)')


def _add_analyze_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the analyze command."""
    parser.add_argument('--tokenizer', '-t', required=True,
                       help='Tokenizer file path or registered name')
    parser.add_argument('--input', '-i', nargs='+',
                       help='Input files or text to analyze')
    parser.add_argument('--text',
                       help='Text to analyze directly')
    parser.add_argument('--output-file',
                       help='Output file for analysis results (default: stdout)')
    parser.add_argument('--metrics', nargs='+',
                       choices=['basic', 'compression', 'coverage', 'quality', 'all'],
                       default=['basic'],
                       help='Metrics to compute')
    parser.add_argument('--save-vocab',
                       help='Save vocabulary to file')


def _add_list_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the list command."""
    parser.add_argument('--domain',
                       help='Filter by domain')
    parser.add_argument('--language',
                       help='Filter by language')
    parser.add_argument('--model-family',
                       help='Filter by model family')
    parser.add_argument('--tags', nargs='+',
                       help='Filter by tags')
    parser.add_argument('--format', choices=['table', 'json', 'names'], default='table',
                       help='Output format')


def _add_info_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the info command."""
    parser.add_argument('tokenizer',
                       help='Tokenizer name or file path')
    parser.add_argument('--format', choices=['text', 'json'], default='text',
                       help='Output format')


def _add_search_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the search command."""
    parser.add_argument('query',
                       help='Search query')
    parser.add_argument('--limit', type=int, default=10,
                       help='Maximum number of results')


def _add_benchmark_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the benchmark command."""
    parser.add_argument('--tokenizers', '-t', nargs='+', required=True,
                       help='Tokenizer names or file paths to benchmark')
    parser.add_argument('--input', '-i', required=True,
                       help='Input file containing test data')
    parser.add_argument('--metrics', nargs='+',
                       choices=['compression', 'speed', 'coverage', 'all'],
                       default=['compression', 'speed'],
                       help='Metrics to benchmark')
    parser.add_argument('--output-file',
                       help='Output file for benchmark results (default: stdout)')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of test samples to use')


def _add_preprocess_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the preprocess command."""
    parser.add_argument('--input', '-i', required=True, nargs='+',
                       help='Input files to preprocess')
    parser.add_argument('--output', '-o', required=True,
                       help='Output file for preprocessed data')
    parser.add_argument('--config-file',
                       help='Preprocessing configuration file')
    parser.add_argument('--min-length', type=int, default=50,
                       help='Minimum text length')
    parser.add_argument('--max-length', type=int, default=1000000,
                       help='Maximum text length')
    parser.add_argument('--remove-duplicates', action='store_true',
                       help='Remove duplicate texts')
    parser.add_argument('--quality-threshold', type=float, default=0.5,
                       help='Quality score threshold')
    parser.add_argument('--workers', type=int, default=mp.cpu_count(),
                       help='Number of parallel workers')


def _add_convert_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the convert command."""
    parser.add_argument('--input', '-i', required=True,
                       help='Input tokenizer file')
    parser.add_argument('--output', '-o', required=True,
                       help='Output file')
    parser.add_argument('--format', choices=['json', 'pickle'], required=True,
                       help='Output format')


def load_tokenizer_from_arg(tokenizer_arg: str) -> BPETokenizer:
    """Load tokenizer from command line argument (file path or registered name)."""
    if Path(tokenizer_arg).exists():
        # Load from file
        return BPETokenizer.load(tokenizer_arg)
    else:
        # Load from registry
        try:
            return load_tokenizer(tokenizer_arg)
        except ValueError:
            logger.error(f"Tokenizer not found: {tokenizer_arg}")
            logger.info("Available tokenizers:")
            for name in list_tokenizers():
                print(f"  - {name}")
            sys.exit(1)


def load_input_data(input_paths: List[str], 
                   format_type: str = 'auto',
                   text_field: str = 'text',
                   max_files: Optional[int] = None) -> List[str]:
    """Load input data from various sources."""
    all_texts = []
    file_count = 0
    
    for path_str in input_paths:
        path = Path(path_str)
        
        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = list(path.rglob('*.txt'))
            files.extend(list(path.rglob('*.jsonl')))
        else:
            # Glob pattern
            files = list(Path('.').glob(path_str))
        
        for file_path in files:
            if max_files and file_count >= max_files:
                break
            
            file_ext = file_path.suffix.lower()
            
            # Auto-detect format
            if format_type == 'auto':
                if file_ext == '.jsonl':
                    current_format = 'jsonl'
                else:
                    current_format = 'txt'
            else:
                current_format = format_type
            
            # Load data
            if current_format == 'jsonl':
                texts = DataLoader.load_jsonl(str(file_path), text_field)
            else:
                text = DataLoader.load_text_file(str(file_path))
                texts = [text] if text else []
            
            all_texts.extend(texts)
            file_count += 1
            
            logger.info(f"Loaded {len(texts)} texts from {file_path}")
    
    logger.info(f"Total loaded texts: {len(all_texts)}")
    return all_texts


def command_train(args) -> None:
    """Handle the train command."""
    logger.info("Starting tokenizer training...")
    
    # Load training data
    texts = load_input_data(args.input, args.format, args.text_field, args.max_files)
    if not texts:
        logger.error("No training data found")
        sys.exit(1)
    
    # Apply preprocessing if requested
    if args.preprocessing:
        logger.info("Applying preprocessing pipeline...")
        pipeline = create_preprocessing_pipeline()
        texts = pipeline.process_batch(texts, max_workers=args.workers)
        logger.info(f"After preprocessing: {len(texts)} texts")
        
        # Print preprocessing statistics
        stats = pipeline.get_statistics()
        logger.info(f"Preprocessing stats: {stats}")
    
    # Load configuration from file if provided
    config_dict = {}
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
    
    # Override with command line arguments
    config_dict.update({
        'vocab_size': args.vocab_size,
        'min_frequency': args.min_frequency,
    })
    
    if args.regex_pattern:
        config_dict['regex_pattern'] = args.regex_pattern
    
    if args.special_tokens:
        special_tokens = config_dict.get('special_tokens', {})
        next_id = max(special_tokens.values()) + 1 if special_tokens else 4
        for token in args.special_tokens:
            if token not in special_tokens:
                special_tokens[token] = next_id
                next_id += 1
        config_dict['special_tokens'] = special_tokens
    
    # Create and train tokenizer
    config = TokenizerConfig(**config_dict)
    tokenizer = BPETokenizer(config)
    
    start_time = time.time()
    tokenizer.train(texts, verbose=True)
    training_time = time.time() - start_time
    
    # Save tokenizer
    tokenizer.save(args.output)
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Tokenizer saved to: {args.output}")
    
    # Print basic statistics
    stats = compute_token_stats(texts[:1000], tokenizer)  # Sample for stats
    logger.info(f"Basic statistics: {stats}")


def command_encode(args) -> None:
    """Handle the encode command."""
    tokenizer = load_tokenizer_from_arg(args.tokenizer)
    
    # Get input text
    if args.text:
        texts = [args.text]
    elif args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        # Read from stdin
        texts = [line.strip() for line in sys.stdin if line.strip()]
    
    if not texts:
        logger.error("No input text provided")
        sys.exit(1)
    
    # Encode texts
    results = []
    for text in texts:
        token_ids = tokenizer.encode(text)
        
        if args.format == 'ids':
            result = ' '.join(map(str, token_ids))
        elif args.format == 'tokens':
            tokens = [tokenizer.id_to_token(tid) or f"<{tid}>" for tid in token_ids]
            result = ' '.join(f'"{token}"' for token in tokens)
        elif args.format == 'json':
            result = json.dumps({
                'text': text,
                'token_ids': token_ids,
                'tokens': [tokenizer.id_to_token(tid) for tid in token_ids]
            })
        
        results.append(result)
    
    # Output results
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(result + '\n')
    else:
        for result in results:
            print(result)


def command_decode(args) -> None:
    """Handle the decode command."""
    tokenizer = load_tokenizer_from_arg(args.tokenizer)
    
    # Get input tokens
    if args.tokens:
        token_sequences = [list(map(int, args.tokens.split()))]
    elif args.input_file:
        token_sequences = []
        with open(args.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        tokens = list(map(int, line.split()))
                        token_sequences.append(tokens)
                    except ValueError:
                        logger.warning(f"Invalid token line: {line}")
    else:
        # Read from stdin
        token_sequences = []
        for line in sys.stdin:
            line = line.strip()
            if line:
                try:
                    tokens = list(map(int, line.split()))
                    token_sequences.append(tokens)
                except ValueError:
                    logger.warning(f"Invalid token line: {line}")
    
    if not token_sequences:
        logger.error("No token sequences provided")
        sys.exit(1)
    
    # Decode token sequences
    results = []
    for tokens in token_sequences:
        decoded_text = tokenizer.decode(tokens)
        
        if args.format == 'text':
            result = decoded_text
        elif args.format == 'json':
            result = json.dumps({
                'token_ids': tokens,
                'decoded_text': decoded_text
            })
        
        results.append(result)
    
    # Output results
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(result + '\n')
    else:
        for result in results:
            print(result)


def command_analyze(args) -> None:
    """Handle the analyze command."""
    tokenizer = load_tokenizer_from_arg(args.tokenizer)
    analyzer = VocabAnalyzer(tokenizer)
    
    # Get input texts
    if args.text:
        texts = [args.text]
    elif args.input:
        texts = load_input_data(args.input)
    else:
        logger.error("No input text or files provided")
        sys.exit(1)
    
    # Compute analysis
    analysis = {}
    
    if 'basic' in args.metrics or 'all' in args.metrics:
        basic_stats = compute_token_stats(texts, tokenizer)
        analysis['basic_statistics'] = basic_stats
    
    if 'compression' in args.metrics or 'all' in args.metrics:
        compression_stats = {}
        for i, text in enumerate(texts[:10]):  # Analyze first 10 texts
            stats = analyzer.analyze_text(text)
            compression_stats[f'text_{i}'] = {
                'compression_ratio': stats['compression_ratio'],
                'text_length': stats['text_length'],
                'num_tokens': stats['num_tokens']
            }
        analysis['compression_analysis'] = compression_stats
    
    if 'coverage' in args.metrics or 'all' in args.metrics:
        coverage_stats = analyzer.analyze_vocab_coverage(texts)
        analysis['coverage_analysis'] = coverage_stats
    
    if 'quality' in args.metrics or 'all' in args.metrics:
        validation_results = validate_tokenizer(tokenizer, texts[:5])
        analysis['quality_analysis'] = validation_results
    
    # Save vocabulary if requested
    if args.save_vocab:
        save_vocab_to_file(tokenizer, args.save_vocab)
        logger.info(f"Vocabulary saved to: {args.save_vocab}")
    
    # Output analysis results
    output_data = json.dumps(analysis, indent=2, ensure_ascii=False)
    
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(output_data)
    else:
        print(output_data)


def command_list(args) -> None:
    """Handle the list command."""
    registry = get_registry()
    tokenizers = registry.list_tokenizers(
        domain=args.domain,
        language=args.language,
        model_family=args.model_family,
        tags=args.tags
    )
    
    if args.format == 'names':
        for name in tokenizers:
            print(name)
    elif args.format == 'json':
        tokenizer_data = []
        for name in tokenizers:
            metadata = registry.get_metadata(name)
            if metadata:
                tokenizer_data.append({
                    'name': name,
                    'description': metadata.description,
                    'vocab_size': metadata.vocab_size,
                    'domain': metadata.domain,
                    'language': metadata.language,
                    'tags': metadata.tags
                })
        print(json.dumps(tokenizer_data, indent=2))
    else:  # table format
        if not tokenizers:
            print("No tokenizers found matching criteria.")
            return
        
        print(f"{'Name':<20} {'Vocab Size':<10} {'Domain':<10} {'Language':<8} {'Description'}")
        print('-' * 80)
        for name in tokenizers:
            metadata = registry.get_metadata(name)
            if metadata:
                print(f"{name:<20} {metadata.vocab_size:<10} {metadata.domain:<10} "
                      f"{metadata.language:<8} {metadata.description[:30]}...")


def command_info(args) -> None:
    """Handle the info command."""
    # Try to load as registered tokenizer first
    registry = get_registry()
    metadata = registry.get_metadata(args.tokenizer)
    
    if metadata:
        # Registered tokenizer
        if args.format == 'json':
            print(json.dumps(metadata.__dict__, indent=2, default=str))
        else:
            print(f"Name: {metadata.name}")
            print(f"Description: {metadata.description}")
            print(f"Vocabulary Size: {metadata.vocab_size}")
            print(f"Domain: {metadata.domain}")
            print(f"Language: {metadata.language}")
            print(f"Model Family: {metadata.model_family or 'N/A'}")
            print(f"Version: {metadata.version}")
            print(f"Created by: {metadata.created_by}")
            print(f"License: {metadata.license}")
            print(f"Tags: {', '.join(metadata.tags) if metadata.tags else 'None'}")
            if metadata.config:
                print(f"Configuration: {json.dumps(metadata.config, indent=2)}")
    else:
        # Try to load as file
        try:
            tokenizer = BPETokenizer.load(args.tokenizer)
            info_data = {
                'vocab_size': tokenizer.get_vocab_size(),
                'num_merges': len(tokenizer.merges),
                'special_tokens': {k: v for k, v in tokenizer.vocab.items() 
                                 if k.startswith('<') and k.endswith('>')},
                'config': tokenizer.config.__dict__ if hasattr(tokenizer, 'config') else None
            }
            
            if args.format == 'json':
                print(json.dumps(info_data, indent=2))
            else:
                print(f"Vocabulary Size: {info_data['vocab_size']}")
                print(f"Number of Merges: {info_data['num_merges']}")
                print(f"Special Tokens: {len(info_data['special_tokens'])}")
                for token, token_id in info_data['special_tokens'].items():
                    print(f"  {token}: {token_id}")
        except Exception as e:
            logger.error(f"Could not load tokenizer: {e}")
            sys.exit(1)


def command_search(args) -> None:
    """Handle the search command."""
    results = search_tokenizers(args.query)
    
    if not results:
        print(f"No tokenizers found matching '{args.query}'")
        return
    
    print(f"Search results for '{args.query}':")
    print()
    
    for name, score in results[:args.limit]:
        metadata = get_registry().get_metadata(name)
        if metadata:
            print(f"{name} (relevance: {score:.2f})")
            print(f"  Description: {metadata.description}")
            print(f"  Domain: {metadata.domain}, Language: {metadata.language}")
            print(f"  Vocab Size: {metadata.vocab_size}")
            print()


def command_benchmark(args) -> None:
    """Handle the benchmark command."""
    # Load test data
    texts = load_input_data([args.input])[:args.num_samples]
    if not texts:
        logger.error("No test data found")
        sys.exit(1)
    
    logger.info(f"Benchmarking {len(args.tokenizers)} tokenizers on {len(texts)} samples")
    
    # Run benchmark
    metrics = args.metrics if 'all' not in args.metrics else ['compression', 'speed', 'coverage']
    results = benchmark_tokenizers(args.tokenizers, texts, metrics)
    
    # Format results
    output_data = {
        'benchmark_config': {
            'tokenizers': args.tokenizers,
            'num_samples': len(texts),
            'metrics': metrics
        },
        'results': results
    }
    
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
    else:
        print(json.dumps(output_data, indent=2))


def command_preprocess(args) -> None:
    """Handle the preprocess command."""
    # Load configuration
    config_dict = {}
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
    
    # Override with command line arguments
    config_dict.update({
        'min_text_length': args.min_length,
        'max_text_length': args.max_length,
        'remove_duplicates': args.remove_duplicates,
    })
    
    config = PreprocessingConfig(**config_dict)
    pipeline = PreprocessingPipeline(config)
    
    # Load input data
    texts = load_input_data(args.input)
    logger.info(f"Loaded {len(texts)} texts for preprocessing")
    
    # Process data
    processed_texts = pipeline.process_batch(texts, max_workers=args.workers)
    
    # Save processed data
    with open(args.output, 'w', encoding='utf-8') as f:
        for text in processed_texts:
            f.write(text + '\n\n')
    
    # Print statistics
    stats = pipeline.get_statistics()
    logger.info(f"Preprocessing completed: {stats}")
    logger.info(f"Output saved to: {args.output}")


def command_convert(args) -> None:
    """Handle the convert command."""
    tokenizer = BPETokenizer.load(args.input)
    
    if args.format == 'json':
        # Save as JSON
        output_path = args.output if args.output.endswith('.json') else args.output + '.json'
        tokenizer.save(output_path)
    elif args.format == 'pickle':
        # Save as pickle
        output_path = args.output if args.output.endswith('.pkl') else args.output + '.pkl'
        tokenizer.save(output_path)
    
    logger.info(f"Tokenizer converted and saved to: {output_path}")


def main():
    """Main CLI entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        # Route to appropriate command handler
        command_handlers = {
            'train': command_train,
            'encode': command_encode,
            'decode': command_decode,
            'analyze': command_analyze,
            'list': command_list,
            'info': command_info,
            'search': command_search,
            'benchmark': command_benchmark,
            'preprocess': command_preprocess,
            'convert': command_convert,
        }
        
        handler = command_handlers.get(args.command)
        if handler:
            handler(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()