# Using the Tokenizer in Training Scripts

This guide shows how to integrate the Advanced BPE Tokenizer Library into machine learning training scripts for various frameworks including PyTorch, TensorFlow, and Hugging Face Transformers.

## Table of Contents
1. [Basic Integration](#basic-integration)
2. [PyTorch Integration](#pytorch-integration)
3. [TensorFlow Integration](#tensorflow-integration)
4. [Hugging Face Integration](#hugging-face-integration)
5. [Data Loading and Preprocessing](#data-loading-and-preprocessing)
6. [Custom Dataset Classes](#custom-dataset-classes)
7. [Distributed Training](#distributed-training)
8. [Best Practices](#best-practices)

## Basic Integration

### Simple Training Data Preparation

```python
from tokenizer import load_tokenizer, train_simple_tokenizer, create_preprocessing_pipeline
import numpy as np
from typing import List, Dict, Tuple

def prepare_training_data(texts: List[str], 
                         tokenizer_name: str = None,
                         max_length: int = 512,
                         custom_tokenizer_path: str = None) -> Dict:
    """
    Prepare training data using the tokenizer library.
    
    Args:
        texts: List of training texts
        tokenizer_name: Name of pretrained tokenizer or None for custom
        max_length: Maximum sequence length
        custom_tokenizer_path: Path to custom tokenizer
    
    Returns:
        Dictionary with tokenized data and metadata
    """
    # Load or create tokenizer
    if custom_tokenizer_path:
        from tokenizer import BPETokenizer
        tokenizer = BPETokenizer.load(custom_tokenizer_path)
    elif tokenizer_name:
        tokenizer = load_tokenizer(tokenizer_name)
    else:
        # Train custom tokenizer
        print("Training custom tokenizer...")
        tokenizer = train_simple_tokenizer(
            texts, 
            vocab_size=32000,
            preprocessing=True
        )
        tokenizer.save("custom_tokenizer.pkl")
    
    # Tokenize all texts
    print(f"Tokenizing {len(texts)} texts...")
    tokenized_texts = []
    attention_masks = []
    
    for text in texts:
        tokens = tokenizer.encode(text)
        
        # Truncate or pad to max_length
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
            mask = [1] * max_length
        else:
            mask = [1] * len(tokens) + [0] * (max_length - len(tokens))
            tokens = tokens + [tokenizer.token_to_id("<|pad|>")] * (max_length - len(tokens))
        
        tokenized_texts.append(tokens)
        attention_masks.append(mask)
    
    return {
        'input_ids': np.array(tokenized_texts),
        'attention_mask': np.array(attention_masks),
        'tokenizer': tokenizer,
        'vocab_size': tokenizer.get_vocab_size(),
        'max_length': max_length
    }

# Example usage
training_texts = [
    "This is the first training example.",
    "Here's another piece of text for training.",
    "Machine learning models need lots of data.",
    # ... more texts
]

data = prepare_training_data(training_texts, tokenizer_name="gpt2-small")
print(f"Prepared data shape: {data['input_ids'].shape}")
print(f"Vocabulary size: {data['vocab_size']}")
```

## PyTorch Integration

### Custom Dataset with Tokenizer

```python
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import load_tokenizer, create_preprocessing_pipeline
import json
from typing import Optional, List, Dict

class TokenizedDataset(Dataset):
    """Custom PyTorch Dataset using the tokenizer library."""
    
    def __init__(self, 
                 texts: List[str], 
                 tokenizer_name: str = "gpt2-small",
                 max_length: int = 512,
                 tokenizer_path: Optional[str] = None,
                 preprocessing: bool = True):
        """
        Initialize dataset with tokenizer.
        
        Args:
            texts: List of input texts
            tokenizer_name: Name of tokenizer to use
            max_length: Maximum sequence length
            tokenizer_path: Path to custom tokenizer
            preprocessing: Whether to apply preprocessing
        """
        self.texts = texts
        self.max_length = max_length
        
        # Apply preprocessing if requested
        if preprocessing:
            pipeline = create_preprocessing_pipeline()
            self.texts = pipeline.process_batch(texts)
            print(f"Preprocessing retained {len(self.texts)}/{len(texts)} texts")
        
        # Load tokenizer
        if tokenizer_path:
            from tokenizer import BPETokenizer
            self.tokenizer = BPETokenizer.load(tokenizer_path)
        else:
            self.tokenizer = load_tokenizer(tokenizer_name)
        
        # Get special token IDs
        self.pad_token_id = self.tokenizer.token_to_id("<|pad|>") or 0
        self.eos_token_id = self.tokenizer.token_to_id("<|endoftext|>") or 0
        
        print(f"Dataset initialized with {len(self.texts)} texts")
        print(f"Vocabulary size: {self.tokenizer.get_vocab_size()}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize text
        tokens = self.tokenizer.encode(text)
        
        # Add EOS token
        tokens = tokens + [self.eos_token_id]
        
        # Create attention mask and handle padding/truncation
        if len(tokens) > self.max_length:
            # Truncate
            input_ids = tokens[:self.max_length]
            attention_mask = [1] * self.max_length
        else:
            # Pad
            input_ids = tokens + [self.pad_token_id] * (self.max_length - len(tokens))
            attention_mask = [1] * len(tokens) + [0] * (self.max_length - len(tokens))
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(input_ids, dtype=torch.long)  # For language modeling
        }

# Complete PyTorch Training Example
def train_pytorch_model():
    """Complete example of training a PyTorch model with the tokenizer."""
    
    # Load training data
    with open('training_data.jsonl', 'r') as f:
        texts = [json.loads(line)['text'] for line in f]
    
    # Create dataset
    dataset = TokenizedDataset(
        texts=texts,
        tokenizer_name="gpt2-small",
        max_length=512,
        preprocessing=True
    )
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Example model (simple transformer)
    import torch.nn as nn
    
    class SimpleTransformer(nn.Module):
        def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, nhead, batch_first=True),
                num_layers
            )
            self.output_proj = nn.Linear(d_model, vocab_size)
        
        def forward(self, input_ids, attention_mask=None):
            seq_len = input_ids.size(1)
            embeddings = self.embedding(input_ids) + self.pos_encoding[:seq_len]
            
            # Create attention mask for transformer
            if attention_mask is not None:
                # Convert to transformer format (inverted, with -inf for masked positions)
                mask = (attention_mask == 0).float() * -1e9
            else:
                mask = None
            
            output = self.transformer(embeddings, src_key_padding_mask=mask)
            return self.output_proj(output)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleTransformer(vocab_size=dataset.tokenizer.get_vocab_size())
    model = model.to(device)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_token_id)
    
    # Training loop
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            
            # Calculate loss (shift for next token prediction)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), 
                           shift_labels.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        print(f'Epoch {epoch} completed. Average loss: {total_loss/len(dataloader):.4f}')
    
    # Save model and tokenizer info
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_vocab_size': dataset.tokenizer.get_vocab_size(),
        'max_length': dataset.max_length,
    }, 'trained_model.pt')
    
    return model, dataset.tokenizer

if __name__ == "__main__":
    model, tokenizer = train_pytorch_model()
```

### Sequence-to-Sequence Dataset

```python
class Seq2SeqDataset(Dataset):
    """Dataset for sequence-to-sequence tasks."""
    
    def __init__(self, 
                 source_texts: List[str],
                 target_texts: List[str],
                 tokenizer_name: str = "gpt2-small",
                 max_length: int = 512):
        
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.max_length = max_length
        
        # Load tokenizer
        self.tokenizer = load_tokenizer(tokenizer_name)
        self.pad_token_id = self.tokenizer.token_to_id("<|pad|>") or 0
        self.sos_token_id = self.tokenizer.token_to_id("<|startoftext|>") or 1
        self.eos_token_id = self.tokenizer.token_to_id("<|endoftext|>") or 0
    
    def __len__(self):
        return len(self.source_texts)
    
    def __getitem__(self, idx):
        source_text = self.source_texts[idx]
        target_text = self.target_texts[idx]
        
        # Tokenize source and target
        source_tokens = self.tokenizer.encode(source_text)
        target_tokens = [self.sos_token_id] + self.tokenizer.encode(target_text) + [self.eos_token_id]
        
        # Handle padding/truncation for both sequences
        def pad_or_truncate(tokens, max_len):
            if len(tokens) > max_len:
                return tokens[:max_len], [1] * max_len
            else:
                mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
                tokens = tokens + [self.pad_token_id] * (max_len - len(tokens))
                return tokens, mask
        
        source_ids, source_mask = pad_or_truncate(source_tokens, self.max_length)
        target_ids, target_mask = pad_or_truncate(target_tokens, self.max_length)
        
        return {
            'source_ids': torch.tensor(source_ids, dtype=torch.long),
            'source_mask': torch.tensor(source_mask, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'target_mask': torch.tensor(target_mask, dtype=torch.long)
        }
```

## TensorFlow Integration

### TensorFlow Dataset Integration

```python
import tensorflow as tf
from tokenizer import load_tokenizer, create_preprocessing_pipeline
import numpy as np

class TensorFlowTokenizedDataset:
    """TensorFlow dataset integration with the tokenizer library."""
    
    def __init__(self, 
                 texts: List[str],
                 tokenizer_name: str = "gpt2-small",
                 max_length: int = 512,
                 batch_size: int = 32):
        
        self.texts = texts
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Load tokenizer
        self.tokenizer = load_tokenizer(tokenizer_name)
        self.pad_token_id = self.tokenizer.token_to_id("<|pad|>") or 0
        
        # Preprocess and tokenize all texts
        self.input_ids, self.attention_masks = self._tokenize_texts()
    
    def _tokenize_texts(self):
        """Tokenize all texts and create attention masks."""
        input_ids = []
        attention_masks = []
        
        for text in self.texts:
            tokens = self.tokenizer.encode(text)
            
            # Pad or truncate
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
                mask = [1] * self.max_length
            else:
                mask = [1] * len(tokens) + [0] * (self.max_length - len(tokens))
                tokens = tokens + [self.pad_token_id] * (self.max_length - len(tokens))
            
            input_ids.append(tokens)
            attention_masks.append(mask)
        
        return np.array(input_ids), np.array(attention_masks)
    
    def create_dataset(self):
        """Create TensorFlow Dataset object."""
        dataset = tf.data.Dataset.from_tensor_slices({
            'input_ids': self.input_ids,
            'attention_mask': self.attention_masks,
            'labels': self.input_ids  # For language modeling
        })
        
        return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

# Complete TensorFlow Training Example
def train_tensorflow_model():
    """Complete TensorFlow training example."""
    
    # Load data
    with open('training_data.jsonl', 'r') as f:
        texts = [json.loads(line)['text'] for line in f]
    
    # Create dataset
    tf_dataset = TensorFlowTokenizedDataset(
        texts=texts,
        tokenizer_name="gpt2-small",
        max_length=512,
        batch_size=16
    )
    
    train_dataset = tf_dataset.create_dataset()
    vocab_size = tf_dataset.tokenizer.get_vocab_size()
    
    # Define model
    class SimpleTransformerTF(tf.keras.Model):
        def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6):
            super().__init__()
            self.d_model = d_model
            self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
            self.pos_encoding = self.positional_encoding(1000, d_model)
            
            self.transformer_layers = [
                tf.keras.layers.MultiHeadAttention(num_heads, d_model // num_heads)
                for _ in range(num_layers)
            ]
            self.layer_norms = [
                tf.keras.layers.LayerNormalization()
                for _ in range(num_layers)
            ]
            self.ffn_layers = [
                tf.keras.Sequential([
                    tf.keras.layers.Dense(d_model * 4, activation='relu'),
                    tf.keras.layers.Dense(d_model)
                ])
                for _ in range(num_layers)
            ]
            
            self.final_layer_norm = tf.keras.layers.LayerNormalization()
            self.output_projection = tf.keras.layers.Dense(vocab_size)
        
        def positional_encoding(self, position, d_model):
            angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, 
                (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
            angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
            angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
            pos_encoding = angle_rads[np.newaxis, ...]
            return tf.cast(pos_encoding, dtype=tf.float32)
        
        def call(self, inputs, training=None):
            input_ids = inputs['input_ids']
            attention_mask = inputs.get('attention_mask')
            
            seq_len = tf.shape(input_ids)[1]
            
            # Embedding + positional encoding
            embeddings = self.embedding(input_ids)
            embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            embeddings += self.pos_encoding[:, :seq_len, :]
            
            # Create attention mask
            if attention_mask is not None:
                attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
                attention_mask = (1.0 - tf.cast(attention_mask, tf.float32)) * -1e9
            
            # Transformer layers
            x = embeddings
            for i in range(len(self.transformer_layers)):
                # Multi-head attention
                attn_output = self.transformer_layers[i](
                    x, x, attention_mask=attention_mask, training=training
                )
                x = self.layer_norms[i](x + attn_output, training=training)
                
                # Feed forward
                ffn_output = self.ffn_layers[i](x, training=training)
                x = self.layer_norms[i](x + ffn_output, training=training)
            
            x = self.final_layer_norm(x, training=training)
            return self.output_projection(x)
    
    # Create and compile model
    model = SimpleTransformerTF(vocab_size)
    
    # Define loss function that ignores padding tokens
    def masked_loss(y_true, y_pred):
        mask = tf.math.logical_not(tf.math.equal(y_true, tf_dataset.pad_token_id))
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
        loss = tf.boolean_mask(loss, mask)
        return tf.reduce_mean(loss)
    
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4),
        loss=masked_loss,
        metrics=['accuracy']
    )
    
    # Training
    history = model.fit(
        train_dataset,
        epochs=3,
        verbose=1
    )
    
    # Save model
    model.save('tensorflow_model')
    
    return model, tf_dataset.tokenizer

# Run training
if __name__ == "__main__":
    model, tokenizer = train_tensorflow_model()
```

## Hugging Face Integration

### Custom Tokenizer Wrapper

```python
from transformers import PreTrainedTokenizer
from tokenizer import BPETokenizer, load_tokenizer
from typing import List, Optional, Dict, Any

class HuggingFaceTokenizerWrapper(PreTrainedTokenizer):
    """
    Wrapper to make our tokenizer compatible with Hugging Face Transformers.
    """
    
    def __init__(self, 
                 tokenizer_name: Optional[str] = None,
                 tokenizer_path: Optional[str] = None,
                 **kwargs):
        
        # Load our tokenizer
        if tokenizer_path:
            self.tokenizer = BPETokenizer.load(tokenizer_path)
        elif tokenizer_name:
            self.tokenizer = load_tokenizer(tokenizer_name)
        else:
            raise ValueError("Must specify either tokenizer_name or tokenizer_path")
        
        # Set up special tokens
        self.pad_token = "<|pad|>"
        self.unk_token = "<|unk|>"
        self.eos_token = "<|endoftext|>"
        self.bos_token = "<|startoftext|>"
        
        super().__init__(
            pad_token=self.pad_token,
            unk_token=self.unk_token,
            eos_token=self.eos_token,
            bos_token=self.bos_token,
            **kwargs
        )
    
    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()
    
    def get_vocab(self):
        return self.tokenizer.get_vocab()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text and return token strings."""
        token_ids = self.tokenizer.encode(text)
        return [self.tokenizer.id_to_token(tid) or self.unk_token for tid in token_ids]
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to ID."""
        return self.tokenizer.token_to_id(token) or self.tokenizer.token_to_id(self.unk_token)
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert ID to token."""
        return self.tokenizer.id_to_token(index) or self.unk_token
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert tokens back to string."""
        # Join tokens and clean up
        return "".join(tokens).replace("Ġ", " ").strip()
    
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """Build model inputs from sequences by adding special tokens."""
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]
    
    def save_pretrained(self, save_directory: str, **kwargs):
        """Save tokenizer to directory."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        self.tokenizer.save(os.path.join(save_directory, "tokenizer.pkl"))
        super().save_pretrained(save_directory, **kwargs)

# Usage with Hugging Face Transformers
def train_with_huggingface():
    """Train using Hugging Face Transformers with custom tokenizer."""
    
    from transformers import (
        GPT2LMHeadModel, 
        GPT2Config, 
        Trainer, 
        TrainingArguments,
        DataCollatorForLanguageModeling
    )
    
    # Initialize custom tokenizer
    tokenizer = HuggingFaceTokenizerWrapper(tokenizer_name="gpt2-small")
    
    # Create model config
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=512,
        n_ctx=512,
        n_embd=768,
        n_layer=12,
        n_head=12,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Initialize model
    model = GPT2LMHeadModel(config)
    
    # Prepare dataset
    class SimpleDataset:
        def __init__(self, texts, tokenizer, max_length=512):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = self.texts[idx]
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': encoding['input_ids'].flatten()
            }
    
    # Load training data
    with open('training_data.jsonl', 'r') as f:
        texts = [json.loads(line)['text'] for line in f]
    
    train_dataset = SimpleDataset(texts, tokenizer)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal language modeling
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        logging_steps=100,
        save_steps=1000,
        evaluation_strategy='no',
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=True if torch.cuda.is_available() else False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    trainer.train()
    
    # Save
    trainer.save_model()
    tokenizer.save_pretrained('./results')
    
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = train_with_huggingface()
```

## Data Loading and Preprocessing

### Efficient Data Pipeline

```python
from tokenizer import create_preprocessing_pipeline, BatchProcessor
import multiprocessing as mp
from pathlib import Path
import json

class DataPipeline:
    """Comprehensive data loading and preprocessing pipeline."""
    
    def __init__(self, 
                 tokenizer_name: str = "gpt2-small",
                 max_length: int = 512,
                 preprocessing: bool = True,
                 batch_size: int = 1000,
                 num_workers: int = None):
        
        self.tokenizer = load_tokenizer(tokenizer_name)
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers or mp.cpu_count()
        
        # Setup preprocessing
        if preprocessing:
            self.preprocessing_pipeline = create_preprocessing_pipeline(
                min_text_length=50,
                max_text_length=50000,
                remove_duplicates=True,
                quality_threshold=0.6
            )
        else:
            self.preprocessing_pipeline = None
        
        # Setup batch processor
        self.batch_processor = BatchProcessor(
            batch_size=batch_size,
            max_workers=self.num_workers
        )
    
    def load_from_files(self, file_paths: List[str]) -> List[str]:
        """Load texts from multiple files."""
        all_texts = []
        
        for file_path in file_paths:
            path = Path(file_path)
            
            if path.suffix == '.jsonl':
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if 'text' in data:
                                all_texts.append(data['text'])
                        except json.JSONDecodeError:
                            continue
            else:
                # Assume plain text
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    # Split by paragraphs
                    paragraphs = text.split('\n\n')
                    all_texts.extend([p.strip() for p in paragraphs if p.strip()])
        
        print(f"Loaded {len(all_texts)} texts from {len(file_paths)} files")
        return all_texts
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """Apply preprocessing pipeline to texts."""
        if not self.preprocessing_pipeline:
            return texts
        
        print(f"Preprocessing {len(texts)} texts...")
        processed = self.preprocessing_pipeline.process_batch(
            texts, 
            max_workers=self.num_workers
        )
        
        stats = self.preprocessing_pipeline.get_statistics()
        print(f"Preprocessing stats: {stats}")
        return processed
    
    def tokenize_batch(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Tokenize a batch of texts efficiently."""
        def tokenize_single(text):
            tokens = self.tokenizer.encode(text)
            
            # Handle padding/truncation
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
                mask = [1] * self.max_length
            else:
                mask = [1] * len(tokens) + [0] * (self.max_length - len(tokens))
                tokens = tokens + [self.tokenizer.token_to_id("<|pad|>")] * (self.max_length - len(tokens))
            
            return tokens, mask
        
        # Process in parallel
        results = self.batch_processor.process_texts_parallel(
            texts, 
            tokenize_single,
            use_processes=True
        )
        
        # Separate tokens and masks
        all_tokens = []
        all_masks = []
        for tokens, mask in results:
            all_tokens.append(tokens)
            all_masks.append(mask)
        
        return {
            'input_ids': np.array(all_tokens),
            'attention_mask': np.array(all_masks)
        }
    
    def create_training_data(self, file_paths: List[str]) -> Dict[str, np.ndarray]:
        """Complete pipeline from files to tokenized data."""
        # Load texts
        texts = self.load_from_files(file_paths)
        
        # Preprocess
        texts = self.preprocess_texts(texts)
        
        # Tokenize in batches for memory efficiency
        all_input_ids = []
        all_attention_masks = []
        
        print(f"Tokenizing {len(texts)} texts in batches...")
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_data = self.tokenize_batch(batch_texts)
            
            all_input_ids.append(batch_data['input_ids'])
            all_attention_masks.append(batch_data['attention_mask'])
            
            if (i // self.batch_size) % 10 == 0:
                print(f"Processed {i + len(batch_texts)}/{len(texts)} texts")
        
        # Concatenate all batches
        final_input_ids = np.vstack(all_input_ids)
        final_attention_masks = np.vstack(all_attention_masks)
        
        print(f"Final data shape: {final_input_ids.shape}")
        
        return {
            'input_ids': final_input_ids,
            'attention_mask': final_attention_masks,
            'tokenizer': self.tokenizer,
            'vocab_size': self.tokenizer.get_vocab_size()
        }

# Usage example
def prepare_large_dataset():
    """Example of preparing a large dataset efficiently."""
    
    # Initialize data pipeline
    pipeline = DataPipeline(
        tokenizer_name="gpt2-small",
        max_length=1024,
        preprocessing=True,
        batch_size=2000,
        num_workers=8
    )
    
    # Prepare file paths
    data_dir = Path("training_data")
    file_paths = list(data_dir.glob("*.jsonl")) + list(data_dir.glob("*.txt"))
    
    # Create training data
    training_data = pipeline.create_training_data([str(p) for p in file_paths])
    
    # Save processed data for reuse
    np.savez_compressed(
        'processed_training_data.npz',
        **training_data
    )
    
    return training_data
```

## Custom Dataset Classes

### Memory-Efficient Dataset for Large Corpora

```python
import mmap
import pickle
from typing import Iterator

class MemoryMappedDataset(Dataset):
    """Memory-mapped dataset for very large corpora."""
    
    def __init__(self, 
                 data_file: str,
                 tokenizer_name: str = "gpt2-small",
                 max_length: int = 512,
                 cache_file: str = None):
        
        self.data_file = data_file
        self.max_length = max_length
        self.tokenizer = load_tokenizer(tokenizer_name)
        
        # Special token IDs
        self.pad_token_id = self.tokenizer.token_to_id("<|pad|>") or 0
        self.eos_token_id = self.tokenizer.token_to_id("<|endoftext|>") or 0
        
        # Build or load index
        self.cache_file = cache_file or f"{data_file}.cache"
        self.line_offsets = self._build_or_load_index()
        
        print(f"Dataset initialized with {len(self.line_offsets)} samples")
    
    def _build_or_load_index(self) -> List[int]:
        """Build or load file line offset index for fast random access."""
        
        if Path(self.cache_file).exists():
            print("Loading cached index...")
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        
        print("Building index...")
        offsets = []
        with open(self.data_file, 'rb') as f:
            offset = 0
            for line in f:
                offsets.append(offset)
                offset += len(line)
        
        # Cache the index
        with open(self.cache_file, 'wb') as f:
            pickle.dump(offsets, f)
        
        return offsets
    
    def __len__(self):
        return len(self.line_offsets)
    
    def __getitem__(self, idx):
        # Get line at specific offset
        with open(self.data_file, 'r', encoding='utf-8') as f:
            f.seek(self.line_offsets[idx])
            line = f.readline().strip()
        
        # Parse JSON if needed
        try:
            import json
            data = json.loads(line)
            text = data.get('text', line)
        except:
            text = line
        
        # Tokenize
        tokens = self.tokenizer.encode(text) + [self.eos_token_id]
        
        # Pad or truncate
        if len(tokens) > self.max_length:
            input_ids = tokens[:self.max_length]
            attention_mask = [1] * self.max_length
        else:
            attention_mask = [1] * len(tokens) + [0] * (self.max_length - len(tokens))
            input_ids = tokens + [self.pad_token_id] * (self.max_length - len(tokens))
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(input_ids, dtype=torch.long)
        }

class StreamingDataset:
    """Streaming dataset that doesn't load everything into memory."""
    
    def __init__(self, 
                 file_paths: List[str],
                 tokenizer_name: str = "gpt2-small",
                 max_length: int = 512,
                 buffer_size: int = 10000,
                 shuffle_buffer: bool = True):
        
        self.file_paths = file_paths
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.shuffle_buffer = shuffle_buffer
        
        # Load tokenizer
        self.tokenizer = load_tokenizer(tokenizer_name)
        self.pad_token_id = self.tokenizer.token_to_id("<|pad|>") or 0
        self.eos_token_id = self.tokenizer.token_to_id("<|endoftext|>") or 0
        
        # Setup preprocessing
        self.preprocessing = create_preprocessing_pipeline(
            min_text_length=20,
            max_text_length=10000
        )
    
    def text_generator(self) -> Iterator[str]:
        """Generate texts from files."""
        for file_path in self.file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.jsonl'):
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if 'text' in data:
                                yield data['text']
                        except:
                            continue
                else:
                    # Process as plain text, split by paragraphs
                    content = f.read()
                    for paragraph in content.split('\n\n'):
                        if paragraph.strip():
                            yield paragraph.strip()
    
    def __iter__(self):
        """Create iterator over processed and tokenized texts."""
        buffer = []
        
        for text in self.text_generator():
            # Preprocess text
            processed = self.preprocessing.process_single_text(text)
            if processed is None:
                continue
            
            # Tokenize
            tokens = self.tokenizer.encode(processed) + [self.eos_token_id]
            
            # Pad or truncate
            if len(tokens) > self.max_length:
                input_ids = tokens[:self.max_length]
                attention_mask = [1] * self.max_length
            else:
                attention_mask = [1] * len(tokens) + [0] * (self.max_length - len(tokens))
                input_ids = tokens + [self.pad_token_id] * (self.max_length - len(tokens))
            
            sample = {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'labels': torch.tensor(input_ids, dtype=torch.long)
            }
            
            buffer.append(sample)
            
            # Yield from buffer when full
            if len(buffer) >= self.buffer_size:
                if self.shuffle_buffer:
                    import random
                    random.shuffle(buffer)
                
                for sample in buffer:
                    yield sample
                buffer = []
        
        # Yield remaining samples
        if buffer:
            if self.shuffle_buffer:
                import random
                random.shuffle(buffer)
            for sample in buffer:
                yield sample
```

## Distributed Training

### PyTorch Distributed Training

```python
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_distributed(rank, world_size):
    """Setup distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set device
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()

def train_distributed(rank, world_size, dataset_path, tokenizer_name):
    """Distributed training function."""
    
    # Setup
    setup_distributed(rank, world_size)
    
    # Create dataset
    dataset = MemoryMappedDataset(
        dataset_path,
        tokenizer_name=tokenizer_name,
        max_length=1024
    )
    
    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=16,  # Per-GPU batch size
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = SimpleTransformer(vocab_size=dataset.tokenizer.get_vocab_size())
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_token_id)
    
    # Training loop
    model.train()
    for epoch in range(3):
        sampler.set_epoch(epoch)  # Important for proper shuffling
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            input_ids = batch['input_ids'].to(rank, non_blocking=True)
            attention_mask = batch['attention_mask'].to(rank, non_blocking=True)
            labels = batch['labels'].to(rank, non_blocking=True)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            
            # Calculate loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), 
                           shift_labels.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0 and rank == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    # Save model (only on rank 0)
    if rank == 0:
        torch.save({
            'model_state_dict': model.module.state_dict(),
            'tokenizer_vocab_size': dataset.tokenizer.get_vocab_size(),
        }, 'distributed_model.pt')
    
    cleanup_distributed()

def main_distributed():
    """Main function for distributed training."""
    world_size = torch.cuda.device_count()
    print(f"Training on {world_size} GPUs")
    
    mp.spawn(
        train_distributed,
        args=(world_size, 'large_dataset.jsonl', 'gpt2-small'),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main_distributed()
```

## Best Practices

### 1. Tokenizer Selection and Configuration

```python
def choose_optimal_tokenizer(domain: str, dataset_size: int) -> str:
    """Choose the best tokenizer for your use case."""
    
    if domain == "code":
        return "code-tokenizer"
    elif domain == "scientific":
        return "scientific-tokenizer"
    elif dataset_size > 10_000_000:  # Large dataset
        return "gpt2-medium"  # Larger vocab for better compression
    else:
        return "gpt2-small"  # Standard choice

def configure_for_domain(domain: str) -> Dict:
    """Get optimal configuration for specific domains."""
    
    configs = {
        "code": {
            "max_length": 2048,  # Code can be longer
            "preprocessing": {
                "preserve_spaces": True,
                "normalize_unicode": False,
                "remove_urls": False,
            }
        },
        "scientific": {
            "max_length": 1024,
            "preprocessing": {
                "fix_html_entities": True,
                "remove_urls": False,  # Keep citation URLs
            }
        },
        "general": {
            "max_length": 512,
            "preprocessing": {
                "remove_duplicates": True,
                "quality_threshold": 0.6,
            }
        }
    }
    
    return configs.get(domain, configs["general"])
```

### 2. Memory and Performance Optimization

```python
def optimize_for_memory(dataset_size: int, available_ram_gb: int) -> Dict:
    """Optimize settings based on available memory."""
    
    if dataset_size > 1_000_000 or available_ram_gb < 16:
        return {
            "use_streaming": True,
            "batch_size": 8,
            "num_workers": 2,
            "pin_memory": False,
            "buffer_size": 1000,
        }
    else:
        return {
            "use_streaming": False,
            "batch_size": 32,
            "num_workers": 4,
            "pin_memory": True,
            "buffer_size": 10000,
        }

def setup_efficient_training(config: Dict):
    """Setup training with memory-efficient settings."""
    
    if config["use_streaming"]:
        # Use streaming dataset
        dataset = StreamingDataset(
            file_paths=config["data_files"],
            buffer_size=config["buffer_size"],
            max_length=config["max_length"]
        )
        # Convert to PyTorch DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            num_workers=0,  # Streaming datasets work better with num_workers=0
        )
    else:
        # Use regular dataset
        dataset = TokenizedDataset(
            texts=load_all_texts(config["data_files"]),
            max_length=config["max_length"]
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"]
        )
    
    return dataloader
```

### 3. Monitoring and Validation

```python
def setup_training_monitoring(tokenizer, validation_texts: List[str]):
    """Setup monitoring and validation during training."""
    
    def validate_tokenizer_quality():
        """Validate tokenizer quality on sample texts."""
        from tokenizer import analyze_tokenization_quality
        
        quality_metrics = analyze_tokenization_quality(
            tokenizer, 
            validation_texts[:100],  # Sample for validation
            detailed=False
        )
        
        return quality_metrics
    
    def log_training_metrics(epoch, batch_idx, loss, model, dataloader):
        """Log comprehensive training metrics."""
        
        if batch_idx % 100 == 0:
            # Basic training metrics
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}")
            
            # Memory usage (if using CUDA)
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
                print(f"GPU Memory: {memory_used:.2f}GB used, {memory_cached:.2f}GB cached")
            
            # Tokenizer quality (periodic check)
            if batch_idx % 1000 == 0:
                quality = validate_tokenizer_quality()
                print(f"Tokenizer Quality - Compression: {quality['compression_ratio']:.2f}, "
                      f"Coverage: {quality['coverage_rate']:.2%}")
    
    return log_training_metrics

# Complete training script with monitoring
def train_with_monitoring():
    """Complete training script with comprehensive monitoring."""
    
    # Load configuration
    config = {
        "domain": "general",
        "dataset_size": 100000,
        "available_ram_gb": 32,
        "data_files": ["training_data.jsonl"],
        "max_length": 512,
        "epochs": 3,
        "validation_texts": ["Sample validation text", "Another example"]
    }
    
    # Optimize for resources
    opt_config = optimize_for_memory(config["dataset_size"], config["available_ram_gb"])
    config.update(opt_config)
    
    # Setup data loading
    dataloader = setup_efficient_training(config)
    
    # Get tokenizer from dataset
    if hasattr(dataloader.dataset, 'tokenizer'):
        tokenizer = dataloader.dataset.tokenizer
    else:
        tokenizer = load_tokenizer("gpt2-small")
    
    # Setup monitoring
    log_metrics = setup_training_monitoring(tokenizer, config["validation_texts"])
    
    # Initialize model
    model = SimpleTransformer(vocab_size=tokenizer.get_vocab_size())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<|pad|>"))
    
    # Training loop with monitoring
    model.train()
    for epoch in range(config["epochs"]):
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            
            # Calculate loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), 
                           shift_labels.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Monitoring and logging
            log_metrics(epoch, batch_idx, loss.item(), model, dataloader)
    
    # Final validation
    print("\nFinal model validation:")
    with torch.no_grad():
        model.eval()
        sample_text = "This is a test of the trained model."
        tokens = tokenizer.encode(sample_text)
        input_ids = torch.tensor([tokens]).to(device)
        
        outputs = model(input_ids)
        predictions = torch.softmax(outputs.logits[0, -1, :], dim=0)
        top_tokens = torch.topk(predictions, 5)
        
        print(f"Input: {sample_text}")
        print("Top 5 next token predictions:")
        for i, (score, token_id) in enumerate(zip(top_tokens.values, top_tokens.indices)):
            token = tokenizer.id_to_token(token_id.item())
            print(f"  {i+1}. {token}: {score:.4f}")
    
    # Save everything
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'tokenizer_vocab_size': tokenizer.get_vocab_size(),
        'final_loss': loss.item(),
    }
    torch.save(save_dict, 'final_trained_model.pt')
    
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = train_with_monitoring()
```

This comprehensive guide shows how to integrate the tokenizer library into various training frameworks with best practices for performance, memory management, and monitoring. The examples cover everything from simple usage to production-ready distributed training setups.