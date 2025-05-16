#!/usr/bin/env python
"""
Script to extract activations from a model for a given dataset.
"""

import os
import argparse
import logging
import torch
import sys
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.activation_utils import extract_activations, create_activation_dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_prepare_dataset(
    dataset_name: str,
    text_column: str,
    split: str = "train",
    num_samples: int = None,
    shuffle: bool = True,
    seed: int = 42
):
    """
    Load and prepare dataset for activation extraction.
    
    Args:
        dataset_name: HuggingFace dataset name or path
        text_column: Column name containing text data
        split: Dataset split to use
        num_samples: Maximum number of samples to use
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        
    Returns:
        List of text strings
    """
    # Load dataset
    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {e}")
        raise
    
    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    
    # Limit samples if requested
    if num_samples and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))
    
    # Extract text column
    if text_column not in dataset.column_names:
        raise ValueError(f"Column '{text_column}' not found in dataset. Available columns: {dataset.column_names}")
    
    texts = dataset[text_column]
    logger.info(f"Loaded {len(texts)} examples from dataset")
    
    return texts


def main():
    """Main entry point for script."""
    parser = argparse.ArgumentParser(description="Extract activations from a model for a given dataset")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFace model name or path")
    parser.add_argument("--model_revision", type=str, default="main",
                        help="Model revision to use")
    parser.add_argument("--layer_idx", type=int, default=-1,
                        help="Layer index to extract activations from")
    
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="HuggingFace dataset name or path")
    parser.add_argument("--text_column", type=str, required=True,
                        help="Column name containing text data")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to use")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Maximum number of samples to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")
    
    # Processing arguments
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for processing")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: auto-detect)")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Data type for computation")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./data/activations",
                        help="Output directory")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Output filename (default: auto-generate)")
    
    args = parser.parse_args()
    
    # Auto-detect device if not specified
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Convert string dtype to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    torch_dtype = dtype_map.get(args.dtype, torch.float32)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Auto-generate output name if not specified
    if args.output_name is None:
        model_short_name = args.model_name.split("/")[-1]
        args.output_name = f"{model_short_name}_layer{args.layer_idx}_activations.pt"
    
    output_path = os.path.join(args.output_dir, args.output_name)
    
    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_name}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            revision=args.model_revision,
            torch_dtype=torch_dtype,
            device_map=args.device
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, 
            revision=args.model_revision
        )
        
        # Add padding token if not already defined
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_name}")
    texts = load_and_prepare_dataset(
        dataset_name=args.dataset_name,
        text_column=args.text_column,
        split=args.split,
        num_samples=args.num_samples,
        shuffle=True,
        seed=args.seed
    )
    
    # Extract activations
    logger.info(f"Extracting activations from layer {args.layer_idx}")
    activations, token_ids = extract_activations(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        layer_idx=args.layer_idx,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device
    )
    
    # Create and save dataset
    logger.info(f"Creating activation dataset")
    _ = create_activation_dataset(
        activations=activations,
        token_ids=token_ids,
        save_path=output_path
    )
    
    logger.info(f"Saved activation dataset to {output_path}")
    logger.info(f"Activations shape: {activations.shape}")


if __name__ == "__main__":
    main() 