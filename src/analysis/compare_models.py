#!/usr/bin/env python
import os
import argparse
import logging
import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.sae import SparseAutoEncoder
from src.utils.activation_utils import extract_activations, get_model_residuals


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_models_and_tokenizer(
    base_model_name: str,
    ft_model_name: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype = torch.float16
):
    """
    Load base and fine-tuned models along with tokenizer.
    
    Args:
        base_model_name: Name or path of the base model
        ft_model_name: Name or path of the fine-tuned model
        device: Device to load models on
        dtype: Data type for model weights
    
    Returns:
        Tuple of (base_model, ft_model, tokenizer)
    """
    logger.info(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map=device
    )
    
    logger.info(f"Loading fine-tuned model: {ft_model_name}")
    ft_model = AutoModelForCausalLM.from_pretrained(
        ft_model_name,
        torch_dtype=dtype,
        device_map=device
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return base_model, ft_model, tokenizer


def extract_paired_activations(
    base_model: AutoModelForCausalLM,
    ft_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    layer_idx: int = -1,
    batch_size: int = 16,
    max_length: int = 512,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract activations from the same layer of both models.
    
    Args:
        base_model: Base model
        ft_model: Fine-tuned model
        tokenizer: Tokenizer to use
        texts: List of text inputs
        layer_idx: Index of the layer to extract from
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        device: Device to use
    
    Returns:
        Tuple of (base_activations, ft_activations)
    """
    logger.info("Extracting activations from base model")
    base_activations, base_tokens = extract_activations(
        model=base_model,
        tokenizer=tokenizer,
        texts=texts,
        layer_idx=layer_idx,
        batch_size=batch_size,
        max_length=max_length,
        device=device
    )
    
    logger.info("Extracting activations from fine-tuned model")
    ft_activations, ft_tokens = extract_activations(
        model=ft_model,
        tokenizer=tokenizer,
        texts=texts,
        layer_idx=layer_idx,
        batch_size=batch_size,
        max_length=max_length,
        device=device
    )
    
    assert all(b == f for b, f in zip(base_tokens, ft_tokens)), "Token mismatch between models"
    
    return base_activations, ft_activations


def compute_delta_sae_activations(
    base_activations: torch.Tensor,
    ft_activations: torch.Tensor,
    base_sae_path: str,
    delta_sae_path: str = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Compute activations of the Delta-SAE or train a new one.
    
    Args:
        base_activations: Activations from base model
        ft_activations: Activations from fine-tuned model
        base_sae_path: Path to pretrained SAE for base model
        delta_sae_path: Path to pretrained Delta-SAE (optional)
        device: Computation device
    
    Returns:
        Dictionary with activation results
    """
    if base_activations.dim() == 3:
        base_activations = base_activations.reshape(-1, base_activations.size(-1))
        ft_activations = ft_activations.reshape(-1, ft_activations.size(-1))
    
    logger.info(f"Loading base SAE from {base_sae_path}")
    base_sae = SparseAutoEncoder.load_model(base_sae_path, device=device)
    base_sae.eval()
    
    with torch.no_grad():
        base_reconstructed, base_features = base_sae(base_activations.to(device))
        base_residual = base_activations.to(device) - base_reconstructed
        
        ft_reconstructed, ft_features = base_sae(ft_activations.to(device))
        ft_residual = ft_activations.to(device) - ft_reconstructed

    delta_features = None
    if delta_sae_path and os.path.exists(delta_sae_path):
        logger.info(f"Loading Delta SAE from {delta_sae_path}")
        delta_sae = SparseAutoEncoder.load_model(delta_sae_path, device=device)
        delta_sae.eval()
        
        with torch.no_grad():
            _, delta_features = delta_sae(ft_residual)
    
    results = {
        "base_features": base_features.cpu(),
        "ft_features": ft_features.cpu(),
        "base_residual": base_residual.cpu(),
        "ft_residual": ft_residual.cpu(),
    }
    
    if delta_features is not None:
        results["delta_features"] = delta_features.cpu()
    
    return results


def analyze_feature_differences(
    results: Dict[str, torch.Tensor],
    output_dir: str = "./analysis_results",
    top_k: int = 20
) -> Dict[str, Any]:
    """
    Analyze differences between base and fine-tuned features.
    
    Args:
        results: Dictionary with activation results
        output_dir: Directory to save analysis results
        top_k: Number of top features to analyze in detail
    
    Returns:
        Dictionary with analysis metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    base_features = results["base_features"]
    ft_features = results["ft_features"]
    
    base_active = (base_features > 0).float().mean(dim=0)
    ft_active = (ft_features > 0).float().mean(dim=0)
    
    base_mean = base_features.mean(dim=0)
    ft_mean = ft_features.mean(dim=0)
    
    mean_diff = (ft_mean - base_mean).abs()
    freq_diff = (ft_active - base_active).abs()
    
    _, top_mean_idx = torch.topk(mean_diff, k=top_k)
    _, top_freq_idx = torch.topk(freq_diff, k=top_k)
    
    base_residual_norm = results["base_residual"].pow(2).sum(dim=1).sqrt().mean().item()
    ft_residual_norm = results["ft_residual"].pow(2).sum(dim=1).sqrt().mean().item()
    
    analysis = {
        "base_active_freq": base_active.numpy(),
        "ft_active_freq": ft_active.numpy(),
        "base_mean_act": base_mean.numpy(),
        "ft_mean_act": ft_mean.numpy(),
        "mean_diff": mean_diff.numpy(),
        "freq_diff": freq_diff.numpy(),
        "top_mean_idx": top_mean_idx.numpy().tolist(),
        "top_freq_idx": top_freq_idx.numpy().tolist(),
        "base_residual_norm": base_residual_norm,
        "ft_residual_norm": ft_residual_norm,
    }
    

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(base_active.numpy(), label="Base Model", alpha=0.6)
    sns.histplot(ft_active.numpy(), label="Fine-tuned Model", alpha=0.6)
    plt.title("Feature Activation Frequency")
    plt.xlabel("Fraction of samples where feature is active")
    plt.ylabel("Count")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    sns.histplot(freq_diff.numpy(), label="Frequency Difference")
    plt.title("Activation Frequency Difference")
    plt.xlabel("Absolute difference in activation frequency")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "activation_frequency.png"))
    plt.close()
    

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(base_mean.numpy(), label="Base Model", alpha=0.6)
    sns.histplot(ft_mean.numpy(), label="Fine-tuned Model", alpha=0.6)
    plt.title("Mean Feature Activation")
    plt.xlabel("Mean activation value")
    plt.ylabel("Count")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    sns.histplot(mean_diff.numpy(), label="Magnitude Difference")
    plt.title("Activation Magnitude Difference")
    plt.xlabel("Absolute difference in mean activation")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "activation_magnitude.png"))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.bar(["Base Model", "Fine-tuned Model"], 
            [base_residual_norm, ft_residual_norm])
    plt.title("Mean Residual L2 Norm")
    plt.xlabel("Model")
    plt.ylabel("L2 Norm")
    plt.savefig(os.path.join(output_dir, "residual_comparison.png"))
    plt.close()
    
    torch.save(analysis, os.path.join(output_dir, "analysis_results.pt"))
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")
    
    logger.info("=== Analysis Summary ===")
    logger.info(f"Base model residual norm: {base_residual_norm:.4f}")
    logger.info(f"Fine-tuned model residual norm: {ft_residual_norm:.4f}")
    logger.info(f"Residual norm increase: {(ft_residual_norm - base_residual_norm):.4f} " + 
               f"({(ft_residual_norm - base_residual_norm) / base_residual_norm:.2%})")
    logger.info(f"Top {top_k} features with largest mean activation change: {top_mean_idx.tolist()}")
    logger.info(f"Top {top_k} features with largest frequency change: {top_freq_idx.tolist()}")
    
    return analysis


def main():
    """Main entry point for script."""
    parser = argparse.ArgumentParser(description="Compare base and fine-tuned models using SAE")
    
    parser.add_argument("--base_model", type=str, required=True,
                        help="Name or path of the base model")
    parser.add_argument("--ft_model", type=str, required=True,
                        help="Name or path of the fine-tuned model")
    parser.add_argument("--layer_idx", type=int, default=-1,
                        help="Layer index to analyze")
    
    parser.add_argument("--base_sae_path", type=str, required=True,
                        help="Path to pretrained SAE for base model")
    parser.add_argument("--delta_sae_path", type=str, default=None,
                        help="Path to pretrained Delta-SAE (optional)")
    
    parser.add_argument("--dataset_name", type=str, default="wikimedia/wikipedia",
                        help="Dataset to use for analysis")
    parser.add_argument("--text_column", type=str, default="text",
                        help="Column containing text data")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to analyze")
    
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for processing")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Number of top features to analyze in detail")
    
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: auto-detect)")
    parser.add_argument("--output_dir", type=str, default="./analysis_results",
                        help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    base_model, ft_model, tokenizer = load_models_and_tokenizer(
        base_model_name=args.base_model,
        ft_model_name=args.ft_model,
        device=args.device
    )
    
    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train")
    
    if args.num_samples and args.num_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(args.num_samples))
    
    texts = dataset[args.text_column]
    logger.info(f"Analyzing {len(texts)} text samples")
    
    base_activations, ft_activations = extract_paired_activations(
        base_model=base_model,
        ft_model=ft_model,
        tokenizer=tokenizer,
        texts=texts,
        layer_idx=args.layer_idx,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device
    )
    
    results = compute_delta_sae_activations(
        base_activations=base_activations,
        ft_activations=ft_activations,
        base_sae_path=args.base_sae_path,
        delta_sae_path=args.delta_sae_path,
        device=args.device
    )
    
    analysis = analyze_feature_differences(
        results=results,
        output_dir=args.output_dir,
        top_k=args.top_k
    )
    
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main() 