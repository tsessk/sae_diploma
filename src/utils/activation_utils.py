import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional, Tuple, Callable, Union, Any
import os
import logging
from tqdm.auto import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActivationHook:
    """Hook for extracting activations from a model layer."""

    def __init__(self, layer, hook_fn=None):
        """
        Args:
            layer: Layer to extract activations from
            hook_fn: Custom hook function (optional)
        """
        self.layer = layer
        self.hook_fn = hook_fn if hook_fn else self._default_hook_fn
        self.activations = None
        self.handle = None
    
    def _default_hook_fn(self, module, input, output):
        """Default hook function that saves output activations."""

        self.activations = output.detach().cpu()
    
    def __enter__(self):
        self.handle = self.layer.register_forward_hook(self.hook_fn)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle:
            self.handle.remove()


def extract_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    layer_idx: int = -1,
    batch_size: int = 32,
    max_length: int = 512,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[torch.Tensor, List[List[int]]]:
    """
    Extract activations from a specific layer of the model for given texts.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        texts: List of text strings to extract activations for
        layer_idx: Index of layer to extract from (negative indices like -1 for last layer)
        batch_size: Batch size for processing
        max_length: Max sequence length for tokenization
        device: Device to run inference on
        
    Returns:
        Tuple of (activations tensor, list of token ids)
    """
    model.eval()
    model.to(device)
    
    if hasattr(model, "transformer"):
        target_module = model.transformer.h[layer_idx] if layer_idx >= 0 else model.transformer.h[layer_idx]
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        target_module = model.model.layers[layer_idx] if layer_idx >= 0 else model.model.layers[layer_idx]
    else:
        raise ValueError(f"Unsupported model architecture: {type(model)}")
    
    all_activations = []
    all_token_ids = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting activations"):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        
        batch_token_ids = inputs["input_ids"].cpu().tolist()
        all_token_ids.extend(batch_token_ids)
        

        with ActivationHook(target_module) as hook:
            with torch.no_grad():
                outputs = model(**inputs)
            
            batch_activations = hook.activations
            all_activations.append(batch_activations)
    
    activations = torch.cat(all_activations, dim=0)
    
    return activations, all_token_ids


def create_activation_dataset(
    activations: torch.Tensor,
    token_ids: List[List[int]] = None,
    save_path: Optional[str] = None,
) -> TensorDataset:
    """
    Create a dataset from extracted activations.
    
    Args:
        activations: Tensor of activations, shape [batch_size, seq_len, hidden_dim]
        token_ids: Optional list of token IDs
        save_path: Optional path to save the dataset
        
    Returns:
        TensorDataset containing the activations
    """
    if activations.dim() == 3:
        flat_activations = activations.reshape(-1, activations.size(-1))
    else:
        flat_activations = activations
    
    dataset = TensorDataset(flat_activations)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'activations': flat_activations,
            'token_ids': token_ids
        }, save_path)
        logger.info(f"Saved activation dataset to {save_path}")
    
    return dataset


def load_activation_dataset(
    path: str,
    device: str = "cpu"
) -> Tuple[TensorDataset, List[List[int]]]:
    """
    Load a saved activation dataset.
    
    Args:
        path: Path to the saved dataset
        device: Device to load the dataset to
        
    Returns:
        Tuple of (TensorDataset, token_ids)
    """
    data = torch.load(path, map_location=device)
    activations = data['activations']
    token_ids = data.get('token_ids', None)
    
    dataset = TensorDataset(activations)
    return dataset, token_ids


def get_model_residuals(
    activations: torch.Tensor,
    sae_model,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> torch.Tensor:
    """
    Compute residuals after SAE reconstruction.
    
    Args:
        activations: Input activations tensor
        sae_model: Trained SAE model
        device: Computation device
        
    Returns:
        Tensor of residuals
    """
    sae_model.eval()
    sae_model.to(device)
    
    all_residuals = []
    batch_size = 256
    
    with torch.no_grad():
        for i in range(0, activations.size(0), batch_size):
            batch = activations[i:i+batch_size].to(device)
            reconstructed, _ = sae_model(batch)
            residual = batch - reconstructed
            all_residuals.append(residual.cpu())
    
    return torch.cat(all_residuals, dim=0) 