#!/usr/bin/env python
import os
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import wandb
from tqdm.auto import tqdm
import json
import sys
from datetime import datetime


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.sae import SAEConfig, SparseAutoEncoder
from src.utils.activation_utils import load_activation_dataset


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_sae(
    activations_path: str,
    architecture: str = "standard",
    activation_fn: str = "relu",
    d_sae: int = 4096,
    l1_coefficient: float = 1e-3,
    k: int = 100,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    num_epochs: int = 20,
    eval_steps: int = 100,
    log_steps: int = 10,
    save_steps: int = 500,
    normalize_decoder: bool = True,
    dtype: str = "float32",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "./sae_checkpoints",
    wandb_project: str = "sparse-autoencoders",
    wandb_entity: str = None,
    log_to_wandb: bool = True,
):
    """
    Train a Sparse Autoencoder (SAE) on model activations.
    
    Args:
        activations_path: Path to the saved activations
        architecture: SAE architecture type ("standard", "topk")
        activation_fn: Activation function ("relu", "gelu")
        d_sae: Number of SAE features to learn
        l1_coefficient: Sparsity regularization coefficient
        k: Number of active features for TopK activation
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        eval_steps: Steps between evaluations
        log_steps: Steps between logging
        save_steps: Steps between checkpoint saves
        normalize_decoder: Whether to normalize decoder weights
        dtype: Data type for computation
        device: Device to train on
        output_dir: Directory to save checkpoints
        wandb_project: W&B project name
        wandb_entity: W&B entity name
        log_to_wandb: Whether to log to W&B
        
    Returns:
        Trained SAE model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    dataset, token_ids = load_activation_dataset(
        path=activations_path,
        device="cpu"
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    
    d_in = dataset[0][0].size(-1)
    logger.info(f"Input dimension: {d_in}")
    
    config = SAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        architecture=architecture,
        activation_fn=activation_fn,
        k=k,
        l1_coefficient=l1_coefficient,
        learning_rate=learning_rate,
        normalize_decoder=normalize_decoder,
        dtype=dtype,
        device=device,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        log_to_wandb=log_to_wandb
    )
    
    model = SparseAutoEncoder(config)
    model.to(device=device)
    
    config_path = os.path.join(output_dir, "sae_config.json")
    config.save(config_path)
    logger.info(f"Saved config to {config_path}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    if log_to_wandb:
        run_name = f"sae_{architecture}_{d_sae}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config={
                "architecture": architecture,
                "activation_fn": activation_fn,
                "d_in": d_in,
                "d_sae": d_sae,
                "l1_coefficient": l1_coefficient,
                "k": k,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "normalize_decoder": normalize_decoder
            }
        )
        wandb.watch(model, log="all", log_freq=log_steps)
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch_idx, (x,) in enumerate(progress_bar):
            x = x.to(device=device)
            
            optimizer.zero_grad()
            loss_dict = model.calculate_loss(x)
            loss = loss_dict["total_loss"]
            
            loss.backward()
            optimizer.step()
            
            if normalize_decoder:
                model.normalize_decoder_weights()
            
            global_step += 1
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
            if log_to_wandb and global_step % log_steps == 0:
                wandb.log({
                    "train/total_loss": loss_dict["total_loss"].item(),
                    "train/mse_loss": loss_dict["mse_loss"].item(),
                    "train/l1_loss": loss_dict["l1_loss"].item(),
                    "epoch": epoch,
                    "global_step": global_step
                })
            
            if global_step % eval_steps == 0:
                eval_loss = evaluate_sae(model, data_loader, device, max_batches=10)
                if log_to_wandb:
                    wandb.log({
                        "eval/total_loss": eval_loss,
                        "epoch": epoch,
                        "global_step": global_step
                    })
                
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    model_path = os.path.join(output_dir, "sae_best_model.pt")
                    model.save_model(model_path)
                    logger.info(f"Saved best model to {model_path} (loss: {best_loss:.4f})")
            
            if global_step % save_steps == 0:
                checkpoint_path = os.path.join(output_dir, f"sae_checkpoint_step_{global_step}.pt")
                model.save_model(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        avg_epoch_loss = epoch_loss / len(data_loader)
        logger.info(f"Epoch {epoch}/{num_epochs} - Avg Loss: {avg_epoch_loss:.4f}")
        
        epoch_checkpoint_path = os.path.join(output_dir, f"sae_checkpoint_epoch_{epoch}.pt")
        model.save_model(epoch_checkpoint_path)
    
    final_model_path = os.path.join(output_dir, "sae_final_model.pt")
    model.save_model(final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    if log_to_wandb:
        wandb.finish()
    
    return model


def evaluate_sae(model, data_loader, device, max_batches=None):
    """Evaluate the SAE model on validation data."""
    model.eval()
    total_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, (x,) in enumerate(data_loader):
            if max_batches and batch_idx >= max_batches:
                break
                
            x = x.to(device)
            loss_dict = model.calculate_loss(x)
            total_loss += loss_dict["total_loss"].item()
            batch_count += 1
    
    return total_loss / batch_count if batch_count > 0 else float('inf')


def main():
    """Main entry point for script."""
    parser = argparse.ArgumentParser(description="Train a Sparse Autoencoder (SAE) on model activations")
    

    parser.add_argument("--activations_path", type=str, required=True,
                        help="Path to saved activations")
    
    parser.add_argument("--architecture", type=str, default="standard", 
                        choices=["standard", "topk"],
                        help="SAE architecture type")
    parser.add_argument("--activation_fn", type=str, default="relu",
                        choices=["relu", "gelu"],
                        help="Activation function")
    parser.add_argument("--d_sae", type=int, default=4096,
                        help="Number of SAE features to learn")
    parser.add_argument("--l1_coefficient", type=float, default=1e-3,
                        help="Sparsity regularization coefficient")
    parser.add_argument("--k", type=int, default=100,
                        help="Number of active features for TopK activation")
    parser.add_argument("--normalize_decoder", action="store_true", default=True,
                        help="Whether to normalize decoder weights")
    
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate for optimizer")
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Steps between evaluations")
    parser.add_argument("--log_steps", type=int, default=10,
                        help="Steps between logging")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Steps between checkpoint saves")
    
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Data type for computation")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to train on (default: auto-detect)")
    parser.add_argument("--output_dir", type=str, default="./sae_checkpoints",
                        help="Directory to save checkpoints")
    
    parser.add_argument("--wandb_project", type=str, default="sparse-autoencoders",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity name")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")
    
    args = parser.parse_args()
    
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_sae(
        activations_path=args.activations_path,
        architecture=args.architecture,
        activation_fn=args.activation_fn,
        d_sae=args.d_sae,
        l1_coefficient=args.l1_coefficient,
        k=args.k,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        eval_steps=args.eval_steps,
        log_steps=args.log_steps,
        save_steps=args.save_steps,
        normalize_decoder=args.normalize_decoder,
        dtype=args.dtype,
        device=args.device,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        log_to_wandb=not args.no_wandb
    )


if __name__ == "__main__":
    main() 