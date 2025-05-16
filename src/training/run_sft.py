#!/usr/bin/env python
"""
Script for Supervised Fine-Tuning (SFT) of language models.
"""

import os
import argparse
import logging
import torch
import sys
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
import wandb

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_sft(
    model_name: str,
    dataset_name: str,
    text_column: str = "text",
    target_column: str = "summary",
    output_dir: str = "./sft_output",
    hub_model_id: str = None,
    push_to_hub: bool = False,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-5,
    max_steps: int = None,
    logging_steps: int = 10,
    save_steps: int = 100,
    eval_steps: int = 50,
    warmup_steps: int = 0,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    bf16: bool = True,
    fp16: bool = False,
    optim: str = "adamw_torch",
    lr_scheduler_type: str = "cosine",
    max_seq_length: int = 512,
    packing: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_chat_format: bool = True,
    report_to_wandb: bool = True,
    wandb_project: str = "sft-project",
    wandb_entity: str = None,
    seed: int = 42,
    use_peft: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_flash_attention: bool = True
):
    """
    Run Supervised Fine-Tuning (SFT) on a language model.
    
    Args:
        model_name: Name or path of pretrained model
        dataset_name: Name or path of dataset for fine-tuning
        text_column: Column name containing the input text
        target_column: Column name containing the target text
        output_dir: Directory to save model checkpoints
        hub_model_id: HuggingFace Hub model ID for uploading
        push_to_hub: Whether to push model to HuggingFace Hub
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Training batch size per device
        per_device_eval_batch_size: Evaluation batch size per device
        gradient_accumulation_steps: Number of steps for gradient accumulation
        learning_rate: Learning rate for optimizer
        max_steps: Maximum number of training steps (overrides epochs)
        logging_steps: Steps between logging
        save_steps: Steps between saving checkpoints
        eval_steps: Steps between evaluations
        warmup_steps: Steps for learning rate warmup
        warmup_ratio: Ratio of total steps for learning rate warmup
        weight_decay: Weight decay for optimizer
        max_grad_norm: Maximum gradient norm for clipping
        bf16: Whether to use bfloat16 precision
        fp16: Whether to use float16 precision
        optim: Optimizer type
        lr_scheduler_type: Learning rate scheduler type
        max_seq_length: Maximum sequence length
        packing: Whether to use packing for efficient training
        device: Device to train on
        use_chat_format: Whether to use chat format for the model
        report_to_wandb: Whether to report to W&B
        wandb_project: W&B project name
        wandb_entity: W&B entity name
        seed: Random seed
        use_peft: Whether to use PEFT for parameter-efficient fine-tuning
        lora_r: LoRA r parameter
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout
        use_flash_attention: Whether to use Flash Attention if available
    
    Returns:
        Fine-tuned model path
    """
    # Initialize W&B if requested
    if report_to_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=f"sft-{model_name.split('/')[-1]}",
            config={
                "model_name": model_name,
                "dataset_name": dataset_name,
                "num_train_epochs": num_train_epochs,
                "learning_rate": learning_rate,
                "per_device_train_batch_size": per_device_train_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "max_seq_length": max_seq_length,
                "use_peft": use_peft,
                "lora_r": lora_r if use_peft else None,
                "lora_alpha": lora_alpha if use_peft else None,
            }
        )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32),
            trust_remote_code=True,
            use_flash_attention_2=use_flash_attention
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Set pad token if not defined
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Set up chat format if requested
        if use_chat_format:
            model, tokenizer = setup_chat_format(model, tokenizer)
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    # Load dataset
    logger.info(f"Loading dataset: {dataset_name}")
    try:
        dataset = load_dataset(dataset_name)
        
        # Check if dataset has train and test splits
        if "test" not in dataset and "validation" not in dataset:
            # Split dataset if necessary
            dataset = dataset["train"].train_test_split(test_size=0.1, seed=seed)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise
        
    logger.info(f"Dataset loaded with {len(dataset['train'])} training examples")
    
    # Prepare SFT configuration
    sft_config = SFTConfig(
        output_dir=output_dir,
        
        # Training parameters
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_steps=max_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        
        # Optimizer settings
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        bf16=bf16,
        fp16=fp16,
        optim=optim,
        lr_scheduler_type=lr_scheduler_type,
        
        # Data processing
        max_seq_length=max_seq_length,
        packing=packing,
        
        # HuggingFace Hub integration
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        
        # Reporting and logging
        report_to="wandb" if report_to_wandb else "none",
        
        # Seed for reproducibility
        seed=seed,
        
        # PEFT settings
        use_peft=use_peft,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    
    # Create SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"] if "test" in dataset else dataset.get("validation"),
        tokenizer=tokenizer,
        dataset_text_field=text_column,
        dataset_summary_field=target_column
    )
    
    # Run training
    logger.info("Starting training")
    train_result = trainer.train()
    
    # Save final model
    logger.info("Saving final model")
    trainer.save_model()
    
    # Log and save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Evaluate the model
    if trainer.is_world_process_zero() and (dataset["test"] is not None or dataset.get("validation") is not None):
        logger.info("Evaluating model")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    # Close wandb if used
    if report_to_wandb:
        wandb.finish()
    
    logger.info(f"Training complete! Model saved to {output_dir}")
    return output_dir


def main():
    """Main entry point for script."""
    parser = argparse.ArgumentParser(description="Run Supervised Fine-Tuning (SFT) on a language model")
    
    # Basic arguments
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name or path of pretrained model")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name or path of dataset for fine-tuning")
    parser.add_argument("--text_column", type=str, default="text",
                        help="Column name containing input text")
    parser.add_argument("--target_column", type=str, default="summary",
                        help="Column name containing target text")
    parser.add_argument("--output_dir", type=str, default="./sft_output",
                        help="Directory to save model checkpoints")
    
    # HuggingFace Hub integration
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="HuggingFace Hub model ID for uploading")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Whether to push model to HuggingFace Hub")
    
    # Training parameters
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of steps for gradient accumulation")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate for optimizer")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum number of training steps (overrides epochs)")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Steps between logging")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Steps between saving checkpoints")
    parser.add_argument("--eval_steps", type=int, default=50,
                        help="Steps between evaluations")
    
    # Optimizer settings
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Steps for learning rate warmup")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Ratio of total steps for learning rate warmup")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--bf16", action="store_true",
                        help="Whether to use bfloat16 precision")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use float16 precision")
    parser.add_argument("--optim", type=str, default="adamw_torch",
                        help="Optimizer type")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="Learning rate scheduler type")
    
    # Data processing
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--packing", action="store_true",
                        help="Whether to use packing for efficient training")
    
    # System settings
    parser.add_argument("--device", type=str, default=None,
                        help="Device to train on (default: auto-detect)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--use_chat_format", action="store_true",
                        help="Whether to use chat format for the model")
    
    # W&B integration
    parser.add_argument("--report_to_wandb", action="store_true",
                        help="Whether to report to W&B")
    parser.add_argument("--wandb_project", type=str, default="sft-project",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity name")
    
    # PEFT settings
    parser.add_argument("--use_peft", action="store_true",
                        help="Whether to use PEFT for parameter-efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA r parameter")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    
    # Flash attention
    parser.add_argument("--use_flash_attention", action="store_true",
                        help="Whether to use Flash Attention if available")
    
    args = parser.parse_args()
    
    # Auto-detect device if not specified
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Run SFT
    run_sft(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        text_column=args.text_column,
        target_column=args.target_column,
        output_dir=args.output_dir,
        hub_model_id=args.hub_model_id,
        push_to_hub=args.push_to_hub,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        fp16=args.fp16,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        max_seq_length=args.max_seq_length,
        packing=args.packing,
        device=args.device,
        use_chat_format=args.use_chat_format,
        report_to_wandb=args.report_to_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        seed=args.seed,
        use_peft=args.use_peft,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_flash_attention=args.use_flash_attention
    )


if __name__ == "__main__":
    main() 