"""
train_model.py
========

Production training script for `GPTForCausalLM`.

Features:
  - Device auto-selection (CUDA, MPS, or CPU)
  - Full MLflow experiment tracking (hyperparameters, loss curves, and model artifacts)
  - Cosine annealing learning rate scheduler with warm-up
  - Gradient clipping to prevent exploding gradients
  - Periodic evaluation on validation split
  - Automatic checkpoint saving via `save_pretrained()`

Usage:
------
    python train_model.py
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .causal_lm import GPTForCausalLM
from .config import GPTConfig
from data.dataset import create_dataloader


def get_device() -> torch.device:
    """Select the best available hardware accelerator.

    Returns:
        torch.device: Resolved PyTorch device (`cuda`, `mps`, or `cpu`).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Creates a learning rate schedule with linear warmup and cosine decay.

    Args:
        optimizer: The PyTorch optimizer to schedule.
        num_warmup_steps: Number of steps for linear learning rate warmup.
        num_training_steps: Total number of training steps.
        min_lr_ratio: Minimum LR as a fraction of maximum LR. Defaults to 0.1.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: Configured PyTorch scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(
    model: GPTForCausalLM, val_loader: DataLoader, device: torch.device
) -> float:
    """Evaluates loss over a validation dataloader without updating weights.

    Args:
        model: The `GPTForCausalLM` instance being trained.
        val_loader: DataLoader containing validation data.
        device: Target execution device.

    Returns:
        float: Average cross-entropy loss over the validation dataset.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            outputs = model(input_ids=x_val, labels=y_val)
            total_loss += outputs.loss.item()

    return total_loss / max(1, len(val_loader))


def train(
    config: GPTConfig,
    train_data,
    val_data,
    max_length: int,
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 5e-4,
    weight_decay: float = 0.1,
    max_grad_norm: float = 1.0,
    warmup_ratio: float = 0.05,
    output_dir: str = "./checkpoints/gpt2",
    experiment_name: str = "GPT_Causal_LM",
) -> GPTForCausalLM:
    """Trains a `GPTForCausalLM` model with validation, MLflow tracking, and checkpointing.

    Args:
        config: Model architecture definition (`GPTConfig`).
        train_dataset: PyTorch dataset yielding `(input_ids, label_ids)` tuples.
        val_dataset: Validation dataset of identical structure to `train_dataset`.
        num_epochs: Number of complete training epochs. Defaults to 10.
        batch_size: Micro-batch size per optimization step. Defaults to 8.
        learning_rate: Peak learning rate for AdamW. Defaults to 5e-4.
        weight_decay: Weight decay coefficient for regularization. Defaults to 0.1.
        max_grad_norm: Maximum norm threshold for gradient clipping. Defaults to 1.0.
        warmup_ratio: Fraction of total steps devoted to LR warmup. Defaults to 0.05.
        output_dir: Path where `save_pretrained()` writes config & weights.
        experiment_name: Name of MLflow experiment room. Defaults to "GPT_Causal_LM".

    Returns:
        GPTForCausalLM: Trained model instance placed in `eval()` mode on CPU.
    """
    device = get_device()
    print(f"[Info] Training device resolved to: {device}")

    # 1. Prepare Data Loaders
    train_loader = create_dataloader(train_data, batch_size=batch_size, max_length=max_length)
    val_loader = create_dataloader(val_data, batch_size, max_length)

    # 2. Instantiate Model and Transfer to Device
    model = GPTForCausalLM(config).to(device)

    # 3. Setup Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    mlflow.set_tracking_uri("sqlite:///C:/Users/user/Documents/AI-EngineeringRoadmap/llm-from-scratch/02_tiny_llm/mlflow.db")

    # 4. MLflow Experiment Initialization
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Log Architecture and Hyperparameters
        mlflow.log_params(
            {
                "context_length": config.context_length,
                "vocab_size": config.vocab_size,
                "n_layer": config.n_layers,
                "n_head": config.n_heads,
                "n_embd": config.emb_dim,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "peak_learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "max_grad_norm": max_grad_norm,
                "warmup_steps": warmup_steps,
                "total_steps": total_steps,
                "device": str(device),
            }
        )

        global_step = 0

        # 5. Training Loop
        for epoch in range(num_epochs):
            model.train()
            epoch_train_loss = 0.0

            for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()

                # Forward pass - CausalLM calculates loss internally
                outputs = model(input_ids=x_batch, labels=y_batch)
                loss = outputs.loss

                loss.backward()

                # Gradient Clipping
                if max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()

                # Metrics Logging
                current_lr = scheduler.get_last_lr()[0]
                batch_loss = loss.item()
                epoch_train_loss += batch_loss

                mlflow.log_metrics(
                    {"train/step_loss": batch_loss, "train/learning_rate": current_lr},
                    step=global_step,
                )

                global_step += 1

            # Epoch Metrics & Validation Evaluation
            avg_epoch_train_loss = epoch_train_loss / len(train_loader)
            val_loss = evaluate(model, val_loader, device)

            mlflow.log_metrics(
                {
                    "train/epoch_loss": avg_epoch_train_loss,
                    "val/epoch_loss": val_loss,
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch + 1:02d}/{num_epochs:02d} | "
                f"Train Loss: {avg_epoch_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {current_lr:.2e}"
            )

        # 6. Save Model Artifacts
        save_path = Path(output_dir)
        model.save_pretrained(str(save_path))
        mlflow.log_artifacts(str(save_path), artifact_path="model_checkpoint")
        print(f"[Success] Checkpoint successfully saved to '{save_path.resolve()}'")

    model.to("cpu")
    model.eval()
    return model