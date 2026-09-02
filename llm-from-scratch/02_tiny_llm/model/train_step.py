"""
train_model.py
========

Production training script for `GPTForCausalLM`.

Features:
  - Device auto-selection (CUDA, MPS, or CPU)
  - Full MLflow experiment tracking (hyperparameters, loss curves, and model artifacts)
  - Periodic evaluation on validation split
  - Tracks cumulative tokens seen, logged alongside train/val loss so you
    can plot loss against tokens seen (not just step count or epoch)
  - Automatic checkpoint saving via `save_pretrained()`

NOTE: The learning rate scheduler (cosine annealing + warmup) and
gradient clipping have been removed for now -- flat learning rate,
no clipping -- to keep the loop simple while you get comfortable with
the core train/val loss behavior. We'll add both back in later once
you understand what each one is doing and why.

Usage:
------
    python train_model.py
"""

from __future__ import annotations

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
    output_dir: str = "./checkpoints/gpt2",
    experiment_name: str = "GPT_Causal_LM",
) -> GPTForCausalLM:
    """Trains a `GPTForCausalLM` model with validation, MLflow tracking, and checkpointing.

    Args:
        config: Model architecture definition (`GPTConfig`).
        train_data: Raw training text.
        val_data: Raw validation text.
        max_length: Context window / sequence length for each training example.
        num_epochs: Number of complete training epochs. Defaults to 10.
        batch_size: Micro-batch size per optimization step. Defaults to 8.
        learning_rate: Flat learning rate for AdamW (no scheduler). Defaults to 5e-4.
        weight_decay: Weight decay coefficient for regularization. Defaults to 0.1.
        output_dir: Path where `save_pretrained()` writes config & weights.
        experiment_name: Name of MLflow experiment room. Defaults to "GPT_Causal_LM".

    Returns:
        GPTForCausalLM: Trained model instance placed in `eval()` mode on CPU.
    """
    device = get_device()
    print(f"[Info] Training device resolved to: {device}")

    # 1. Prepare Data Loaders
    train_loader = create_dataloader(train_data, batch_size=batch_size, max_length=max_length, stride=config.context_length)
    val_loader = create_dataloader(val_data, batch_size, max_length, stride=config.context_length, shuffle=False)

    # 2. Instantiate Model and Transfer to Device
    model = GPTForCausalLM(config).to(device)

    # 3. Setup Optimizer (flat learning rate -- no scheduler for now)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
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
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "device": str(device),
            }
        )

        global_step = 0
        tokens_seen = 0
        best_val_loss = float("inf")
        best_epoch = -1

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
                optimizer.step()

                # Count tokens actually seen this step (batch_size * seq_len)
                tokens_seen += x_batch.numel()

                # Metrics Logging
                batch_loss = loss.item()
                epoch_train_loss += batch_loss

                mlflow.log_metrics(
                    {
                        "train/step_loss": batch_loss,
                        "train/tokens_seen": tokens_seen,
                    },
                    step=global_step,
                )

                global_step += 1

            # Epoch Metrics & Validation Evaluation
            avg_epoch_train_loss = epoch_train_loss / len(train_loader)
            val_loss = evaluate(model, val_loader, device)

            # Log train loss, val loss, AND tokens_seen together at the
            # same step (epoch) so they can be plotted against each
            # other -- e.g. loss vs. tokens seen, not just loss vs. epoch.
            mlflow.log_metrics(
                {
                    "train/epoch_loss": avg_epoch_train_loss,
                    "val/epoch_loss": val_loss,
                    "epoch/tokens_seen": tokens_seen,
                },
                step=epoch,
            )

            is_best = val_loss < best_val_loss
            print(
                f"Epoch {epoch + 1:02d}/{num_epochs:02d} | "
                f"Train Loss: {avg_epoch_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Tokens Seen: {tokens_seen:,}"
                + (" | New best val loss" if is_best else "")
            )

            # Save a checkpoint every time val loss hits a new minimum,
            # so the "best" model isn't overwritten by later, more
            # overfit epochs. Kept in a separate directory from the
            # final checkpoint so both remain available afterward.
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch
                best_save_path = Path(output_dir) / "best"
                model.save_pretrained(str(best_save_path))
                mlflow.log_metrics({"val/best_epoch_loss": best_val_loss}, step=epoch)

        mlflow.log_params({"best_epoch": best_epoch, "best_val_loss": best_val_loss})
        print(
            f"[Info] Best val loss: {best_val_loss:.4f} at epoch {best_epoch + 1:02d} "
            f"(checkpoint saved to '{(Path(output_dir) / 'best').resolve()}')"
        )

        # 6. Save Final-Epoch Model Artifacts (in addition to the best one above)
        save_path = Path(output_dir) / "final"
        model.save_pretrained(str(save_path))
        mlflow.log_artifacts(str(Path(output_dir)), artifact_path="model_checkpoint")
        print(f"[Success] Final checkpoint successfully saved to '{save_path.resolve()}'")

    model.to("cpu")
    model.eval()
    return model