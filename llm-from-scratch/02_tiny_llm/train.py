"""
Training loop for the tiny GPT model.

Run:
    python train.py --config configs/default.yaml
"""
import mlflow
import torch


def train(cfg, model, train_loader, val_loader, device):
    mlflow.set_tracking_uri("../../experiments/mlruns")  # adjust relative path per stage
    mlflow.set_experiment("02_tiny_llm")

    with mlflow.start_run():
        mlflow.log_params(cfg)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.1)
        )

        global_step = 0
        for epoch in range(cfg["num_epochs"]):
            model.train()
            for input_batch, target_batch in train_loader:
                optimizer.zero_grad()
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward()
                optimizer.step()
                global_step += 1

                if global_step % cfg["eval_freq"] == 0:
                    model.eval()
                    with torch.no_grad():
                        train_loss = calc_loss_loader(train_loader, model, device, cfg["eval_iter"])
                        val_loss = calc_loss_loader(val_loader, model, device, cfg["eval_iter"])
                    mlflow.log_metrics(
                        {"train_loss": train_loss, "val_loss": val_loss}, step=global_step
                    )
                    model.train()

        # log the final checkpoint as an MLflow artifact -- separate from Git LFS,
        # useful for comparing runs without polluting the repo
        torch.save(model.state_dict(), "checkpoint.pt")
        mlflow.log_artifact("checkpoint.pt")


def calc_loss_batch(input_batch, target_batch, model, device):
    import torch.nn.functional as F
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    return F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches


if __name__ == "__main__":
    raise NotImplementedError("Wire up model/dataloaders for this stage, then call train(cfg, ...)")
