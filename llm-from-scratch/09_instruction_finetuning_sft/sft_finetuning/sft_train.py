"""
Supervised fine-tuning (SFT) training loop.

Run:
    python train.py --config configs/default.yaml
"""
import mlflow


def train(cfg):
    mlflow.set_tracking_uri("../experiments/mlruns")
    mlflow.set_experiment("08_instruction_finetuning_sft")
    with mlflow.start_run():
        mlflow.log_params(cfg)
        raise NotImplementedError("TODO: wire up model, data, and training loop for this stage")


if __name__ == "__main__":
    raise NotImplementedError("TODO: argparse --config, load yaml, call train(cfg)")
