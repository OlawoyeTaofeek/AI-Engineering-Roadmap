"""
gpt2_weight_loader.py

Utilities for fetching OpenAI's publicly released GPT-2 checkpoint files and
converting them from TensorFlow's checkpoint format into a plain nested
Python dict of NumPy arrays, ready to be loaded into a from-scratch model
implementation (e.g. PyTorch).

This is an original implementation built around the same public GPT-2
checkpoint files OpenAI released (https://github.com/openai/gpt-2), but
structured differently from other reference implementations you may have
seen: a small `GPT2Downloader` class handles fetching with mirror fallback,
and checkpoint-variable names are parsed with a regex instead of manual
string slicing.

Typical usage:
    >>> loader = GPT2WeightLoader(models_root="gpt2_checkpoints")
    >>> settings, params = loader.load("124M")
    >>> params["wte"].shape
    (50257, 768)
    >>> params["blocks"][0]["attn"]["c_attn"]["w"].shape
    (768, 2304)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import requests
import tensorflow as tf
from tqdm import tqdm
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VALID_MODEL_SIZES = ("124M", "355M", "774M", "1558M")

# Files that make up one GPT-2 checkpoint release.
CHECKPOINT_FILENAMES = (
    "checkpoint",
    "encoder.json",
    "hparams.json",
    "model.ckpt.data-00000-of-00001",
    "model.ckpt.index",
    "model.ckpt.meta",
    "vocab.bpe",
)

# A checkpoint variable name looks like "model/h3/attn/c_attn/w".
# This pattern splits it into: an optional block index ("h3") and the
# remaining path segments that describe where the tensor lives inside
# that block (or at the top level, if there's no block index).
_BLOCK_PREFIX_RE = re.compile(r"^h(?P<layer>\d+)$")


@dataclass(frozen=True)
class MirrorSource:
    """A single place a checkpoint file can be downloaded from."""

    label: str
    base_url: str

    def url_for(self, model_size: str, filename: str) -> str:
        return f"{self.base_url}/{model_size}/{filename}"


@dataclass(frozen=True)
class GPT2Sources:
    """The set of mirrors to try, in priority order."""

    mirrors: tuple[MirrorSource, ...] = field(
        default_factory=lambda: (
            MirrorSource("openai", "https://openaipublic.blob.core.windows.net/gpt-2/models"),
            MirrorSource("backup", "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"),
        )
    )


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

class GPT2Downloader:
    """
    Fetches GPT-2 checkpoint files to disk, trying each configured mirror
    in order until one succeeds. Skips re-downloading a file whose local
    size already matches the remote Content-Length header.
    """

    def __init__(self, sources: GPT2Sources | None = None, chunk_size_bytes: int = 1024):
        self.sources = sources or GPT2Sources()
        self.chunk_size_bytes = chunk_size_bytes

    def fetch(self, model_size: str, filename: str, destination: Path) -> None:
        """Download a single checkpoint file, trying every mirror in turn."""
        last_error: Exception | None = None

        for mirror in self.sources.mirrors:
            url = mirror.url_for(model_size, filename)
            try:
                self._download_one(url, destination, source_label=mirror.label)
                return
            except requests.exceptions.RequestException as exc:
                logger.warning("Mirror '%s' failed for %s: %s", mirror.label, filename, exc)
                last_error = exc
                continue

        raise RuntimeError(
            f"Could not download '{filename}' for model size '{model_size}' from any "
            f"configured mirror. Last error: {last_error}"
        )

    def _download_one(self, url: str, destination: Path, source_label: str) -> None:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        remote_size = int(response.headers.get("Content-Length", 0))
        if self._already_downloaded(destination, remote_size):
            logger.info("Skipping %s (already present, size matches)", destination.name)
            return

        destination.parent.mkdir(parents=True, exist_ok=True)
        with tqdm(
            total=remote_size,
            unit="iB",
            unit_scale=True,
            desc=f"[{source_label}] {destination.name}",
        ) as progress:
            with destination.open("wb") as f:
                for chunk in response.iter_content(chunk_size=self.chunk_size_bytes):
                    if not chunk:
                        continue
                    f.write(chunk)
                    progress.update(len(chunk))

    @staticmethod
    def _already_downloaded(destination: Path, remote_size: int) -> bool:
        if not destination.exists():
            return False
        return remote_size > 0 and destination.stat().st_size == remote_size


# ---------------------------------------------------------------------------
# Checkpoint parsing
# ---------------------------------------------------------------------------

def parse_checkpoint_variable_name(name: str) -> tuple[int | None, list[str]]:
    """
    Break a TensorFlow checkpoint variable name into a (layer_index, path)
    pair describing where its value belongs in the output dict.

    GPT-2 checkpoint variables always start with a "model/" prefix, followed
    either by a per-block segment like "h3" (meaning: this tensor belongs to
    transformer block 3) or a top-level name like "wte" / "ln_f".

    Examples
    --------
    >>> parse_checkpoint_variable_name("model/h3/attn/c_attn/w")
    (3, ['attn', 'c_attn', 'w'])
    >>> parse_checkpoint_variable_name("model/wte")
    (None, ['wte'])
    >>> parse_checkpoint_variable_name("model/ln_f/g")
    (None, ['ln_f', 'g'])
    """
    segments = name.split("/")[1:]  # drop the leading "model" segment
    if not segments:
        raise ValueError(f"Unexpected checkpoint variable name with no path: {name!r}")

    block_match = _BLOCK_PREFIX_RE.match(segments[0])
    if block_match:
        return int(block_match.group("layer")), segments[1:]
    return None, segments


def _place_value(root: dict[str, Any], path: list[str], value: np.ndarray) -> None:
    """
    Walk (creating as needed) nested dicts under `root` following `path`,
    and store `value` at the final key.

    Written recursively rather than with an iterative setdefault loop, so
    each level of nesting is explicit:

    >>> d = {}
    >>> _place_value(d, ["attn", "c_attn", "w"], np.zeros((2, 2)))
    >>> d["attn"]["c_attn"]["w"].shape
    (2, 2)
    """
    if len(path) == 1:
        root[path[0]] = value
        return
    child = root.setdefault(path[0], {})
    _place_value(child, path[1:], value)


def load_params_from_checkpoint(checkpoint_path: str, num_layers: int) -> dict[str, Any]:
    """
    Read every variable out of a TensorFlow checkpoint and reassemble them
    into a nested Python dict:

        {
            "wte": <array>, "wpe": <array>,
            "ln_f": {"g": <array>, "b": <array>},
            "blocks": [
                {"attn": {"c_attn": {"w": ..., "b": ...}, ...}, "ln_1": {...}, ...},
                ...  # one entry per transformer block
            ],
        }

    Parameters
    ----------
    checkpoint_path:
        Path to the TensorFlow checkpoint (as returned by
        ``tf.train.latest_checkpoint``), without file extension.
    num_layers:
        Number of transformer blocks in this GPT-2 size (from hparams.json's
        "n_layer" field) — used to pre-size the `blocks` list.
    """
    params: dict[str, Any] = {"blocks": [{} for _ in range(num_layers)]}

    for variable_name, _shape in tf.train.list_variables(checkpoint_path):
        raw_value = tf.train.load_variable(checkpoint_path, variable_name)
        value = np.squeeze(raw_value)  # drop any singleton dimensions TF added

        layer_index, path = parse_checkpoint_variable_name(variable_name)
        target = params["blocks"][layer_index] if layer_index is not None else params
        _place_value(target, path, value)

    return params


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

class GPT2WeightLoader:
    """
    High-level entry point: downloads a GPT-2 checkpoint (if not already
    present) and loads it into a (settings, params) pair ready for use in a
    custom model implementation.
    """

    def __init__(self, models_root: str | Path, downloader: GPT2Downloader | None = None):
        self.models_root = Path(models_root)
        self.downloader = downloader or GPT2Downloader()

    def load(self, model_size: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Ensure the given GPT-2 size is downloaded, then parse it.

        Parameters
        ----------
        model_size:
            One of "124M", "355M", "774M", "1558M".

        Returns
        -------
        (settings, params):
            `settings` is the parsed contents of hparams.json (n_layer,
            n_head, n_embd, etc). `params` is the nested weight dict from
            `load_params_from_checkpoint`.
        """
        if model_size not in VALID_MODEL_SIZES:
            raise ValueError(
                f"Unsupported model_size {model_size!r}; expected one of {VALID_MODEL_SIZES}"
            )

        model_dir = self.models_root / model_size
        self._ensure_downloaded(model_size, model_dir)

        settings_path = model_dir / "hparams.json"
        with settings_path.open("r", encoding="utf-8") as f:
            settings = json.load(f)

        checkpoint_path = tf.train.latest_checkpoint(str(model_dir))
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No TensorFlow checkpoint found in {model_dir} after download — "
                "the download may have failed partway through."
            )

        params = load_params_from_checkpoint(checkpoint_path, num_layers=settings["n_layer"])
        return settings, params

    def _ensure_downloaded(self, model_size: str, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        for filename in CHECKPOINT_FILENAMES:
            self.downloader.fetch(model_size, filename, model_dir / filename)

def assign(target: nn.Parameters, source: np.ndarray, name: str = "") -> nn.Parameter:
    """
    Copies `source` into `target`, raising immediately on any shape
    mismatch rather than letting PyTorch silently broadcast or fail
    later during a forward pass with a confusing error far from the
    real cause.
    """
    if target.shape != source.shape:
        raise ValueError(f"Shape mismatch for name: {name}. target: {target.shape}, and source: {source.shape}")
    return nn.Parameter(torch.tensor(source))


def load_weights_into_gpt(gpt, params):
    """
    Load GPT-2 TensorFlow checkpoint weights into a from-scratch PyTorch GPT model.

    Copies each tensor from `params` (the nested dict produced by
    parse_checkpoint_variable_name + _place_value) into the corresponding
    parameter of `gpt`, using `assign` to validate shapes at every step.

    Handles three checkpoint-to-PyTorch conversions along the way:
      - Conv1D-style weights are transposed (.T) to match nn.Linear's
        [out_features, in_features] convention.
      - The combined Q/K/V weight and bias tensors (`c_attn`) are split
        into three equal thirds along the last axis.
      - The output head reuses the token embedding weight (`wte`), since
        the original GPT-2 checkpoint ties input and output embeddings.

    Parameters
    ----------
    gpt : GPTModel
        A freshly constructed model whose parameter shapes already match
        the target GPT-2 config (so `assign`'s shape check succeeds).
    params : dict
        Nested dict of checkpoint weights, keyed as:
        params['wte'], params['wpe'], params['ln_f']['g'/'b'],
        params['blocks'][i]['ln_1'/'ln_2']['g'/'b'],
        params['blocks'][i]['attn']['c_attn'/'c_proj']['w'/'b'],
        params['blocks'][i]['mlp']['c_fc'/'c_proj']['w'/'b'].

    Returns
    -------
    None
        Mutates `gpt` in place.
    """
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for i in range(len(params['blocks'])):
        query_w, key_w, value_w = np.split(
            params['blocks'][i]['attn']['c_attn']['w'], 3, axis=-1
        )
        gpt.trf_blocks[i].attn.W_query.weight = assign(
            gpt.trf_blocks[i].attn.W_query.weight, query_w.T
        )
        gpt.trf_blocks[i].attn.W_key.weight = assign(
            gpt.trf_blocks[i].attn.W_key.weight, key_w.T
        )
        gpt.trf_blocks[i].attn.W_value.weight = assign(
            gpt.trf_blocks[i].attn.W_value.weight, value_w.T
        )

        query_b, key_b, value_b = np.split(
            params['blocks'][i]['attn']['c_attn']['b'], 3, axis=-1
        )
        gpt.trf_blocks[i].attn.W_query.bias = assign(
            gpt.trf_blocks[i].attn.W_query.bias, query_b
        )
        gpt.trf_blocks[i].attn.W_key.bias = assign(
            gpt.trf_blocks[i].attn.W_key.bias, key_b
        )
        gpt.trf_blocks[i].attn.W_value.bias = assign(
            gpt.trf_blocks[i].attn.W_value.bias, value_b
        )

        gpt.trf_blocks[i].attn.out_proj.weight = assign(
            gpt.trf_blocks[i].attn.out_proj.weight,
            params['blocks'][i]['attn']['c_proj']['w'].T
        )
        gpt.trf_blocks[i].attn.out_proj.bias = assign(
            gpt.trf_blocks[i].attn.out_proj.bias,
            params['blocks'][i]['attn']['c_proj']['b']
        )

        gpt.trf_blocks[i].ffn.net[0].weight = assign(
            gpt.trf_blocks[i].ffn.net[0].weight,
            params['blocks'][i]['mlp']['c_fc']['w'].T
        )
        gpt.trf_blocks[i].ffn.net[0].bias = assign(
            gpt.trf_blocks[i].ffn.net[0].bias,
            params['blocks'][i]['mlp']['c_fc']['b']
        )

        gpt.trf_blocks[i].ffn.net[2].weight = assign(
            gpt.trf_blocks[i].ffn.net[2].weight,
            params['blocks'][i]['mlp']['c_proj']['w'].T
        )
        gpt.trf_blocks[i].ffn.net[2].bias = assign(
            gpt.trf_blocks[i].ffn.net[2].bias,
            params['blocks'][i]['mlp']['c_proj']['b']
        )

        gpt.trf_blocks[i].norm1.scale = assign(
            gpt.trf_blocks[i].norm1.scale,
            params['blocks'][i]['ln_1']['g']
        )
        gpt.trf_blocks[i].norm1.shift = assign(
            gpt.trf_blocks[i].norm1.shift,
            params['blocks'][i]['ln_1']['b']
        )

        gpt.trf_blocks[i].norm2.scale = assign(
            gpt.trf_blocks[i].norm2.scale,
            params['blocks'][i]['ln_2']['g']
        )
        gpt.trf_blocks[i].norm2.shift = assign(
            gpt.trf_blocks[i].norm2.shift,
            params['blocks'][i]['ln_2']['b']
        )

    # Top-level params — outside the loop, only set once
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params['ln_f']['g'])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params['ln_f']['b'])
    gpt.out_head.weight = assign(gpt.out_head.weight, params['wte'])


# ---------------------------------------------------------------------------
# Convenience function mirroring the class-based API for quick scripts
# ---------------------------------------------------------------------------

def download_and_load_gpt2(model_size: str, models_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Thin functional wrapper around GPT2WeightLoader, for one-off scripts."""
    return GPT2WeightLoader(models_root=models_dir).load(model_size)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    settings, params = download_and_load_gpt2("124M", models_dir="gpt2_checkpoints")
    print("n_layer:", settings["n_layer"])
    print("wte shape:", params["wte"].shape)
    print("block 0 keys:", list(params["blocks"][0].keys()))