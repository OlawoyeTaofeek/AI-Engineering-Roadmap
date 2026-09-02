"""
Quantization from scratch using PyTorch (no bitsandbytes / no HF quantization libs).

Covers:
  1. Symmetric  quantization (per-tensor)
  2. Asymmetric quantization (per-tensor)
  3. Per-channel variants of both
  4. A quantized Linear layer (weights quantized, matmul done in fake-quant fp32)
  5. Error (MSE) comparison between the methods

Run:
    python quantization_from_scratch.py
"""

import torch


# ---------------------------------------------------------------------------
# 1. SYMMETRIC QUANTIZATION  (z = 0, range centered on 0)
# ---------------------------------------------------------------------------
def symmetric_quantize(r: torch.Tensor, num_bits: int = 8):
    """
    q = round(r / s)
    r = s * q
    Range used: [-(2^(b-1)-1), 2^(b-1)-1]   e.g. INT8 -> [-127, 127]
    """
    qmax = 2 ** (num_bits - 1) - 1          # 127 for int8
    max_val = r.abs().max()
    scale = max_val / qmax
    scale = torch.clamp(scale, min=1e-8)    # avoid div-by-zero on all-zero tensors

    q = torch.round(r / scale)
    q = torch.clamp(q, -qmax, qmax)
    return q.to(torch.int8), scale


def symmetric_dequantize(q: torch.Tensor, scale: torch.Tensor):
    return q.to(torch.float32) * scale


# ---------------------------------------------------------------------------
# 2. ASYMMETRIC QUANTIZATION (zero-point shifts the range)
# ---------------------------------------------------------------------------
def asymmetric_quantize(r: torch.Tensor, num_bits: int = 8):
    """
    q = round(r / s) + z
    r = s * (q - z)
    Range used: [0, 2^b - 1]   e.g. UINT8 -> [0, 255]
    """
    qmin, qmax = 0, 2 ** num_bits - 1       # 0..255 for uint8
    min_val, max_val = r.min(), r.max()

    scale = (max_val - min_val) / (qmax - qmin)
    scale = torch.clamp(scale, min=1e-8)

    zero_point = qmin - torch.round(min_val / scale)
    zero_point = torch.clamp(zero_point, qmin, qmax)

    q = torch.round(r / scale) + zero_point
    q = torch.clamp(q, qmin, qmax)
    return q.to(torch.uint8), scale, zero_point


def asymmetric_dequantize(q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor):
    return (q.to(torch.float32) - zero_point) * scale


# ---------------------------------------------------------------------------
# 3. PER-CHANNEL versions (one scale/zero-point per output channel/row)
#    This is what GPTQ/AWQ-style quantizers use instead of one scale for
#    the whole tensor -- much better accuracy for weight matrices.
# ---------------------------------------------------------------------------
def symmetric_quantize_per_channel(r: torch.Tensor, num_bits: int = 8, dim: int = 0):
    qmax = 2 ** (num_bits - 1) - 1
    max_val = r.abs().amax(dim=[d for d in range(r.dim()) if d != dim], keepdim=True)
    scale = torch.clamp(max_val / qmax, min=1e-8)

    q = torch.round(r / scale)
    q = torch.clamp(q, -qmax, qmax)
    return q.to(torch.int8), scale  # scale shape: broadcastable per-channel


def symmetric_dequantize_per_channel(q: torch.Tensor, scale: torch.Tensor):
    return q.to(torch.float32) * scale


# ---------------------------------------------------------------------------
# 4. A "quantized" Linear layer
#    In real INT8 kernels the matmul itself runs in integer arithmetic.
#    Here we do FAKE quantization: store weights as int8 + scale, but
#    dequantize on the fly before matmul (this is how QAT / PTQ error
#    simulation normally works, and it's easy to verify by hand).
# ---------------------------------------------------------------------------
class QuantizedLinear(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear, num_bits: int = 8, per_channel: bool = True):
        super().__init__()
        w = linear.weight.data  # shape [out_features, in_features]

        if per_channel:
            q_w, scale = symmetric_quantize_per_channel(w, num_bits, dim=0)
        else:
            q_w, scale = symmetric_quantize(w, num_bits)

        self.register_buffer("q_weight", q_w)
        self.register_buffer("scale", scale)
        self.bias = linear.bias

    def forward(self, x):
        # dequantize weight back to fp32, then run normal matmul
        w_dequant = symmetric_dequantize_per_channel(self.q_weight, self.scale)
        return torch.nn.functional.linear(x, w_dequant, self.bias)


# ---------------------------------------------------------------------------
# DEMO / SANITY CHECK
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    print("=" * 70)
    print("1) Symmetric vs Asymmetric on a SKEWED (post-ReLU-like) tensor")
    print("=" * 70)
    r = torch.relu(torch.randn(1000) * 2)  # all >= 0, skewed distribution

    q_sym, s_sym = symmetric_quantize(r, num_bits=8)
    r_sym = symmetric_dequantize(q_sym, s_sym)

    q_asym, s_asym, z_asym = asymmetric_quantize(r, num_bits=8)
    r_asym = asymmetric_dequantize(q_asym, s_asym, z_asym)

    mse_sym = torch.mean((r - r_sym) ** 2).item()
    mse_asym = torch.mean((r - r_asym) ** 2).item()

    print(f"Symmetric   scale={s_sym.item():.5f}                MSE={mse_sym:.6f}")
    print(f"Asymmetric  scale={s_asym.item():.5f} zp={z_asym.item():.0f}   MSE={mse_asym:.6f}")
    print("-> Asymmetric should win here since data is one-sided (skewed).\n")

    print("=" * 70)
    print("2) Symmetric vs Asymmetric on a CENTERED tensor (typical weights)")
    print("=" * 70)
    r2 = torch.randn(1000)  # roughly centered around 0

    q_sym2, s_sym2 = symmetric_quantize(r2, num_bits=8)
    r_sym2 = symmetric_dequantize(q_sym2, s_sym2)

    q_asym2, s_asym2, z_asym2 = asymmetric_quantize(r2, num_bits=8)
    r_asym2 = asymmetric_dequantize(q_asym2, s_asym2, z_asym2)

    mse_sym2 = torch.mean((r2 - r_sym2) ** 2).item()
    mse_asym2 = torch.mean((r2 - r_asym2) ** 2).item()

    print(f"Symmetric   MSE={mse_sym2:.6f}")
    print(f"Asymmetric  MSE={mse_asym2:.6f}")
    print("-> Roughly similar here since data is already centered.\n")

    print("=" * 70)
    print("3) Per-tensor vs Per-channel quantization on a weight matrix")
    print("=" * 70)
    w = torch.randn(64, 128)
    w[0] *= 20  # give one output channel a much larger range (common in real nets)

    q_pt, s_pt = symmetric_quantize(w, num_bits=8)               # single scale for whole tensor
    w_pt = symmetric_dequantize(q_pt, s_pt)

    q_pc, s_pc = symmetric_quantize_per_channel(w, num_bits=8, dim=0)  # scale per row
    w_pc = symmetric_dequantize_per_channel(q_pc, s_pc)

    mse_pt = torch.mean((w - w_pt) ** 2).item()
    mse_pc = torch.mean((w - w_pc) ** 2).item()

    print(f"Per-tensor  MSE={mse_pt:.6f}")
    print(f"Per-channel MSE={mse_pc:.6f}")
    print("-> Per-channel should win: the outlier row no longer blows up")
    print("   the scale used for every other row.\n")

    print("=" * 70)
    print("4) QuantizedLinear layer vs real nn.Linear (numerical check)")
    print("=" * 70)
    linear = torch.nn.Linear(128, 64)
    qlinear = QuantizedLinear(linear, num_bits=8, per_channel=True)

    x = torch.randn(4, 128)
    out_fp32 = linear(x)
    out_quant = qlinear(x)

    rel_err = (out_fp32 - out_quant).abs().mean() / out_fp32.abs().mean()
    print(f"Mean relative error vs full-precision output: {rel_err.item():.4%}")
    print("-> Small single-digit-percent error is expected/normal for INT8.")
