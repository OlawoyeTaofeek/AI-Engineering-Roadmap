"""
Module 1 — Quantization from scratch.

We implement:
  1. Symmetric quantization (for weights)
  2. Asymmetric quantization (for activations)
  3. Per-tensor vs per-channel granularity
  4. Error measurement (SQNR)

Run: python quantizer.py
"""
import torch


# ----------------------------------------------------------------------
# 1. SYMMETRIC QUANTIZATION
# ----------------------------------------------------------------------
def symmetric_quantize(x: torch.Tensor, num_bits: int = 8, dim: int = None):
    """
    Symmetric affine quantization: zero_point is always 0.
    Range is [-2^(b-1)+1, 2^(b-1)-1]  (we use the "restricted" range so
    that -qmax*scale and +qmax*scale are both representable — this is
    what PyTorch/TensorRT do for symmetric weight quant).

    Args:
        x: tensor to quantize
        num_bits: bit width (e.g. 8 for int8)
        dim: if given, compute one scale per slice along this dim
             (per-channel quantization). If None -> per-tensor.
    Returns:
        q: quantized integer tensor (stored as int8/int32 for generality)
        scale: the scale(s) used
    """
    qmax = 2 ** (num_bits - 1) - 1  # e.g. 127 for int8

    if dim is None:
        max_val = x.abs().max().clamp(min=1e-8)
        scale = max_val / qmax
    else:
        # per-channel: reduce over all dims except `dim`
        reduce_dims = [d for d in range(x.dim()) if d != dim]
        max_val = x.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8)
        scale = max_val / qmax

    q = torch.round(x / scale).clamp(-qmax - 1, qmax)
    return q.to(torch.int8), scale


def symmetric_dequantize(q: torch.Tensor, scale: torch.Tensor):
    return q.float() * scale


# ----------------------------------------------------------------------
# 2. ASYMMETRIC (AFFINE) QUANTIZATION
# ----------------------------------------------------------------------
def asymmetric_quantize(x: torch.Tensor, num_bits: int = 8):
    """
    Full affine quantization with a zero_point. Used for activations
    whose distribution isn't centered at 0 (e.g. post-ReLU >= 0).

    q = round(x/scale) + zero_point,   scale = (max-min)/(qmax-qmin)
    """
    qmin, qmax = 0, 2 ** num_bits - 1  # e.g. [0, 255] for uint8

    min_val = x.min()
    max_val = x.max()
    # guard against degenerate ranges
    if (max_val - min_val).abs() < 1e-8:
        max_val = min_val + 1e-8

    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - torch.round(min_val / scale)
    zero_point = zero_point.clamp(qmin, qmax)

    q = torch.round(x / scale + zero_point).clamp(qmin, qmax)
    return q.to(torch.uint8), scale, zero_point


def asymmetric_dequantize(q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor):
    return (q.float() - zero_point) * scale


# ----------------------------------------------------------------------
# 3. ERROR METRIC — Signal-to-Quantization-Noise Ratio
# ----------------------------------------------------------------------
def sqnr(x_fp32: torch.Tensor, x_dequant: torch.Tensor) -> float:
    """Higher SQNR (dB) = better. Rule of thumb: >30dB is usually 'safe'."""
    signal_power = (x_fp32 ** 2).mean()
    noise_power = ((x_fp32 - x_dequant) ** 2).mean().clamp(min=1e-12)
    return 10 * torch.log10(signal_power / noise_power).item()


# ----------------------------------------------------------------------
# DEMO: quantize a real-ish weight matrix, per-tensor vs per-channel
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # Simulate a linear layer's weight: out_features=64, in_features=256
    # Real weights aren't uniform - they're roughly Gaussian with a few
    # outlier channels (common in trained nets, especially LLMs).
    W = torch.randn(64, 256) * 0.02
    W[3] *= 15   # one "outlier" output channel, like real trained weights

    print("=" * 60)
    print("SYMMETRIC QUANTIZATION — PER-TENSOR vs PER-CHANNEL")
    print("=" * 60)

    # Per-tensor: one scale for the WHOLE matrix
    q_pt, scale_pt = symmetric_quantize(W, num_bits=8, dim=None)
    W_hat_pt = symmetric_dequantize(q_pt, scale_pt)
    print(f"Per-tensor scale: {scale_pt.item():.6f}")
    print(f"Per-tensor SQNR:  {sqnr(W, W_hat_pt):.2f} dB")

    # Per-channel: one scale PER OUTPUT CHANNEL (dim=0)
    q_pc, scale_pc = symmetric_quantize(W, num_bits=8, dim=0)
    W_hat_pc = symmetric_dequantize(q_pc, scale_pc)
    print(f"Per-channel SQNR: {sqnr(W, W_hat_pc):.2f} dB")
    print()
    print(">> Notice: the single outlier channel (row 3) forces the whole")
    print(">> per-tensor scale to be huge, wasting resolution on the other")
    print(">> 63 channels. Per-channel quantization isolates the damage.")
    print(f">> Row 3 alone SQNR (per-tensor): "
          f"{sqnr(W[3], W_hat_pt[3]):.2f} dB vs a normal row (row 0): "
          f"{sqnr(W[0], W_hat_pt[0]):.2f} dB")
    print(f">> Row 0 SQNR (per-channel): {sqnr(W[0], W_hat_pc[0]):.2f} dB "
          f"<- much better, unaffected by row 3's outlier")

    print()
    print("=" * 60)
    print("ASYMMETRIC QUANTIZATION — for post-ReLU activations")
    print("=" * 60)
    # Simulate activations after a ReLU: all >= 0, skewed distribution
    act = torch.relu(torch.randn(1000) * 2 + 1)
    q_a, scale_a, zp_a = asymmetric_quantize(act, num_bits=8)
    act_hat = asymmetric_dequantize(q_a, scale_a, zp_a)
    print(f"scale={scale_a.item():.6f}  zero_point={zp_a.item():.1f}")
    print(f"Asymmetric SQNR: {sqnr(act, act_hat):.2f} dB")

    # Compare: what if we (wrongly) used symmetric quant on this activation?
    q_s, scale_s = symmetric_quantize(act, num_bits=8)
    act_hat_s = symmetric_dequantize(q_s, scale_s)
    print(f"Symmetric SQNR (wrong choice for ReLU output): "
          f"{sqnr(act, act_hat_s):.2f} dB  <- wastes half the int8 range on negatives that don't exist")

    print()
    print("=" * 60)
    print("VALIDATE AGAINST PYTORCH'S BUILT-IN QUANTIZATION")
    print("=" * 60)
    # torch's own per-tensor affine quant, for sanity-check
    torch_q = torch.quantize_per_tensor(W, scale=scale_pt.item(), zero_point=0, dtype=torch.qint8)
    torch_dq = torch_q.dequantize()
    diff = (torch_dq - W_hat_pt).abs().max().item()
    print(f"Max diff between our impl and torch.quantize_per_tensor: {diff:.8f}")
    print("(should be ~0 — confirms our from-scratch math matches PyTorch internals)")
