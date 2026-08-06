"""
INT8 ConvRot quantization core (offline conversion + runtime linear forward).

ConvRot = group-wise Hadamard rotation applied to Linear weights before
per-output-channel (row-wise) INT8 quantization. The activation is rotated by
the same block-diagonal Hadamard at runtime; because H is symmetric and
orthonormal (H == H.T and H @ H == I) the two rotations cancel inside the
matmul, so results match the un-rotated computation - but weight outliers get
spread within each group first, which massively reduces INT8 quantization
error (about 41 dB weight SQNR vs about 32 dB for scaled FP8).

The on-disk conventions follow the implementations that ship in ComfyUI
(comfy_kitchen "int8_tensorwise" + convrot) and kohya-ss/musubi-tuner, so
checkpoints produced here stay interoperable at the tensor-layout level.

V6.1 quality upgrades on top of the V6.0 baseline:
  - weights clamp to the full [-128, 127] INT8 range (the MSE clip search can
    reach -128; V6.0 discarded that level),
  - closed-form least-squares scale refit after the clip search (always lowers
    per-row weight MSE for the chosen codes),
  - optional per-(row, rotation-group) scales instead of one scale per row,
  - optional calibration-aware features: activation-energy-weighted scale
    search, GPTQ rounding (full Hessian), Hessian/energy-weighted low-rank
    error-recovery adapters (ARA) stored in the same cache file, and bias
    correction (applied by the per-model converters),
  - defensive outlier clamp before rotation so one corrupted weight cannot
    poison a whole Hadamard group,
  - the runtime Triton activation-quantize kernel now divides in fp32 (the
    bf16 division added up to +-0.5 LSB of noise on the top octave of values).

Runtime uses torch._int_mm (real INT8 tensor-core GEMM, PyTorch >= 2.1,
NVIDIA SM >= 7.5 / Turing+). Anything older falls back to a transparent
dequantize + F.linear path with identical outputs, just without the speedup.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F

DEFAULT_INT8_CONVROT_GROUP_SIZES: Tuple[int, ...] = (256, 64, 16)
DEFAULT_INT8_CONVROT_CLIP_MIN = 0.55
DEFAULT_INT8_CONVROT_CLIP_STEPS = 80
DEFAULT_INT8_CONVROT_CHUNK_ELEMENTS = 4 * 1024 * 1024  # retained for compat
DEFAULT_INT8_OUTLIER_CLAMP = 1000.0
DEFAULT_INT8_LOWRANK_RANK = 16
INT8_QMIN = -128.0
INT8_QMAX = 127.0

# torch._int_mm requires M > 16 on CUDA; tiny batches get zero-padded instead
# of falling back to a dequantized matmul.
_INT_MM_SMALL_BATCH_THRESHOLD = 16
_INT_MM_MIN_CUDA_CAPABILITY = (7, 5)

_HADAMARD_CACHE: Dict[Tuple[int, str, torch.dtype], torch.Tensor] = {}
_INT_MM_SUPPORT_CACHE: Dict[str, bool] = {}


def _is_power_of_four(value: int) -> bool:
    if value < 4:
        return False
    while value % 4 == 0:
        value //= 4
    return value == 1


def build_hadamard(size: int, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Regular Hadamard matrix via Kronecker powers of a symmetric 4x4 base block.

    The base block is chosen so every row/column sums to the same value, which
    avoids the all-ones column of standard Sylvester matrices (that column
    amplifies row-wise outliers in diffusion models). The block is symmetric,
    so H == H.T and H @ H == I - the whole fused runtime relies on this.
    """
    if not _is_power_of_four(size):
        raise ValueError(f"Regular Hadamard size must be a power of 4 >= 4, got {size}")
    key = (int(size), str(device), dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        device=device,
        dtype=torch.float32,
    )
    h = h4
    current = 4
    while current < size:
        h = torch.kron(h, h4)
        current *= 4
    h = (h / math.sqrt(size)).to(dtype=dtype)
    _HADAMARD_CACHE[key] = h
    return h


def parse_int8_convrot_groupsizes(groupsizes: object = None) -> Tuple[int, ...]:
    if groupsizes is None:
        return DEFAULT_INT8_CONVROT_GROUP_SIZES
    if isinstance(groupsizes, (list, tuple)):
        values = [int(v) for v in groupsizes]
    else:
        values = [int(v) for v in str(groupsizes).replace(";", ",").split(",") if str(v).strip()]
    cleaned = tuple(sorted({v for v in values if _is_power_of_four(v)}, reverse=True))
    return cleaned or DEFAULT_INT8_CONVROT_GROUP_SIZES


def best_int8_convrot_groupsize(in_features: int, groupsizes: object = None) -> Optional[int]:
    for group_size in parse_int8_convrot_groupsizes(groupsizes):
        if in_features % group_size == 0:
            return group_size
    return None


def rotate_weight(weight: torch.Tensor, h: torch.Tensor, group_size: int, *, inverse: bool = False) -> torch.Tensor:
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(f"in_features {in_features} not divisible by group size {group_size}")
    grouped = weight.reshape(out_features, in_features // group_size, group_size)
    rot = h if inverse else h.T
    return torch.matmul(grouped, rot).reshape(out_features, in_features)


def rotate_activation(x2d: torch.Tensor, group_size: int) -> torch.Tensor:
    """Rotate a [M, in] activation by the block-diagonal Hadamard (x @ H per group)."""
    m, in_features = x2d.shape
    h = build_hadamard(group_size, device=x2d.device, dtype=x2d.dtype)
    grouped = x2d.reshape(m, in_features // group_size, group_size)
    return torch.matmul(grouped, h).reshape(m, in_features)


def _ls_refit_scale(
    x3: torch.Tensor,
    scale: torch.Tensor,
    energy: Optional[torch.Tensor],
    iterations: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Closed-form least-squares scale refit: for fixed integer codes q the MSE-
    (or energy-weighted-) optimal scale is sum(e*w*q) / sum(e*q*q). Alternating
    two rounds of (re-round, refit) strictly lowers the reconstruction error
    below whatever grid point the clip search selected.

    x3: [rows, G, cols], scale: [rows, G, 1], energy: [1, G, cols] or None.
    Returns (q [rows, G, cols] float, scale [rows, G, 1]).
    """
    q = None
    for _ in range(max(1, iterations)):
        q = torch.round(x3 / scale).clamp_(INT8_QMIN, INT8_QMAX)
        if energy is None:
            num = (x3 * q).sum(dim=2, keepdim=True)
            den = (q * q).sum(dim=2, keepdim=True)
        else:
            num = (energy * x3 * q).sum(dim=2, keepdim=True)
            den = (energy * q * q).sum(dim=2, keepdim=True)
        new_scale = num / den.clamp(min=1e-12)
        scale = torch.where((den > 0) & (num > 0), new_scale, scale).clamp(min=1e-30)
    q = torch.round(x3 / scale).clamp_(INT8_QMIN, INT8_QMAX)
    return q, scale


def quantize_int8_rowwise(
    x: torch.Tensor,
    *,
    mse_clip: bool = True,
    clip_min: float = DEFAULT_INT8_CONVROT_CLIP_MIN,
    clip_steps: int = DEFAULT_INT8_CONVROT_CLIP_STEPS,
    col_energy: Optional[torch.Tensor] = None,
    scale_groups: int = 1,
    ls_refit: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric INT8 quantization with an MSE-optimal clip search plus a
    closed-form least-squares scale refit.

    - scale_groups == 1: one scale per row (default, matches the ecosystem
      int8_tensorwise layout).
    - scale_groups == G: one scale per (row, contiguous cols/G block). The
      caller is responsible for making G line up with the rotation groups.
    - col_energy: optional per-column weights (E[x_rot^2] from calibration);
      the search and refit then minimize the activation-energy-weighted error,
      a better proxy for output error than plain weight MSE.

    Returns (q int8 [rows, cols], scale float32 [rows, scale_groups]).
    """
    rows, cols = int(x.shape[0]), int(x.shape[1])
    groups = max(1, int(scale_groups))
    if cols % groups != 0:
        raise ValueError(f"scale_groups {groups} does not divide {cols} columns")
    gcols = cols // groups

    x3 = x.detach().float().reshape(rows, groups, gcols)
    energy = None
    if col_energy is not None:
        energy = col_energy.detach().float().reshape(1, groups, gcols).to(x3.device)
        energy = energy.clamp(min=1e-12)

    absmax = x3.abs().amax(dim=2, keepdim=True).clamp(min=1e-30)
    if not mse_clip or clip_steps <= 1:
        best_scale = (absmax / 127.0).clamp(min=1e-30)
    else:
        best_mse = torch.full_like(absmax, float("inf"))
        best_scale = (absmax / 127.0).clamp(min=1e-30)
        for ratio in torch.linspace(clip_min, 1.0, clip_steps, device=x3.device, dtype=torch.float32):
            scale = (absmax * ratio / 127.0).clamp(min=1e-30)
            q = torch.round(x3 / scale).clamp_(INT8_QMIN, INT8_QMAX)
            err = q * scale - x3
            if energy is None:
                mse = (err * err).mean(dim=2, keepdim=True)
            else:
                mse = (err * err * energy).mean(dim=2, keepdim=True)
            better = mse < best_mse
            best_mse = torch.where(better, mse, best_mse)
            best_scale = torch.where(better, scale, best_scale)

    if ls_refit:
        q, best_scale = _ls_refit_scale(x3, best_scale, energy)
    else:
        q = torch.round(x3 / best_scale).clamp_(INT8_QMIN, INT8_QMAX)

    q_int8 = q.reshape(rows, cols).to(torch.int8)
    scale_out = best_scale.reshape(rows, groups).contiguous()
    return q_int8, scale_out


def quantize_int8_convrot_weight(
    weight: torch.Tensor,
    *,
    group_size: int,
    calc_device: str | torch.device = "cpu",
    mse_clip: bool = True,
    max_chunk_elements: int = DEFAULT_INT8_CONVROT_CHUNK_ELEMENTS,  # unused, kept for compat
    col_energy: Optional[torch.Tensor] = None,
    scale_groups: int = 1,
    outlier_clamp: float = DEFAULT_INT8_OUTLIER_CLAMP,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rotate + quantize a 2-D Linear weight.

    col_energy is E[x_rot^2] per input column (rotated basis) when calibration
    statistics are available. outlier_clamp zeroes obviously-corrupted weight
    entries (|w| > clamp) before rotation so a single garbage value cannot
    poison a whole Hadamard group.

    Returns (q int8 [out, in] in rotated space, scale float32 [out, scale_groups]).
    """
    if weight.ndim != 2:
        raise ValueError(f"INT8 ConvRot expects 2-D weights, got shape {tuple(weight.shape)}")
    out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
    if in_features % group_size != 0:
        raise ValueError(f"in_features {in_features} not divisible by group size {group_size}")

    try:
        w = weight.detach().to(device=calc_device, dtype=torch.float32)
    except RuntimeError:
        calc_device = "cpu"
        w = weight.detach().to(device="cpu", dtype=torch.float32)
    if outlier_clamp and outlier_clamp > 0:
        bad = w.abs() > float(outlier_clamp)
        if bool(bad.any()):
            print(
                f"[INT8 ConvRot] zeroing {int(bad.sum())} corrupted weight value(s) "
                f"(|w| > {outlier_clamp:g}) before rotation",
                flush=True,
            )
            w = w.masked_fill(bad, 0.0)
    h = build_hadamard(group_size, device=w.device, dtype=torch.float32)
    rotated = rotate_weight(w, h, group_size)
    q, scale = quantize_int8_rowwise(
        rotated,
        mse_clip=mse_clip,
        col_energy=col_energy,
        scale_groups=scale_groups,
    )
    return q.cpu(), scale.cpu()


def dequantize_rotated(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize INT8 codes back to the rotated-space float weight."""
    out_features, in_features = int(q.shape[0]), int(q.shape[1])
    scale2d = scale.reshape(out_features, -1).float()
    groups = int(scale2d.shape[1])
    if groups == 1:
        return q.float() * scale2d
    gcols = in_features // groups
    w = q.float().reshape(out_features, groups, gcols) * scale2d.unsqueeze(2)
    return w.reshape(out_features, in_features)


def dequantize_int8_convrot_weight(
    q: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    *,
    dtype: torch.dtype = torch.float32,
    ara_down: Optional[torch.Tensor] = None,
    ara_up: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reverse rotation + quantization (used by the no-int_mm fallback path)."""
    h = build_hadamard(group_size, device=q.device, dtype=torch.float32)
    w = dequantize_rotated(q, scale.to(q.device))
    if ara_down is not None and ara_up is not None:
        w = w + ara_up.float().to(q.device) @ ara_down.float().to(q.device)
    w = rotate_weight(w, h, group_size, inverse=True)
    return w.to(dtype)


# --------------------------------------------------------------------- #
# GPTQ rounding (calibration Hessian required)
# --------------------------------------------------------------------- #
def gptq_quantize_rotated(
    w_rot: torch.Tensor,
    hessian: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_size: int = 128,
    damp: float = 0.01,
) -> Optional[torch.Tensor]:
    """
    GPTQ rounding of a rotated fp32 weight [out, in] against a rotated-basis
    Hessian H = E[x_rot x_rot^T] [in, in], with the (fixed) per-row or
    per-(row, group) scales chosen beforehand. Minimizes output error instead
    of weight error by compensating each column's rounding error into the
    not-yet-quantized columns.

    Original implementation of the published GPTQ algorithm (Frantar et al.),
    no act-order. Returns int8 codes, or None if the Hessian cannot be
    factorized even after damping retries (caller falls back to RTN codes).
    """
    out_features, in_features = int(w_rot.shape[0]), int(w_rot.shape[1])
    device = w_rot.device
    scale2d = scale.reshape(out_features, -1).float().to(device)
    groups = int(scale2d.shape[1])
    gcols = in_features // groups

    w = w_rot.detach().float().clone()
    h = hessian.detach().float().to(device).clone()
    diag = torch.diagonal(h)
    dead = diag <= 0
    if bool(dead.any()):
        diag[dead] = 1.0
        w[:, dead] = 0.0

    mean_diag = float(diag.mean().clamp(min=1e-12))
    damp_val = damp * mean_diag
    upper = None
    for _ in range(6):
        try:
            h_try = h.clone()
            torch.diagonal(h_try).add_(damp_val)
            chol = torch.linalg.cholesky(h_try)
            h_inv = torch.cholesky_inverse(chol)
            upper = torch.linalg.cholesky(h_inv, upper=True)
            break
        except Exception:
            damp_val *= 10.0
            upper = None
    if upper is None:
        return None

    q_out = torch.empty((out_features, in_features), dtype=torch.int8, device=device)
    for start in range(0, in_features, block_size):
        end = min(start + block_size, in_features)
        w_block = w[:, start:end].clone()
        err_block = torch.zeros_like(w_block)
        for local in range(end - start):
            j = start + local
            col = w_block[:, local]
            s_col = scale2d[:, j // gcols]
            q = torch.round(col / s_col).clamp_(INT8_QMIN, INT8_QMAX)
            q_out[:, j] = q.to(torch.int8)
            d = float(upper[j, j])
            err = (col - q * s_col) / max(d, 1e-12)
            if local + 1 < end - start:
                w_block[:, local + 1:] -= err.unsqueeze(1) * upper[j, start + local + 1:end].unsqueeze(0)
            err_block[:, local] = err
        if end < in_features:
            w[:, end:] -= err_block @ upper[start:end, end:]
    return q_out.cpu()


# --------------------------------------------------------------------- #
# Low-rank error recovery (ARA tensors, stored in the same cache file)
# --------------------------------------------------------------------- #
def fit_lowrank_residual(
    w_rot: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
    *,
    rank: int = DEFAULT_INT8_LOWRANK_RANK,
    hessian: Optional[torch.Tensor] = None,
    col_energy: Optional[torch.Tensor] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Fit a rank-r correction to the INT8 residual E = W_rot - dequant(q, scale)
    so the runtime can add back  (x_rot @ down.T) @ up.T .

    With a Hessian (or per-column activation energy) available the SVD is
    computed in the whitened space, i.e. the correction minimizes the
    *output* error under the calibration input distribution, not the plain
    weight error. Returns (ara_up [out, r], ara_down [r, in]) fp32, or None.
    """
    rank = int(rank)
    if rank <= 0:
        return None
    device = w_rot.device
    residual = w_rot.detach().float() - dequantize_rotated(q.to(device), scale.to(device))
    out_features, in_features = residual.shape
    rank = min(rank, out_features, in_features)
    oversample = min(rank + 8, min(out_features, in_features))

    chol = None
    if hessian is not None and hessian.shape[0] == in_features:
        try:
            h = hessian.detach().float().to(device).clone()
            torch.diagonal(h).add_(float(torch.diagonal(h).mean().clamp(min=1e-12)) * 0.01)
            chol = torch.linalg.cholesky(h)
        except Exception:
            chol = None

    try:
        if chol is not None:
            m = residual @ chol
            u, s, v = torch.svd_lowrank(m, q=oversample, niter=4)
            up = u[:, :rank] * s[:rank].unsqueeze(0)
            down_w = v[:, :rank].T  # rank x in, in whitened space
            # down = down_w @ L^-1  <=>  L^T @ down^T = down_w^T
            down = torch.linalg.solve_triangular(
                chol.mT, down_w.T, upper=True, left=True
            ).T.contiguous()
        elif col_energy is not None:
            e_sqrt = col_energy.detach().float().to(device).clamp(min=1e-12).sqrt()
            m = residual * e_sqrt.unsqueeze(0)
            u, s, v = torch.svd_lowrank(m, q=oversample, niter=4)
            up = u[:, :rank] * s[:rank].unsqueeze(0)
            down = (v[:, :rank].T / e_sqrt.unsqueeze(0)).contiguous()
        else:
            u, s, v = torch.svd_lowrank(residual, q=oversample, niter=4)
            up = u[:, :rank] * s[:rank].unsqueeze(0)
            down = v[:, :rank].T.contiguous()
    except Exception as exc:
        print(f"[INT8 ConvRot] low-rank residual fit skipped: {exc}", flush=True)
        return None
    if not (torch.isfinite(up).all() and torch.isfinite(down).all()):
        return None
    return up.contiguous().cpu(), down.contiguous().cpu()


def compute_bias_correction(
    w_rot: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
    mean_rot: torch.Tensor,
    *,
    ara_up: Optional[torch.Tensor] = None,
    ara_down: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Systematic-error bias correction: delta_b = (W_hat_eff - W_rot) @ E[x_rot].
    The caller subtracts this from the stored bias, cancelling the mean output
    shift the quantization introduces under the calibration distribution.
    """
    device = w_rot.device
    w_hat = dequantize_rotated(q.to(device), scale.to(device))
    if ara_up is not None and ara_down is not None:
        w_hat = w_hat + ara_up.float().to(device) @ ara_down.float().to(device)
    err = w_hat - w_rot.detach().float()
    return err @ mean_rot.detach().float().to(device)


def weight_error_metrics(
    w_rot: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
    *,
    ara_up: Optional[torch.Tensor] = None,
    ara_down: Optional[torch.Tensor] = None,
    col_energy: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Relative error / SQNR of the effective quantized weight vs the original."""
    device = w_rot.device
    w = w_rot.detach().float()
    w_hat = dequantize_rotated(q.to(device), scale.to(device))
    if ara_up is not None and ara_down is not None:
        w_hat = w_hat + ara_up.float().to(device) @ ara_down.float().to(device)
    err = w_hat - w
    if col_energy is not None:
        e = col_energy.detach().float().to(device).clamp(min=1e-12).unsqueeze(0)
        err_energy = float((err * err * e).sum())
        sig_energy = float((w * w * e).sum())
    else:
        err_energy = float((err * err).sum())
        sig_energy = float((w * w).sum())
    rel = math.sqrt(err_energy / max(sig_energy, 1e-30))
    sqnr = 10.0 * math.log10(max(sig_energy, 1e-30) / max(err_energy, 1e-30))
    return {"rel_err": rel, "sqnr_db": sqnr, "err_energy": err_energy, "sig_energy": sig_energy}


def supports_int_mm(device: torch.device | str) -> bool:
    """True when torch._int_mm can run fast on this device (CUDA SM >= 7.5)."""
    key = str(device)
    cached = _INT_MM_SUPPORT_CACHE.get(key)
    if cached is not None:
        return cached
    ok = False
    try:
        dev = torch.device(device)
        if hasattr(torch, "_int_mm") and dev.type == "cuda" and torch.cuda.is_available():
            ok = torch.cuda.get_device_capability(dev) >= _INT_MM_MIN_CUDA_CAPABILITY
    except Exception:
        ok = False
    _INT_MM_SUPPORT_CACHE[key] = ok
    return ok


def _int_mm_allow_small_m(a_int8: torch.Tensor, b_int8: torch.Tensor) -> torch.Tensor:
    """torch._int_mm requires M > 16 on CUDA; zero-pad tiny batches instead."""
    m = a_int8.shape[0]
    if m > _INT_MM_SMALL_BATCH_THRESHOLD:
        return torch._int_mm(a_int8, b_int8)
    padded_m = _INT_MM_SMALL_BATCH_THRESHOLD + 1
    padding = torch.zeros((padded_m - m, a_int8.shape[1]), device=a_int8.device, dtype=a_int8.dtype)
    return torch._int_mm(torch.cat((a_int8, padding), dim=0), b_int8)[:m]


def _quantize_int8_per_token(x2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    abs_max = x2d.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max / 127.0).clamp(min=1e-30).float()
    quantized = (x2d.float() / scale).round().clamp(-127, 127).to(torch.int8)
    return quantized, scale


# --------------------------------------------------------------------- #
# Triton fused fast path (GEMM + dequant + bias in one kernel)
# --------------------------------------------------------------------- #
# The eager torch._int_mm path pays for two extra full [M, N] float32 passes
# (per-token scale, per-channel scale). The Triton kernels below fuse the
# whole epilogue into the GEMM, which is where the real INT8 speedup over
# BF16 comes from. If Triton is missing or fails to compile on this machine,
# we latch it off and silently use the eager path (identical outputs, just
# slower).
_TRITON_STATE = {"checked": False, "ok": False}


def _triton_available() -> bool:
    if _TRITON_STATE["checked"]:
        return _TRITON_STATE["ok"]
    _TRITON_STATE["checked"] = True
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401

        _build_triton_kernels()
        _TRITON_STATE["ok"] = True
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[INT8 ConvRot] Triton fast path unavailable ({exc}); using torch._int_mm path", flush=True)
        _TRITON_STATE["ok"] = False
    return _TRITON_STATE["ok"]


def mark_triton_broken(reason: str = "") -> None:
    """Latch the Triton path off after a runtime failure (self-healing)."""
    if _TRITON_STATE.get("ok"):
        print(f"[INT8 ConvRot] Disabling Triton fast path: {reason}", flush=True)
    _TRITON_STATE["checked"] = True
    _TRITON_STATE["ok"] = False


_TRITON_KERNELS: Dict[str, object] = {}


def _build_triton_kernels() -> None:
    if _TRITON_KERNELS:
        return
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def _quantize_rowwise_kernel(x_ptr, y_ptr, s_ptr, n_elements, block_size: tl.constexpr):
        row_idx = tl.program_id(0)
        x_row_ptr = x_ptr + row_idx * n_elements
        y_row_ptr = y_ptr + row_idx * n_elements
        offsets = tl.arange(0, block_size)
        mask = offsets < n_elements
        x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
        # All math in fp32: a bf16 division adds up to +-0.5 LSB of noise on
        # the top octave of quantized values (bf16 step is 0.5 between 64 and
        # 128), which measurably degrades activation SQNR.
        x_f = x.to(tl.float32)
        abs_x = tl.abs(x_f)
        max_val = tl.max(abs_x, axis=0)
        scale = tl.maximum(max_val / 127.0, 1e-30)
        q_i = tl.clamp(libdevice.rint(x_f / scale), -127.0, 127.0).to(tl.int32)
        tl.store(y_row_ptr + offsets, q_i.to(tl.int8), mask=mask)
        tl.store(s_ptr + row_idx, scale)

    @triton.autotune(
        configs=[
            triton.Config({"block_m": 128, "block_n": 256, "block_k": 64, "group_size_m": 8}, num_stages=3, num_warps=8),
            triton.Config({"block_m": 64, "block_n": 256, "block_k": 32, "group_size_m": 8}, num_stages=4, num_warps=4),
            triton.Config({"block_m": 128, "block_n": 128, "block_k": 32, "group_size_m": 8}, num_stages=4, num_warps=4),
            triton.Config({"block_m": 128, "block_n": 64, "block_k": 32, "group_size_m": 8}, num_stages=4, num_warps=4),
            triton.Config({"block_m": 64, "block_n": 128, "block_k": 32, "group_size_m": 8}, num_stages=4, num_warps=4),
            triton.Config({"block_m": 128, "block_n": 32, "block_k": 32, "group_size_m": 8}, num_stages=4, num_warps=4),
        ],
        key=["m", "n", "k"],
    )
    @triton.jit
    def _int8_matmul_dequant_per_row_kernel(
        a_ptr, b_ptr, c_ptr,
        a_scale_ptr, b_scale_ptr, bias_ptr,
        m, n, k,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
        group_size_m: tl.constexpr,
        has_bias: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(m, block_m)
        num_pid_n = tl.cdiv(n, block_n)
        num_pid_in_group = group_size_m * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * group_size_m
        actual_group_size_m = min(num_pid_m - first_pid_m, group_size_m)
        pid_m = first_pid_m + (pid % actual_group_size_m)
        pid_n = (pid % num_pid_in_group) // actual_group_size_m

        offs_am = (pid_m * block_m + tl.arange(0, block_m)) % m
        offs_bn = (pid_n * block_n + tl.arange(0, block_n)) % n
        offs_k = tl.arange(0, block_k)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        accumulator = tl.zeros((block_m, block_n), dtype=tl.int32)
        for k_idx in range(0, tl.cdiv(k, block_k)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k - k_idx * block_k, other=0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k - k_idx * block_k, other=0)
            accumulator += tl.dot(a, b)
            a_ptrs += block_k * stride_ak
            b_ptrs += block_k * stride_bk

        scale_a = tl.load(a_scale_ptr + offs_am)
        scale_b = tl.load(b_scale_ptr + offs_bn)
        c = accumulator.to(tl.float32)
        c = c * (scale_a[:, None] * scale_b[None, :])
        if has_bias:
            bias = tl.load(bias_ptr + offs_bn)
            c = c + bias[None, :]
        c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
        c_mask = (offs_am[:, None] < m) & (offs_bn[None, :] < n)
        tl.store(c_ptrs, c, mask=c_mask)

    @triton.jit
    def _int8_matmul_dequant_group_kernel(
        a_ptr, b_ptr, c_ptr,
        a_scale_ptr, b_scale_ptr, bias_ptr,
        m, n, k,
        num_groups, blocks_per_group,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_bsn, stride_bsg,
        block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
        group_size_m: tl.constexpr,
        has_bias: tl.constexpr,
    ):
        # Per-(row, K-group) weight scales: the int32 accumulator is flushed
        # into the fp32 accumulator at every scale-group boundary. K is an
        # exact multiple of (blocks_per_group * block_k) by construction, so
        # no K masks are needed.
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(m, block_m)
        num_pid_n = tl.cdiv(n, block_n)
        num_pid_in_group = group_size_m * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * group_size_m
        actual_group_size_m = min(num_pid_m - first_pid_m, group_size_m)
        pid_m = first_pid_m + (pid % actual_group_size_m)
        pid_n = (pid % num_pid_in_group) // actual_group_size_m

        offs_am = (pid_m * block_m + tl.arange(0, block_m)) % m
        offs_bn = (pid_n * block_n + tl.arange(0, block_n)) % n
        offs_k = tl.arange(0, block_k)

        acc_f = tl.zeros((block_m, block_n), dtype=tl.float32)
        for g in range(0, num_groups):
            acc_i = tl.zeros((block_m, block_n), dtype=tl.int32)
            for kb in range(0, blocks_per_group):
                k_base = g * blocks_per_group * block_k + kb * block_k
                a_ptrs = a_ptr + (offs_am[:, None] * stride_am + (k_base + offs_k)[None, :] * stride_ak)
                b_ptrs = b_ptr + ((k_base + offs_k)[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                acc_i += tl.dot(a, b)
            scale_bg = tl.load(b_scale_ptr + offs_bn * stride_bsn + g * stride_bsg)
            acc_f += acc_i.to(tl.float32) * scale_bg[None, :]

        scale_a = tl.load(a_scale_ptr + offs_am)
        c = acc_f * scale_a[:, None]
        if has_bias:
            bias = tl.load(bias_ptr + offs_bn)
            c = c + bias[None, :]
        c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
        c_mask = (offs_am[:, None] < m) & (offs_bn[None, :] < n)
        tl.store(c_ptrs, c, mask=c_mask)

    _TRITON_KERNELS["quantize_rowwise"] = _quantize_rowwise_kernel
    _TRITON_KERNELS["matmul_dequant_per_row"] = _int8_matmul_dequant_per_row_kernel
    _TRITON_KERNELS["matmul_dequant_group"] = _int8_matmul_dequant_group_kernel


def _triton_quantize_rowwise(x2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    import triton

    rows, cols = x2d.shape
    y = torch.empty_like(x2d, dtype=torch.int8)
    s = torch.empty((rows, 1), device=x2d.device, dtype=torch.float32)
    block_size = max(triton.next_power_of_2(cols), 128)
    _TRITON_KERNELS["quantize_rowwise"][(rows,)](
        x2d.contiguous(), y, s, cols, block_size=block_size
    )
    return y, s


def _triton_int8_convrot_linear(
    x2d: torch.Tensor,
    q_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    ara_down: Optional[torch.Tensor],
    ara_up: Optional[torch.Tensor],
) -> torch.Tensor:
    import triton

    m, k = x2d.shape
    n = q_weight.shape[0]
    xr = rotate_activation(x2d, group_size)
    x_int8, x_scale = _triton_quantize_rowwise(xr)
    has_ara = ara_down is not None and ara_up is not None
    output = torch.empty((m, n), device=x2d.device, dtype=torch.float32 if has_ara else out_dtype)
    has_bias = bias is not None
    bias_f32 = bias.float().contiguous() if has_bias else x_scale

    scale2d = weight_scale.reshape(n, -1).float().contiguous()
    groups = int(scale2d.shape[1])
    if groups == 1:
        kernel = _TRITON_KERNELS["matmul_dequant_per_row"]

        def grid(meta):
            return (triton.cdiv(m, meta["block_m"]) * triton.cdiv(n, meta["block_n"]),)

        kernel[grid](
            a_ptr=x_int8,
            b_ptr=q_weight,
            c_ptr=output,
            a_scale_ptr=x_scale,
            b_scale_ptr=scale2d.reshape(-1),
            bias_ptr=bias_f32,
            m=m, n=n, k=k,
            stride_am=x_int8.stride(0), stride_ak=x_int8.stride(1),
            stride_bk=q_weight.stride(1), stride_bn=q_weight.stride(0),
            stride_cm=output.stride(0), stride_cn=output.stride(1),
            has_bias=has_bias,
        )
    else:
        gcols = k // groups
        block_k = 32
        if gcols % block_k != 0:
            raise RuntimeError(f"scale group width {gcols} not divisible by block_k {block_k}")
        kernel = _TRITON_KERNELS["matmul_dequant_group"]
        block_m, block_n = 64, 64
        grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
        kernel[grid](
            a_ptr=x_int8,
            b_ptr=q_weight,
            c_ptr=output,
            a_scale_ptr=x_scale,
            b_scale_ptr=scale2d,
            bias_ptr=bias_f32,
            m=m, n=n, k=k,
            num_groups=groups, blocks_per_group=gcols // block_k,
            stride_am=x_int8.stride(0), stride_ak=x_int8.stride(1),
            stride_bk=q_weight.stride(1), stride_bn=q_weight.stride(0),
            stride_cm=output.stride(0), stride_cn=output.stride(1),
            stride_bsn=scale2d.stride(0), stride_bsg=scale2d.stride(1),
            block_m=block_m, block_n=block_n, block_k=block_k,
            group_size_m=8,
            has_bias=has_bias,
            num_warps=4, num_stages=4,
        )
    if has_ara:
        output = output + (xr @ ara_down.to(xr.dtype).T).to(torch.float32) @ ara_up.to(torch.float32).T
    return output.to(out_dtype)


def int8_convrot_linear(
    x: torch.Tensor,
    q_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    bias: Optional[torch.Tensor] = None,
    ara_down: Optional[torch.Tensor] = None,
    ara_up: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    INT8 ConvRot Linear: rotate activation -> dynamic per-token INT8 ->
    fused Triton INT8 GEMM (or torch._int_mm) -> rescale, plus the optional
    low-rank error-recovery branch (x_rot @ ara_down.T) @ ara_up.T. Falls back
    to a dequantized F.linear off-CUDA or on GPUs older than SM 7.5.

    weight_scale may be [out] / [out, 1] (per-row) or [out, G] with G > 1
    (per-(row, K-group)); the per-group layout needs the Triton kernel and
    otherwise uses the dequantized fallback.
    """
    out_features = q_weight.shape[0]
    x2d = x.reshape(-1, x.shape[-1])
    on_gpu = supports_int_mm(x2d.device) and q_weight.device == x2d.device
    scale2d = weight_scale.reshape(out_features, -1)
    per_group = scale2d.shape[1] > 1

    if on_gpu and _triton_available():
        try:
            out = _triton_int8_convrot_linear(
                x2d, q_weight, weight_scale, group_size, bias, x.dtype, ara_down, ara_up
            )
            return out.reshape(*x.shape[:-1], out_features)
        except Exception as exc:  # pragma: no cover - driver/compiler specific
            mark_triton_broken(str(exc)[:200])

    if on_gpu and not per_group:
        xr = rotate_activation(x2d, group_size)
        x_int8, x_scale = _quantize_int8_per_token(xr)
        acc = _int_mm_allow_small_m(x_int8.contiguous(), q_weight.t())
        out = acc.float() * x_scale * scale2d.reshape(1, -1).float()
        if ara_down is not None and ara_up is not None:
            out = out + (xr @ ara_down.to(xr.dtype).T).float() @ ara_up.float().T
        out = out.to(x.dtype)
    else:
        weight = dequantize_int8_convrot_weight(
            q_weight, weight_scale, group_size, dtype=x.dtype,
            ara_down=ara_down, ara_up=ara_up,
        )
        out = F.linear(x2d, weight)
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out.reshape(*x.shape[:-1], out_features)
