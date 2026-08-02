"""
INT8 ConvRot quantization core (offline conversion + runtime linear forward).

ConvRot = group-wise Hadamard rotation applied to Linear weights before
per-output-channel (row-wise) INT8 quantization. The activation is rotated by
the same block-diagonal Hadamard at runtime; because H is symmetric and
orthonormal (H == H.T and H @ H == I) the two rotations cancel inside the
matmul, so results match the un-rotated computation - but weight outliers get
spread within each group first, which massively reduces INT8 quantization
error (about 41 dB weight SQNR vs about 32 dB for scaled FP8).

The math and on-disk conventions follow the implementations that ship in
ComfyUI (comfy_kitchen "int8_tensorwise" + convrot) and kohya-ss/musubi-tuner
(modules/int8_convrot_utils.py, integrated from musubi-tuner PR #1008 /
arXiv:2512.03673), so checkpoints produced here stay interoperable.

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
DEFAULT_INT8_CONVROT_CHUNK_ELEMENTS = 4 * 1024 * 1024

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


def quantize_int8_rowwise(
    x: torch.Tensor,
    *,
    mse_clip: bool = True,
    clip_min: float = DEFAULT_INT8_CONVROT_CLIP_MIN,
    clip_steps: int = DEFAULT_INT8_CONVROT_CLIP_STEPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-row symmetric INT8 quantization with optional MSE-optimal clip search.

    Returns (q int8 [rows, cols], scale float32 [rows, 1]).
    """
    absmax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-30)
    if not mse_clip or clip_steps <= 1:
        scale = (absmax.float() / 127.0).clamp(min=1e-30)
        q = (x.float() / scale).round().clamp(-127, 127).to(torch.int8)
        return q, scale

    best_mse = torch.full_like(absmax, float("inf"), dtype=torch.float32)
    best_scale = (absmax.float() / 127.0).clamp(min=1e-30)
    best_q: Optional[torch.Tensor] = None
    x_f32 = x.float()
    for ratio in torch.linspace(clip_min, 1.0, clip_steps, device=x.device, dtype=torch.float32):
        scale = (absmax.float() * ratio / 127.0).clamp(min=1e-30)
        q = (x_f32 / scale).round().clamp(-127, 127).to(torch.int8)
        mse = ((q.float() * scale - x_f32) ** 2).mean(dim=1, keepdim=True)
        better = mse < best_mse
        best_mse = torch.where(better, mse, best_mse)
        best_scale = torch.where(better, scale, best_scale)
        best_q = q if best_q is None else torch.where(better.expand_as(q), q, best_q)
    return best_q, best_scale


def _row_slices(out_features: int, in_features: int, max_chunk_elements: int) -> Iterable[slice]:
    rows_per_chunk = max(1, int(max_chunk_elements // max(1, in_features)))
    for start in range(0, out_features, rows_per_chunk):
        yield slice(start, min(start + rows_per_chunk, out_features))


def quantize_int8_convrot_weight(
    weight: torch.Tensor,
    *,
    group_size: int,
    calc_device: str | torch.device = "cpu",
    mse_clip: bool = True,
    max_chunk_elements: int = DEFAULT_INT8_CONVROT_CHUNK_ELEMENTS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rotate + quantize a 2-D Linear weight. Chunked by rows so a large weight
    never fully materializes in float32.

    Returns (q int8 [out, in] in rotated space, scale float32 [out, 1]).
    """
    if weight.ndim != 2:
        raise ValueError(f"INT8 ConvRot expects 2-D weights, got shape {tuple(weight.shape)}")
    out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
    if in_features % group_size != 0:
        raise ValueError(f"in_features {in_features} not divisible by group size {group_size}")

    h = build_hadamard(group_size, device=calc_device, dtype=torch.float32)
    q_out = torch.empty((out_features, in_features), dtype=torch.int8)
    scale_out = torch.empty((out_features, 1), dtype=torch.float32)
    for row_slice in _row_slices(out_features, in_features, max_chunk_elements):
        w_chunk = weight[row_slice].to(device=calc_device, dtype=torch.float32)
        rotated = rotate_weight(w_chunk, h, group_size)
        q_chunk, scale_chunk = quantize_int8_rowwise(rotated, mse_clip=mse_clip)
        q_out[row_slice].copy_(q_chunk.cpu())
        scale_out[row_slice].copy_(scale_chunk.cpu())
    return q_out, scale_out


def dequantize_int8_convrot_weight(
    q: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reverse the rotation + quantization (used by the no-int_mm fallback path)."""
    h = build_hadamard(group_size, device=q.device, dtype=torch.float32)
    w = q.float() * scale.float()
    w = rotate_weight(w, h, group_size, inverse=True)
    return w.to(dtype)


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
# (per-token scale, per-channel scale). The Triton kernel below - ported from
# ComfyUI's comfy_kitchen backend - fuses the whole epilogue into the GEMM,
# which is where the real INT8 speedup over BF16 comes from. If Triton is
# missing or fails to compile on this machine, we latch it off and silently
# use the eager path (identical outputs, just slower).
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
    def _quantize_rowwise_kernel(x_ptr, y_ptr, s_ptr, n_elements, block_size: tl.constexpr, input_dtype_code: tl.constexpr):
        row_idx = tl.program_id(0)
        x_row_ptr = x_ptr + row_idx * n_elements
        y_row_ptr = y_ptr + row_idx * n_elements
        offsets = tl.arange(0, block_size)
        mask = offsets < n_elements
        x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
        abs_x = tl.abs(x)
        max_val = tl.max(abs_x, axis=0)
        scale = tl.maximum(max_val / 127.0, 1e-30)
        if input_dtype_code == 1:
            q_f = (x / scale.to(tl.float16)).to(tl.float16)
        elif input_dtype_code == 2:
            q_f = (x / scale.to(tl.bfloat16)).to(tl.bfloat16)
        else:
            q_f = x / scale
        q_i = tl.clamp(libdevice.rint(q_f.to(tl.float32)), -127.0, 127.0).to(tl.int32)
        tl.store(y_row_ptr + offsets, q_i.to(tl.int8), mask=mask)
        tl.store(s_ptr + row_idx, scale.to(tl.float32))

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

    _TRITON_KERNELS["quantize_rowwise"] = _quantize_rowwise_kernel
    _TRITON_KERNELS["matmul_dequant_per_row"] = _int8_matmul_dequant_per_row_kernel


def _triton_quantize_rowwise(x2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    import triton

    rows, cols = x2d.shape
    y = torch.empty_like(x2d, dtype=torch.int8)
    s = torch.empty((rows, 1), device=x2d.device, dtype=torch.float32)
    input_dtype_code = 1 if x2d.dtype == torch.float16 else 2 if x2d.dtype == torch.bfloat16 else 0
    block_size = max(triton.next_power_of_2(cols), 128)
    _TRITON_KERNELS["quantize_rowwise"][(rows,)](
        x2d.contiguous(), y, s, cols, block_size=block_size, input_dtype_code=input_dtype_code
    )
    return y, s


def _triton_int8_convrot_linear(
    x2d: torch.Tensor,
    q_weight: torch.Tensor,
    weight_scale_flat: torch.Tensor,
    group_size: int,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    import triton

    m, k = x2d.shape
    n = q_weight.shape[0]
    xr = rotate_activation(x2d, group_size)
    x_int8, x_scale = _triton_quantize_rowwise(xr)
    output = torch.empty((m, n), device=x2d.device, dtype=out_dtype)
    kernel = _TRITON_KERNELS["matmul_dequant_per_row"]
    has_bias = bias is not None

    def grid(meta):
        return (triton.cdiv(m, meta["block_m"]) * triton.cdiv(n, meta["block_n"]),)

    kernel[grid](
        a_ptr=x_int8,
        b_ptr=q_weight,
        c_ptr=output,
        a_scale_ptr=x_scale,
        b_scale_ptr=weight_scale_flat,
        bias_ptr=bias if has_bias else x_int8,
        m=m, n=n, k=k,
        stride_am=x_int8.stride(0), stride_ak=x_int8.stride(1),
        stride_bk=q_weight.stride(1), stride_bn=q_weight.stride(0),
        stride_cm=output.stride(0), stride_cn=output.stride(1),
        has_bias=has_bias,
    )
    return output


def int8_convrot_linear(
    x: torch.Tensor,
    q_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    INT8 ConvRot Linear: rotate activation -> dynamic per-token INT8 ->
    fused Triton INT8 GEMM (or torch._int_mm) -> rescale. Falls back to a
    dequantized F.linear off-CUDA or on GPUs older than SM 7.5.
    """
    out_features = q_weight.shape[0]
    x2d = x.reshape(-1, x.shape[-1])
    on_gpu = supports_int_mm(x2d.device) and q_weight.device == x2d.device

    if on_gpu and _triton_available():
        try:
            out = _triton_int8_convrot_linear(
                x2d,
                q_weight,
                weight_scale.reshape(-1).contiguous(),
                group_size,
                bias,
                x.dtype,
            )
            return out.reshape(*x.shape[:-1], out_features)
        except Exception as exc:  # pragma: no cover - driver/compiler specific
            mark_triton_broken(str(exc)[:200])

    if on_gpu:
        xr = rotate_activation(x2d, group_size)
        x_int8, x_scale = _quantize_int8_per_token(xr)
        acc = _int_mm_allow_small_m(x_int8.contiguous(), q_weight.t())
        out = acc.float() * x_scale * weight_scale.reshape(1, -1).float()
        out = out.to(x.dtype)
    else:
        weight = dequantize_int8_convrot_weight(q_weight, weight_scale, group_size, dtype=x.dtype)
        out = F.linear(x2d, weight)
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out.reshape(*x.shape[:-1], out_features)
