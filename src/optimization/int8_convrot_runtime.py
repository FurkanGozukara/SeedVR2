"""
On-the-fly INT8 ConvRot quantization for the SeedVR2 DiT.

After the DiT state dict is loaded (bf16/fp16 safetensors, never GGUF), every
eligible nn.Linear weight is rotated by a block-diagonal Hadamard, quantized
to INT8 with per-row MSE-optimized scales, and the layer forward is patched to
run a fused INT8 GEMM (Triton / torch._int_mm). Roughly halves DiT weight VRAM
versus bf16 and is faster on NVIDIA Turing (SM 7.5) or newer; older GPUs fall
back to a transparent dequantized matmul.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.optimization.int8_convrot import (
    best_int8_convrot_groupsize,
    int8_convrot_linear,
    quantize_int8_convrot_weight,
)

# Keep the patch-in/patch-out projections and small embedding MLPs in the
# original dtype: they are tiny (no speed win) and quality-sensitive.
_SKIP_SUFFIXES = ("vid_in.proj", "vid_out.proj", "txt_in")
_SKIP_PREFIXES = ("emb_in.",)


def _patch_linear(module: nn.Linear, group_size: int, q: torch.Tensor, scale: torch.Tensor) -> None:
    module.weight = nn.Parameter(q, requires_grad=False)  # int8 params cannot require grad
    module.register_buffer("scale_weight", scale, persistent=False)
    module._int8_convrot_gs = int(group_size)

    def int8_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        return int8_convrot_linear(x, self.weight, self.scale_weight, self._int8_convrot_gs, self.bias)

    module.forward = int8_forward.__get__(module, type(module))
    module._seedvr2_int8_convrot = True


def apply_int8_convrot_to_model(model: nn.Module, debug=None, model_type: str = "DiT") -> int:
    """Quantize eligible Linear layers in-place. Returns the number patched."""
    target = model.dit_model if hasattr(model, "dit_model") else model
    calc_device = "cuda" if torch.cuda.is_available() else "cpu"
    patched = 0
    skipped = 0
    for name, module in list(target.named_modules()):
        if not isinstance(module, nn.Linear) or getattr(module, "_seedvr2_int8_convrot", False):
            continue
        if name.endswith(_SKIP_SUFFIXES) or name.startswith(_SKIP_PREFIXES):
            skipped += 1
            continue
        if module.weight is None or module.weight.device.type == "meta" or module.weight.dtype == torch.int8:
            skipped += 1
            continue
        group_size = best_int8_convrot_groupsize(int(module.in_features))
        if group_size is None:
            skipped += 1
            continue
        device = module.weight.device
        q, scale = quantize_int8_convrot_weight(
            module.weight.detach(), group_size=group_size, calc_device=calc_device, mse_clip=True
        )
        _patch_linear(module, group_size, q.to(device), scale.to(device))
        patched += 1
    message = f"INT8 ConvRot: quantized {patched} {model_type} Linear layers ({skipped} skipped)"
    if debug is not None:
        try:
            debug.log(message, category="precision", force=True)
        except Exception:
            print(f"[SeedVR2] {message}", flush=True)
    else:
        print(f"[SeedVR2] {message}", flush=True)
    return patched


def count_int8_convrot_linears(model: nn.Module) -> int:
    target = model.dit_model if hasattr(model, "dit_model") else model
    return sum(
        1
        for module in target.modules()
        if isinstance(module, nn.Linear) and getattr(module, "_seedvr2_int8_convrot", False)
    )
