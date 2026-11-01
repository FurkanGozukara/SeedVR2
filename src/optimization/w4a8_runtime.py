"""
Grouped W4A8 ConvRot DiT checkpoints (ComfyUI "asym_w4a8_int8" layout).

Checkpoints such as ussoewwin/SeedVR2-ConvRot-INT8-and-w4a8 store each quantized Linear as
``weight`` (two 4-bit codes per int8 byte, [N, K/2]), ``weight_s_rel`` (FP8 per-16-channel
relative scales), ``weight_s_channel`` (FP32 per-row scales), ``weight_codebook`` (16 levels)
and a ``comfy_quant`` JSON record. Weights are ConvRot-rotated (Hadamard groups of 256).
The forward runs comfy_kitchen's fused dequant + INT8 GEMM (``w4a8_int8_linear``), which
rotates and quantizes the activation per token.
"""

from __future__ import annotations

import json
from typing import Dict

import torch
import torch.nn as nn

W4A8_FORMAT = "asym_w4a8_int8"
_QUANT_SUFFIX = ".comfy_quant"
_DROP_KEYS = ("positive_conditioning", "negative_conditioning")


def _record(tensor: torch.Tensor) -> dict:
    try:
        return json.loads(bytes(tensor.cpu().to(torch.uint8).tolist()).decode("utf-8"))
    except Exception:
        return {}


def w4a8_layers(state_dict: Dict[str, torch.Tensor]) -> Dict[str, dict]:
    """Layer name -> quant params for every asym_w4a8_int8 layer in the state dict."""
    layers = {}
    for key, value in state_dict.items():
        if not key.endswith(_QUANT_SUFFIX):
            continue
        rec = _record(value)
        if rec.get("format") == W4A8_FORMAT:
            params = rec.get("params") or {}
            layers[key[: -len(_QUANT_SUFFIX)]] = {
                "group_size": int(params.get("group_size", 16)),
                "convrot_groupsize": int(rec.get("convrot_groupsize", params.get("convrot_groupsize", 256))),
            }
    return layers


def w4a8_unavailable_reason(device: torch.device) -> "str | None":
    """None when comfy_kitchen's CUDA W4A8 kernel can run on `device`, else why not."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return "the W4A8 kernels run on NVIDIA GPUs only"
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) < (8, 0):
        return f"compute capability {major}.{minor}; the W4A8 kernels need 8.0 or newer (RTX 30 series or later)"
    try:
        import comfy_kitchen.backends.cuda as ck_cuda
    except Exception as exc:  # noqa: BLE001 - reported to the user
        return f"comfy_kitchen could not be loaded: {exc}"
    if not getattr(ck_cuda, "_EXT_AVAILABLE", False) or not hasattr(ck_cuda, "w4a8_int8_linear"):
        return f"comfy_kitchen CUDA backend: {getattr(ck_cuda, '_EXT_ERROR', None) or 'no W4A8 kernel'}"
    return None


def is_w4a8_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any(k.endswith(_QUANT_SUFFIX) for k in state_dict) and bool(w4a8_layers(state_dict))


def _w4a8_forward(self, x: torch.Tensor) -> torch.Tensor:
    from comfy_kitchen.tensor.w4a8_int8 import w4a8_int8_linear

    out_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    x2d = x.reshape(-1, x.shape[-1])
    if x2d.dtype != out_dtype:
        x2d = x2d.to(out_dtype)
    bias = self.bias.to(out_dtype) if self.bias is not None else None
    out = w4a8_int8_linear(
        x2d, self.weight, self.weight_s_rel, self.weight_s_channel,
        codebook=self.weight_codebook, bias=bias,
        group_size=self._w4a8_group_size, convrot_groupsize=self._w4a8_convrot_groupsize,
        out_dtype=out_dtype,
    )
    return out.reshape(*x.shape[:-1], out.shape[-1])


def prepare_model_for_w4a8(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> int:
    """Turn the matching nn.Linear modules of a (meta) DiT into W4A8 layers ready for
    load_state_dict(assign=True), and drop the non-tensor records from the state dict."""
    layers = w4a8_layers(state_dict)
    modules = dict(model.named_modules())
    missing = [n for n in layers if not isinstance(modules.get(n), nn.Linear)]
    if missing:
        raise RuntimeError(f"W4A8 checkpoint does not match the DiT; unknown layers: {missing[:8]}")
    for name, q in layers.items():
        m = modules[name]
        for suffix, dtype_key in (("weight_s_rel", None), ("weight_s_channel", None), ("weight_codebook", None)):
            t = state_dict[f"{name}.{suffix}"]
            m.register_buffer(suffix, torch.empty(tuple(t.shape), dtype=t.dtype, device="meta"), persistent=True)
        w = state_dict[f"{name}.weight"]
        m.weight = nn.Parameter(torch.empty(tuple(w.shape), dtype=w.dtype, device="meta"), requires_grad=False)
        m._w4a8_group_size = q["group_size"]
        m._w4a8_convrot_groupsize = q["convrot_groupsize"]
        m._seedvr2_w4a8 = True
        m.forward = _w4a8_forward.__get__(m, type(m))
    for key in [k for k in state_dict if k.endswith(_QUANT_SUFFIX) or k in _DROP_KEYS]:
        del state_dict[key]
    return len(layers)
