"""
VAE Fused Path Module for SeedVR2 (port of sh202603 d3254c722b; enabled with SEEDVR2_FUSED_VAE=1)

Opt-in acceleration (fused_vae) that swaps the execution path of the VAE without
changing its model definition, using comfy-kitchen kernels:
- GroupNorm + SiLU: one fused per-frame pass (group_norm_silu_pad3d), NDHWC output
- Causal 3D convolutions: fp16-accumulate kernel (fp16_conv3d), NDHWC in and out
- The tensors between them stay in NDHWC (channels_last_3d), including the frames
  a causal convolution carries over from the previous temporal slice and the
  pixel shuffle of the upsamplers

On GeForce GPUs the tensor core rate with fp32 accumulation is half of the fp16
accumulation rate, so the convolution kernel itself is ~2x faster. It only pays off
when its input is already NDHWC, hence the fused GroupNorm + SiLU in front of it.

Requirements: NVIDIA CUDA device and comfy-kitchen with its CUDA backend. The kernels
need fp16 inputs and weights, so the VAE runs in fp16 instead of the compute dtype.
comfy-kitchen is optional: when it is missing the standard path runs unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

_REQUIRED_KERNELS = ("fp16_conv3d", "group_norm_silu_pad3d")

# Lazily imported comfy_kitchen module (None until the fused path is requested)
_kitchen = None

# Single-frame (image) VAE calls run the standard kernels: the fused kernels' first-call cost is not
# amortized on one frame (measured: image encode 0.73 -> 13.1 s, decode 0.89 -> 3.5 s on an RTX 5090).
_SUSPENDED = [False]


def fused_vae_requested() -> bool:
    """On by default (WP-SEEDVR2 row 8: video VAE encode/decode 43-47 % faster, ground-truth metrics
    unchanged); SEEDVR2_FUSED_VAE=0 keeps the standard path. Falls back by itself when comfy-kitchen
    or its CUDA kernels are unavailable."""
    import os
    return os.environ.get("SEEDVR2_FUSED_VAE", "1").strip() != "0"


def get_vae_fused_path_unsupported_reason(device: torch.device) -> Optional[str]:
    """
    Check whether the VAE fused path can run on the given device.
    Imports comfy-kitchen on first use.

    Args:
        device: Inference device of the VAE

    Returns:
        None if supported, otherwise a human-readable reason
    """
    global _kitchen

    device = torch.device(device)
    if device.type != "cuda" or getattr(torch.version, "hip", None) is not None:
        return "requires an NVIDIA CUDA device"

    if _kitchen is None:
        try:
            import comfy_kitchen
        except Exception as e:
            return f"requires comfy-kitchen (pip install comfy-kitchen): {e}"

        # Without the CUDA backend comfy-kitchen falls back to plain torch ops,
        # which would be slower than the standard path
        try:
            backend = comfy_kitchen.list_backends().get("cuda", {})
        except Exception as e:
            return f"could not query comfy-kitchen backends: {e}"
        if not backend.get("available", False) or backend.get("disabled", False):
            return f"comfy-kitchen CUDA backend unavailable ({backend.get('unavailable_reason')})"
        missing = [name for name in _REQUIRED_KERNELS if name not in backend.get("capabilities", [])]
        if missing:
            return f"comfy-kitchen CUDA backend lacks {', '.join(missing)} (update comfy-kitchen)"
        _kitchen = comfy_kitchen

    return None


def _is_fusable_conv(conv: nn.Module) -> bool:
    """Shapes the fp16-accumulate kernel serves; other convolutions keep the standard path."""
    return (
        conv.groups == 1
        and tuple(conv.dilation) == (1, 1, 1)
        and conv.padding_mode == "zeros"
        and (conv.in_channels % 8 == 0 or conv.in_channels < 8)
        and conv.out_channels % 8 == 0
    )


def enable_vae_fused_path(vae: nn.Module) -> Tuple[int, int]:
    """
    Switch a materialized VAE to the fused execution path.

    Sets the `fused_path` flag on the causal convolutions the kernel serves, on the
    modules that run GroupNorm + SiLU in front of them and on the upsamplers, and stores
    the convolution weights in NDHWC. The model definition and state dict keys are unchanged.

    Args:
        vae: Materialized VAE with fp16 weights

    Returns:
        Tuple of (convolutions switched, norm + activation owners switched)
    """
    # Import here to avoid circular dependency
    from ..models.video_vae_v3.modules.attn_video_vae import Decoder3D, Encoder3D, ResnetBlock3D, Upsample3D
    from ..models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d

    convs = 0
    owners = 0
    for module in vae.modules():
        if isinstance(module, InflatedCausalConv3d):
            if _is_fusable_conv(module):
                module.weight.data = module.weight.data.contiguous(memory_format=torch.channels_last_3d)
                module.fused_path = True
                convs += 1
        elif isinstance(module, (ResnetBlock3D, Encoder3D, Decoder3D)):
            module.fused_path = True
            owners += 1
        elif isinstance(module, Upsample3D):
            # Pixel shuffle written out in NDHWC for the convolution that follows
            module.fused_path = True
    vae.fused_path = True
    return convs, owners


def is_fusable_norm_silu(norm_layer: nn.Module, act: nn.Module, x: torch.Tensor) -> bool:
    """Whether norm_layer followed by act on x can run as one fused GroupNorm + SiLU pass."""
    return (
        not _SUSPENDED[0] and
        type(norm_layer) is nn.GroupNorm
        and norm_layer.affine
        and isinstance(act, nn.SiLU)
        and x.ndim == 5
        and x.dtype == torch.float16
    )


def fused_group_norm_silu(norm_layer: nn.GroupNorm, x: torch.Tensor) -> torch.Tensor:
    """
    Per-frame GroupNorm followed by SiLU in one pass.

    Args:
        norm_layer: GroupNorm applied per frame
        x: [B, C, T, H, W] fp16 tensor

    Returns:
        [B, C, T, H, W] fp16 tensor in NDHWC (channels_last_3d)
    """
    return _kitchen.group_norm_silu_pad3d(
        x, norm_layer.weight, norm_layer.bias, norm_layer.num_groups, norm_layer.eps,
        pad=(0, 0, 0, 0, 0), silu=True
    )


def fp16_accum_conv3d(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor],
                      stride: Tuple[int, int, int], padding: Tuple[int, int, int]) -> torch.Tensor:
    """
    Zero-padded 3D convolution with fp16 accumulation.

    Args:
        input: [B, C, T, H, W] fp16 tensor (NDHWC avoids a layout conversion)
        weight: fp16 convolution weight
        bias: fp16 bias or None
        stride: Convolution stride (T, H, W)
        padding: Zero padding (T, H, W) applied on both sides

    Returns:
        fp16 convolution output in NDHWC (channels_last_3d)
    """
    if any(padding):
        input = F.pad(input, (padding[2], padding[2], padding[1], padding[1], padding[0], padding[0]))
    return _kitchen.fp16_conv3d(input, weight, bias, stride=stride)


def match_channels_last_3d(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Convert tensor to NDHWC if reference is NDHWC, so that concatenating them keeps NDHWC."""
    cl = torch.channels_last_3d
    if reference.is_contiguous(memory_format=cl) and not tensor.is_contiguous(memory_format=cl):
        return tensor.contiguous(memory_format=cl)
    return tensor
