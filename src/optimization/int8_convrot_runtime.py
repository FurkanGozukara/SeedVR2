"""
Persistent single-file INT8 ConvRot caching for the SeedVR2 DiT.

After the DiT state dict is loaded (bf16/fp16 safetensors, never GGUF), every
eligible nn.Linear weight is rotated by a block-diagonal Hadamard, quantized
to INT8 with per-row MSE-optimized + least-squares-refit scales, optionally
GPTQ-rounded / ARA-corrected when calibration artifacts exist, and the layer
forward is patched to run a fused INT8 GEMM (Triton / torch._int_mm).
Roughly halves DiT weight VRAM versus bf16 and is faster on NVIDIA Turing
(SM 7.5) or newer; older GPUs fall back to a transparent dequantized matmul.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

from src.optimization.int8_convert_engine import Int8ConversionEngine
from src.optimization.int8_convrot import int8_convrot_linear

# Belt-and-suspenders on top of the generic layer policy (which already keeps
# the patch-in/patch-out projections and embedding MLPs in high precision).
_SKIP_SUFFIXES = ("vid_in.proj", "vid_out.proj", "txt_in")
_SKIP_PREFIXES = ("emb_in.",)
INT8_CACHE_SUFFIX = "_int8_convrot.safetensors"
# v2: V6.1 pipeline (generic sensitive-layer policy, -128 range, LS scale
# refit, optional calibration features, ARA low-rank recovery, budgeted
# rescue) and portable validation for shipped caches.
INT8_CACHE_FORMAT = "seedvr2-dit-int8-convrot-v2"
INT8_GROUPSIZE_KEY_SUFFIX = ".int8_convrot_groupsize"
INT8_ARA_DOWN_SUFFIX = ".int8_ara_down"
INT8_ARA_UP_SUFFIX = ".int8_ara_up"


def int8_convrot_cache_path(source_checkpoint: str | Path) -> Path:
    source = Path(source_checkpoint)
    stem = source.name[: -len(".safetensors")] if source.name.lower().endswith(".safetensors") else source.stem
    models_dir = Path(__file__).resolve().parents[2] / "models"
    return models_dir / f"{stem}{INT8_CACHE_SUFFIX}"


def _source_metadata(source_checkpoint: str | Path) -> Dict[str, str]:
    source = Path(source_checkpoint)
    stat = source.stat()
    return {
        "source": str(source.resolve()),
        "source_size": str(stat.st_size),
        "source_mtime_ns": str(stat.st_mtime_ns),
    }


def has_valid_int8_convrot_cache(cache_path: str | Path, source_checkpoint: str | Path) -> bool:
    """
    Portable validation: format marker + group-size keys. When the source
    checkpoint exists its size must match the recorded one; a missing source
    is accepted so shipped caches load on machines that only downloaded the
    INT8 file.
    """
    cache = Path(cache_path)
    if not cache.is_file():
        return False
    try:
        with safe_open(str(cache), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            has_groups = any(key.endswith(INT8_GROUPSIZE_KEY_SUFFIX) for key in handle.keys())
        if (
            metadata.get("int8_convrot_format") != INT8_CACHE_FORMAT
            or metadata.get("seedvr2_int8_convrot") != "true"
            or not has_groups
        ):
            return False
        source = Path(source_checkpoint)
        if not source.is_file():
            return True
        recorded = metadata.get("source_size")
        return recorded is None or recorded == str(source.stat().st_size)
    except Exception:
        return False


def _bind_int8_forward(module: nn.Linear) -> None:
    def int8_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        group_size = getattr(self, "_int8_convrot_gs", None)
        if group_size is None:
            group_size = int(self.int8_convrot_groupsize.item())
            self._int8_convrot_gs = group_size
        return int8_convrot_linear(
            x,
            self.weight,
            self.scale_weight,
            group_size,
            self.bias,
            getattr(self, "int8_ara_down", None),
            getattr(self, "int8_ara_up", None),
        )

    module.forward = int8_forward.__get__(module, type(module))
    module._seedvr2_int8_convrot = True


def _patch_linear(module: nn.Linear, result) -> None:
    device = module.weight.device
    module.weight = nn.Parameter(result.q.to(device), requires_grad=False)
    module.register_buffer("scale_weight", result.scale.to(device), persistent=False)
    module._int8_convrot_gs = int(result.group_size)
    if result.ara_down is not None and result.ara_up is not None:
        module.register_buffer("int8_ara_down", result.ara_down.to(device), persistent=False)
        module.register_buffer("int8_ara_up", result.ara_up.to(device), persistent=False)
    if result.bias_delta is not None and module.bias is not None:
        with torch.no_grad():
            module.bias.data = (
                module.bias.data.float() - result.bias_delta.to(device)
            ).to(module.bias.dtype)
    _bind_int8_forward(module)


def prepare_model_for_int8_convrot_cache(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> int:
    """Prepare a meta-initialized DiT for assigning cached INT8 tensors."""
    quantized_layers = {
        key[: -len(INT8_GROUPSIZE_KEY_SUFFIX)]
        for key in state_dict
        if key.endswith(INT8_GROUPSIZE_KEY_SUFFIX)
    }
    modules = dict(model.named_modules())
    missing_layers = [
        name
        for name in quantized_layers
        if name not in modules or not isinstance(modules[name], nn.Linear)
    ]
    if missing_layers:
        raise RuntimeError(
            f"SeedVR2 INT8 cache architecture mismatch; unknown layers: {missing_layers[:8]}"
        )
    patched = 0
    for name, module in modules.items():
        if name not in quantized_layers or not isinstance(module, nn.Linear):
            continue
        scale = state_dict[f"{name}.scale_weight"]
        module.register_buffer(
            "scale_weight",
            torch.empty(tuple(scale.shape), dtype=torch.float32, device="meta"),
            persistent=True,
        )
        module.register_buffer(
            "int8_convrot_groupsize",
            torch.empty((), dtype=torch.int32, device="meta"),
            persistent=True,
        )
        ara_down = state_dict.get(f"{name}{INT8_ARA_DOWN_SUFFIX}")
        ara_up = state_dict.get(f"{name}{INT8_ARA_UP_SUFFIX}")
        if ara_down is not None and ara_up is not None:
            module.register_buffer(
                "int8_ara_down",
                torch.empty(tuple(ara_down.shape), dtype=ara_down.dtype, device="meta"),
                persistent=True,
            )
            module.register_buffer(
                "int8_ara_up",
                torch.empty(tuple(ara_up.shape), dtype=ara_up.dtype, device="meta"),
                persistent=True,
            )
        module.weight.requires_grad_(False)
        _bind_int8_forward(module)
        patched += 1
    if patched != len(quantized_layers):
        raise RuntimeError(
            f"SeedVR2 INT8 cache architecture mismatch: found {len(quantized_layers)} cached layers, "
            f"patched {patched}"
        )
    return patched


def save_int8_convrot_cache(
    model: nn.Module,
    cache_path: str | Path,
    source_checkpoint: str | Path,
) -> Path:
    """Save the complete DiT, but no VAE, as one atomic safetensors file."""
    output = Path(cache_path)
    state: Dict[str, torch.Tensor] = {
        key: tensor.detach().cpu().contiguous()
        for key, tensor in model.state_dict().items()
    }
    quantized = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or not getattr(module, "_seedvr2_int8_convrot", False):
            continue
        prefix = f"{name}." if name else ""
        state[prefix + "scale_weight"] = module.scale_weight.detach().float().cpu().contiguous()
        state[prefix + "int8_convrot_groupsize"] = torch.tensor(
            int(module._int8_convrot_gs), dtype=torch.int32
        )
        ara_down = getattr(module, "int8_ara_down", None)
        ara_up = getattr(module, "int8_ara_up", None)
        if ara_down is not None and ara_up is not None:
            state[name + INT8_ARA_DOWN_SUFFIX] = ara_down.detach().cpu().contiguous()
            state[name + INT8_ARA_UP_SUFFIX] = ara_up.detach().cpu().contiguous()
        quantized += 1
    if not quantized:
        raise RuntimeError("Refusing to save an empty SeedVR2 INT8 ConvRot cache")

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output.with_name(output.name + f".{os.getpid()}.tmp")
    metadata = {
        "format": "pt",
        "int8_convrot_format": INT8_CACHE_FORMAT,
        "seedvr2_int8_convrot": "true",
        "component": "dit",
        "quantized_linear_layers": str(quantized),
        "scale_dtype": "float32",
        "rotation": "hadamard-regular",
        "mse_clip": "true",
        **_source_metadata(source_checkpoint),
    }
    report = getattr(model, "_int8_conversion_report", None)
    if report:
        metadata["conversion_report"] = str(report)
    try:
        save_file(state, str(temp_path), metadata=metadata)
        os.replace(temp_path, output)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        del state
    return output


def apply_int8_convrot_to_model(
    model: nn.Module,
    debug=None,
    model_type: str = "DiT",
    source_checkpoint: str | Path | None = None,
) -> int:
    """Quantize eligible Linear layers in-place. Returns the number patched."""
    target = model.dit_model if hasattr(model, "dit_model") else model
    engine = Int8ConversionEngine(source_checkpoint, log_prefix="[SeedVR2 INT8]")
    candidates: Dict[str, nn.Linear] = {}
    shapes: Dict[str, tuple] = {}
    skipped = 0
    for name, module in target.named_modules():
        if not isinstance(module, nn.Linear) or getattr(module, "_seedvr2_int8_convrot", False):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or weight.device.type == "meta" or weight.dtype == torch.int8:
            skipped += 1
            continue
        if weight.ndim != 2:
            skipped += 1
            continue
        candidates[name] = module
        shapes[name] = (int(weight.shape[0]), int(weight.shape[1]))

    decisions = engine.plan(
        shapes,
        extra_skip_prefixes=_SKIP_PREFIXES,
        extra_skip_suffixes=_SKIP_SUFFIXES,
    )
    results = {}
    for name, module in candidates.items():
        decision = decisions.get(name)
        if decision is None or not decision.quantize:
            skipped += 1
            continue
        results[name] = engine.quantize_layer(
            name,
            module.weight,
            decision.group_size,
            has_bias=module.bias is not None,
            source_bytes_per_element=module.weight.element_size(),
        )
    rescued = engine.select_rescue(results)
    patched = 0
    for name, result in results.items():
        if name in rescued:
            skipped += 1
            continue
        _patch_linear(candidates[name], result)
        patched += 1
    target._int8_conversion_report = engine.report.metadata_json()
    if hasattr(model, "dit_model"):
        model._int8_conversion_report = target._int8_conversion_report
    print(engine.summary_line(results, rescued), flush=True)

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
