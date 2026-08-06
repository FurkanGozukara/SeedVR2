"""
INT8 ConvRot calibration capture.

Run any normal BF16/FP16 upscale once with the environment variable
SECOURSES_INT8_CALIBRATE=1 and this module records, for every Linear the INT8
policy would quantize, statistics of the *rotated* activations feeding it:

  - mean       E[x_rot]      (bias correction)
  - energy     E[x_rot^2]    (energy-weighted scale search, weighted low-rank)
  - max_m      largest token count seen (Rule 5 small-batch skip)

and, when SECOURSES_INT8_HESSIAN=1 is also set, the full Hessian
H = sum(x_rot^T x_rot) per layer (GPTQ + Hessian-weighted ARA), limited to
layers with in_features <= SECOURSES_INT8_HESSIAN_MAX_IN (default 6144)
because H is [in, in] fp32.

Artifacts are written next to the source checkpoint:

  <checkpoint>.int8_calib.safetensors     (small - can be shipped)
  <checkpoint>.int8_hessian.safetensors   (large - local only, used once at
                                           conversion time and never needed
                                           by end users: the shipped cache
                                           has its effect baked in)

The converters pick these files up automatically when present; without them
every data-free improvement still applies.
"""

from __future__ import annotations

import atexit
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

try:
    from shared.int8_convrot import best_int8_convrot_groupsize, build_hadamard
    from shared.int8_layer_policy import plan_int8_layers
except ImportError:  # pragma: no cover - per-model tree copies
    try:
        from src.models.int8_convrot import best_int8_convrot_groupsize, build_hadamard
        from src.models.int8_layer_policy import plan_int8_layers
    except ImportError:
        from src.optimization.int8_convrot import best_int8_convrot_groupsize, build_hadamard
        from src.optimization.int8_layer_policy import plan_int8_layers

CALIB_SUFFIX = ".int8_calib.safetensors"
HESSIAN_SUFFIX = ".int8_hessian.safetensors"
CALIB_FORMAT = "secourses-int8-calib-v1"
HESSIAN_FORMAT = "secourses-int8-hessian-v1"
_HESSIAN_SAMPLE_ROWS = 16384

_ACTIVE_COLLECTORS = []


def calib_path_for(checkpoint_path: str | Path) -> Path:
    return Path(str(checkpoint_path) + CALIB_SUFFIX)


def hessian_path_for(checkpoint_path: str | Path) -> Path:
    return Path(str(checkpoint_path) + HESSIAN_SUFFIX)


def calibration_env_enabled() -> bool:
    return os.environ.get("SECOURSES_INT8_CALIBRATE", "").strip() not in ("", "0", "false", "off")


def hessian_env_enabled() -> bool:
    return os.environ.get("SECOURSES_INT8_HESSIAN", "").strip() not in ("", "0", "false", "off")


def hessian_max_in() -> int:
    try:
        return int(os.environ.get("SECOURSES_INT8_HESSIAN_MAX_IN", "6144"))
    except ValueError:
        return 6144


class Int8CalibrationCollector:
    """Forward-pre-hook based collector of rotated-activation statistics."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        targets: Dict[str, int],
        *,
        collect_hessian: bool = False,
        hessian_in_limit: int = 6144,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.targets = dict(targets)
        self.collect_hessian = collect_hessian
        self.hessian_in_limit = int(hessian_in_limit)
        self._handles = []
        self._state: Dict[str, Dict[str, object]] = {}
        self._finalized = False

    def attach(self, model: nn.Module) -> int:
        attached = 0
        for name, module in model.named_modules():
            group_size = self.targets.get(name)
            if group_size is None or not isinstance(module, nn.Linear):
                continue
            weight = getattr(module, "weight", None)
            if weight is None or weight.dtype == torch.int8:
                continue
            handle = module.register_forward_pre_hook(self._make_hook(name, int(group_size)))
            self._handles.append(handle)
            attached += 1
        if attached:
            print(
                f"[INT8 calib] recording activation statistics for {attached} Linear layers "
                f"(hessian={'on' if self.collect_hessian else 'off'})",
                flush=True,
            )
        return attached

    def _make_hook(self, name: str, group_size: int):
        def hook(module: nn.Linear, args):
            try:
                x = args[0]
                if not torch.is_tensor(x):
                    return
                with torch.no_grad():
                    in_features = int(module.in_features)
                    x2d = x.detach().reshape(-1, x.shape[-1])
                    if x2d.shape[-1] != in_features or x2d.numel() == 0:
                        return
                    x2d = x2d.float()
                    h = build_hadamard(group_size, device=x2d.device, dtype=torch.float32)
                    grouped = x2d.reshape(x2d.shape[0], in_features // group_size, group_size)
                    xr = torch.matmul(grouped, h).reshape(x2d.shape[0], in_features)
                    entry = self._state.get(name)
                    if entry is None:
                        entry = {
                            "count": 0,
                            "sum": torch.zeros(in_features, dtype=torch.float64, device=xr.device),
                            "sumsq": torch.zeros(in_features, dtype=torch.float64, device=xr.device),
                            "max_m": 0,
                            "group_size": group_size,
                            "hessian": None,
                        }
                        self._state[name] = entry
                    entry["count"] = int(entry["count"]) + int(xr.shape[0])
                    entry["sum"] += xr.sum(dim=0, dtype=torch.float64)
                    entry["sumsq"] += (xr * xr).sum(dim=0, dtype=torch.float64)
                    entry["max_m"] = max(int(entry["max_m"]), int(xr.shape[0]))
                    if self.collect_hessian and in_features <= self.hessian_in_limit:
                        rows = xr
                        if rows.shape[0] > _HESSIAN_SAMPLE_ROWS:
                            idx = torch.randperm(rows.shape[0], device=rows.device)[:_HESSIAN_SAMPLE_ROWS]
                            rows = rows.index_select(0, idx)
                        xtx = rows.T @ rows
                        if entry["hessian"] is None:
                            entry["hessian"] = xtx.to("cpu", dtype=torch.float32)
                            entry["hessian_rows"] = int(rows.shape[0])
                        else:
                            entry["hessian"] += xtx.to("cpu", dtype=torch.float32)
                            entry["hessian_rows"] = int(entry.get("hessian_rows", 0)) + int(rows.shape[0])
            except Exception as exc:  # never break inference because of calibration
                print(f"[INT8 calib] hook error on {name}: {exc}", flush=True)

        return hook

    def finalize(self) -> Optional[Path]:
        if self._finalized:
            return None
        self._finalized = True
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles.clear()
        if not self._state:
            print("[INT8 calib] no activation statistics were recorded", flush=True)
            return None

        from safetensors.torch import save_file

        calib_state: Dict[str, torch.Tensor] = {}
        hessian_state: Dict[str, torch.Tensor] = {}
        for name, entry in self._state.items():
            count = max(int(entry["count"]), 1)
            mean = (entry["sum"] / count).to(torch.float32).cpu()
            energy = (entry["sumsq"] / count).to(torch.float32).cpu()
            calib_state[f"{name}.mean"] = mean.contiguous()
            calib_state[f"{name}.energy"] = energy.contiguous()
            calib_state[f"{name}.max_m"] = torch.tensor(int(entry["max_m"]), dtype=torch.int64)
            calib_state[f"{name}.count"] = torch.tensor(count, dtype=torch.int64)
            calib_state[f"{name}.group_size"] = torch.tensor(int(entry["group_size"]), dtype=torch.int32)
            hessian = entry.get("hessian")
            if hessian is not None:
                rows = max(int(entry.get("hessian_rows", 1)), 1)
                hessian_state[f"{name}.hessian"] = (hessian / float(rows)).contiguous()

        calib_file = calib_path_for(self.checkpoint_path)
        save_file(calib_state, str(calib_file), metadata={"format": CALIB_FORMAT})
        print(f"[INT8 calib] wrote {calib_file} ({len(self._state)} layers)", flush=True)
        if hessian_state:
            hessian_file = hessian_path_for(self.checkpoint_path)
            save_file(hessian_state, str(hessian_file), metadata={"format": HESSIAN_FORMAT})
            print(f"[INT8 calib] wrote {hessian_file} ({len(hessian_state)} Hessians)", flush=True)
        self._state.clear()
        return calib_file


def maybe_start_calibration(model: nn.Module, checkpoint_path: str | Path) -> Optional[Int8CalibrationCollector]:
    """
    Env-gated entry point used by the model loaders: when the calibration env
    var is set and the model is a plain high-precision model, attach hooks for
    every layer the INT8 policy would quantize and register an atexit save.
    """
    if not calibration_env_enabled():
        return None
    shapes: Dict[str, tuple] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and getattr(module, "weight", None) is not None:
            weight = module.weight
            if weight.ndim == 2 and weight.dtype != torch.int8:
                shapes[name] = (int(weight.shape[0]), int(weight.shape[1]))
    decisions = plan_int8_layers(shapes, best_groupsize_fn=best_int8_convrot_groupsize)
    targets = {
        name: decision.group_size
        for name, decision in decisions.items()
        if decision.quantize and decision.group_size
    }
    if not targets:
        print("[INT8 calib] policy selected no layers on this model; nothing to record", flush=True)
        return None
    collector = Int8CalibrationCollector(
        checkpoint_path,
        targets,
        collect_hessian=hessian_env_enabled(),
        hessian_in_limit=hessian_max_in(),
    )
    if collector.attach(model) == 0:
        return None
    _ACTIVE_COLLECTORS.append(collector)
    atexit.register(collector.finalize)
    return collector


def load_int8_calibration(checkpoint_path: str | Path) -> Optional[Dict[str, Dict[str, object]]]:
    """Load per-layer calibration stats written by the collector, or None."""
    calib_file = calib_path_for(checkpoint_path)
    if not calib_file.is_file():
        return None
    try:
        from safetensors import safe_open

        stats: Dict[str, Dict[str, object]] = {}
        with safe_open(str(calib_file), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if metadata.get("format") != CALIB_FORMAT:
                return None
            for key in handle.keys():
                name, _, field = key.rpartition(".")
                if not name:
                    continue
                entry = stats.setdefault(name, {})
                tensor = handle.get_tensor(key)
                if field in ("max_m", "count", "group_size"):
                    entry[field] = int(tensor.item())
                else:
                    entry[field] = tensor
        print(f"[INT8 calib] using calibration statistics: {calib_file} ({len(stats)} layers)", flush=True)
        return stats
    except Exception as exc:
        print(f"[INT8 calib] could not read {calib_file}: {exc}", flush=True)
        return None


class HessianIndex:
    """Lazy reader for the (large) per-layer Hessian file."""

    def __init__(self, path: Path):
        self._path = path
        from safetensors import safe_open

        self._handle = safe_open(str(path), framework="pt", device="cpu")
        self._keys = set(self._handle.keys())

    def get(self, name: str) -> Optional[torch.Tensor]:
        key = f"{name}.hessian"
        if key not in self._keys:
            return None
        return self._handle.get_tensor(key)


def load_int8_hessians(checkpoint_path: str | Path) -> Optional[HessianIndex]:
    hessian_file = hessian_path_for(checkpoint_path)
    if not hessian_file.is_file():
        return None
    try:
        index = HessianIndex(hessian_file)
        print(f"[INT8 calib] using calibration Hessians: {hessian_file}", flush=True)
        return index
    except Exception as exc:
        print(f"[INT8 calib] could not read {hessian_file}: {exc}", flush=True)
        return None
