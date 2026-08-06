"""
Shared INT8 ConvRot conversion engine.

Drives the full V6.1 quantization pipeline for one checkpoint:

  policy plan -> outlier clamp -> rotate -> (energy-weighted) MSE clip search
  -> least-squares scale refit -> GPTQ rounding (when a calibration Hessian
  exists) -> low-rank error-recovery fit (ARA) -> bias correction -> damage
  metrics -> budgeted rescue of the worst layers back to high precision.

The engine is model-agnostic; the per-model converters (SparkVSR state-dict
streaming, FlashVSR / SeedVR2 live modules) feed it (name, weight) pairs and
apply the returned tensors to their own cache format.

Environment knobs (all optional):
  SECOURSES_INT8_SCALE_GROUPS = "rot"  per-(row, rotation-group) weight scales
                                        (default: one scale per row)
  SECOURSES_INT8_LOWRANK_RANK = int    ARA rank, 0 disables (default 16)
  SECOURSES_INT8_RESCUE_MB    = float  Rule-4 rescue budget (default 256)
  SECOURSES_INT8_GPTQ         = 0      disable GPTQ even if Hessians exist
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Set, Tuple

import torch

try:
    from shared.int8_convrot import (
        DEFAULT_INT8_LOWRANK_RANK,
        best_int8_convrot_groupsize,
        build_hadamard,
        compute_bias_correction,
        fit_lowrank_residual,
        gptq_quantize_rotated,
        quantize_int8_rowwise,
        rotate_weight,
        weight_error_metrics,
    )
    from shared.int8_layer_policy import (
        DEFAULT_RESCUE_BUDGET_MB,
        LayerDecision,
        plan_int8_layers,
        select_rescue_layers,
        summarize_plan,
    )
    from shared.int8_calibration import load_int8_calibration, load_int8_hessians
except ImportError:  # pragma: no cover - per-model tree copies
    try:
        from src.models.int8_convrot import (
            DEFAULT_INT8_LOWRANK_RANK,
            best_int8_convrot_groupsize,
            build_hadamard,
            compute_bias_correction,
            fit_lowrank_residual,
            gptq_quantize_rotated,
            quantize_int8_rowwise,
            rotate_weight,
            weight_error_metrics,
        )
        from src.models.int8_layer_policy import (
            DEFAULT_RESCUE_BUDGET_MB,
            LayerDecision,
            plan_int8_layers,
            select_rescue_layers,
            summarize_plan,
        )
        from src.models.int8_calibration import load_int8_calibration, load_int8_hessians
    except ImportError:
        from src.optimization.int8_convrot import (
            DEFAULT_INT8_LOWRANK_RANK,
            best_int8_convrot_groupsize,
            build_hadamard,
            compute_bias_correction,
            fit_lowrank_residual,
            gptq_quantize_rotated,
            quantize_int8_rowwise,
            rotate_weight,
            weight_error_metrics,
        )
        from src.optimization.int8_layer_policy import (
            DEFAULT_RESCUE_BUDGET_MB,
            LayerDecision,
            plan_int8_layers,
            select_rescue_layers,
            summarize_plan,
        )
        from src.optimization.int8_calibration import load_int8_calibration, load_int8_hessians

DEFAULT_OUTLIER_CLAMP = 1000.0


def _env_scale_groups_mode() -> str:
    value = os.environ.get("SECOURSES_INT8_SCALE_GROUPS", "").strip().lower()
    return "rot" if value in ("rot", "group", "groups", "1", "on", "true") else "row"


def _env_lowrank_rank() -> int:
    try:
        return max(0, int(os.environ.get("SECOURSES_INT8_LOWRANK_RANK", str(DEFAULT_INT8_LOWRANK_RANK))))
    except ValueError:
        return DEFAULT_INT8_LOWRANK_RANK


def _env_rescue_budget_bytes() -> int:
    try:
        mb = float(os.environ.get("SECOURSES_INT8_RESCUE_MB", str(DEFAULT_RESCUE_BUDGET_MB)))
    except ValueError:
        mb = DEFAULT_RESCUE_BUDGET_MB
    return int(max(0.0, mb) * 1024 * 1024)


def _env_gptq_enabled() -> bool:
    return os.environ.get("SECOURSES_INT8_GPTQ", "1").strip() not in ("0", "false", "off")


@dataclass
class LayerQuantResult:
    name: str
    group_size: int
    q: torch.Tensor                    # int8 [out, in], rotated space, CPU
    scale: torch.Tensor                # fp32 [out, G], CPU
    ara_up: Optional[torch.Tensor]     # bf16 [out, r], CPU
    ara_down: Optional[torch.Tensor]   # bf16 [r, in], CPU
    bias_delta: Optional[torch.Tensor] # fp32 [out], CPU; subtract from bias
    rel_err: float
    sqnr_db: float
    extra_bytes: int                   # cost of keeping this layer high precision
    used_gptq: bool = False
    used_energy: bool = False


@dataclass
class ConversionReport:
    decisions: Dict[str, LayerDecision] = field(default_factory=dict)
    rescued: Set[str] = field(default_factory=set)
    features: Dict[str, object] = field(default_factory=dict)

    def metadata_json(self) -> str:
        payload = {
            "policy": summarize_plan(self.decisions),
            "rescued": sorted(self.rescued),
            "skipped": {
                name: decision.reason
                for name, decision in sorted(self.decisions.items())
                if not decision.quantize
            },
            "features": self.features,
        }
        return json.dumps(payload, separators=(",", ":"))


class Int8ConversionEngine:
    def __init__(
        self,
        source_checkpoint: str | os.PathLike | None,
        *,
        calc_device: str = "cpu",
        log_prefix: str = "[INT8 ConvRot]",
    ) -> None:
        if calc_device == "cpu" and torch.cuda.is_available():
            calc_device = "cuda"
        self.calc_device = calc_device
        self.log_prefix = log_prefix
        self.scale_groups_mode = _env_scale_groups_mode()
        self.lowrank_rank = _env_lowrank_rank()
        self.rescue_budget_bytes = _env_rescue_budget_bytes()
        self.stats = load_int8_calibration(source_checkpoint) if source_checkpoint else None
        self.hessians = None
        if source_checkpoint and _env_gptq_enabled():
            self.hessians = load_int8_hessians(source_checkpoint)
        self.report = ConversionReport()
        self.report.features = {
            "clip_search": "mse-0.55-1.0-80",
            "ls_refit": True,
            "qmin": -128,
            "scale_groups": self.scale_groups_mode,
            "lowrank_rank": self.lowrank_rank,
            "rescue_budget_mb": round(self.rescue_budget_bytes / (1024 * 1024), 1),
            "gptq": bool(self.hessians is not None),
            "calibrated": bool(self.stats is not None),
        }

    # ------------------------------------------------------------------ #
    def plan(
        self,
        shapes: Mapping[str, Tuple[int, int]],
        *,
        extra_skip_prefixes: Sequence[str] = (),
        extra_skip_suffixes: Sequence[str] = (),
    ) -> Dict[str, LayerDecision]:
        max_m_stats = None
        if self.stats:
            max_m_stats = {
                name: entry["max_m"]
                for name, entry in self.stats.items()
                if isinstance(entry.get("max_m"), int)
            }
        decisions = plan_int8_layers(
            shapes,
            extra_skip_prefixes=extra_skip_prefixes,
            extra_skip_suffixes=extra_skip_suffixes,
            max_m_stats=max_m_stats,
            best_groupsize_fn=best_int8_convrot_groupsize,
        )
        self.report.decisions = decisions
        summary = summarize_plan(decisions)
        print(f"{self.log_prefix} layer policy: {summary}", flush=True)
        return decisions

    # ------------------------------------------------------------------ #
    def _layer_stats(self, name: str, group_size: int, in_features: int):
        mean = energy = None
        if self.stats:
            entry = self.stats.get(name)
            if entry is not None and int(entry.get("group_size", group_size)) == group_size:
                mean_t = entry.get("mean")
                energy_t = entry.get("energy")
                if torch.is_tensor(mean_t) and mean_t.numel() == in_features:
                    mean = mean_t
                if torch.is_tensor(energy_t) and energy_t.numel() == in_features:
                    energy = energy_t
        return mean, energy

    def quantize_layer(
        self,
        name: str,
        weight: torch.Tensor,
        group_size: int,
        *,
        has_bias: bool = False,
        source_bytes_per_element: int = 2,
    ) -> LayerQuantResult:
        out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
        mean, energy = self._layer_stats(name, group_size, in_features)

        scale_groups = 1
        if self.scale_groups_mode == "rot" and group_size >= 64:
            scale_groups = in_features // group_size

        try:
            w = weight.detach().to(device=self.calc_device, dtype=torch.float32)
        except RuntimeError:
            w = weight.detach().to(device="cpu", dtype=torch.float32)
        bad = w.abs() > DEFAULT_OUTLIER_CLAMP
        if bool(bad.any()):
            print(
                f"{self.log_prefix} {name}: zeroing {int(bad.sum())} corrupted weight value(s) before rotation",
                flush=True,
            )
            w = w.masked_fill(bad, 0.0)
        h = build_hadamard(group_size, device=w.device, dtype=torch.float32)
        w_rot = rotate_weight(w, h, group_size)
        del w

        energy_dev = energy.to(w_rot.device) if energy is not None else None
        q, scale = quantize_int8_rowwise(
            w_rot,
            mse_clip=True,
            col_energy=energy_dev,
            scale_groups=scale_groups,
        )

        used_gptq = False
        if self.hessians is not None:
            hessian = self.hessians.get(name)
            if hessian is not None and hessian.shape[0] == in_features:
                try:
                    q_gptq = gptq_quantize_rotated(
                        w_rot, hessian.to(w_rot.device), scale.to(w_rot.device)
                    )
                except RuntimeError:
                    q_gptq = None
                if q_gptq is not None:
                    q = q_gptq
                    used_gptq = True

        ara_up = ara_down = None
        if self.lowrank_rank > 0:
            hessian = None
            if self.hessians is not None:
                hessian_t = self.hessians.get(name)
                if hessian_t is not None and hessian_t.shape[0] == in_features:
                    hessian = hessian_t.to(w_rot.device)
            pair = fit_lowrank_residual(
                w_rot,
                q.to(w_rot.device),
                scale.to(w_rot.device),
                rank=self.lowrank_rank,
                hessian=hessian,
                col_energy=energy_dev,
            )
            if pair is not None:
                ara_up_f, ara_down_f = pair
                ara_up = ara_up_f.to(torch.bfloat16).contiguous()
                ara_down = ara_down_f.to(torch.bfloat16).contiguous()

        bias_delta = None
        if has_bias and mean is not None:
            bias_delta = compute_bias_correction(
                w_rot,
                q.to(w_rot.device),
                scale.to(w_rot.device),
                mean,
                ara_up=ara_up,
                ara_down=ara_down,
            ).cpu()

        metrics = weight_error_metrics(
            w_rot,
            q.to(w_rot.device),
            scale.to(w_rot.device),
            ara_up=ara_up,
            ara_down=ara_down,
            col_energy=energy_dev,
        )
        del w_rot

        return LayerQuantResult(
            name=name,
            group_size=group_size,
            q=q.cpu().contiguous(),
            scale=scale.cpu().contiguous(),
            ara_up=ara_up.cpu() if ara_up is not None else None,
            ara_down=ara_down.cpu() if ara_down is not None else None,
            bias_delta=bias_delta,
            rel_err=float(metrics["rel_err"]),
            sqnr_db=float(metrics["sqnr_db"]),
            extra_bytes=out_features * in_features * max(1, source_bytes_per_element - 1),
            used_gptq=used_gptq,
            used_energy=energy is not None,
        )

    # ------------------------------------------------------------------ #
    def select_rescue(self, results: Mapping[str, LayerQuantResult]) -> Set[str]:
        damage = {
            name: (result.rel_err, result.extra_bytes)
            for name, result in results.items()
        }
        rescued = select_rescue_layers(damage, self.rescue_budget_bytes)
        self.report.rescued = rescued
        if rescued:
            worst = sorted(rescued, key=lambda n: -results[n].rel_err)[:8]
            details = ", ".join(f"{n} ({results[n].rel_err * 100:.2f}%)" for n in worst)
            print(
                f"{self.log_prefix} rescuing {len(rescued)} highest-damage layer(s) "
                f"back to source precision: {details}",
                flush=True,
            )
        return rescued

    def summary_line(self, results: Mapping[str, LayerQuantResult], rescued: Set[str]) -> str:
        kept = [r for name, r in results.items() if name not in rescued]
        if not kept:
            return f"{self.log_prefix} no layers quantized"
        err_energy = sum(r.rel_err ** 2 * r.extra_bytes for r in kept)
        sig = sum(r.extra_bytes for r in kept)
        mean_sqnr = sum(r.sqnr_db for r in kept) / len(kept)
        gptq_count = sum(1 for r in kept if r.used_gptq)
        ara_count = sum(1 for r in kept if r.ara_up is not None)
        return (
            f"{self.log_prefix} quantized {len(kept)} layers "
            f"(mean weight SQNR {mean_sqnr:.1f} dB, param-weighted rel err "
            f"{(err_energy / max(sig, 1)) ** 0.5 * 100:.3f}%, GPTQ on {gptq_count}, "
            f"ARA on {ara_count}, rescued {len(rescued)})"
        )
