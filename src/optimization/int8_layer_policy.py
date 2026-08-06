"""
Model-agnostic INT8 ConvRot layer-selection policy.

Every public quantization recipe for diffusion transformers converges on the
same structural rule: quantize the repeated transformer blocks, keep the
conditioning / IO periphery (patch and text inputs, timestep embedders,
modulation projections, final head) in high precision. This module encodes
that rule generically so no per-model hand list is needed:

  Rule 1  only quantize Linears inside a repeated block stack (a name prefix
          followed by >= MIN_STACK_LEN distinct integer indices),
  Rule 2  inside a stack, skip segments that look like modulation / norm /
          embedding layers,
  Rule 3  skip layers with no INT8 upside (min(out, in) < min_dim) and layers
          whose in_features has no power-of-4 Hadamard group,
  Rule 4  (rescue) after quantization, promote the layers with the worst
          measured damage back to the original dtype under a byte budget,
  Rule 5  (optional, needs one calibration run) skip layers that only ever see
          tiny GEMMs (max token count <= 16): they are per-sample conditioning
          layers, INT8 buys nothing there and costs quality.

The policy works purely on (name, weight shape) pairs so the same code drives
state-dict converters, live-module converters and the benchmark tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

MIN_STACK_LEN = 4
DEFAULT_MIN_DIM = 256
SMALL_BATCH_M = 16
DEFAULT_RESCUE_BUDGET_MB = 256.0

# Segment substrings that mark conditioning / modulation layers *inside* a
# block stack (checked per dot-segment below the block index, lowercase).
# IO-style layers (patch/txt/vid projections, heads, final layers) live
# outside the block stacks and are already excluded by Rule 1 - listing their
# tokens here would false-positive on ordinary in-block write-back
# projections (e.g. SeedVR2 names its attention output Linears "proj_out").
_SENSITIVE_SEGMENT_SUBSTRINGS = (
    "norm",
    "mod",        # modulation, img_mod, txt_mod, adaln_modulation
    "ada",        # adaLN
    "emb",        # per-block embedders
    "time",       # per-block timestep projections
    "cond",       # condition embedders
)


@dataclass
class LayerDecision:
    quantize: bool
    reason: str
    group_size: Optional[int] = None


def _segments(name: str) -> Tuple[str, ...]:
    return tuple(seg for seg in name.split(".") if seg)


def detect_block_stacks(names: Iterable[str]) -> Set[str]:
    """
    Find repeated-block prefixes: a prefix P such that names of the form
    "P.<int>.<rest>" exist for >= MIN_STACK_LEN distinct integers.
    """
    indices: Dict[str, Set[int]] = {}
    for name in names:
        segs = _segments(name)
        for i, seg in enumerate(segs):
            if seg.isdigit() and i > 0:
                prefix = ".".join(segs[:i])
                indices.setdefault(prefix, set()).add(int(seg))
    return {prefix for prefix, idx in indices.items() if len(idx) >= MIN_STACK_LEN}


def _inside_stack(name: str, stacks: Set[str]) -> bool:
    segs = _segments(name)
    for i, seg in enumerate(segs):
        if seg.isdigit() and i > 0 and ".".join(segs[:i]) in stacks:
            return True
    return False


def _sensitive_segment(name: str, stacks: Set[str]) -> Optional[str]:
    """Check the segments *below* the block index for sensitive tokens."""
    segs = _segments(name)
    start = 0
    for i, seg in enumerate(segs):
        if seg.isdigit() and i > 0 and ".".join(segs[:i]) in stacks:
            start = i + 1
            break
    for seg in segs[start:]:
        low = seg.lower()
        for token in _SENSITIVE_SEGMENT_SUBSTRINGS:
            if token in low:
                return seg
    return None


def plan_int8_layers(
    shapes: Mapping[str, Tuple[int, int]],
    *,
    groupsizes: object = None,
    min_dim: int = DEFAULT_MIN_DIM,
    extra_skip_prefixes: Sequence[str] = (),
    extra_skip_suffixes: Sequence[str] = (),
    max_m_stats: Optional[Mapping[str, int]] = None,
    best_groupsize_fn=None,
) -> Dict[str, LayerDecision]:
    """
    Decide, for every 2-D Linear weight, whether it should be INT8 ConvRot
    quantized. `shapes` maps module name (no ".weight" suffix) to
    (out_features, in_features). `max_m_stats` optionally maps module name to
    the maximum activation token count observed during calibration (Rule 5).
    """
    if best_groupsize_fn is None:
        try:
            from shared.int8_convrot import best_int8_convrot_groupsize as best_groupsize_fn
        except ImportError:
            try:
                from src.models.int8_convrot import best_int8_convrot_groupsize as best_groupsize_fn
            except ImportError:
                from src.optimization.int8_convrot import best_int8_convrot_groupsize as best_groupsize_fn

    stacks = detect_block_stacks(shapes.keys())
    decisions: Dict[str, LayerDecision] = {}
    for name, (out_features, in_features) in shapes.items():
        if any(name.startswith(p) for p in extra_skip_prefixes):
            decisions[name] = LayerDecision(False, "skip-prefix")
            continue
        if any(name.endswith(s) for s in extra_skip_suffixes):
            decisions[name] = LayerDecision(False, "skip-suffix")
            continue
        if not _inside_stack(name, stacks):
            decisions[name] = LayerDecision(False, "outside-block-stack")
            continue
        sensitive = _sensitive_segment(name, stacks)
        if sensitive is not None:
            decisions[name] = LayerDecision(False, f"sensitive:{sensitive}")
            continue
        if min(int(out_features), int(in_features)) < int(min_dim):
            decisions[name] = LayerDecision(False, "tiny-gemm")
            continue
        group_size = best_groupsize_fn(int(in_features), groupsizes)
        if group_size is None:
            decisions[name] = LayerDecision(False, "no-hadamard-group")
            continue
        if max_m_stats is not None:
            max_m = max_m_stats.get(name)
            if max_m is not None and int(max_m) <= SMALL_BATCH_M:
                decisions[name] = LayerDecision(False, f"small-batch:m<={SMALL_BATCH_M}")
                continue
        decisions[name] = LayerDecision(True, "quantize", int(group_size))
    return decisions


def select_rescue_layers(
    damage: Mapping[str, Tuple[float, int]],
    budget_bytes: int,
) -> Set[str]:
    """
    Rule 4: given per-layer (relative_error, extra_bytes_if_kept_high_precision),
    greedily promote the highest-damage layers back to the original dtype until
    the byte budget is exhausted.
    """
    if budget_bytes <= 0:
        return set()
    ranked = sorted(damage.items(), key=lambda kv: (-kv[1][0], kv[1][1], kv[0]))
    rescued: Set[str] = set()
    used = 0
    for name, (rel_err, extra_bytes) in ranked:
        if rel_err <= 0:
            break
        if used + int(extra_bytes) <= budget_bytes:
            rescued.add(name)
            used += int(extra_bytes)
    return rescued


def summarize_plan(decisions: Mapping[str, LayerDecision]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for decision in decisions.values():
        key = decision.reason if not decision.quantize else "quantize"
        summary[key] = summary.get(key, 0) + 1
    return summary
