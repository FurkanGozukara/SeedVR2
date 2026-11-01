"""
Peak-VRAM helpers for the SeedVR2 DiT forward pass.

The DiT runs one-step inference over every latent token of a batch at once (about 106k
tokens for 49 frames at 1080p), so a few full-length temporaries set the phase's peak:
- the MLP hidden activation (tokens x 4*dim, twice: projection and activation),
- RoPE, which rotates q and k in float32 through several full-size temporaries,
- the per-window SDPA outputs collected in a list and concatenated.

These helpers process the long token axis in chunks and write into one output buffer.
Every operation involved works row by row (per token), so each token's result is computed
by the same kernels on the same values. Chunking applies only in eager inference
(no autograd, not while torch.compile traces), so compiled graphs are unchanged.

Environment overrides (A/B testing; the defaults are the measured choice):
    SEEDVR2_MLP_CHUNK_TOKENS   tokens per MLP chunk, 0 disables (default 8192)
    SEEDVR2_ROPE_CHUNK_TOKENS  tokens per RoPE chunk, 0 disables (default 4096)
    SEEDVR2_SDPA_BATCH_WINDOWS 1 = one SDPA call per run of equal-length windows,
                               0 = one call per window (default 0)
"""

from __future__ import annotations

import os
from typing import Callable

import torch
import torch.nn.functional as F


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


MLP_CHUNK_TOKENS = _env_int("SEEDVR2_MLP_CHUNK_TOKENS", 8192)
ROPE_CHUNK_TOKENS = _env_int("SEEDVR2_ROPE_CHUNK_TOKENS", 4096)
SDPA_BATCH_WINDOWS = _env_int("SEEDVR2_SDPA_BATCH_WINDOWS", 0) > 0


def _eager_inference() -> bool:
    if torch.is_grad_enabled():
        return False
    try:
        return not torch.compiler.is_compiling()
    except Exception:  # pragma: no cover - very old torch
        return True


def should_chunk(x: torch.Tensor, chunk: int) -> bool:
    return chunk > 0 and x.dim() >= 2 and x.shape[0] > chunk and _eager_inference()


def linear_fn(linear: torch.nn.Module, like: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """The layer as a callable for repeated chunk calls. GGUF layers dequantize their
    weight on every call, so dequantize once here and reuse it for every chunk."""
    getw = getattr(linear, "get_dequantized_weight_for_compute", None)
    if getw is not None and getattr(linear, "quantized_weight", None) is not None:
        weight = getw(like)
        bias = linear.bias
        return lambda t: F.linear(t, weight, bias)
    return linear


def chunked_rows(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, chunk: int) -> torch.Tensor:
    """fn(x) for a row-wise fn, computed `chunk` rows at a time into one output buffer."""
    n = x.shape[0]
    out = None
    for start in range(0, n, chunk):
        y = fn(x[start:start + chunk])
        if out is None:
            out = y.new_empty((n, *y.shape[1:]))
        out[start:start + y.shape[0]].copy_(y)
        del y
    return out


def rope_apply(apply_rotary_emb, freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Rotate t (L, heads, dim) with freqs (L, rot) exactly as the reference path
    `apply_rotary_emb(freqs, rearrange(t, "L h d -> h L d").float()).to(t.dtype)` does,
    in token chunks written back into t, so the float32 temporaries cover one chunk."""
    chunk = ROPE_CHUNK_TOKENS
    if should_chunk(t, chunk) and freqs.shape[0] == t.shape[0]:
        for start in range(0, t.shape[0], chunk):
            end = min(start + chunk, t.shape[0])
            part = t[start:end].permute(1, 0, 2)  # h L d
            rotated = apply_rotary_emb(freqs[start:end], part.float()).to(t.dtype)
            t[start:end] = rotated.permute(1, 0, 2)
            del part, rotated
        return t
    rotated = apply_rotary_emb(freqs, t.permute(1, 0, 2).float()).to(t.dtype)
    return rotated.permute(1, 0, 2)


def _equal_length_runs(q_lens, k_lens):
    """(first, last) window index spans over which (q_len, k_len) stays constant."""
    runs = []
    start = 0
    n = len(q_lens)
    for i in range(1, n + 1):
        if i == n or q_lens[i] != q_lens[start] or k_lens[i] != k_lens[start]:
            runs.append((start, i))
            start = i
    return runs


def varlen_sdpa(q, k, v, cu_seqlens_q, cu_seqlens_k, dropout_p=0.0, causal=False, deterministic=False):
    """
    Variable-length attention with PyTorch SDPA, one sequence (window) per call, written
    into one preallocated output instead of collecting every window output and
    concatenating them (the list plus the concatenation doubled the output's memory).
    With SEEDVR2_SDPA_BATCH_WINDOWS=1 (eager only), adjacent windows of equal length run
    as one batched SDPA call.
    """
    drop = dropout_p if not deterministic else 0.0
    if SDPA_BATCH_WINDOWS and _eager_inference():
        cu_q = cu_seqlens_q.long().cpu().tolist()
        cu_k = cu_seqlens_k.long().cpu().tolist()
        q_lens = [b - a for a, b in zip(cu_q, cu_q[1:])]
        k_lens = [b - a for a, b in zip(cu_k, cu_k[1:])]
        out = None
        for first, last in _equal_length_runs(q_lens, k_lens):
            n = last - first
            lq, lk = q_lens[first], k_lens[first]
            if lq == 0:
                continue
            qs, qe = cu_q[first], cu_q[last]
            ks, ke = cu_k[first], cu_k[last]
            q_i = q[qs:qe].reshape(n, lq, *q.shape[1:]).transpose(1, 2)
            k_i = k[ks:ke].reshape(n, lk, *k.shape[1:]).transpose(1, 2)
            v_i = v[ks:ke].reshape(n, lk, *v.shape[1:]).transpose(1, 2)
            o_i = F.scaled_dot_product_attention(q_i, k_i, v_i, dropout_p=drop, is_causal=causal)
            heads, head_dim = o_i.shape[1], o_i.shape[-1]  # (n, heads, lq, dim)
            o_i = o_i.transpose(1, 2).reshape(n * lq, heads, head_dim)
            if out is None:
                out = o_i.new_empty((q.shape[0], *o_i.shape[1:]))
            out[qs:qe].copy_(o_i)
            del q_i, k_i, v_i, o_i
        if out is None:
            out = q.new_empty((q.shape[0], q.shape[1], v.shape[-1]))
        return out

    # Split with tensor_split (no .item(): compile-friendly, as the reference path)
    q_splits = torch.tensor_split(q, cu_seqlens_q[1:-1].long().cpu(), dim=0)
    k_splits = torch.tensor_split(k, cu_seqlens_k[1:-1].long().cpu(), dim=0)
    v_splits = torch.tensor_split(v, cu_seqlens_k[1:-1].long().cpu(), dim=0)
    out = None
    offset = 0
    for q_i, k_i, v_i in zip(q_splits, k_splits, v_splits):
        o_i = F.scaled_dot_product_attention(
            q_i.permute(1, 0, 2).unsqueeze(0),
            k_i.permute(1, 0, 2).unsqueeze(0),
            v_i.permute(1, 0, 2).unsqueeze(0),
            dropout_p=drop,
            is_causal=causal,
        ).squeeze(0).permute(1, 0, 2)
        if out is None:
            out = o_i.new_empty((q.shape[0], *o_i.shape[1:]))
        length = o_i.shape[0]
        out[offset:offset + length].copy_(o_i)
        offset += length
        del o_i
    if out is None:
        out = q.new_empty((q.shape[0], q.shape[1], v.shape[-1]))
    return out
