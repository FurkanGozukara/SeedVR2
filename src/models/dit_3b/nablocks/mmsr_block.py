# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

import os
from typing import Tuple
import torch
import torch.nn as nn

# from ..cache import Cache
from ....common.cache import Cache

from .attention.mmattn import NaSwinAttention
from ..mm import MMArg
from ..modulation import ada_layer_type
from ..normalization import norm_layer_type
from ..mm import MMArg, MMModule
from ..mlp import get_mlp
    

# fp16 compute only: residual streams kept in fp32 between blocks (all | txt | off)
_FP16_RESIDUAL = os.environ.get("SEEDVR2_FP16_RESIDUAL", "all").strip().lower()


def _finite(x: torch.Tensor) -> torch.Tensor:
    """fp16 branch outputs past the fp16 range become inf; keep them at the range limit so they
    cannot turn into NaN in the next norm (inf * 0)."""
    return torch.nan_to_num(x, nan=0.0, posinf=65504.0, neginf=-65504.0) if x.dtype == torch.float16 else x


class NaMMSRTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        vid_dim: int,
        txt_dim: int,
        emb_dim: int,
        heads: int,
        head_dim: int,
        expand_ratio: int,
        norm: norm_layer_type,
        norm_eps: float,
        ada: ada_layer_type,
        qk_bias: bool,
        qk_norm: norm_layer_type,
        mlp_type: str,
        shared_weights: bool,
        rope_type: str,
        rope_dim: int,
        is_last_layer: bool,
        attention_mode: str = 'sdpa',
        **kwargs,
    ):
        super().__init__()
        dim = MMArg(vid_dim, txt_dim)
        self.attn_norm = MMModule(norm, dim=dim, eps=norm_eps, elementwise_affine=False, shared_weights=shared_weights,)

        self.attn = NaSwinAttention(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            heads=heads,
            head_dim=head_dim,
            qk_bias=qk_bias,
            qk_norm=qk_norm,
            qk_norm_eps=norm_eps,
            rope_type=rope_type,
            rope_dim=rope_dim,
            shared_weights=shared_weights,
            attention_mode=attention_mode,
            window=kwargs.pop("window", None),
            window_method=kwargs.pop("window_method", None),
        )

        self.mlp_norm = MMModule(norm, dim=dim, eps=norm_eps, elementwise_affine=False, shared_weights=shared_weights, vid_only=is_last_layer)
        self.mlp = MMModule(
            get_mlp(mlp_type),
            dim=dim,
            expand_ratio=expand_ratio,
            shared_weights=shared_weights,
            vid_only=is_last_layer
        )
        self.ada = MMModule(ada, dim=dim, emb_dim=emb_dim, layers=["attn", "mlp"], shared_weights=shared_weights, vid_only=is_last_layer)
        self.is_last_layer = is_last_layer

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        emb: torch.FloatTensor,
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.LongTensor,
        torch.LongTensor,
    ]:
        hid_len = MMArg(
            cache("vid_len", lambda: vid_shape.prod(-1)),
            cache("txt_len", lambda: txt_shape.prod(-1)),
        )
        ada_kwargs = {
            "emb": emb,
            "hid_len": hid_len,
            "cache": cache,
            "branch_tag": MMArg("vid", "txt"),
        }

        if emb.dtype == torch.float16 and _FP16_RESIDUAL != "off":
            return self._forward_fp32_residual(vid, txt, vid_shape, txt_shape, cache, ada_kwargs)

        vid_attn, txt_attn = self.attn_norm(vid, txt)

        vid_attn, txt_attn = self.ada(vid_attn, txt_attn, layer="attn", mode="in", **ada_kwargs)
        vid_attn, txt_attn = self.attn(vid_attn, txt_attn, vid_shape, txt_shape, cache)
        vid_attn, txt_attn = self.ada(vid_attn, txt_attn, layer="attn", mode="out", **ada_kwargs)
        vid_attn, txt_attn = (vid_attn + vid), (txt_attn + txt)

        vid_mlp, txt_mlp = self.mlp_norm(vid_attn, txt_attn)
        # ADD BY NUMZ
        if vid_mlp.dtype != vid_attn.dtype:
            vid_mlp = vid_mlp.to(vid_attn.dtype)
        if txt_mlp.dtype != txt_attn.dtype:
            txt_mlp = txt_mlp.to(txt_attn.dtype)
        # END BY NUMZ
        vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, layer="mlp", mode="in", **ada_kwargs)
        vid_mlp, txt_mlp = self.mlp(vid_mlp, txt_mlp)
        vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, layer="mlp", mode="out", **ada_kwargs)
        vid_mlp, txt_mlp = (vid_mlp + vid_attn), (txt_mlp + txt_attn)

        return vid_mlp, txt_mlp, vid_shape, txt_shape

    def _forward_fp32_residual(self, vid, txt, vid_shape, txt_shape, cache, ada_kwargs):
        """fp16 compute (GPUs without bf16 GEMM): keep the residual stream(s) in fp32 between
        blocks; norms run on them in fp32 and everything after a norm runs in fp16 (see dit_7b)."""
        half = torch.float16
        keep_vid = _FP16_RESIDUAL == "all"
        vid = vid.float() if keep_vid else vid
        txt = txt.float()
        vid_attn, txt_attn = self.attn_norm(vid, txt)
        vid_attn, txt_attn = vid_attn.to(half), txt_attn.to(half)
        vid_attn, txt_attn = self.ada(vid_attn, txt_attn, layer="attn", mode="in", **ada_kwargs)
        vid_attn, txt_attn = self.attn(vid_attn, txt_attn, vid_shape, txt_shape, cache)
        vid_attn, txt_attn = self.ada(vid_attn, txt_attn, layer="attn", mode="out", **ada_kwargs)
        vid = vid + _finite(vid_attn).to(vid.dtype)
        txt = txt + _finite(txt_attn).float()
        del vid_attn, txt_attn

        vid_mlp, txt_mlp = self.mlp_norm(vid, txt)
        vid_mlp, txt_mlp = vid_mlp.to(half), txt_mlp.to(half)
        vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, layer="mlp", mode="in", **ada_kwargs)
        vid_mlp, txt_mlp = self.mlp(vid_mlp, txt_mlp)
        vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, layer="mlp", mode="out", **ada_kwargs)
        vid = vid + _finite(vid_mlp).to(vid.dtype)
        txt = txt + _finite(txt_mlp).float()
        return vid, txt, vid_shape, txt_shape
