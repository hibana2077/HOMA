""" HOMA‑TinyViT – TinyViT backbone enhanced with HOMA Channel Gating

This file integrates **High‑Order Moment Aggregation (HOMA)** blocks into the
original *TinyViT* architecture (Microsoft Cream repo) so that the network can
leverage higher‑order statistical representations in the early stages while
retaining TinyViT’s efficient windowed self‑attention.

Key integration points
~~~~~~~~~~~~~~~~~~~~~~
1. **HOMABlock / HOMAChannelGate** are copied verbatim from *homa_models.py* with
the 2025‑07‑05‑b stabilisation fix.
2. **HomaPatchEmbed** – wraps TinyViT’s `PatchEmbed` with a channel‑wise gate
   driven by HOMA features.
3. **HomaTinyVit** – sub‑classes the original `TinyVit` class, replacing
   `patch_embed` with `HomaPatchEmbed`. Everything else (stages, attention,
   classifier head, cfgs, etc.) is inherited unchanged.
4. **Factory helpers** – `_create_homa_tiny_vit` mirrors TinyViT’s factory so we
   can register variants like `homa_tiny_vit_11m_224` and call
   `timm.create_model()` seamlessly.

Author: (converted by ChatGPT, 2025‑07‑06)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, List, Union
from functools import partial

# -----------------------------------------------------------------------------
# Import TinyViT internals (assumes tiny_vit.py is in PYTHONPATH / same package)
# -----------------------------------------------------------------------------
from timm.models.tiny_vit import (
    TinyVit, PatchEmbed, _cfg, generate_default_cfgs, register_model,
    build_model_with_cfg, checkpoint_filter_fn, IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD, LayerNorm2d, NormMlpClassifierHead, resize_rel_pos_bias_table_levit,
)

from timm.layers import DropPath

################################################################################
# HOMA blocks – copied (lightly polished) from original implementation
################################################################################

class HOMABlock(nn.Module):
    """High‑Order Moment Aggregation (HOMA).

    Args:
        in_ch:   Number of input channels **C**.
        out_dim: Size of fused moment feature.
        rank:    Projection rank for each moment order.
        orders:  Tuple with the moment orders to use (2, 3, 4 supported).
    """

    def __init__(
        self,
        in_ch: int,
        out_dim: int = 2048,
        rank: int = 64,
        orders: Tuple[int, ...] = (2, 3, 4),
    ) -> None:
        super().__init__()
        self.orders = orders

        # Per‑order linear projections ------------------------------------------------
        proj_dims = {2: in_ch ** 2, 3: in_ch, 4: in_ch}
        self.proj = nn.ModuleDict({
            str(k): nn.Linear(proj_dims[k], rank, bias=False) for k in orders
        })

        # Early fuse + Mish -----------------------------------------------------------
        self.early_fuse_layer_norm = nn.LayerNorm(rank * len(orders), elementwise_affine=False)
        self.fuse = nn.Sequential(
            nn.Linear(rank * len(orders), out_dim),
            nn.Mish(inplace=True),
        )

        # Random tensor for 3rd‑order moments (unit norm) -----------------------------
        if 3 in orders:
            rand = torch.randn(1, in_ch, 1, 1)
            rand = rand / rand.norm(p=2)
            self.register_buffer("random_tensor", rand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        x_norm = x - x.mean(dim=(-1, -2), keepdim=True)  # zero‑mean per sample

        feats: List[torch.Tensor] = []
        for order in self.orders:
            if order == 2:
                x_flat = x_norm.view(B, C, -1)            # (B,C,N)
                N = x_flat.size(-1)
                cov = torch.bmm(x_flat, x_flat.transpose(1, 2)) / N
                feats.append(self.proj["2"](cov.reshape(B, -1)))
            elif order == 3:
                m3 = (x_norm * self.random_tensor).sum(dim=(-1, -2))
                feats.append(self.proj["3"](m3))
            elif order == 4:
                x_mean = x_norm.mean(dim=(-1, -2), keepdim=True)
                cumul4 = (x_norm ** 2).mean(dim=(-1, -2)) - 3 * (
                    x_mean.squeeze(-1).squeeze(-1) ** 2
                )
                feats.append(self.proj["4"](cumul4))

        fused = self.early_fuse_layer_norm(torch.cat(feats, dim=-1))
        return self.fuse(fused)


class HOMAChannelGate(nn.Module):
    """Squeeze‑and‑Excite style channel gate driven by HOMA features."""

    def __init__(self, in_ch: int, out_dim=128, rank=32, orders=(2, 3, 4), reduction=16):
        super().__init__()
        self.homa = HOMABlock(in_ch, out_dim, rank, orders)
        hidden = max(in_ch // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(out_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.fc(self.homa(x)).view(x.size(0), -1, 1, 1)
        return x * s

################################################################################
# HOMA‑aware PatchEmbed
################################################################################

class HomaPatchEmbed(PatchEmbed):
    """Patch embedding + channel gate.

    After TinyViT’s two stride‑2 convolutions, apply a HOMA‑driven channel gate
    to adaptively modulate patch features before they enter the transformer.
    """

    def __init__(self, in_chs: int, out_chs: int, act_layer, homa_gate_kws: Optional[Dict] = None):
        super().__init__(in_chs, out_chs, act_layer)
        homa_gate_kws = homa_gate_kws or {}
        self.gate = HOMAChannelGate(out_chs, **homa_gate_kws)

    def forward(self, x):
        x = super().forward(x)
        x = self.gate(x)
        return x

################################################################################
# TinyViT with HOMA patch embedding
################################################################################

class HomaTinyVit(TinyVit):
    """TinyViT variant that embeds HOMA channel gating in the stem."""

    def __init__(self, *args, homa_gate_kws: Optional[Dict] = None, **kwargs):
        # Build the base TinyViT first -------------------------------------------
        super().__init__(*args, **kwargs)

        # Replace patch embed with HOMA‑enhanced version --------------------------
        homa_gate_kws = homa_gate_kws or dict(out_dim=128, rank=32)
        self.patch_embed = HomaPatchEmbed(
            in_chs=3,  # TinyViT default in_chans
            out_chs=self.patch_embed.conv2.conv.out_channels,
            act_layer=nn.GELU,  # TinyViT default
            homa_gate_kws=homa_gate_kws,
        )
        # (feature_info[0] remains correct since channel count unchanged)

    # Nothing else needs to be overridden – forward passes are the same.

################################################################################
# Factory helpers & registry
################################################################################

def _create_homa_tiny_vit(variant: str, pretrained: bool = False, **kwargs):
    out_indices = kwargs.pop("out_indices", (0, 1, 2, 3))
    model = build_model_with_cfg(
        HomaTinyVit,
        variant,
        pretrained,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs,
    )
    return model


@register_model
def homa_tiny_vit_5m_224(pretrained: bool = False, **kwargs):
    model_kwargs = dict(
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=0.0,
    )
    model_kwargs.update(kwargs)
    return _create_homa_tiny_vit("homa_tiny_vit_5m_224", pretrained, **model_kwargs)


@register_model
def homa_tiny_vit_11m_224(pretrained: bool = False, **kwargs):
    model_kwargs = dict(
        embed_dims=[64, 128, 256, 448],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 8, 14],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=0.1,
    )
    model_kwargs.update(kwargs)
    return _create_homa_tiny_vit("homa_tiny_vit_11m_224", pretrained, **model_kwargs)


@register_model
def homa_tiny_vit_21m_224(pretrained: bool = False, **kwargs):
    model_kwargs = dict(
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=0.2,
    )
    model_kwargs.update(kwargs)
    return _create_homa_tiny_vit("homa_tiny_vit_21m_224", pretrained, **model_kwargs)

################################################################################
# Quick demo
################################################################################

def _quick_demo():
    m = HomaTinyVit(
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=0.0,
        num_classes=10,
    )
    x = torch.randn(2, 3, 224, 224)
    logits = m(x)
    print("logits", logits.shape)


if __name__ == "__main__":
    _quick_demo()
