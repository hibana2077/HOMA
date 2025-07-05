# homa_models.py
"""
PyTorch implementation of the HOMA family of CNNs (Small, Base, Large).

Each variant consists of a lightweight convolutional backbone whose feature
maps are recalibrated by High‑Order Moment Attention (HOMA).  The extra branch
computes 2nd‑, 3rd‑ and 4th‑order statistics (see `HOMABlock`) and produces a
channel‑wise gate,類似於 SE‑Net 的重標定方式。

Author: ChatGPT (2025‑07‑05)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Iterable

# -----------------------------------------------------------------------------
# HOMA building blocks
# -----------------------------------------------------------------------------

class HOMABlock(nn.Module):
    """High‑Order Moment Aggregation (HOMA) block.

    Args:
        in_ch:   Number of input channels C.
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
        self.rank = rank

        # Projection input sizes per order
        proj_dims = {2: in_ch ** 2, 3: in_ch, 4: in_ch}

        self.proj = nn.ModuleDict(
            {str(k): nn.Linear(proj_dims[k], rank, bias=False) for k in orders}
        )

        self.fuse = nn.Sequential(
            nn.Linear(rank * len(orders), out_dim),
            nn.Mish(inplace=True),
        )

        if 3 in orders:
            self.register_buffer("random_tensor", torch.randn(1, in_ch, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,C,H,W)
        B, C, H, W = x.size()
        x_norm = x - x.mean(dim=(-1, -2), keepdim=True)

        feats: List[torch.Tensor] = []
        for order in self.orders:
            if order == 2:
                x_flat = x_norm.view(B, C, -1)
                cov = torch.bmm(x_flat, x_flat.transpose(1, 2)).view(B, -1)
                feats.append(self.proj["2"](cov))
            elif order == 3:
                m3 = (x_norm * self.random_tensor).sum(dim=(-1, -2))
                feats.append(self.proj["3"](m3))
            elif order == 4:
                x_mean = x_norm.mean(dim=(-1, -2), keepdim=True)
                cumul4 = (x_norm ** 2).mean(dim=(-1, -2)) - 3 * (x_mean.squeeze(-1).squeeze(-1) ** 2)
                feats.append(self.proj["4"](cumul4))

        h = torch.cat(feats, dim=-1)
        return self.fuse(h)


class ConvBNAct(nn.Sequential):
    """3×3 Conv → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, groups: int = 1):
        p = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, p, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Mish(inplace=True),
        )


class HOMAChannelGate(nn.Module):
    """Channel‑wise gating using HOMA features (類 SE‑Net)."""

    def __init__(
        self,
        in_ch: int,
        out_dim: int = 128,
        rank: int = 32,
        orders: Tuple[int, ...] = (2, 3, 4),
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.homa = HOMABlock(in_ch, out_dim, rank, orders)
        self.fc = nn.Sequential(
            nn.Linear(out_dim, max(in_ch // reduction, 4)),
            nn.Mish(inplace=True),
            nn.Linear(max(in_ch // reduction, 4), in_ch),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        s = self.homa(x)                 # (B, out_dim)
        s = self.fc(s).view(b, c, 1, 1)  # (B, C, 1, 1)
        return x * s


class HOMAStage(nn.Module):
    """Conv blocks followed by HOMA attention."""

    def __init__(self, in_ch: int, out_ch: int, num_blocks: int, stride: int, gate_kws: dict):
        super().__init__()
        blks: List[nn.Module] = [ConvBNAct(in_ch, out_ch, stride=stride)]
        blks.extend(ConvBNAct(out_ch, out_ch) for _ in range(num_blocks - 1))
        self.blocks = nn.Sequential(*blks)
        self.gate = HOMAChannelGate(out_ch, **gate_kws)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(self.blocks(x))


# -----------------------------------------------------------------------------
# Core network
# -----------------------------------------------------------------------------

class _HOMANet(nn.Module):
    def __init__(self, stage_cfg: Iterable[Tuple[int, int]], num_classes: int = 1000, gate_kws: dict | None = None):
        super().__init__()
        gate_kws = gate_kws or {}

        self.stem = nn.Sequential(
            ConvBNAct(3, 32, stride=2),
            ConvBNAct(32, 32),
            ConvBNAct(32, 64),
        )

        stages: List[nn.Module] = []
        in_ch = 64
        for out_ch, n_block in stage_cfg:
            stages.append(HOMAStage(in_ch, out_ch, n_block, stride=2, gate_kws=gate_kws))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# -----------------------------------------------------------------------------
# Public factories
# -----------------------------------------------------------------------------

def HOMA_Small(num_classes: int = 1000) -> _HOMANet:
    stage_cfg = [(128, 2), (256, 2), (512, 2)]
    gate_kws = dict(out_dim=128, rank=32, orders=(2, 3, 4))
    return _HOMANet(stage_cfg, num_classes, gate_kws)


def HOMA_Base(num_classes: int = 1000) -> _HOMANet:
    stage_cfg = [(128, 3), (256, 3), (512, 3)]
    gate_kws = dict(out_dim=256, rank=64, orders=(2, 3, 4))
    return _HOMANet(stage_cfg, num_classes, gate_kws)


def HOMA_Large(num_classes: int = 1000) -> _HOMANet:
    stage_cfg = [(128, 3), (256, 4), (512, 6)]
    gate_kws = dict(out_dim=512, rank=64, orders=(2, 3, 4))
    return _HOMANet(stage_cfg, num_classes, gate_kws)


__all__ = [
    "HOMABlock",
    "HOMA_Small",
    "HOMA_Base",
    "HOMA_Large",
]

# -----------------------------------------------------------------------------
# Quick usage example (unit test style)
# -----------------------------------------------------------------------------

def _quick_demo():
    """Run a forward pass to verify the model outputs."""

    model = HOMA_Base(num_classes=10)  # create model
    model.eval()
    x = torch.randn(4, 3, 224, 224)    # dummy batch
    with torch.no_grad():
        logits = model(x)
    print("logits.shape:", logits.shape)  # should be (4, 10)


if __name__ == "__main__":
    _quick_demo()