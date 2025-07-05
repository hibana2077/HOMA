# homa_models.py
"""
PyTorch implementation of the HOMA family of CNNs (Small, Base, Large).

Fix 2025‑07‑05‑b: Stabilize second‑order moment (divide by N) and normalise
random tensor to mitigate gradient explosion.
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
    """High‑Order Moment Aggregation (HOMA).

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

        proj_dims = {2: in_ch ** 2, 3: in_ch, 4: in_ch}
        self.proj = nn.ModuleDict({str(k): nn.Linear(proj_dims[k], rank, bias=False) for k in orders})
        self.fuse = nn.Sequential(nn.Linear(rank * len(orders), out_dim), nn.Mish(inplace=True))

        if 3 in orders:
            rand = torch.randn(1, in_ch, 1, 1)
            rand = rand / rand.norm(p=2)  # scale to unit norm to stabilise magnitude
            self.register_buffer("random_tensor", rand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,C,H,W)
        B, C, H, W = x.size()
        x_norm = x - x.mean(dim=(-1, -2), keepdim=True)

        feats: List[torch.Tensor] = []
        for order in self.orders:
            if order == 2:
                x_flat = x_norm.view(B, C, -1)            # (B,C,N)
                N = x_flat.size(-1)
                cov = torch.bmm(x_flat, x_flat.transpose(1, 2)) / N  # unbiased estimate
                feats.append(self.proj["2"](cov.reshape(B, -1)))

            elif order == 3:
                m3 = (x_norm * self.random_tensor).sum(dim=(-1, -2))
                feats.append(self.proj["3"](m3))

            elif order == 4:
                x_mean = x_norm.mean(dim=(-1, -2), keepdim=True)
                cumul4 = (x_norm ** 2).mean(dim=(-1, -2)) - 3 * (x_mean.squeeze(-1).squeeze(-1) ** 2)
                feats.append(self.proj["4"](cumul4))

        return self.fuse(torch.cat(feats, dim=-1))


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, groups: int = 1):
        p = k // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Mish(inplace=True),
        )


class HOMAChannelGate(nn.Module):
    def __init__(self, in_ch: int, out_dim=128, rank=32, orders=(2, 3, 4), reduction=16):
        super().__init__()
        self.homa = HOMABlock(in_ch, out_dim, rank, orders)
        self.fc = nn.Sequential(
            nn.Linear(out_dim, max(in_ch // reduction, 4)), nn.Mish(inplace=True), nn.Linear(max(in_ch // reduction, 4), in_ch), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.fc(self.homa(x)).view(x.size(0), -1, 1, 1)
        return x * s


class HOMAStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_blk: int, stride: int, gate_kws: dict):
        super().__init__()
        blks = [ConvBNAct(in_ch, out_ch, s=stride)] + [ConvBNAct(out_ch, out_ch) for _ in range(n_blk - 1)]
        self.blocks = nn.Sequential(*blks)
        self.gate = HOMAChannelGate(out_ch, **gate_kws)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(self.blocks(x))


class _HOMANet(nn.Module):
    def __init__(self, stage_cfg: Iterable[Tuple[int, int]], num_classes=1000, gate_kws=None):
        super().__init__()
        gate_kws = gate_kws or {}

        self.stem = nn.Sequential(ConvBNAct(3, 32, s=2), ConvBNAct(32, 32), ConvBNAct(32, 64))
        stages: List[nn.Module] = []
        in_ch = 64
        for out_ch, n in stage_cfg:
            stages.append(HOMAStage(in_ch, out_ch, n, stride=2, gate_kws=gate_kws))
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
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# factory helpers -------------------------------------------------------------


def HOMA_Small(num_classes=1000):
    """ About 15M parameters """
    return _HOMANet([(128, 2), (256, 2), (512, 2)], num_classes, gate_kws=dict(out_dim=128, rank=32))


def HOMA_Base(num_classes=1000):
    """ About 30M parameters """
    return _HOMANet([(128, 3), (256, 3), (512, 3)], num_classes, gate_kws=dict(out_dim=256, rank=64))


def HOMA_Large(num_classes=1000):
    """ About 60M parameters """
    return _HOMANet([(128, 3), (256, 4), (512, 6)], num_classes, gate_kws=dict(out_dim=512, rank=64))


__all__ = ["HOMABlock", "HOMA_Small", "HOMA_Base", "HOMA_Large"]


# -----------------------------------------------------------------------------
# Quick demo (also shows gradient clipping)
# -----------------------------------------------------------------------------

def _quick_demo():
    model = HOMA_Base(num_classes=10)
    x = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, 10, (4,))
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), 1e-3)

    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optim.step()
    print("loss", loss.item())


if __name__ == "__main__":
    _quick_demo()