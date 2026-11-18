"""
② 単眼 3D 骨格補正モデル

入力 : (B, T, J, 4) = 正規化済み (x, y, z) + 信頼度 conf の時系列
出力 : 補正後の (x, y, z) と補正量 Δ

設計方針:
  - 2D (x, y) は信頼できるので「ベース」として残し、Δxy は tanh で小さく有界にする。
  - z は Δz を自由に出力し、時系列文脈（前後 ~1 秒）と物理制約ロスで深度を復元する。
  - 最終層をゼロ初期化して学習開始時は恒等写像（入力そのまま）から出発する。
  - バックボーンは軽量な非因果 dilated 1D-CNN (TCN) と Transformer エンコーダーを選択可能。
    TCN は全畳み込みなので任意長の動画をそのまま推論できる。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad,
                               padding_mode="replicate")
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv1d(channels, channels, 1)
        self.norm2 = nn.GroupNorm(1, channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        h = self.drop(self.act(self.norm1(self.conv1(x))))
        h = self.drop(self.act(self.norm2(self.conv2(h))))
        return x + h


class TCNBackbone(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_blocks: int, kernel_size: int, dropout: float):
        super().__init__()
        self.inp = nn.Conv1d(in_dim, hidden, 1)
        self.blocks = nn.ModuleList(
            ResidualTemporalBlock(hidden, kernel_size, 2 ** i, dropout) for i in range(num_blocks)
        )
        # 受容野 = 1 + (k-1) * Σ 2^i  (k=3, 4 blocks → 31 フレーム ≈ 1 秒 @30fps)
        self.receptive_field = 1 + (kernel_size - 1) * sum(2 ** i for i in range(num_blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, D) → (B, T, H)
        h = self.inp(x.transpose(1, 2))
        for blk in self.blocks:
            h = blk(h)
        return h.transpose(1, 2)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.shape[1]].unsqueeze(0)


class TransformerBackbone(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_layers: int, num_heads: int, dropout: float):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.pos = SinusoidalPositionalEncoding(hidden)
        layer = nn.TransformerEncoderLayer(hidden, num_heads, hidden * 2, dropout, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(self.pos(self.inp(x)), src_key_padding_mask=pad_mask)


class SkeletonRefiner(nn.Module):
    """単眼 3D 骨格補正モデル。

    Args:
        num_joints: 関節数 (MediaPipe = 33)
        backbone: "tcn" | "transformer"
        max_xy_offset: Δxy の上限（正規化単位 = 体幹長）。2D はほぼ維持する。
    """

    def __init__(self, num_joints: int = 33, in_feats: int = 4, hidden: int = 128, num_blocks: int = 4,
                 kernel_size: int = 3, num_heads: int = 4, dropout: float = 0.1, backbone: str = "tcn",
                 max_xy_offset: float = 0.1):
        super().__init__()
        self.num_joints = num_joints
        self.max_xy_offset = max_xy_offset
        in_dim = num_joints * in_feats
        if backbone == "tcn":
            self.backbone = TCNBackbone(in_dim, hidden, num_blocks, kernel_size, dropout)
        elif backbone == "transformer":
            self.backbone = TransformerBackbone(in_dim, hidden, num_blocks, num_heads, dropout)
        else:
            raise ValueError(f"unknown backbone: {backbone}")
        self.backbone_type = backbone
        self.head = nn.Linear(hidden, num_joints * 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """x: (B, T, J, 4) → {"coords": (B,T,J,3), "delta": (B,T,J,3)}"""
        B, T, J, F = x.shape
        feats = x.reshape(B, T, J * F)
        if self.backbone_type == "transformer":
            h = self.backbone(feats, pad_mask)
        else:
            h = self.backbone(feats)
        raw = self.head(h).reshape(B, T, J, 3)
        delta = torch.cat([self.max_xy_offset * torch.tanh(raw[..., :2] / self.max_xy_offset),
                           raw[..., 2:]], dim=-1)
        coords = x[..., :3] + delta
        return {"coords": coords, "delta": delta}

    @torch.no_grad()
    def refine_sequence(self, coords: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
        """1 動画 (T,J,3)+(T,J) を丸ごと補正して (T,J,3) を返す推論用ヘルパー。"""
        self.eval()
        x = torch.cat([coords, conf.unsqueeze(-1)], dim=-1).unsqueeze(0)
        return self.forward(x)["coords"][0]


if __name__ == "__main__":
    for bb in ("tcn", "transformer"):
        m = SkeletonRefiner(backbone=bb)
        x = torch.randn(2, 48, 33, 4)
        out = m(x)
        n = sum(p.numel() for p in m.parameters())
        print(bb, out["coords"].shape, f"params={n:,}", "identity at init:",
              torch.allclose(out["coords"], x[..., :3]))
