from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class _Backbone(nn.Module):
    def __init__(self, input_dim: int, tcn_channels: Sequence[int], dropout: float) -> None:
        super().__init__()
        channels = [input_dim, *tcn_channels]
        layers: list[nn.Module] = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            layers.extend(
                [
                    nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
        self.net = nn.Sequential(*layers)
        self.output_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class TCNBiLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_size: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.2,
        tcn_channels: Sequence[int] = (64, 128),
    ) -> None:
        super().__init__()
        self.backbone = _Backbone(input_dim, tcn_channels, dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.output_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


class DualBranchTCNBiLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_size: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.2,
        tcn_channels: Sequence[int] = (64, 128),
    ) -> None:
        super().__init__()
        if input_dim % 2 != 0:
            raise ValueError(f'DualBranchTCNBiLSTM expects even input_dim, got {input_dim}')
        branch_dim = input_dim // 2
        self.branch_dim = branch_dim
        self.color_backbone = _Backbone(branch_dim, tcn_channels, dropout)
        self.depth_backbone = _Backbone(branch_dim, tcn_channels, dropout)
        branch_out = self.color_backbone.output_dim
        self.color_pool = nn.AdaptiveAvgPool1d(1)
        self.depth_pool = nn.AdaptiveAvgPool1d(1)
        self.fusion = nn.Sequential(
            nn.Linear(branch_out * 2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        color_x, depth_x = torch.split(x, self.branch_dim, dim=2)
        color_x = self.color_backbone(color_x).transpose(1, 2)
        depth_x = self.depth_backbone(depth_x).transpose(1, 2)
        color_x = self.color_pool(color_x).squeeze(-1)
        depth_x = self.depth_pool(depth_x).squeeze(-1)
        x = torch.cat([color_x, depth_x], dim=1)
        x = self.fusion(x)
        return self.classifier(x)


class GatedFusionTCNBiLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_size: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.2,
        tcn_channels: Sequence[int] = (64, 128),
    ) -> None:
        super().__init__()
        if input_dim % 2 != 0:
            raise ValueError(f'GatedFusionTCNBiLSTM expects even input_dim, got {input_dim}')
        branch_dim = input_dim // 2
        self.branch_dim = branch_dim
        self.color_backbone = _Backbone(branch_dim, tcn_channels, dropout)
        self.depth_backbone = _Backbone(branch_dim, tcn_channels, dropout)
        branch_out = self.color_backbone.output_dim
        self.color_pool = nn.AdaptiveAvgPool1d(1)
        self.depth_pool = nn.AdaptiveAvgPool1d(1)
        fused_dim = branch_out
        self.gate = nn.Sequential(
            nn.Linear(branch_out * 2, fused_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        color_x, depth_x = torch.split(x, self.branch_dim, dim=2)
        color_x = self.color_pool(self.color_backbone(color_x).transpose(1, 2)).squeeze(-1)
        depth_x = self.depth_pool(self.depth_backbone(depth_x).transpose(1, 2)).squeeze(-1)
        gate = self.gate(torch.cat([color_x, depth_x], dim=1))
        fused = gate * color_x + (1.0 - gate) * depth_x
        return self.classifier(fused)


class AttentionFusionTCNBiLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_size: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.2,
        tcn_channels: Sequence[int] = (64, 128),
    ) -> None:
        super().__init__()
        if input_dim % 2 != 0:
            raise ValueError(f'AttentionFusionTCNBiLSTM expects even input_dim, got {input_dim}')
        branch_dim = input_dim // 2
        self.branch_dim = branch_dim
        self.color_backbone = _Backbone(branch_dim, tcn_channels, dropout)
        self.depth_backbone = _Backbone(branch_dim, tcn_channels, dropout)
        branch_out = self.color_backbone.output_dim
        self.color_pool = nn.AdaptiveAvgPool1d(1)
        self.depth_pool = nn.AdaptiveAvgPool1d(1)
        self.attention = nn.Sequential(
            nn.Linear(branch_out * 2, branch_out),
            nn.Tanh(),
            nn.Linear(branch_out, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(branch_out, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        color_x, depth_x = torch.split(x, self.branch_dim, dim=2)
        color_x = self.color_pool(self.color_backbone(color_x).transpose(1, 2)).squeeze(-1)
        depth_x = self.depth_pool(self.depth_backbone(depth_x).transpose(1, 2)).squeeze(-1)
        fused = torch.cat([color_x, depth_x], dim=1)
        attn_logits = self.attention(fused)
        attn_weights = torch.softmax(attn_logits, dim=1)
        fused = attn_weights[:, :1] * color_x + attn_weights[:, 1:] * depth_x
        return self.classifier(fused)
