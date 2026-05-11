from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


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
        self.backbone = nn.Sequential(*layers)

        lstm_input_dim = channels[-1]
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.backbone(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        return self.classifier(x)
