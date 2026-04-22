from __future__ import annotations

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
    ) -> None:
        super().__init__()
        # TCN 部分：先用 1D 卷积在时间维上提取局部时序特征
        # 这里先做一个轻量版骨架，后续拿到数据后可以继续加深或改成多层残差块
        self.backbone = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # BiLSTM 部分：对卷积后的序列特征再做双向时序建模
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        # 分类头：将最后时刻的双向特征映射到类别空间
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入张量格式约定为 [batch, seq_len, feature_dim]
        # Conv1d 需要 [batch, channels, seq_len]，因此先交换维度
        x = x.transpose(1, 2)
        x = self.backbone(x)
        # 再转回 LSTM 所需的序列格式 [batch, seq_len, channels]
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        # 这里简单取最后一个时刻的输出做分类
        x = x[:, -1, :]
        return self.classifier(x)
