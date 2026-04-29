from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class Sample:
    # 单条样本的特征文件路径，例如 npy 文件
    feature_path: Path
    # 类别标签编号，后续会和 label_map 对齐
    label: int
    # 可选元信息，比如 split、translator、原始编号等
    meta: dict | None = None


class SignLanguageDataset(Dataset):
    # 直接读取处理后的特征文件，供训练脚本使用
    def __init__(self, samples: list[Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        features = np.load(sample.feature_path)
        features_tensor = torch.from_numpy(features).float()
        label_tensor = torch.tensor(sample.label, dtype=torch.long)
        return features_tensor, label_tensor, sample.meta or {}
