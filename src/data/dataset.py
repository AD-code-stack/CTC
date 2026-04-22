from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Sample:
    # 单条样本的特征文件路径，例如 npy / pkl / pt
    feature_path: Path
    # 类别标签编号，后续会和 label_map 对齐
    label: int
    # 可选元信息，比如采集人、场景、时间段等
    meta: dict[str, Any] | None = None


class SignLanguageDataset:
    # 这里先做最轻量的 Dataset 占位类
    # 后续拿到真实数据后，可以扩展为：读取特征文件、做归一化、返回 tensor
    def __init__(self, samples: list[Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]
