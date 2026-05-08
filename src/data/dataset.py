from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


@dataclass(slots=True)
class Sample:
    # 单标签分类样本
    feature_path: Path
    label: int
    meta: dict | None = None


@dataclass(slots=True)
class SequenceSample:
    # 连续识别样本，token_ids 为变长序列
    feature_path: Path
    token_ids: list[int]
    meta: dict | None = None


class SignLanguageDataset(Dataset):
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


class SignLanguageSequenceDataset(Dataset):
    def __init__(self, samples: list[SequenceSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        features = np.load(sample.feature_path)
        features_tensor = torch.from_numpy(features).float()
        token_tensor = torch.tensor(sample.token_ids, dtype=torch.long)
        token_length = len(sample.token_ids)
        return features_tensor, token_tensor, token_length, sample.meta or {}


def collate_sequence_batch(batch):
    features, token_tensors, token_lengths, metas = zip(*batch)
    feature_batch = torch.stack(features, dim=0)
    padded_tokens = pad_sequence(token_tensors, batch_first=True, padding_value=0)
    token_length_tensor = torch.tensor(token_lengths, dtype=torch.long)
    return feature_batch, padded_tokens, token_length_tensor, list(metas)
