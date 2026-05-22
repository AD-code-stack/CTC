from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class Sample:
    feature_path: Path
    label: int
    meta: dict[str, Any] | None = None


@dataclass(slots=True)
class SequenceSample:
    feature_path: Path
    token_ids: list[int]
    meta: dict[str, Any] | None = None


@dataclass(slots=True)
class IsolatedWordSample:
    feature_path: Path
    label_id: int
    label_name: str
    sample_id: str
    meta: dict[str, Any] | None = None


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


class IsolatedWordDataset(Dataset):
    def __init__(self, samples: list[IsolatedWordSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        features = np.load(sample.feature_path)
        features_tensor = torch.from_numpy(features).float()
        label_tensor = torch.tensor(sample.label_id, dtype=torch.long)
        meta = sample.meta or {}
        meta = {**meta, 'sample_id': sample.sample_id, 'label_name': sample.label_name}
        return features_tensor, label_tensor, meta


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
        input_length = int(features.shape[0])
        target_length = len(sample.token_ids)
        return features_tensor, token_tensor, input_length, target_length, sample.meta or {}
