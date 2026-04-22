from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    # 固定随机种子，保证实验尽量可复现
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
