from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    # 统一读取 YAML 配置文件，方便后续训练脚本直接复用
    with Path(path).open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_json(path: str | Path, data: Any) -> None:
    # 统一保存 JSON 结果，例如标签映射、评估指标、实验记录等
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
