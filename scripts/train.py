from __future__ import annotations

from pathlib import Path

from src.utils.io import load_yaml
from src.utils.train_utils import set_seed


def main() -> None:
    # 读取统一配置文件，后续训练、验证、导出都尽量从配置驱动
    config = load_yaml(Path(__file__).resolve().parents[1] / 'src' / 'configs' / 'default.yaml')
    set_seed(config['project']['seed'])
    # 当前阶段只是训练骨架验证，先打印关键配置信息
    print('Training scaffold loaded successfully.')
    print(f"Project: {config['project']['name']}")
    print(f"Sequence length: {config['data']['sequence_length']}")
    print(f"Model: {config['model']['name']}")


if __name__ == '__main__':
    main()
