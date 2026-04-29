from __future__ import annotations

from pathlib import Path

from src.data.dataset import FeatureDataset
from src.models.tcn_bilstm import TCNBiLSTM
from src.utils.io import load_yaml
from src.utils.train_utils import set_seed


def main() -> None:
    # 读取统一配置文件，后续训练、验证、导出都尽量从配置驱动
    config = load_yaml(Path(__file__).resolve().parents[1] / 'src' / 'configs' / 'default.yaml')
    set_seed(config['project']['seed'])

    manifest_path = Path(__file__).resolve().parents[1] / config['data']['processed_records']
    if manifest_path.exists():
        dataset = FeatureDataset(manifest_path)
        print(f'Loaded processed dataset: {len(dataset)} samples')
        if len(dataset) > 0:
            first = dataset[0]
            print(f"First sample split: {first['split']}, feature shape: {first['features'].shape}")
    else:
        print(f'Processed manifest not found: {manifest_path}')

    model = TCNBiLSTM(
        input_dim=config['model'].get('input_dim', config['data']['feature_dim']),
        num_classes=config['model']['num_classes'],
        hidden_size=config['model']['hidden_size'],
        lstm_layers=config['model']['lstm_layers'],
        dropout=config['model']['dropout'],
    )
    print('Training scaffold loaded successfully.')
    print(f"Project: {config['project']['name']}")
    print(f"Sequence length: {config['data']['sequence_length']}")
    print(f"Model: {config['model']['name']}")
    print(f'Model parameters ready: {sum(p.numel() for p in model.parameters())}')


if __name__ == '__main__':
    main()
