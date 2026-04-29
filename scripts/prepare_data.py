from __future__ import annotations

import argparse
from pathlib import Path

from src.data.feature_pipeline import build_processed_dataset
from src.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare CE-CSL processed feature data.')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file.')
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    config_path = Path(args.config) if args.config else base / 'src' / 'configs' / 'default.yaml'
    config = load_yaml(config_path)

    raw_dir = base / config['data']['raw_dir']
    processed_dir = base / config['data']['processed_dir']
    sequence_length = int(config['data']['sequence_length'])
    feature_dim = int(config['data']['feature_dim'])

    if not raw_dir.exists():
        print(f'Raw dataset directory not found: {raw_dir}')
        print('Please place the CE-CSL dataset under data/raw/CE-CSL/')
        return

    items = build_processed_dataset(
        raw_root=raw_dir,
        processed_root=processed_dir,
        sequence_length=sequence_length,
        feature_dim=feature_dim,
    )

    print(f'Loaded {len(items)} records from CE-CSL.')
    print(f'Processed metadata saved to: {processed_dir}')
    print(f'Feature files saved to: {processed_dir / "features"}')


if __name__ == '__main__':
    main()
