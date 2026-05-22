from __future__ import annotations

import argparse
import sys
from pathlib import Path

base = Path(__file__).resolve().parents[1]
if str(base) not in sys.path:
    sys.path.insert(0, str(base))

from src.data.slr_isolated import build_isolated_word_dataset
from src.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare isolated-word SLR processed data.')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file.')
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    config_path = Path(args.config) if args.config else base / 'src' / 'configs' / 'default.yaml'
    config = load_yaml(config_path)

    raw_dir = base / config['data']['raw_dir']
    processed_dir = base / config['data']['processed_dir']
    sequence_length = int(config['data']['sequence_length'])
    dictionary_file = base / config['data']['dictionary_file']

    if not raw_dir.exists():
        print(f'Raw dataset directory not found: {raw_dir}')
        return

    items = build_isolated_word_dataset(
        raw_root=raw_dir,
        processed_root=processed_dir,
        dictionary_file=dictionary_file if dictionary_file.exists() else None,
        sequence_length=sequence_length,
        split_ratio=(float(config['data']['train_ratio']), float(config['data']['val_ratio']), float(config['data']['test_ratio'])),
    )

    print(f'Loaded {len(items)} isolated-word records.')
    print(f'Processed metadata saved to: {processed_dir}')
    print(f'Feature files saved to: {processed_dir / "features"}')


if __name__ == '__main__':
    main()
