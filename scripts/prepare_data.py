from __future__ import annotations

import argparse
import sys
from pathlib import Path

base = Path(__file__).resolve().parents[1]
if str(base) not in sys.path:
    sys.path.insert(0, str(base))

from src.data.slr_isolated import build_isolated_word_dataset
from src.utils.io import load_json, load_yaml


def _resolve_path(base: Path, path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _validate_processed_dataset(processed_dir: Path) -> None:
    manifest_dir = processed_dir / 'manifests'
    label_map_path = processed_dir / 'labels' / 'label_map.json'
    summary_path = processed_dir / 'stats' / 'dataset_summary.json'
    feature_dir = processed_dir / 'features'

    required_paths = [
        manifest_dir / 'train.json',
        manifest_dir / 'val.json',
        manifest_dir / 'test.json',
        label_map_path,
        summary_path,
        feature_dir,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise RuntimeError('Processed dataset validation failed. Missing: ' + ', '.join(missing))

    train_records = load_json(manifest_dir / 'train.json')
    val_records = load_json(manifest_dir / 'val.json')
    test_records = load_json(manifest_dir / 'test.json')
    label_map = load_json(label_map_path)
    summary = load_json(summary_path)

    all_records = train_records + val_records + test_records
    if not all_records:
        raise RuntimeError('Processed dataset validation failed. No records were generated.')

    for split_name, records in [('train', train_records), ('val', val_records), ('test', test_records)]:
        for record in records:
            feature_path = Path(record['feature_path'])
            if not feature_path.is_absolute():
                feature_path = Path(__file__).resolve().parents[1] / feature_path
            if not feature_path.exists():
                raise RuntimeError(f'Processed dataset validation failed. Missing feature file for {split_name}: {feature_path}')
            if int(record['label_id']) not in label_map.values():
                raise RuntimeError(f'Processed dataset validation failed. Invalid label_id in {split_name}: {record}')

    print('Processed dataset validation passed.')
    print(f"  Total records: {summary.get('total', len(all_records))}")
    print(f"  Classes: {summary.get('num_classes', len(label_map))}")
    print(f"  Modalities: {summary.get('modalities', ['color'])}")
    print(f"  Train/Val/Test: {len(train_records)}/{len(val_records)}/{len(test_records)}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare isolated-word SLR processed data.')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file.')
    parser.add_argument('--depth-dir', type=str, default=None, help='Optional depth raw directory path.')
    parser.add_argument('--max-class', type=int, default=None, help='Only keep classes with numeric directory name <= max-class.')
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    config_path = Path(args.config) if args.config else base / 'src' / 'configs' / 'default.yaml'
    config = load_yaml(config_path)

    raw_dir = base / config['data']['raw_dir']
    processed_dir = base / config['data']['processed_dir']
    sequence_length = int(config['data']['sequence_length'])
    dictionary_file = base / config['data']['dictionary_file']
    depth_dir_value = args.depth_dir or config['data'].get('depth_raw_dir')
    depth_dir = _resolve_path(base, depth_dir_value) if depth_dir_value else None

    if not raw_dir.exists():
        print(f'Raw dataset directory not found: {raw_dir}')
        return

    if not dictionary_file.exists():
        raise FileNotFoundError(f'Dictionary file not found: {dictionary_file}')

    if args.max_class is not None:
        class_dir = raw_dir / f'{int(args.max_class):03d}'
        if class_dir.exists():
            print(f'Limiting dataset to classes <= {int(args.max_class):03d}')
        else:
            print(f'Warning: class directory not found for max-class={args.max_class}, using raw_dir as-is.')

    items = build_isolated_word_dataset(
        raw_root=raw_dir,
        processed_root=processed_dir,
        dictionary_file=dictionary_file,
        sequence_length=sequence_length,
        split_ratio=(float(config['data']['train_ratio']), float(config['data']['val_ratio']), float(config['data']['test_ratio'])),
        seed=int(config['project'].get('seed', 42)),
        depth_root=depth_dir,
        max_class=args.max_class,
    )

    print(f'Loaded {len(items)} isolated-word records.')
    print(f'Processed metadata saved to: {processed_dir}')
    print(f'Feature files saved to: {processed_dir / "features"}')
    _validate_processed_dataset(processed_dir)


if __name__ == '__main__':
    main()
