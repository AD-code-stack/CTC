from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.data.ce_csl import CECSLRecord, build_index, load_records, save_records
from src.utils.io import load_yaml, save_json


def _stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    return int(digest[:8], 16)


def _detect_video_frames(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)
    return np.stack(frames, axis=0)


def _resample_indices(num_frames: int, target_length: int) -> np.ndarray:
    if num_frames <= 0:
        return np.zeros(target_length, dtype=np.int64)
    if num_frames == 1:
        return np.zeros(target_length, dtype=np.int64)
    return np.linspace(0, num_frames - 1, target_length).round().astype(np.int64)


def _extract_features(video_path: Path, sequence_length: int, feature_dim: int) -> np.ndarray:
    frames = _detect_video_frames(video_path)
    if frames.size == 0:
        return np.zeros((sequence_length, feature_dim), dtype=np.float32)

    indices = _resample_indices(len(frames), sequence_length)
    features = []
    seed = _stable_seed(video_path.as_posix())
    rng = np.random.default_rng(seed)

    for idx in indices:
        frame = frames[int(idx)]
        resized = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        histogram = cv2.calcHist([gray], [0], None, [feature_dim], [0, 256]).flatten().astype(np.float32)
        histogram /= max(histogram.sum(), 1.0)
        noise = rng.normal(0.0, 0.005, size=feature_dim).astype(np.float32)
        features.append(histogram + noise)

    return np.stack(features, axis=0)


def _copy_or_save_feature(feature_path: Path, feature_array: np.ndarray) -> None:
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(feature_path, feature_array.astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare CE-CSL processed feature data.')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file.')
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    config_path = Path(args.config) if args.config else base / 'src' / 'configs' / 'default.yaml'
    config = load_yaml(config_path)

    raw_dir = base / config['data']['raw_dir']
    processed_dir = base / config['data']['processed_dir']
    features_dir = processed_dir / 'features'
    features_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f'Raw dataset directory not found: {raw_dir}')
        print('Please place the CE-CSL dataset under data/raw/CE-CSL/')
        return

    records = load_records(raw_dir)
    processed_records = []
    skipped_records = []
    sequence_length = int(config['data']['sequence_length'])
    feature_dim = int(config['data']['feature_dim'])

    for record in tqdm(records, desc='Extracting features'):
        data = record.__dict__.copy()
        video_path = data.get('video_path')
        if not video_path:
            skipped_records.append(data['number'])
            processed_records.append(data)
            continue

        video_path_obj = Path(video_path)
        feature_path = features_dir / data['split'] / data['translator'] / f"{data['number']}.npy"
        feature_array = _extract_features(video_path_obj, sequence_length, feature_dim)
        _copy_or_save_feature(feature_path, feature_array)
        data['feature_path'] = str(feature_path)
        processed_records.append(data)

    save_records(processed_dir / 'ce_csl_records.json', [CECSLRecord(**item) for item in processed_records])
    save_json(processed_dir / 'labels.json', build_index(records))
    split_summary = {
        'splits': {split: sum(1 for item in processed_records if item['split'] == split) for split in ('train', 'dev', 'test')},
        'missing_videos': skipped_records,
        'feature_dir': str(features_dir),
    }
    save_json(processed_dir / 'split_summary.json', split_summary)

    print(f'Loaded {len(records)} records from CE-CSL.')
    print(f'Missing video files: {len(skipped_records)}')
    if skipped_records:
        print('Example missing IDs:', ', '.join(skipped_records[:10]))
    print(f'Processed metadata saved to: {processed_dir}')
    print(f'Feature files saved to: {features_dir}')


if __name__ == '__main__':
    main()
