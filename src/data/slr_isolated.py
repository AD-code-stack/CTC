from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.io import save_json


@dataclass(slots=True)
class IsolatedWordItem:
    sample_id: str
    feature_path: str
    label_id: int
    label_name: str
    source_path: str
    num_frames: int
    feature_dim: int
    split: str


DEFAULT_SEQUENCE_LENGTH = 32
DEFAULT_FEATURE_DIM = 50


def load_dictionary(dictionary_path: str | Path) -> dict[str, str]:
    dictionary_path = Path(dictionary_path)
    mapping: dict[str, str] = {}
    if not dictionary_path.exists():
        return mapping

    with dictionary_path.open('r', encoding='utf-8-sig', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                mapping[parts[0]] = ' '.join(parts[1:])
    return mapping


def _parse_numeric_line(line: str) -> list[float] | None:
    line = line.strip()
    if not line:
        return None
    parts = line.replace(',', ' ').split()
    values: list[float] = []
    for part in parts:
        try:
            values.append(float(part))
        except ValueError:
            return None
    return values if values else None


def load_skeleton_sequence(txt_path: str | Path) -> np.ndarray:
    txt_path = Path(txt_path)
    frames: list[list[float]] = []
    with txt_path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            values = _parse_numeric_line(line)
            if values is not None:
                frames.append(values)
    if not frames:
        return np.zeros((0, DEFAULT_FEATURE_DIM), dtype=np.float32)

    max_dim = max(len(row) for row in frames)
    arr = np.zeros((len(frames), max_dim), dtype=np.float32)
    for i, row in enumerate(frames):
        arr[i, : len(row)] = np.asarray(row, dtype=np.float32)
    return arr


def resample_sequence(seq: np.ndarray, target_length: int) -> np.ndarray:
    if seq.ndim != 2:
        raise ValueError(f'Expected 2D sequence, got shape {seq.shape}')
    if seq.shape[0] == 0:
        return np.zeros((target_length, seq.shape[1]), dtype=np.float32)
    if seq.shape[0] == target_length:
        return seq.astype(np.float32)
    indices = np.linspace(0, seq.shape[0] - 1, target_length).round().astype(np.int64)
    return seq[indices].astype(np.float32)


def scan_isolated_word_files(raw_root: str | Path) -> list[Path]:
    raw_root = Path(raw_root)
    candidates: list[Path] = []
    if raw_root.is_file() and raw_root.suffix.lower() == '.txt':
        return [raw_root]
    for ext in ('*.txt', '*.TXT'):
        candidates.extend(raw_root.rglob(ext))
    return sorted({p.resolve() for p in candidates if p.is_file()})


def _infer_label_from_parent(txt_path: Path, class_dir_to_label: dict[str, str]) -> str | None:
    parent_name = txt_path.parent.name
    return class_dir_to_label.get(parent_name)


def _infer_label_from_filename(txt_path: Path, dictionary: dict[str, str]) -> str | None:
    stem = txt_path.stem
    for key, label in dictionary.items():
        if key in stem:
            return label
    return None


def _parse_class_dir_mapping(class_dirs: list[Path], dictionary: dict[str, str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for class_dir in class_dirs:
        label = dictionary.get(class_dir.name)
        if label is None and class_dir.name.isdigit():
            label = dictionary.get(f'{int(class_dir.name):06d}')
        if label is None:
            try:
                label = dictionary.get(f'{int(class_dir.name):06d}')
            except ValueError:
                label = None
        if label is not None:
            mapping[class_dir.name] = label
    return mapping


def _discover_samples(raw_root: Path, dictionary: dict[str, str]) -> list[tuple[Path, str]]:
    samples: list[tuple[Path, str]] = []
    class_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir()])
    if class_dirs:
        class_dir_to_label = _parse_class_dir_mapping(class_dirs, dictionary)
        for class_dir in class_dirs:
            label_name = class_dir_to_label.get(class_dir.name)
            for txt_path in sorted(class_dir.rglob('*.txt')):
                if not txt_path.is_file():
                    continue
                if label_name is None:
                    label_name = _infer_label_from_filename(txt_path, dictionary)
                if label_name is None:
                    label_name = class_dir.name
                samples.append((txt_path, label_name))
    else:
        for txt_path in scan_isolated_word_files(raw_root):
            label_name = _infer_label_from_filename(txt_path, dictionary) or txt_path.parent.name
            samples.append((txt_path, label_name))
    return samples


def build_isolated_word_dataset(
    raw_root: str | Path,
    processed_root: str | Path,
    dictionary_file: str | Path | None = None,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> list[IsolatedWordItem]:
    raw_root = Path(raw_root)
    processed_root = Path(processed_root)
    feature_root = processed_root / 'features'
    manifest_root = processed_root / 'manifests'
    label_root = processed_root / 'labels'
    stats_root = processed_root / 'stats'
    feature_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)
    stats_root.mkdir(parents=True, exist_ok=True)

    dictionary = load_dictionary(dictionary_file) if dictionary_file else {}
    samples = _discover_samples(raw_root, dictionary)

    items: list[IsolatedWordItem] = []
    label_names: list[str] = []
    seen_ids: set[str] = set()

    for txt_path, label_name in samples:
        seq = load_skeleton_sequence(txt_path)
        if seq.size == 0:
            continue
        seq = resample_sequence(seq, sequence_length)
        sample_id = txt_path.parent.name + '_' + txt_path.stem
        if sample_id in seen_ids:
            sample_id = f'{sample_id}_{len(seen_ids)}'
        seen_ids.add(sample_id)
        label_names.append(label_name)
        items.append(
            IsolatedWordItem(
                sample_id=sample_id,
                feature_path=f'data/processed/features/{sample_id}.npy',
                label_id=-1,
                label_name=label_name,
                source_path=str(txt_path),
                num_frames=int(seq.shape[0]),
                feature_dim=int(seq.shape[1]),
                split='unknown',
            )
        )
        np.save(feature_root / f'{sample_id}.npy', seq)

    unique_labels = sorted(set(label_names))
    label_map = {name: i for i, name in enumerate(unique_labels)}
    for item in items:
        item.label_id = label_map[item.label_name]

    rng = np.random.default_rng(seed)
    items_by_label: dict[str, list[IsolatedWordItem]] = {}
    for item in items:
        items_by_label.setdefault(item.label_name, []).append(item)

    shuffled_items: list[IsolatedWordItem] = []
    for label_name in sorted(items_by_label):
        label_items = items_by_label[label_name]
        indices = np.arange(len(label_items))
        rng.shuffle(indices)
        shuffled_items.extend([label_items[i] for i in indices])

    split_buckets: dict[str, list[IsolatedWordItem]] = {'train': [], 'val': [], 'test': []}
    for label_name in sorted(items_by_label):
        label_items = items_by_label[label_name]
        indices = np.arange(len(label_items))
        rng.shuffle(indices)
        label_items = [label_items[i] for i in indices]

        n = len(label_items)
        train_end = int(n * split_ratio[0])
        val_end = train_end + int(n * split_ratio[1])
        for i, item in enumerate(label_items):
            if i < train_end:
                split_buckets['train'].append(item)
            elif i < val_end:
                split_buckets['val'].append(item)
            else:
                split_buckets['test'].append(item)

    for split_name, split_items in split_buckets.items():
        for item in split_items:
            item.split = split_name
            item.feature_path = f'data/processed/features/{item.sample_id}.npy'

    for split_name, split_items in split_buckets.items():
        for item in split_items:
            item.split = split_name
            item.feature_path = f'data/processed/features/{item.sample_id}.npy'

    manifest = [asdict(item) for item in split_buckets['train'] + split_buckets['val'] + split_buckets['test']]
    save_json(processed_root / 'isolated_word_manifest.json', manifest)
    save_json(label_root / 'label_map.json', label_map)
    save_json(
        stats_root / 'dataset_summary.json',
        {
            'total': len(items),
            'num_classes': len(label_map),
            'sequence_length': sequence_length,
            'feature_dim': int(items[0].feature_dim) if items else 0,
            'dictionary_file': str(dictionary_file) if dictionary_file else None,
            'raw_root': str(raw_root),
            'split_ratio': {'train': split_ratio[0], 'val': split_ratio[1], 'test': split_ratio[2]},
            'seed': seed,
        },
    )

    for split in ('train', 'val', 'test'):
        split_items = [asdict(item) for item in split_buckets[split]]
        save_json(manifest_root / f'{split}.json', split_items)

    return split_buckets['train'] + split_buckets['val'] + split_buckets['test']
