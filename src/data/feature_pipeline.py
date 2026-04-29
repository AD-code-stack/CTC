from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.utils.io import save_json


@dataclass(slots=True)
class CECSLItem:
    split: str
    number: str
    translator: str
    chinese_sentence: str
    gloss: str
    note: str
    video_path: str
    feature_path: str
    num_frames: int
    feature_dim: int


FEATURE_COLUMNS = [
    'Number',
    'Translator',
    'Chinese Sentences',
    'Gloss',
    'Note',
]


def load_split_rows(raw_root: str | Path) -> dict[str, list[dict[str, str]]]:
    raw_root = Path(raw_root)
    splits: dict[str, list[dict[str, str]]] = {'train': [], 'dev': [], 'test': []}

    for split in splits:
        csv_path = raw_root / 'label' / f'{split}.csv'
        if not csv_path.exists():
            continue
        with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                splits[split].append(row)
    return splits


def resolve_video_path(video_root: str | Path, split: str, translator: str, number: str) -> Path | None:
    video_root = Path(video_root)
    split_dir = video_root / split / translator
    if not split_dir.exists():
        return None
    candidates = sorted(split_dir.glob(f'{number}.*'))
    if candidates:
        return candidates[0]
    nested = sorted(split_dir.glob(f'**/{number}.*'))
    return nested[0] if nested else None


def _center_crop(frame: np.ndarray, target_ratio: float = 1.0) -> np.ndarray:
    height, width = frame.shape[:2]
    current_ratio = width / max(height, 1)
    if abs(current_ratio - target_ratio) < 1e-3:
        return frame
    if current_ratio > target_ratio:
        new_width = int(height * target_ratio)
        start = (width - new_width) // 2
        return frame[:, start:start + new_width]
    new_height = int(width / target_ratio)
    start = (height - new_height) // 2
    return frame[start:start + new_height, :]


def extract_basic_video_features(video_path: str | Path, max_frames: int = 32, feature_dim: int = 84) -> np.ndarray:
    """Extract lightweight per-frame features from a video.

    The feature vector includes:
    - resized grayscale appearance features
    - frame-to-frame motion magnitude summary

    This keeps the pipeline self-contained and avoids requiring a specific
    landmark model during preprocessing.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')

    frames: list[np.ndarray] = []
    prev_gray: np.ndarray | None = None
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = _center_crop(frame, target_ratio=1.0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (int(np.sqrt(feature_dim)), int(np.sqrt(feature_dim))), interpolation=cv2.INTER_AREA)
        flat = resized.astype(np.float32).reshape(-1) / 255.0
        if flat.size < feature_dim:
            flat = np.pad(flat, (0, feature_dim - flat.size))
        else:
            flat = flat[:feature_dim]

        motion = 0.0 if prev_gray is None else float(np.mean(cv2.absdiff(gray, prev_gray)) / 255.0)
        feature = np.concatenate([flat[:-1], np.array([motion], dtype=np.float32)])
        frames.append(feature.astype(np.float32))
        prev_gray = gray

    cap.release()

    if not frames:
        return np.zeros((max_frames, feature_dim), dtype=np.float32)

    seq = np.stack(frames, axis=0)
    if seq.shape[0] < max_frames:
        pad = np.zeros((max_frames - seq.shape[0], feature_dim), dtype=np.float32)
        seq = np.concatenate([seq, pad], axis=0)
    elif seq.shape[0] > max_frames:
        seq = seq[:max_frames]
    return seq.astype(np.float32)


def _safe_gloss(gloss: str, translator: str, number: str) -> str:
    text = gloss.strip()
    if text:
        return text
    return f'{translator}_{number}'


def build_processed_dataset(
    raw_root: str | Path,
    processed_root: str | Path,
    sequence_length: int = 32,
    feature_dim: int = 84,
) -> list[CECSLItem]:
    raw_root = Path(raw_root)
    processed_root = Path(processed_root)
    feature_root = processed_root / 'features'
    feature_root.mkdir(parents=True, exist_ok=True)

    split_rows = load_split_rows(raw_root)
    items: list[CECSLItem] = []
    missing_videos: list[str] = []

    for split, rows in split_rows.items():
        split_feature_dir = feature_root / split
        split_feature_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            number = (row.get('Number') or '').strip()
            translator = (row.get('Translator') or '').strip()
            chinese_sentence = (row.get('Chinese Sentences') or '').strip()
            gloss = (row.get('Gloss') or '').strip()
            note = (row.get('Note') or '').strip()
            video_path = resolve_video_path(raw_root / 'video', split, translator, number)
            if video_path is None:
                missing_videos.append(f'{split}:{translator}:{number}')
                continue

            feature_path = split_feature_dir / f'{number}.npy'
            features = extract_basic_video_features(video_path, max_frames=sequence_length, feature_dim=feature_dim)
            np.save(feature_path, features)

            items.append(
                CECSLItem(
                    split=split,
                    number=number,
                    translator=translator,
                    chinese_sentence=chinese_sentence,
                    gloss=_safe_gloss(gloss, translator, number),
                    note=note,
                    video_path=str(video_path),
                    feature_path=str(feature_path),
                    num_frames=int(features.shape[0]),
                    feature_dim=int(features.shape[1]),
                )
            )

    manifest = [asdict(item) for item in items]
    save_json(processed_root / 'ce_csl_manifest.json', manifest)
    save_json(
        processed_root / 'ce_csl_summary.json',
        {
            'total': len(items),
            'splits': {split: sum(1 for item in items if item.split == split) for split in ('train', 'dev', 'test')},
            'feature_dir': str(feature_root),
            'sequence_length': sequence_length,
            'feature_dim': feature_dim,
            'missing_videos': missing_videos,
        },
    )
    return items
