from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

try:
    import mediapipe as mp
except ModuleNotFoundError as exc:  # pragma: no cover - import-time environment issue
    mp = None
    mp_solutions = None
    _MEDIAPIPE_IMPORT_ERROR = exc
else:
    _MEDIAPIPE_IMPORT_ERROR = None
    mp_solutions = getattr(mp, 'solutions', None)
    if mp_solutions is None:
        try:
            from mediapipe.python import solutions as mp_solutions  # type: ignore
        except Exception:
            mp_solutions = None

if mp_solutions is None:
    _MEDIAPIPE_SOLUTIONS_ERROR = (
        'MediaPipe is installed but the hands solutions API is unavailable. '
        'Please reinstall a compatible build, for example: pip install --force-reinstall mediapipe==0.10.14'
    )
else:
    _MEDIAPIPE_SOLUTIONS_ERROR = None
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


_HAND_LANDMARK_COUNT = 21
_HAND_FEATURE_DIM = _HAND_LANDMARK_COUNT * 2 * 2  # left hand + right hand, each with x/y


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


def _uniform_frame_indices(num_frames: int, target_length: int) -> np.ndarray:
    if num_frames <= 0:
        return np.zeros(target_length, dtype=np.int64)
    if num_frames == 1:
        return np.zeros(target_length, dtype=np.int64)
    return np.linspace(0, num_frames - 1, target_length).round().astype(np.int64)


def _hand_landmarks_to_vector(
    results: Any,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    left_hand = np.zeros((_HAND_LANDMARK_COUNT, 2), dtype=np.float32)
    right_hand = np.zeros((_HAND_LANDMARK_COUNT, 2), dtype=np.float32)

    if not results.multi_hand_landmarks:
        return np.concatenate([left_hand.reshape(-1), right_hand.reshape(-1)], axis=0)

    handedness_list = results.multi_handedness or []
    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        label = ''
        if idx < len(handedness_list) and handedness_list[idx].classification:
            label = handedness_list[idx].classification[0].label.lower()

        target = right_hand if label == 'right' else left_hand
        for lm_idx, landmark in enumerate(hand_landmarks.landmark[:_HAND_LANDMARK_COUNT]):
            target[lm_idx, 0] = float(landmark.x)
            target[lm_idx, 1] = float(landmark.y)

    return np.concatenate([left_hand.reshape(-1), right_hand.reshape(-1)], axis=0)


def extract_keypoint_features(video_path: str | Path, max_frames: int = 32) -> np.ndarray:
    """Extract MediaPipe hand keypoint sequences from a video.

    Each frame is represented by 42 keypoints (left hand + right hand),
    and each keypoint stores normalized x/y coordinates, so the final
    feature dimension is 84.
    """
    if mp_solutions is None:
        raise ImportError(_MEDIAPIPE_SOLUTIONS_ERROR or 'MediaPipe solutions API is unavailable')

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = _uniform_frame_indices(total_frames, max_frames)
    sampled_features: list[np.ndarray] = []

    with mp_solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
    ) as hands:
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok:
                sampled_features.append(np.zeros(_HAND_FEATURE_DIM, dtype=np.float32))
                continue

            frame = _center_crop(frame, target_ratio=1.0)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            results = hands.process(rgb)
            feature = _hand_landmarks_to_vector(results, width, height)
            sampled_features.append(feature.astype(np.float32))

    cap.release()

    if not sampled_features:
        return np.zeros((max_frames, _HAND_FEATURE_DIM), dtype=np.float32)

    seq = np.stack(sampled_features, axis=0)
    if seq.shape[0] < max_frames:
        pad = np.zeros((max_frames - seq.shape[0], _HAND_FEATURE_DIM), dtype=np.float32)
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
        for row in tqdm(rows, desc=f'Processing {split}', leave=False):
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
            features = extract_keypoint_features(video_path, max_frames=sequence_length)
            if features.shape[1] != feature_dim:
                raise ValueError(
                    f'Feature dimension mismatch for {video_path}: '
                    f'expected {feature_dim}, got {features.shape[1]}'
                )
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
            'feature_type': 'mediapipe_hand_keypoints',
            'missing_videos': missing_videos,
        },
    )
    return items
