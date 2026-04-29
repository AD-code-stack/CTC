from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.utils.io import save_json


@dataclass(slots=True)
class CECSLRecord:
    split: str
    number: str
    translator: str
    chinese_sentence: str
    gloss: str
    note: str
    video_path: str | None
    feature_path: str | None


@dataclass(slots=True)
class CECSLFeatureConfig:
    # 输出每个视频统一采样到的帧数
    sequence_length: int = 32
    # 每帧特征维度：468x2 face + 21x2 left hand + 21x2 right hand + 33x2 pose = 1048? 
    # 这里保存为关键点展开后的向量，维度由实际检测结果决定，预留给训练端自适应读取
    feature_dim: int = 0


def _normalize_text(value: str | None) -> str:
    return (value or '').strip()


def load_records(raw_root: str | Path) -> list[CECSLRecord]:
    raw_root = Path(raw_root)
    records: list[CECSLRecord] = []

    for split in ('train', 'dev', 'test'):
        csv_path = raw_root / 'label' / f'{split}.csv'
        video_split_dir = raw_root / 'video' / split
        if not csv_path.exists():
            continue

        with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                number = _normalize_text(row.get('Number'))
                translator = _normalize_text(row.get('Translator'))
                chinese_sentence = _normalize_text(row.get('Chinese Sentences'))
                gloss = _normalize_text(row.get('Gloss'))
                note = _normalize_text(row.get('Note'))
                video_path = _resolve_video_path(video_split_dir, translator, number)
                records.append(
                    CECSLRecord(
                        split=split,
                        number=number,
                        translator=translator,
                        chinese_sentence=chinese_sentence,
                        gloss=gloss,
                        note=note,
                        video_path=str(video_path) if video_path else None,
                        feature_path=None,
                    )
                )
    return records


def _resolve_video_path(split_dir: Path, translator: str, number: str) -> Path | None:
    if not split_dir.exists() or not translator:
        return None

    translator_dir = split_dir / translator
    if not translator_dir.exists():
        return None

    for pattern in (f'{number}.*', f'**/{number}.*'):
        candidates = sorted(translator_dir.glob(pattern))
        if candidates:
            return candidates[0]
    return None


def build_index(records: list[CECSLRecord]) -> dict[str, Any]:
    return {
        'num_records': len(records),
        'splits': {
            split: sum(1 for record in records if record.split == split)
            for split in ('train', 'dev', 'test')
        },
        'translators': sorted({record.translator for record in records if record.translator}),
    }


def save_records(path: str | Path, records: list[CECSLRecord]) -> None:
    save_json(path, [asdict(record) for record in records])


def to_dicts(records: list[CECSLRecord]) -> list[dict[str, Any]]:
    return [asdict(record) for record in records]
