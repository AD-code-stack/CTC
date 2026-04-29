from __future__ import annotations

import csv
from dataclasses import dataclass
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


CSV_COLUMNS = [
    'Number',
    'Translator',
    'Chinese Sentences',
    'Gloss',
    'Note',
]


def _normalize_note(value: str | None) -> str:
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
                number = (row.get('Number') or '').strip()
                translator = (row.get('Translator') or '').strip()
                chinese_sentence = (row.get('Chinese Sentences') or '').strip()
                gloss = (row.get('Gloss') or '').strip()
                note = _normalize_note(row.get('Note'))
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

    candidates = sorted(translator_dir.glob(f'{number}.*'))
    if candidates:
        return candidates[0]

    nested_candidates = sorted(translator_dir.glob(f'**/{number}.*'))
    return nested_candidates[0] if nested_candidates else None


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
    payload = [record.__dict__ for record in records]
    save_json(path, payload)
