from __future__ import annotations

from pathlib import Path

from src.data.ce_csl import build_index, load_records, save_records
from src.utils.io import save_json


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    raw_dir = base / 'data' / 'raw' / 'CE-CSL'
    processed_dir = base / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f'Raw dataset directory not found: {raw_dir}')
        print('Please place the CE-CSL dataset under data/raw/CE-CSL/')
        return

    records = load_records(raw_dir)
    save_records(processed_dir / 'ce_csl_records.json', records)
    save_json(processed_dir / 'labels.json', build_index(records))

    missing_videos = [record.number for record in records if record.video_path is None]
    print(f'Loaded {len(records)} records from CE-CSL.')
    print(f'Missing video files: {len(missing_videos)}')
    if missing_videos:
        print('Example missing IDs:', ', '.join(missing_videos[:10]))
    print(f'Processed metadata saved to: {processed_dir}')


if __name__ == '__main__':
    main()
