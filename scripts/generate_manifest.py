"""Generate manifest from existing feature files.

This script creates the ce_csl_manifest.json file when raw data is unavailable
but processed feature files already exist.
"""
from __future__ import annotations

import sys
from pathlib import Path

base = Path(__file__).resolve().parents[1]
if str(base) not in sys.path:
    sys.path.insert(0, str(base))

from src.utils.io import save_json


def generate_manifest_from_features(processed_root: Path) -> list[dict]:
    """Generate manifest entries from existing .npy feature files."""
    manifest: list[dict] = []
    feature_root = processed_root / 'features'

    for split_dir in feature_root.iterdir():
        if not split_dir.is_dir():
            continue
        split = split_dir.name

        for feat_file in sorted(split_dir.glob('*.npy')):
            # Parse filename: e.g., "dev-00001.npy" -> split="dev", number="00001"
            # or "00001.npy" -> split from dir, number="00001"
            name = feat_file.stem
            if '-' in name:
                file_split, number = name.split('-', 1)
                # Verify it matches the directory name
                if file_split != split:
                    print(f'Warning: filename split "{file_split}" != dir "{split}", using dir')
                    split_used = split
                    number_used = name
                else:
                    split_used = split
                    number_used = number
            else:
                split_used = split
                number_used = name

            # Determine feature_path relative to processed_root
            feature_path = str(feat_file)

            # Use number as a placeholder gloss token so samples aren't skipped
            # In production, this should come from the raw label CSV
            placeholder_token = f'gloss_{number_used}'
            manifest.append({
                'split': split_used,
                'number': number_used,
                'translator': 'unknown',
                'chinese_sentence': '',
                'gloss': placeholder_token,
                'gloss_tokens': [placeholder_token],
                'note': 'placeholder - raw labels not available',
                'video_path': '',
                'feature_path': feature_path,
                'num_frames': 32,
                'feature_dim': 84,
            })

    return manifest


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    processed_root = base / 'data' / 'processed'
    manifest_path = processed_root / 'ce_csl_manifest.json'

    if not processed_root.exists():
        print(f'Error: Processed directory not found: {processed_root}')
        return

    manifest = generate_manifest_from_features(processed_root)

    if not manifest:
        print('No feature files found. Cannot generate manifest.')
        return

    # Group by split for summary
    splits = {}
    for item in manifest:
        split = item['split']
        splits[split] = splits.get(split, 0) + 1

    save_json(manifest_path, manifest)
    print(f'Generated manifest with {len(manifest)} entries:')
    for split, count in sorted(splits.items()):
        print(f'  {split}: {count}')
    print(f'Saved to: {manifest_path}')


if __name__ == '__main__':
    main()
