from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.data.dataset import SequenceSample, SignLanguageSequenceDataset, collate_sequence_batch
from src.models.tcn_bilstm import TCNBiLSTM
from src.utils.io import load_json, load_yaml, save_json
from src.utils.train_utils import set_seed


def _load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f'Processed manifest not found: {manifest_path}')
    data = load_json(manifest_path)
    if not isinstance(data, list):
        raise ValueError('Manifest must be a list of records')
    return data


def _build_sequence_samples(manifest: list[dict[str, Any]]) -> tuple[list[SequenceSample], dict[str, int]]:
    token_names: list[str] = []
    for item in manifest:
        tokens = item.get('gloss_tokens') or []
        token_names.extend(str(token).strip() for token in tokens if str(token).strip())

    token_map = {token: idx + 1 for idx, token in enumerate(sorted(set(token_names)))}

    samples: list[SequenceSample] = []
    for item in manifest:
        feature_path = item.get('feature_path')
        tokens = [str(token).strip() for token in (item.get('gloss_tokens') or []) if str(token).strip()]
        if not feature_path or not tokens:
            continue
        token_ids = [token_map[token] for token in tokens if token in token_map]
        if not token_ids:
            continue
        samples.append(
            SequenceSample(
                feature_path=Path(feature_path),
                token_ids=token_ids,
                meta={
                    'split': item.get('split'),
                    'number': item.get('number'),
                    'translator': item.get('translator'),
                    'gloss': item.get('gloss'),
                    'gloss_tokens': tokens,
                },
            )
        )
    return samples, token_map


def main() -> None:
    config = load_yaml(Path(__file__).resolve().parents[1] / 'src' / 'configs' / 'default.yaml')
    set_seed(config['project']['seed'])

    base_dir = Path(__file__).resolve().parents[1]
    manifest_path = base_dir / config['data']['processed_records']
    manifest = _load_manifest(manifest_path)
    samples, token_map = _build_sequence_samples(manifest)

    print(f'Loaded processed manifest: {len(manifest)} records')
    print(f'Usable sequence samples: {len(samples)}')
    print(f'Gloss tokens: {len(token_map)}')
    if samples:
        print(f'First sample: {samples[0].feature_path} -> first tokens {samples[0].token_ids[:8]}')

    device_name = config['train']['device']
    device = torch.device(device_name if torch.cuda.is_available() or device_name == 'cpu' else 'cpu')
    print(f'Using device: {device}')

    model = TCNBiLSTM(
        input_dim=config['model'].get('input_dim', config['data']['feature_dim']),
        num_classes=max(len(token_map) + 1, config['model']['num_classes']),
        hidden_size=config['model']['hidden_size'],
        lstm_layers=config['model']['lstm_layers'],
        dropout=config['model']['dropout'],
    ).to(device)
    print(f'Model ready: {sum(p.numel() for p in model.parameters())} parameters')

    dataset = SignLanguageSequenceDataset(samples)
    print(f'Sequence dataset ready: {len(dataset)} samples')
    if len(dataset) < 3:
        raise ValueError('Need at least 3 sequence samples to continue')

    train_size = max(int(len(dataset) * 0.7), 1)
    val_size = max(int(len(dataset) * 0.2), 1)
    test_size = len(dataset) - train_size - val_size
    if test_size <= 0:
        test_size = 1
        train_size = max(train_size - 1, 1)
    if train_size + val_size + test_size != len(dataset):
        val_size = max(len(dataset) - train_size - test_size, 1)

    generator = torch.Generator().manual_seed(config['project']['seed'])
    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, val_size, len(dataset) - train_size - val_size], generator=generator)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config['train']['batch_size'], shuffle=True, num_workers=config['train']['num_workers'], collate_fn=collate_sequence_batch)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config['train']['batch_size'], shuffle=False, num_workers=config['train']['num_workers'], collate_fn=collate_sequence_batch)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config['train']['batch_size'], shuffle=False, num_workers=config['train']['num_workers'], collate_fn=collate_sequence_batch)

    work_dir = base_dir / config['project']['work_dir']
    work_dir.mkdir(parents=True, exist_ok=True)
    save_json(work_dir / 'token_map.json', token_map)
    save_json(
        work_dir / 'sequence_prep_summary.json',
        {
            'num_manifest_records': len(manifest),
            'num_samples': len(samples),
            'num_tokens': len(token_map),
            'input_shape': [config['data']['sequence_length'], config['data']['feature_dim']],
            'label_type': 'gloss_sequence',
            'split_sizes': {'train': train_size, 'val': val_size, 'test': test_size},
            'note': 'Sequence labels are prepared for future CTC/decoder training while keeping the current TCN-BiLSTM backbone.',
        },
    )
    print(f'Token map saved to: {work_dir / "token_map.json"}')
    print(f'Sequence prep summary saved to: {work_dir / "sequence_prep_summary.json"}')
    print(f'Train/val/test: {train_size}/{val_size}/{test_size}')
    print(f'Batch example shapes: features [B, 32, 84], token_ids [B, L]')


if __name__ == '__main__':
    main()
