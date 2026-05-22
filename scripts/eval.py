from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import IsolatedWordDataset, Sample
from src.models.tcn_bilstm import TCNBiLSTM
from src.utils.io import load_json, load_yaml, save_json


def _resolve_path(base: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _load_samples(base: Path, manifest_path: Path):
    records = load_json(manifest_path)
    samples: list[Sample] = []
    for record in records:
        samples.append(
            Sample(
                feature_path=_resolve_path(base, record['feature_path']),
                label=int(record['label_id']),
                meta={
                    'sample_id': record.get('sample_id'),
                    'label_name': record.get('label_name'),
                    'source_path': record.get('source_path'),
                    'split': record.get('split'),
                },
            )
        )
    return samples


def _metrics(preds: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor):
    total = labels.numel()
    acc = float((preds == labels).float().mean().item()) if total else 0.0
    top5 = 0.0
    if total:
        k = min(5, logits.shape[1])
        top5 = float(logits.topk(k, dim=1).indices.eq(labels.unsqueeze(1)).any(dim=1).float().mean().item())
    num_classes = int(logits.shape[1]) if logits.numel() else 0
    macro_f1 = 0.0
    if total and num_classes:
        f1s = []
        for c in range(num_classes):
            tp = ((preds == c) & (labels == c)).sum().item()
            fp = ((preds == c) & (labels != c)).sum().item()
            fn = ((preds != c) & (labels == c)).sum().item()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1s.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
        macro_f1 = float(sum(f1s) / len(f1s))
    return {'accuracy': acc, 'top5_accuracy': top5, 'macro_f1': macro_f1}


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate isolated-word SLR classifier.')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file.')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint. Default best_model.pt.')
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    config_path = Path(args.config) if args.config else base / 'src' / 'configs' / 'default.yaml'
    config = load_yaml(config_path)
    processed_dir = _resolve_path(base, config['data']['processed_dir'])
    work_dir = _resolve_path(base, config['project']['work_dir']) / 'isolated_word'
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else work_dir / 'best_model.pt'

    test_manifest = processed_dir / 'manifests' / 'test.json'
    samples = _load_samples(base, test_manifest)
    if not samples:
        raise RuntimeError('No test samples found.')

    label_map = load_json(processed_dir / 'labels' / 'label_map.json')
    num_classes = len(label_map)

    loader = DataLoader(
        IsolatedWordDataset(samples),
        batch_size=int(config['train']['batch_size']),
        shuffle=False,
        num_workers=int(config['train']['num_workers']),
    )

    device = torch.device(config['train']['device'] if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TCNBiLSTM(
        input_dim=int(config['model']['input_dim']),
        num_classes=num_classes,
        hidden_size=int(config['model']['hidden_size']),
        lstm_layers=int(config['model']['lstm_layers']),
        dropout=float(config['model']['dropout']),
        tcn_channels=config['model']['tcn_channels'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    all_preds = []
    all_labels = []
    all_logits = []
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for features, labels, _meta in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            total_count += int(labels.size(0))
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())
            all_logits.append(logits.cpu())

    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)
    logits = torch.cat(all_logits, dim=0)
    metrics = {'loss': total_loss / total_count if total_count else 0.0, **_metrics(preds, labels, logits)}
    save_json(work_dir / 'test_metrics.json', metrics)
    print(metrics)


if __name__ == '__main__':
    main()
