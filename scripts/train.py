from __future__ import annotations

import argparse
import sys
from pathlib import Path

base = Path(__file__).resolve().parents[1]
if str(base) not in sys.path:
    sys.path.insert(0, str(base))

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import IsolatedWordDataset, IsolatedWordSample
from src.models.tcn_bilstm import DualBranchTCNBiLSTM, GatedFusionTCNBiLSTM, TCNBiLSTM
from src.utils.io import load_json, load_yaml, save_json


def _resolve_path(base: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _load_split_samples(base: Path, manifest_path: Path):
    records = load_json(manifest_path)
    samples: list[IsolatedWordSample] = []
    for record in records:
        feature_path = _resolve_path(base, record['feature_path'])
        samples.append(
            IsolatedWordSample(
                feature_path=feature_path,
                label_id=int(record['label_id']),
                label_name=record.get('label_name', ''),
                sample_id=record.get('sample_id', ''),
                meta={
                    'source_path': record.get('source_path'),
                    'split': record.get('split'),
                },
            )
        )
    return samples


def _accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    if labels.numel() == 0:
        return 0.0
    return float((preds == labels).float().mean().item())


def _topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    if labels.numel() == 0:
        return 0.0
    k = min(k, logits.shape[-1])
    topk = logits.topk(k, dim=1).indices
    correct = topk.eq(labels.unsqueeze(1)).any(dim=1).float().mean().item()
    return float(correct)


def _macro_f1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    if labels.numel() == 0 or num_classes <= 0:
        return 0.0
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(sum(f1s) / len(f1s))


def _run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0
    all_preds = []
    all_labels = []
    all_logits = []

    for features, labels, _meta in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        loss = criterion(logits, labels)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        total_count += int(labels.size(0))
        all_preds.append(logits.argmax(dim=1).detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_logits.append(logits.detach().cpu())

    if total_count == 0:
        return {'loss': 0.0, 'accuracy': 0.0, 'top5_accuracy': 0.0, 'macro_f1': 0.0}

    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)
    logits = torch.cat(all_logits, dim=0)
    return {
        'loss': total_loss / total_count,
        'accuracy': _accuracy(preds, labels),
        'top5_accuracy': _topk_accuracy(logits, labels, k=5),
        'macro_f1': _macro_f1(preds, labels, int(logits.shape[1])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Train isolated-word SLR classifier.')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file.')
    parser.add_argument('--fusion', choices=['auto', 'dual', 'gated', 'single'], default='auto', help='Model fusion mode.')
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    config_path = Path(args.config) if args.config else base / 'src' / 'configs' / 'default.yaml'
    config = load_yaml(config_path)

    processed_dir = _resolve_path(base, config['data']['processed_dir'])
    work_dir = _resolve_path(base, config['project']['work_dir']) / 'isolated_word'
    work_dir.mkdir(parents=True, exist_ok=True)

    summary_path = processed_dir / 'stats' / 'dataset_summary.json'
    summary = load_json(summary_path) if summary_path.exists() else {}
    modalities = summary.get('modalities', ['color'])

    train_manifest = processed_dir / 'manifests' / 'train.json'
    val_manifest = processed_dir / 'manifests' / 'val.json'
    test_manifest = processed_dir / 'manifests' / 'test.json'

    train_samples = _load_split_samples(base, train_manifest)
    val_samples = _load_split_samples(base, val_manifest)
    test_samples = _load_split_samples(base, test_manifest)

    if not train_samples:
        raise RuntimeError('No training samples found. Please run prepare_data.py first.')

    label_map = load_json(processed_dir / 'labels' / 'label_map.json')
    num_classes = len(label_map)
    feature_dim = int(summary.get('feature_dim', config['model'].get('input_dim', 50)))
    config['model']['input_dim'] = feature_dim
    print(f'Loaded processed dataset modalities: {modalities}')
    print(f'Using input feature dim: {feature_dim}')

    train_loader = DataLoader(IsolatedWordDataset(train_samples), batch_size=int(config['train']['batch_size']), shuffle=True, num_workers=int(config['train']['num_workers']))
    val_loader = DataLoader(IsolatedWordDataset(val_samples), batch_size=int(config['train']['batch_size']), shuffle=False, num_workers=int(config['train']['num_workers']))
    test_loader = DataLoader(IsolatedWordDataset(test_samples), batch_size=int(config['train']['batch_size']), shuffle=False, num_workers=int(config['train']['num_workers']))

    device = torch.device(config['train']['device'] if torch.cuda.is_available() else 'cpu')
    use_dual_branch = len(modalities) >= 2 and feature_dim % 2 == 0
    fusion_mode = args.fusion
    if fusion_mode == 'auto':
        fusion_mode = 'gated' if use_dual_branch else 'single'

    if fusion_mode == 'gated' and use_dual_branch:
        print('Using GatedFusionTCNBiLSTM for fused modalities.')
        model = GatedFusionTCNBiLSTM(
            input_dim=feature_dim,
            num_classes=num_classes,
            hidden_size=int(config['model']['hidden_size']),
            lstm_layers=int(config['model']['lstm_layers']),
            dropout=float(config['model']['dropout']),
            tcn_channels=config['model']['tcn_channels'],
        ).to(device)
    elif fusion_mode == 'dual' and use_dual_branch:
        print('Using DualBranchTCNBiLSTM for fused modalities.')
        model = DualBranchTCNBiLSTM(
            input_dim=feature_dim,
            num_classes=num_classes,
            hidden_size=int(config['model']['hidden_size']),
            lstm_layers=int(config['model']['lstm_layers']),
            dropout=float(config['model']['dropout']),
            tcn_channels=config['model']['tcn_channels'],
        ).to(device)
    else:
        model = TCNBiLSTM(
            input_dim=feature_dim,
            num_classes=num_classes,
            hidden_size=int(config['model']['hidden_size']),
            lstm_layers=int(config['model']['lstm_layers']),
            dropout=float(config['model']['dropout']),
            tcn_channels=config['model']['tcn_channels'],
        ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['train']['lr']), weight_decay=float(config['train']['weight_decay']))
    scheduler = None
    if config['train'].get('scheduler') == 'multisteplr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['train'].get('milestone_epochs', [20, 35]), gamma=float(config['train'].get('gamma', 0.2)))
    elif config['train'].get('scheduler') == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config['train']['epochs']), eta_min=float(config['train'].get('min_lr', 1e-5)))

    best_val_acc = -1.0
    history: list[dict[str, float | int]] = []
    epochs = int(config['train']['epochs'])
    patience = int(config['train'].get('patience', 10))
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_metrics = _run_epoch(model, val_loader, criterion, device, optimizer=None)
        if scheduler is not None:
            scheduler.step()

        row = {'epoch': epoch, **{f'train_{k}': v for k, v in train_metrics.items()}, **{f'val_{k}': v for k, v in val_metrics.items()}}
        history.append(row)
        print(f"Epoch {epoch:03d} | train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} | val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.4f} f1 {val_metrics['macro_f1']:.4f}")

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            torch.save({'model_state_dict': model.state_dict(), 'num_classes': num_classes, 'config': config}, work_dir / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered at epoch {epoch}.')
                break

    save_json(work_dir / 'history.json', history)
    torch.save({'model_state_dict': model.state_dict(), 'num_classes': num_classes, 'config': config}, work_dir / 'last_model.pt')

    test_metrics = _run_epoch(model, test_loader, criterion, device, optimizer=None)
    save_json(work_dir / 'final_metrics.json', test_metrics)
    print('Test metrics:', test_metrics)


if __name__ == '__main__':
    main()
