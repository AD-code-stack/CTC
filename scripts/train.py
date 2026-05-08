from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.data.dataset import Sample, SignLanguageDataset
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


def _build_samples(manifest: list[dict[str, Any]]) -> tuple[list[Sample], dict[str, int]]:
    label_names = []
    for item in manifest:
        gloss = str(item.get('gloss') or '').strip()
        if gloss:
            label_names.append(gloss)
    label_map = {label: idx for idx, label in enumerate(sorted(set(label_names)))}

    samples: list[Sample] = []
    for item in manifest:
        feature_path = item.get('feature_path')
        gloss = str(item.get('gloss') or '').strip()
        if not feature_path or not gloss:
            continue
        if gloss not in label_map:
            continue
        samples.append(
            Sample(
                feature_path=Path(feature_path),
                label=label_map[gloss],
                meta={
                    'split': item.get('split'),
                    'number': item.get('number'),
                    'translator': item.get('translator'),
                    'gloss': gloss,
                },
            )
        )
    return samples, label_map


def _make_dataloaders(samples: list[Sample], batch_size: int, seed: int, num_workers: int):
    dataset = SignLanguageDataset(samples)
    total = len(dataset)
    if total < 3:
        raise ValueError('Need at least 3 samples to split train/val/test')

    train_size = max(int(total * 0.7), 1)
    val_size = max(int(total * 0.2), 1)
    test_size = total - train_size - val_size
    if test_size <= 0:
        test_size = 1
        train_size = max(train_size - 1, 1)
    if train_size + val_size + test_size != total:
        val_size = max(total - train_size - test_size, 1)

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, total - train_size - val_size], generator=generator)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False),
    )


def _run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    y_true, y_pred = [], []
    for features, labels, _ in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    return total_loss / max(len(loader.dataset), 1), accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')


def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for features, labels, _ in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    return total_loss / max(len(loader.dataset), 1), accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')


def main() -> None:
    config = load_yaml(Path(__file__).resolve().parents[1] / 'src' / 'configs' / 'default.yaml')
    set_seed(config['project']['seed'])

    base_dir = Path(__file__).resolve().parents[1]
    manifest_path = base_dir / config['data']['processed_records']
    manifest = _load_manifest(manifest_path)
    samples, label_map = _build_samples(manifest)
    print(f'Loaded processed manifest: {len(manifest)} records')
    print(f'Usable training samples: {len(samples)}')
    print(f'Label classes: {len(label_map)}')
    if samples:
        print(f'First sample: {samples[0].feature_path} -> label {samples[0].label}')

    device_name = config['train']['device']
    device = torch.device(device_name if torch.cuda.is_available() or device_name == 'cpu' else 'cpu')
    print(f'Using device: {device}')

    train_loader, val_loader, test_loader = _make_dataloaders(
        samples,
        batch_size=config['train']['batch_size'],
        seed=config['project']['seed'],
        num_workers=config['train']['num_workers'],
    )

    model = TCNBiLSTM(
        input_dim=config['model'].get('input_dim', config['data']['feature_dim']),
        num_classes=max(len(label_map), config['model']['num_classes']),
        hidden_size=config['model']['hidden_size'],
        lstm_layers=config['model']['lstm_layers'],
        dropout=config['model']['dropout'],
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay'],
    )

    best_val_f1 = -1.0
    best_state = None
    history = []
    for epoch in range(1, config['train']['epochs'] + 1):
        train_loss, train_acc, train_f1 = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = _evaluate(model, val_loader, criterion, device)
        history.append(
            {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'train_f1': train_f1,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
            }
        )
        print(
            f'Epoch {epoch:03d} | '
            f'train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | '
            f'val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}'
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {'model': model.state_dict(), 'label_map': label_map, 'config': config}

    test_loss, test_acc, test_f1 = _evaluate(model, test_loader, criterion, device)
    print(f'Test | loss={test_loss:.4f} acc={test_acc:.4f} f1={test_f1:.4f}')

    work_dir = base_dir / config['project']['work_dir']
    work_dir.mkdir(parents=True, exist_ok=True)
    save_json(work_dir / 'train_history.json', history)
    save_json(work_dir / 'label_map.json', label_map)
    if best_state is not None:
        torch.save(best_state, work_dir / 'best_model.pt')
        print(f'Best model saved to: {work_dir / "best_model.pt"}')


if __name__ == '__main__':
    main()
