from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

base = Path(__file__).resolve().parents[1]
if str(base) not in sys.path:
    sys.path.insert(0, str(base))

import torch
import torch.nn.functional as F
from torch import nn

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


def _greedy_decode(logits: torch.Tensor, blank_id: int = 0) -> list[list[int]]:
    pred_ids = logits.argmax(dim=-1)
    sequences: list[list[int]] = []
    for seq in pred_ids:
        output: list[int] = []
        prev = None
        for idx in seq.tolist():
            if idx == blank_id:
                prev = None
                continue
            if idx != prev:
                output.append(idx)
            prev = idx
        sequences.append(output)
    return sequences


def _edit_distance(seq1: list[int], seq2: list[int]) -> int:
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n]


def _decode_to_tokens(seq: list[int], id_to_token: dict[int, str]) -> str:
    return '/'.join(id_to_token.get(idx, f'<unk:{idx}>') for idx in seq)


def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, list[dict[str, str]]]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_batches = 0
    total_distance = 0
    total_target_tokens = 0
    preview: list[dict[str, str]] = []

    for features, _padded_tokens, flat_targets, input_lengths, target_lengths, metas in loader:
        features = features.to(device)
        flat_targets = flat_targets.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = target_lengths.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(features)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = criterion(log_probs, flat_targets, input_lengths, target_lengths)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

        decoded = _greedy_decode(logits.detach().cpu())
        start = 0
        for i, target_len in enumerate(target_lengths.detach().cpu().tolist()):
            target_seq = flat_targets[start:start + target_len].detach().cpu().tolist()
            pred_seq = decoded[i]
            total_distance += _edit_distance(pred_seq, target_seq)
            total_target_tokens += max(len(target_seq), 1)
            if len(preview) < 3:
                preview.append(
                    {
                        'number': str(metas[i].get('number', 'unknown')),
                        'target_ids': str(target_seq),
                        'pred_ids': str(pred_seq),
                    }
                )
            start += target_len

    avg_loss = total_loss / max(total_batches, 1)
    token_error_rate = total_distance / max(total_target_tokens, 1)
    return avg_loss, token_error_rate, preview


def main() -> None:
    config = load_yaml(Path(__file__).resolve().parents[1] / 'src' / 'configs' / 'default.yaml')
    set_seed(config['project']['seed'])

    base_dir = Path(__file__).resolve().parents[1]
    manifest_path = base_dir / config['data']['processed_records']
    manifest = _load_manifest(manifest_path)
    samples, token_map = _build_sequence_samples(manifest)
    id_to_token = {idx: token for token, idx in token_map.items()}

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
        num_classes=len(token_map) + 1,
        hidden_size=config['model']['hidden_size'],
        lstm_layers=config['model']['lstm_layers'],
        dropout=config['model']['dropout'],
        tcn_channels=config['model'].get('tcn_channels', [64, 128]),
    ).to(device)
    print(f'Model ready: {sum(p.numel() for p in model.parameters())} parameters')

    dataset = SignLanguageSequenceDataset(samples)
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
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, len(dataset) - train_size - val_size],
        generator=generator,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers'],
        collate_fn=collate_sequence_batch,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['train']['num_workers'],
        collate_fn=collate_sequence_batch,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['train']['num_workers'],
        collate_fn=collate_sequence_batch,
    )

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
            'note': 'Initial CTC training baseline over gloss token sequences.',
        },
    )

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay'],
    )

    best_val_ter = float('inf')
    history: list[dict[str, float]] = []
    epochs = int(config['train']['epochs'])

    for epoch in range(1, epochs + 1):
        train_loss, train_ter, train_preview = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_ter, val_preview = _run_epoch(model, val_loader, criterion, device, optimizer=None)
        history.append(
            {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_token_error_rate': train_ter,
                'val_loss': val_loss,
                'val_token_error_rate': val_ter,
            }
        )
        print(
            f'Epoch {epoch:03d} | '
            f'train_loss={train_loss:.4f} train_TER={train_ter:.4f} | '
            f'val_loss={val_loss:.4f} val_TER={val_ter:.4f}'
        )
        if val_preview:
            preview = val_preview[0]
            target_ids = eval(preview['target_ids'])
            pred_ids = eval(preview['pred_ids'])
            print('  sample:', preview['number'])
            print('  target:', _decode_to_tokens(target_ids, id_to_token))
            print('  pred  :', _decode_to_tokens(pred_ids, id_to_token))

        latest_ckpt = work_dir / 'latest.pt'
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'token_map': token_map,
                'config': config,
                'history': history,
            },
            latest_ckpt,
        )
        if val_ter < best_val_ter:
            best_val_ter = val_ter
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'token_map': token_map,
                    'config': config,
                    'history': history,
                },
                work_dir / 'best.pt',
            )

    test_loss, test_ter, test_preview = _run_epoch(model, test_loader, criterion, device, optimizer=None)
    print(f'Test | loss={test_loss:.4f} token_error_rate={test_ter:.4f}')
    if test_preview:
        preview = test_preview[0]
        target_ids = eval(preview['target_ids'])
        pred_ids = eval(preview['pred_ids'])
        print('  test sample:', preview['number'])
        print('  target:', _decode_to_tokens(target_ids, id_to_token))
        print('  pred  :', _decode_to_tokens(pred_ids, id_to_token))

    save_json(work_dir / 'train_history.json', history)
    print(f'Artifacts saved to: {work_dir}')


if __name__ == '__main__':
    main()
