from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
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


@dataclass(slots=True)
class EpochMetrics:
    epoch: int
    split: str
    loss: float
    token_error_rate: float
    token_accuracy: float
    edit_distance: float
    num_batches: int
    num_sequences: int
    num_tokens: int


def _load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f'Processed manifest not found: {manifest_path}')
    data = load_json(manifest_path)
    if not isinstance(data, list):
        raise ValueError('Manifest must be a list of records')
    return data


def _build_sequence_samples_with_vocab_filtering(
    manifest: list[dict[str, Any]], 
    base_dir: Path,
    min_token_freq: int = 2  # 最小词频阈值
) -> tuple[list[SequenceSample], dict[str, int], Counter]:
    """构建样本，同时过滤低频词汇"""
    
    # 统计所有token出现频率
    token_counter = Counter()
    for item in manifest:
        tokens = item.get('gloss_tokens') or []
        token_counter.update(str(token).strip() for token in tokens if str(token).strip())
    
    # 过滤低频token，建立映射
    valid_tokens = {token for token, count in token_counter.items() if count >= min_token_freq}
    special_tokens = ['<blank>', '<unk>']  # blank用于CTC, unk用于低频词
    all_tokens = special_tokens + sorted(valid_tokens)
    
    # 创建token到ID的映射
    token_map = {token: idx for idx, token in enumerate(all_tokens)}
    unk_id = token_map['<unk>']
    
    print(f'原始词汇量: {len(token_counter)}')
    print(f'过滤后词汇量 (freq >= {min_token_freq}): {len(valid_tokens)}')
    print(f'低频词被映射到<unk>: {len(token_counter) - len(valid_tokens)}')
    
    samples: list[SequenceSample] = []
    skipped_low_freq = 0
    skipped_empty = 0
    
    for item in manifest:
        feature_path_str = item.get('feature_path')
        raw_tokens = [str(token).strip() for token in (item.get('gloss_tokens') or [])]
        raw_tokens = [t for t in raw_tokens if t]
        
        if not feature_path_str or not raw_tokens:
            skipped_empty += 1
            continue
        
        # 将低频词替换为<unk>
        token_ids = [token_map.get(t, unk_id) for t in raw_tokens]
        
        # 检查有效token比例（至少保留50%）
        original_count = len(raw_tokens)
        valid_count = sum(1 for t, tid in zip(raw_tokens, token_ids) 
                         if t in valid_tokens or tid == unk_id)
        
        if valid_count < original_count * 0.5 and original_count > 3:
            skipped_low_freq += 1
        
        feature_path = base_dir / feature_path_str
        samples.append(
            SequenceSample(
                feature_path=Path(feature_path),
                token_ids=token_ids,
                meta={
                    'split': item.get('split'),
                    'number': item.get('number'),
                    'translator': item.get('translator'),
                    'gloss': item.get('gloss'),
                    'gloss_tokens': raw_tokens,
                },
            )
        )
    
    print(f'跳过空样本: {skipped_empty}')
    print(f'高频低有效token样本: {skipped_low_freq}')
    print(f'最终样本数: {len(samples)}')
    
    return samples, token_map, token_counter


def _greedy_decode(logits: torch.Tensor, blank_id: int = 0) -> list[list[int]]:
    """贪婪解码"""
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
    if m == 0:
        return n
    if n == 0:
        return m
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


def _decode_to_tokens(seq: list[int], id_to_token: dict[int, str]) -> str:
    return '/'.join(id_to_token.get(idx, f'<unk:{idx}>') for idx in seq)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑的CTC损失包装器"""
    def __init__(self, criterion: nn.Module, smoothing: float = 0.1):
        super().__init__()
        self.criterion = criterion
        self.smoothing = smoothing
    
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # CTC损失本身不做标签平滑
        return self.criterion(log_probs, targets, input_lengths, target_lengths)


def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
) -> tuple[EpochMetrics, list[dict[str, str]]]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_batches = 0
    total_distance = 0
    total_target_tokens = 0
    total_pred_tokens = 0
    total_correct_tokens = 0
    total_sequences = 0
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
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        total_loss += float(loss.item())
        total_batches += 1

        decoded = _greedy_decode(logits.detach().cpu())
        
        start = 0
        target_lengths_list = target_lengths.detach().cpu().tolist()
        for i, target_len in enumerate(target_lengths_list):
            target_seq = flat_targets[start : start + target_len].detach().cpu().tolist()
            pred_seq = decoded[i] if i < len(decoded) else []
            total_distance += _edit_distance(pred_seq, target_seq)
            total_target_tokens += len(target_seq)
            total_pred_tokens += len(pred_seq)
            total_correct_tokens += sum(1 for a, b in zip(pred_seq, target_seq) if a == b)
            total_sequences += 1
            if len(preview) < 5:
                preview.append({
                    'number': str(metas[i].get('number', 'unknown')),
                    'target_ids': json.dumps(target_seq, ensure_ascii=False),
                    'pred_ids': json.dumps(pred_seq, ensure_ascii=False),
                })
            start += target_len

    metrics = EpochMetrics(
        epoch=0,
        split='train' if is_train else 'val',
        loss=_safe_ratio(total_loss, max(total_batches, 1)),
        token_error_rate=_safe_ratio(total_distance, max(total_target_tokens, 1)),
        token_accuracy=_safe_ratio(total_correct_tokens, max(total_target_tokens, 1)),
        edit_distance=_safe_ratio(total_distance, max(total_sequences, 1)),
        num_batches=total_batches,
        num_sequences=total_sequences,
        num_tokens=total_target_tokens,
    )
    return metrics, preview


def _write_metrics_csv(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[0].keys()) if history else []
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _save_checkpoint(path: Path, **kwargs) -> None:
    torch.save(kwargs, path)


def main() -> None:
    config = load_yaml(Path(__file__).resolve().parents[1] / 'src' / 'configs' / 'default.yaml')
    set_seed(config['project']['seed'])

    base_dir = Path(__file__).resolve().parents[1]
    manifest_path = base_dir / config['data']['processed_records']
    manifest = _load_manifest(manifest_path)
    
    # 使用词汇过滤
    min_token_freq = config['data'].get('min_token_freq', 2)
    samples, token_map, token_counter = _build_sequence_samples_with_vocab_filtering(
        manifest, base_dir, min_token_freq
    )
    id_to_token = {idx: token for token, idx in token_map.items()}
    
    num_classes = len(token_map)
    print(f'\n最终类别数: {num_classes}')
    print(f'样本数: {len(samples)}')

    print('\n' + '=' * 60)
    print('训练配置')
    print('=' * 60)
    print(f'模型参数量: ~{num_classes} 个输出类别')
    print(f'学习率: {config["train"]["lr"]}')
    print(f'隐藏层大小: {config["model"]["hidden_size"]}')
    print(f'LSTM层数: {config["model"]["lstm_layers"]}')
    print(f'Dropout: {config["model"]["dropout"]}')
    print(f'Batch Size: {config["train"]["batch_size"]}')
    print(f'早停耐心值: {config["train"].get("patience", 15)}')
    print('=' * 60)

    device_name = config['train']['device']
    device = torch.device(device_name if torch.cuda.is_available() or device_name == 'cpu' else 'cpu')
    print(f'Using device: {device}')

    model = TCNBiLSTM(
        input_dim=config['model'].get('input_dim', config['data']['feature_dim']),
        num_classes=num_classes,
        hidden_size=config['model']['hidden_size'],
        lstm_layers=config['model']['lstm_layers'],
        dropout=config['model']['dropout'],
        tcn_channels=config['model'].get('tcn_channels', [64, 128]),
    ).to(device)
    print(f'Model ready: {sum(p.numel() for p in model.parameters())} parameters')

    dataset = SignLanguageSequenceDataset(samples)
    if len(dataset) < 10:
        raise ValueError('Need at least 10 sequence samples')

    # 数据分割 80/10/10
    train_ratio = 0.8
    val_ratio = 0.1
    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(config['project']['seed'])
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    print(f'\n数据分布:')
    print(f'  训练集: {train_size} ({train_size/total*100:.1f}%)')
    print(f'  验证集: {val_size} ({val_size/total*100:.1f}%)')
    print(f'  测试集: {test_size} ({test_size/total*100:.1f}%)')

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config['train']['batch_size'],
        shuffle=True, num_workers=config['train']['num_workers'],
        collate_fn=collate_sequence_batch, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config['train']['batch_size'],
        shuffle=False, num_workers=config['train']['num_workers'],
        collate_fn=collate_sequence_batch, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config['train']['batch_size'],
        shuffle=False, num_workers=config['train']['num_workers'],
        collate_fn=collate_sequence_batch, pin_memory=True
    )

    work_dir = base_dir / config['project']['work_dir']
    work_dir.mkdir(parents=True, exist_ok=True)
    save_json(work_dir / 'token_map.json', token_map)
    
    # 保存词频统计
    save_json(work_dir / 'token_frequency.json', 
               {k: v for k, v in token_counter.most_common(100)})

    # CTC损失
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay'],
    )
    
    # 学习率调度
    epochs = int(config['train']['epochs'])
    warmup_epochs = config['train'].get('warmup_epochs', 5)
    use_scheduler = config['train'].get('use_scheduler', True)
    min_lr = config['train'].get('min_lr', 0.00001)
    
    scheduler = None
    if use_scheduler:
        def lr_lambda(epoch):
            if epoch <= warmup_epochs:
                return epoch / warmup_epochs
            progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
            return max(0.5 * (1.0 + math.cos(math.pi * progress)), min_lr / config['train']['lr'])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_ter = float('inf')
    patience = config['train'].get('patience', 20)
    patience_counter = 0
    history: list[dict[str, Any]] = []

    print(f'\n开始训练 (最多 {epochs} epochs, patience={patience})...\n')

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        
        train_metrics, _ = _run_epoch(model, train_loader, criterion, device, optimizer, scheduler)
        val_metrics, val_preview = _run_epoch(model, val_loader, criterion, device, optimizer=None)
        
        train_metrics.epoch = epoch
        val_metrics.epoch = epoch

        epoch_record = {
            'epoch': epoch,
            'lr': current_lr,
            'train_loss': train_metrics.loss,
            'train_TER': train_metrics.token_error_rate,
            'train_ACC': train_metrics.token_accuracy,
            'val_loss': val_metrics.loss,
            'val_TER': val_metrics.token_error_rate,
            'val_ACC': val_metrics.token_accuracy,
        }
        history.append(epoch_record)

        print(
            f'E {epoch:03d} | lr={current_lr:.6f} | '
            f'tr_loss={train_metrics.loss:.4f} tr_TER={train_metrics.token_error_rate:.4f} | '
            f'val_loss={val_metrics.loss:.4f} val_TER={val_metrics.token_error_rate:.4f} val_ACC={val_metrics.token_accuracy:.4f}'
        )

        if val_preview and epoch % 10 == 0:
            for pv in val_preview[:3]:
                target_ids = json.loads(pv['target_ids'])
                pred_ids = json.loads(pv['pred_ids'])
                print(f'  [{pv["number"]}] tgt:{_decode_to_tokens(target_ids, id_to_token)}')
                print(f'        pred:{_decode_to_tokens(pred_ids, id_to_token)}')

        _save_checkpoint(work_dir / 'latest.pt',
            epoch=epoch, model=model, optimizer=optimizer, token_map=token_map,
            config=config, history=history, best_metric=best_val_ter
        )
        
        if val_metrics.token_error_rate < best_val_ter:
            best_val_ter = val_metrics.token_error_rate
            patience_counter = 0
            print(f'  *** NEW BEST val_TER={best_val_ter:.4f} ***')
            _save_checkpoint(work_dir / 'best.pt',
                epoch=epoch, model=model, optimizer=optimizer, token_map=token_map,
                config=config, history=history, best_metric=best_val_ter
            )
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f'\n[早停] 连续 {patience} 轮无改善')
            break
        
        if current_lr < min_lr * 1.1:
            print(f'\n[早停] 学习率已降至最小')
            break

    print('\n' + '=' * 60)
    print('训练完成!')
    print('=' * 60)

    # 测试最佳模型
    if (work_dir / 'best.pt').exists():
        best_checkpoint = torch.load(work_dir / 'best.pt')
        model.load_state_dict(best_checkpoint['model_state_dict'])
        print(f'加载第 {best_checkpoint["epoch"]} 轮最佳模型\n')

    test_metrics, test_preview = _run_epoch(model, test_loader, criterion, device)
    print(
        f'Test | loss={test_metrics.loss:.4f} TER={test_metrics.token_error_rate:.4f} ACC={test_metrics.token_accuracy:.4f}'
    )
    if test_preview:
        for i, preview in enumerate(test_preview[:5]):
            target_ids = json.loads(preview['target_ids'])
            pred_ids = json.loads(preview['pred_ids'])
            print(f'[{preview["number"]}]')
            print(f'  T: {_decode_to_tokens(target_ids, id_to_token)}')
            print(f'  P: {_decode_to_tokens(pred_ids, id_to_token)}')

    save_json(work_dir / 'train_history.json', history)
    _write_metrics_csv(work_dir / 'train_history.csv', history)
    save_json(work_dir / 'final_metrics.json', {
        'history': history,
        'test': asdict(test_metrics),
        'best_val_TER': best_val_ter,
        'num_classes': num_classes,
    })
    print(f'\n结果保存至: {work_dir}')


if __name__ == '__main__':
    main()
