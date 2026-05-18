from __future__ import annotations

import csv
import json
import math
import sys
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


def _build_sequence_samples(manifest: list[dict[str, Any]], base_dir: Path) -> tuple[list[SequenceSample], dict[str, int]]:
    token_names: list[str] = []
    for item in manifest:
        tokens = item.get('gloss_tokens') or []
        token_names.extend(str(token).strip() for token in tokens if str(token).strip())

    token_map = {token: idx + 1 for idx, token in enumerate(sorted(set(token_names)))}

    samples: list[SequenceSample] = []
    for item in manifest:
        feature_path_str = item.get('feature_path')
        tokens = [str(token).strip() for token in (item.get('gloss_tokens') or []) if str(token).strip()]
        if not feature_path_str or not tokens:
            continue
        # 拼接项目根目录，使用相对路径
        feature_path = base_dir / feature_path_str
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
    """贪婪解码：选择每个时间步概率最高的类别"""
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


def _beam_search_decode(log_probs: torch.Tensor, blank_id: int = 0, beam_width: int = 5) -> list[list[int]]:
    """束搜索解码：考虑多个候选路径，更好的CTC解码"""
    T, B, C = log_probs.shape  # 时间步, 批次, 类别数
    
    batch_sequences = []
    for b in range(B):
        # 每个样本单独束搜索
        beams = [{'seq': [], 'score': 0.0, 'last_char': None}]
        
        for t in range(T):
            all_candidates = []
            for beam in beams:
                probs = log_probs[t, b].cpu()
                top_k = probs.topk(beam_width)
                for idx, prob in zip(top_k.indices, top_k.values):
                    idx = idx.item()
                    prob = prob.item()
                    
                    if idx == blank_id:
                        # Blank字符：不添加到序列
                        new_beam = {
                            'seq': beam['seq'].copy(),
                            'score': beam['score'] + prob,
                            'last_char': None
                        }
                    elif idx == beam['last_char']:
                        # 与上一个字符相同（不是blank）
                        # 只在之前的beam上累加分数
                        new_beam = {
                            'seq': beam['seq'].copy(),
                            'score': beam['score'] + prob,
                            'last_char': idx
                        }
                    else:
                        # 新字符
                        new_seq = beam['seq'] + [idx]
                        new_beam = {
                            'seq': new_seq,
                            'score': beam['score'] + prob,
                            'last_char': idx
                        }
                    all_candidates.append(new_beam)
            
            # 保留top beam_width个
            all_candidates.sort(key=lambda x: x['score'], reverse=True)
            beams = all_candidates[:beam_width]
        
        # 返回最好的beam
        if beams:
            batch_sequences.append(beams[0]['seq'])
        else:
            batch_sequences.append([])
    
    return batch_sequences


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
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n]


def _decode_to_tokens(seq: list[int], id_to_token: dict[int, str]) -> str:
    return '/'.join(id_to_token.get(idx, f'<unk:{idx}>') for idx in seq)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    use_beam: bool = False,
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
                # 梯度裁剪：防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step()

        total_loss += float(loss.item())
        total_batches += 1

        # 使用beam search解码（验证时使用）
        if use_beam and not is_train:
            log_probs_detach = log_probs.detach().cpu()
            decoded = _beam_search_decode(log_probs_detach, blank_id=0, beam_width=5)
        else:
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
            if len(preview) < 3:
                preview.append(
                    {
                        'number': str(metas[i].get('number', 'unknown')),
                        'target_ids': json.dumps(target_seq, ensure_ascii=False),
                        'pred_ids': json.dumps(pred_seq, ensure_ascii=False),
                    }
                )
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


def _save_checkpoint(path: Path, *, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, 
                     scheduler: Any, token_map: dict[str, int], config: dict[str, Any], 
                     history: list[dict[str, Any]], best_metric: float) -> None:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'token_map': token_map,
        'config': config,
        'history': history,
        'best_metric': best_metric,
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(checkpoint, path)


def main() -> None:
    config = load_yaml(Path(__file__).resolve().parents[1] / 'src' / 'configs' / 'default.yaml')
    set_seed(config['project']['seed'])

    base_dir = Path(__file__).resolve().parents[1]
    manifest_path = base_dir / config['data']['processed_records']
    manifest = _load_manifest(manifest_path)
    samples, token_map = _build_sequence_samples(manifest, base_dir)
    id_to_token = {idx: token for token, idx in token_map.items()}

    print('=' * 60)
    print('训练配置')
    print('=' * 60)
    print(f'样本数: {len(samples)}')
    print(f'词汇类别数: {len(token_map)}')
    print(f'学习率: {config["train"]["lr"]}')
    print(f'隐藏层大小: {config["model"]["hidden_size"]}')
    print(f'LSTM层数: {config["model"]["lstm_layers"]}')
    print(f'TCN通道: {config["model"]["tcn_channels"]}')
    print(f'Dropout: {config["model"]["dropout"]}')
    print(f'Batch Size: {config["train"]["batch_size"]}')
    print(f'早停耐心值: {config["train"].get("patience", 10)}')
    print('=' * 60)

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

    # 调整数据分割比例：80/10/10
    train_ratio = float(config['data'].get('train_ratio', 0.8))
    val_ratio = float(config['data'].get('val_ratio', 0.1))
    test_ratio = float(config['data'].get('test_ratio', 0.1))
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError('Data split ratios must be positive')
    train_ratio /= ratio_sum
    val_ratio /= ratio_sum
    test_ratio /= ratio_sum

    total = len(dataset)
    train_size = max(int(total * train_ratio), 1)
    val_size = max(int(total * val_ratio), 1)
    test_size = total - train_size - val_size
    if test_size < 1:
        test_size = 1
        if train_size > val_size:
            train_size -= 1
        else:
            val_size -= 1
    if train_size < 1 or val_size < 1 or test_size < 1:
        raise ValueError('Not enough samples for train/val/test split')
    if train_size + val_size + test_size != total:
        val_size += total - (train_size + val_size + test_size)
    if train_size + val_size + test_size != total:
        raise ValueError('Failed to build an exact dataset split')

    generator = torch.Generator().manual_seed(config['project']['seed'])
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )
    
    # 打印数据分布
    print(f'\n数据分布:')
    print(f'  训练集: {train_size} ({train_size/total*100:.1f}%)')
    print(f'  验证集: {val_size} ({val_size/total*100:.1f}%)')
    print(f'  测试集: {test_size} ({test_size/total*100:.1f}%)')
    print()

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers'],
        collate_fn=collate_sequence_batch,
        pin_memory=True,  # 加速数据传输到GPU
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['train']['num_workers'],
        collate_fn=collate_sequence_batch,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['train']['num_workers'],
        collate_fn=collate_sequence_batch,
        pin_memory=True,
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
            'split_ratios': {'train': train_ratio, 'val': val_ratio, 'test': test_ratio},
            'note': 'CTC training with improved hyperparameters.',
        },
    )

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay'],
        betas=(0.9, 0.999),
    )
    
    # 学习率调度器：余弦退火
    epochs = int(config['train']['epochs'])
    warmup_epochs = config['train'].get('warmup_epochs', 5)
    use_scheduler = config['train'].get('use_scheduler', True)
    min_lr = config['train'].get('min_lr', 0.00001)
    
    scheduler = None
    if use_scheduler:
        # 先线性预热，再余弦退火
        def lr_lambda(epoch):
            if epoch <= warmup_epochs:
                return epoch / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
                return max(0.5 * (1.0 + math.cos(math.pi * progress)), min_lr / config['train']['lr'])
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float('inf')
    best_val_ter = float('inf')
    patience = config['train'].get('patience', 15)
    patience_counter = 0
    history: list[dict[str, Any]] = []

    print(f'\n开始训练 (最多 {epochs} epochs, 早停 patience={patience})...\n')

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        
        train_metrics, train_preview = _run_epoch(
            model, train_loader, criterion, device, optimizer, scheduler
        )
        val_metrics, val_preview = _run_epoch(
            model, val_loader, criterion, device, optimizer=None, use_beam=True
        )
        
        # ReduceLROnPlateau调度器需要基于验证损失更新
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics.loss)
        
        train_metrics.epoch = epoch
        train_metrics.split = 'train'
        val_metrics.epoch = epoch
        val_metrics.split = 'val'

        epoch_record = {
            'epoch': epoch,
            'lr': current_lr,
            'train_loss': train_metrics.loss,
            'train_token_error_rate': train_metrics.token_error_rate,
            'train_token_accuracy': train_metrics.token_accuracy,
            'val_loss': val_metrics.loss,
            'val_token_error_rate': val_metrics.token_error_rate,
            'val_token_accuracy': val_metrics.token_accuracy,
        }
        history.append(epoch_record)

        print(
            f'Epoch {epoch:03d} | lr={current_lr:.6f} | '
            f'train_loss={train_metrics.loss:.4f} train_TER={train_metrics.token_error_rate:.4f} '
            f'train_ACC={train_metrics.token_accuracy:.4f} | '
            f'val_loss={val_metrics.loss:.4f} val_TER={val_metrics.token_error_rate:.4f} '
            f'val_ACC={val_metrics.token_accuracy:.4f}'
        )

        if val_preview:
            preview = val_preview[0]
            target_ids = json.loads(preview['target_ids'])
            pred_ids = json.loads(preview['pred_ids'])
            print(f'  sample: {preview["number"]}')
            print(f'  target: {_decode_to_tokens(target_ids, id_to_token)}')
            print(f'  pred  : {_decode_to_tokens(pred_ids, id_to_token)}')

        # 保存最新checkpoint
        _save_checkpoint(
            work_dir / 'latest.pt',
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            token_map=token_map,
            config=config,
            history=history,
            best_metric=best_val_ter,
        )
        
        # 保存最佳模型（基于TER）
        if val_metrics.token_error_rate < best_val_ter:
            best_val_ter = val_metrics.token_error_rate
            patience_counter = 0
            print(f'  [NEW BEST] val_TER={best_val_ter:.4f}')
            _save_checkpoint(
                work_dir / 'best.pt',
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                token_map=token_map,
                config=config,
                history=history,
                best_metric=best_val_ter,
            )
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= patience:
            print(f'\n[早停] 验证损失连续 {patience} 个epoch没有改善，停止训练')
            break
        
        # 学习率过低检查
        if current_lr < min_lr * 1.1:
            print(f'\n[早停] 学习率已降至最小值 {min_lr}')
            break
        
        print()

    print('\n' + '=' * 60)
    print('训练完成!')
    print('=' * 60)

    # 加载最佳模型进行测试
    best_checkpoint = torch.load(work_dir / 'best.pt')
    model.load_state_dict(best_checkpoint['model_state_dict'])
    print(f'使用第 {best_checkpoint["epoch"]} 轮的模型进行测试\n')

    test_metrics, test_preview = _run_epoch(
        model, test_loader, criterion, device, optimizer=None, use_beam=True
    )
    print(
        f'Test | loss={test_metrics.loss:.4f} token_error_rate={test_metrics.token_error_rate:.4f} '
        f'token_accuracy={test_metrics.token_accuracy:.4f}'
    )
    if test_preview:
        for i, preview in enumerate(test_preview[:3]):
            target_ids = json.loads(preview['target_ids'])
            pred_ids = json.loads(preview['pred_ids'])
            print(f'  [{i+1}] sample: {preview["number"]}')
            print(f'      target: {_decode_to_tokens(target_ids, id_to_token)}')
            print(f'      pred  : {_decode_to_tokens(pred_ids, id_to_token)}')

    save_json(work_dir / 'train_history.json', history)
    _write_metrics_csv(work_dir / 'train_history.csv', history)
    save_json(
        work_dir / 'final_metrics.json',
        {
            'history': history,
            'test': asdict(test_metrics),
            'best_val_token_error_rate': best_val_ter,
            'best_epoch': best_checkpoint['epoch'],
            'num_epochs_trained': len(history),
        },
    )
    print(f'\nArtifacts saved to: {work_dir}')
    print(f'  - best.pt: 最佳验证TER的模型')
    print(f'  - latest.pt: 最后一轮的模型')
    print(f'  - train_history.json: 训练历史')
    print(f'  - token_map.json: 词汇表映射')


if __name__ == '__main__':
    main()
