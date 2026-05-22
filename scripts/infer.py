from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data.slr_isolated import load_skeleton_sequence, resample_sequence
from src.models.tcn_bilstm import TCNBiLSTM
from src.utils.io import load_json, load_yaml


def _resolve_path(base: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def main() -> None:
    parser = argparse.ArgumentParser(description='Infer isolated-word SLR label from txt feature file.')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file.')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint. Default best_model.pt.')
    parser.add_argument('--input', type=str, required=True, help='Path to skeleton txt or npy file.')
    parser.add_argument('--topk', type=int, default=5, help='Top-k predictions to print.')
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    config_path = Path(args.config) if args.config else base / 'src' / 'configs' / 'default.yaml'
    config = load_yaml(config_path)
    processed_dir = _resolve_path(base, config['data']['processed_dir'])
    work_dir = _resolve_path(base, config['project']['work_dir']) / 'isolated_word'
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else work_dir / 'best_model.pt'

    label_map = load_json(processed_dir / 'labels' / 'label_map.json')
    id_to_label = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)

    input_path = Path(args.input)
    if input_path.suffix.lower() == '.npy':
        seq = np.load(input_path)
    else:
        seq = load_skeleton_sequence(input_path)
    seq = resample_sequence(seq, int(config['data']['sequence_length']))
    x = torch.from_numpy(seq).float().unsqueeze(0)

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

    with torch.no_grad():
        logits = model(x.to(device))
        probs = torch.softmax(logits, dim=1)[0]
        topk = min(args.topk, probs.numel())
        values, indices = torch.topk(probs, k=topk)

    for rank, (idx, score) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
        print(f'{rank}. {id_to_label.get(idx, str(idx))}  score={score:.4f}')


if __name__ == '__main__':
    main()
