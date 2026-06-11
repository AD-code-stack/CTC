from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.models.tcn_bilstm import TCNBiLSTM
from src.utils.io import load_json, load_yaml, save_json


def _resolve_path(base: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _build_model(config: dict, num_classes: int, fusion: str) -> torch.nn.Module:
    model_cfg = config['model']
    feature_dim = int(model_cfg.get('input_dim', 50))
    hidden_size = int(model_cfg['hidden_size'])
    lstm_layers = int(model_cfg['lstm_layers'])
    dropout = float(model_cfg['dropout'])
    tcn_channels = model_cfg['tcn_channels']

    return TCNBiLSTM(
        input_dim=feature_dim,
        num_classes=num_classes,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        dropout=dropout,
        tcn_channels=tcn_channels,
    )


def export_torchscript(model: torch.nn.Module, example_input: torch.Tensor, output_path: Path) -> None:
    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)
        traced = torch.jit.freeze(traced)
        traced.save(str(output_path))


def export_onnx(model: torch.nn.Module, example_input: torch.Tensor, output_path: Path) -> None:
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            example_input,
            str(output_path),
            input_names=['input'],
            output_names=['logits'],
            opset_version=17,
            dynamic_axes={
                'input': {0: 'batch', 1: 'time'},
                'logits': {0: 'batch'},
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description='Export isolated-word SLR model for deployment.')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file.')
    parser.add_argument('--checkpoint', type=str, default=None, required=True, help='Path to best_model.pt or last_model.pt.')
    parser.add_argument('--output-dir', type=str, default='exports', help='Directory to store exported artifacts.')
    parser.add_argument('--fusion', choices=['single'], default='single', help='Model fusion type to export.')
    parser.add_argument('--format', choices=['torchscript', 'onnx', 'both'], default='both', help='Export format.')
    parser.add_argument('--device', type=str, default='cpu', help='Export device.')
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    config_path = Path(args.config) if args.config else base / 'src' / 'configs' / 'default.yaml'
    config = load_yaml(config_path)

    checkpoint_path = _resolve_path(base, args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    processed_dir = _resolve_path(base, config['data']['processed_dir'])
    label_map = load_json(processed_dir / 'labels' / 'label_map.json')
    num_classes = len(label_map)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    saved_config = checkpoint.get('config', config)

    model = _build_model(saved_config, num_classes=num_classes, fusion=args.fusion)
    model.load_state_dict(state_dict)
    model.to(args.device)

    feature_dim = int(saved_config['model'].get('input_dim', config['model'].get('input_dim', 50)))
    example_input = torch.zeros(1, int(config['data'].get('sequence_length', 32)), feature_dim, device=args.device)

    output_dir = _resolve_path(base, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        'checkpoint': str(checkpoint_path),
        'format': args.format,
        'fusion': args.fusion,
        'feature_dim': feature_dim,
        'sequence_length': int(config['data'].get('sequence_length', 32)),
        'num_classes': num_classes,
        'labels': label_map,
        'raw_dir': str(config['data']['raw_dir']),
        'processed_dir': str(config['data']['processed_dir']),
    }
    save_json(output_dir / 'export_metadata.json', metadata)

    if args.format in {'torchscript', 'both'}:
        export_torchscript(model, example_input, output_dir / 'model_ts.pt')
    if args.format in {'onnx', 'both'}:
        export_onnx(model, example_input, output_dir / 'model.onnx')

    config_copy = dict(config)
    config_copy['model'] = dict(config['model'])
    config_copy['model']['input_dim'] = feature_dim
    config_copy['model']['num_classes'] = num_classes
    save_json(output_dir / 'export_config.json', config_copy)

    print(f'Export completed. Files saved to: {output_dir}')


if __name__ == '__main__':
    main()
