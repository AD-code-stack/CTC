from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml


def _parse_grid_value(value: str) -> list[Any]:
    items = [item.strip() for item in value.split(',') if item.strip()]
    parsed: list[Any] = []
    for item in items:
        lower = item.lower()
        if lower in {'true', 'false'}:
            parsed.append(lower == 'true')
            continue
        try:
            if '.' in item or 'e' in lower:
                parsed.append(float(item))
            else:
                parsed.append(int(item))
        except ValueError:
            parsed.append(item)
    return parsed


def _set_nested(config: dict[str, Any], key_path: str, value: Any) -> None:
    keys = key_path.split('.')
    cur: dict[str, Any] = config
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def _format_value_for_name(value: Any) -> str:
    if isinstance(value, float):
        return f'{value:g}'.replace('.', 'p')
    return str(value).replace('/', '_')


def _run_training(project_root: Path, config_path: Path, log_path: Path) -> None:
    cmd = [sys.executable, '-u', 'scripts/train.py', '--config', str(config_path)]
    with log_path.open('w', encoding='utf-8') as log_file:
        subprocess.run(cmd, cwd=project_root, check=True, stdout=log_file, stderr=subprocess.STDOUT)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a small grid search for isolated-word training.')
    parser.add_argument('--config', type=str, default=None, help='Base YAML config file.')
    parser.add_argument('--grid', action='append', default=[], help='Grid item like train.lr=0.001,0.0005')
    parser.add_argument('--results-dir', type=str, default=None, help='Directory to store grid search results.')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    base_config_path = Path(args.config) if args.config else project_root / 'src' / 'configs' / 'default.yaml'
    results_dir = Path(args.results_dir) if args.results_dir else project_root / 'experiments' / 'grid_search'
    results_dir.mkdir(parents=True, exist_ok=True)

    with base_config_path.open('r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)

    grid_items: list[tuple[str, list[Any]]] = []
    for item in args.grid:
        if '=' not in item:
            raise ValueError(f'Invalid grid spec: {item}')
        key, raw_values = item.split('=', 1)
        grid_items.append((key.strip(), _parse_grid_value(raw_values)))

    if not grid_items:
        raise ValueError('At least one --grid option is required.')

    keys = [key for key, _ in grid_items]
    value_lists = [values for _, values in grid_items]
    combinations = list(itertools.product(*value_lists))

    summary: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for run_idx, combo in enumerate(combinations, start=1):
        config = json.loads(json.dumps(base_config))
        params = dict(zip(keys, combo))
        for key_path, value in params.items():
            _set_nested(config, key_path, value)

        run_name = '_'.join(f'{key.replace(".", "-")}_{_format_value_for_name(value)}' for key, value in params.items())
        run_dir = results_dir / f'run_{run_idx:03d}_{run_name}'
        run_dir.mkdir(parents=True, exist_ok=True)

        with NamedTemporaryFile('w', suffix='.yaml', delete=False, encoding='utf-8') as tmp:
            yaml.safe_dump(config, tmp, allow_unicode=True, sort_keys=False)
            tmp_config_path = Path(tmp.name)

        isolated_dir = project_root / 'experiments' / 'isolated_word'
        if isolated_dir.exists():
            shutil.rmtree(isolated_dir)

        log_path = run_dir / 'train.log'
        try:
            _run_training(project_root, tmp_config_path, log_path)
        finally:
            if tmp_config_path.exists():
                tmp_config_path.unlink()

        metrics_path = isolated_dir / 'final_metrics.json'
        history_path = isolated_dir / 'history.json'
        best_model_path = isolated_dir / 'best_model.pt'
        last_model_path = isolated_dir / 'last_model.pt'

        if not metrics_path.exists():
            raise RuntimeError(f'Grid search run failed, missing metrics: {metrics_path}')

        with metrics_path.open('r', encoding='utf-8') as f:
            metrics = json.load(f)

        run_result = {
            'run': run_idx,
            'params': params,
            'metrics': metrics,
            'run_dir': str(run_dir),
        }
        summary.append(run_result)

        if best is None or metrics.get('accuracy', 0.0) > best['metrics'].get('accuracy', 0.0):
            best = run_result

        if history_path.exists():
            shutil.copy2(history_path, run_dir / 'history.json')
        if metrics_path.exists():
            shutil.copy2(metrics_path, run_dir / 'final_metrics.json')
        if best_model_path.exists():
            shutil.copy2(best_model_path, run_dir / 'best_model.pt')
        if last_model_path.exists():
            shutil.copy2(last_model_path, run_dir / 'last_model.pt')

        archived_dir = run_dir / 'isolated_word'
        if archived_dir.exists():
            shutil.rmtree(archived_dir)
        if isolated_dir.exists():
            shutil.move(str(isolated_dir), str(archived_dir))

        print(f'Run {run_idx}/{len(combinations)} finished: acc={metrics.get("accuracy"):.4f}, params={params}')

    assert best is not None
    with (results_dir / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (results_dir / 'best_result.json').open('w', encoding='utf-8') as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print('Grid search completed.')
    print(f'Best accuracy: {best["metrics"].get("accuracy"):.4f}')
    print(f'Best params: {best["params"]}')
    print(f'Results saved to: {results_dir}')


if __name__ == '__main__':
    main()
