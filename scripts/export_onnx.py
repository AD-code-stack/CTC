from __future__ import annotations

from pathlib import Path


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    export_dir = base / 'exports'
    export_dir.mkdir(parents=True, exist_ok=True)
    print(f'Export placeholder directory: {export_dir}')


if __name__ == '__main__':
    main()
