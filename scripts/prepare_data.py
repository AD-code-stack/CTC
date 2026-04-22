from __future__ import annotations

from pathlib import Path


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    (base / 'data' / 'processed').mkdir(parents=True, exist_ok=True)
    print('Data preparation scaffold ready.')


if __name__ == '__main__':
    main()
