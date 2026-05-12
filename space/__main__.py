"""Точка входа ``python -m space`` — краткая справка по подмодулям."""

from __future__ import annotations

import sys


def main() -> None:
    print(
        "Пакет space — подкоманды:\n"
        "  python -m space.landscape -o <dir> --param MIN MAX STEP (x4) ...  — сэмплинг → landscape.csv + meta.json\n"
        "  python -m space.visualizer <dir>  — браузерный UI (Streamlit) для каталога\n"
        "  python -m space.plot ...  — статичный PNG (matplotlib), как раньше python -m space\n",
        flush=True,
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
