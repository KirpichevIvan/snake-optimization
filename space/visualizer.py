from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def streamlit_app_path() -> Path:
    return Path(__file__).resolve().parents[1] / "ui" / "space_app.py"


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Браузерный UI (Streamlit + Plotly) для каталога с landscape.csv и meta.json",
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Каталог с результатами space.landscape (по умолчанию из SNAKE_LANDSCAPE_DIR или текущая папка)",
    )
    ns = parser.parse_args(argv)

    data_dir = ns.data_dir
    if data_dir is None:
        env = os.environ.get("SNAKE_LANDSCAPE_DIR")
        data_dir = Path(env).resolve() if env else Path.cwd().resolve()
    else:
        data_dir = Path(data_dir).resolve()

    from space.landscape import META_FILENAME, CSV_FILENAME, load_landscape_dir

    if not (data_dir / META_FILENAME).is_file() or not (data_dir / CSV_FILENAME).is_file():
        print(
            f"В каталоге нет {META_FILENAME} или {CSV_FILENAME}: {data_dir}",
            file=sys.stderr,
        )
        return 1

    load_landscape_dir(data_dir)

    app = streamlit_app_path()
    if not app.is_file():
        print(f"Не найден {app}", file=sys.stderr)
        return 1

    os.environ["SNAKE_LANDSCAPE_DIR"] = str(data_dir)
    cmd = [sys.executable, "-m", "streamlit", "run", str(app)]
    print("Запуск:", " ".join(cmd), "| данные:", data_dir, flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
