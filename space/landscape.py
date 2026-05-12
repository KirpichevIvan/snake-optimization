from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from player.rollout import simulate_packed

from space.grid_model import PARAM_LABELS, Grid4D

META_FILENAME = "meta.json"
CSV_FILENAME = "landscape.csv"
PACKAGE_VERSION = 1


def _axis_points(lo: float, hi: float, step: float) -> np.ndarray:
    if step <= 0.0:
        msg = "Шаг сетки должен быть > 0"
        raise ValueError(msg)
    if lo > hi:
        msg = "min должен быть <= max"
        raise ValueError(msg)
    pts = np.arange(lo, hi + step * 1e-12, step, dtype=np.float64)
    return pts[pts <= hi + 1e-9]


def build_grid(
    bounds: list[tuple[float, float, float]],
) -> tuple[np.ndarray, tuple[int, int, int, int], list[np.ndarray]]:
    """
    ``bounds`` — четыре тройки ``(min, max, step)``.
    Возвращает ``(grid, shape, axes)``.
    """
    if len(bounds) != 4:
        msg = "Нужно ровно 4 диапазона (min, max, step)"
        raise ValueError(msg)
    axes = [_axis_points(lo, hi, st) for lo, hi, st in bounds]
    shape = tuple(int(len(a)) for a in axes)
    mg = np.meshgrid(*axes, indexing="ij")
    stacked = np.stack([m.ravel() for m in mg], axis=1)
    return stacked.astype(np.float64), (shape[0], shape[1], shape[2], shape[3]), axes


def _stable_seed(idx: int, theta: tuple[float, float, float, float]) -> int:
    payload = repr(theta).encode("utf-8") + b"|" + str(idx).encode("ascii")
    h = hashlib.sha256(payload).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFF_FFFF


def _packed_task(
    args: tuple[int, int, tuple[int, int], tuple[float, float, float, float]],
) -> float:
    idx, max_steps, field_size, theta = args
    seed = _stable_seed(idx, theta)
    packed = (theta, max_steps, field_size, seed)
    return simulate_packed(packed)


def evaluate_j_grid(
    grid: np.ndarray,
    *,
    max_steps: int,
    field_size: tuple[int, int],
    workers: int,
    progress: bool = True,
) -> np.ndarray:
    n = grid.shape[0]
    thetas = [tuple(float(x) for x in grid[i]) for i in range(n)]
    tasks = [(i, max_steps, field_size, thetas[i]) for i in range(n)]
    w = max(1, workers) if workers > 0 else max(1, os.cpu_count() or 4)
    if w <= 1:
        it = tqdm(tasks, desc="Rollouts J(theta)", unit="pt", disable=not progress)
        return np.array([_packed_task(t) for t in it], dtype=np.float64)
    chunksize = max(1, n // (w * 8))
    with ProcessPoolExecutor(max_workers=w) as pool:
        it = pool.map(_packed_task, tasks, chunksize=chunksize)
        if progress:
            it = tqdm(it, total=n, desc="Rollouts J(theta)", unit="pt")
        out = list(it)
    return np.asarray(out, dtype=np.float64)


def compute_grid4d(
    bounds: list[tuple[float, float, float]],
    *,
    max_steps: int,
    field_size: tuple[int, int],
    workers: int,
    progress: bool,
) -> Grid4D:
    grid, shape, axes = build_grid(bounds)
    j = evaluate_j_grid(
        grid,
        max_steps=max_steps,
        field_size=field_size,
        workers=workers,
        progress=progress,
    )
    tensor = j.reshape(shape, order="C")
    return Grid4D(axes=axes, values=tensor)


def save_landscape_dir(
    out_dir: Path,
    bounds: list[tuple[float, float, float]],
    *,
    max_steps: int,
    field_size: tuple[int, int],
    workers: int,
    progress: bool,
) -> None:
    """Считает J по сетке и сохраняет ``landscape.csv`` + ``meta.json`` в ``out_dir``."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    grid, shape, axes = build_grid(bounds)
    j = evaluate_j_grid(
        grid,
        max_steps=max_steps,
        field_size=field_size,
        workers=workers,
        progress=progress,
    )
    meta: dict[str, Any] = {
        "version": PACKAGE_VERSION,
        "max_steps": int(max_steps),
        "field_height": int(field_size[0]),
        "field_width": int(field_size[1]),
        "workers": int(workers),
        "bounds": [[float(a), float(b), float(c)] for a, b, c in bounds],
        "param_labels": list(PARAM_LABELS),
        "shape": [int(s) for s in shape],
        "n_points": int(grid.shape[0]),
    }
    (out_dir / META_FILENAME).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = out_dir / CSV_FILENAME
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(list(PARAM_LABELS) + ["J"])
        for i in range(grid.shape[0]):
            row = grid[i]
            w.writerow([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(j[i])])


def load_landscape_dir(data_dir: Path) -> Grid4D:
    """Читает пакет из ``data_dir`` (``meta.json`` + ``landscape.csv``)."""
    data_dir = Path(data_dir).resolve()
    meta_path = data_dir / META_FILENAME
    csv_path = data_dir / CSV_FILENAME
    if not meta_path.is_file():
        msg = f"Нет файла {META_FILENAME}: {meta_path}"
        raise FileNotFoundError(msg)
    if not csv_path.is_file():
        msg = f"Нет файла {CSV_FILENAME}: {csv_path}"
        raise FileNotFoundError(msg)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ver = int(meta.get("version", 0))
    if ver != PACKAGE_VERSION:
        msg = f"Неподдерживаемая версия meta.json: {ver} (ожидается {PACKAGE_VERSION})"
        raise ValueError(msg)

    bounds_raw = meta["bounds"]
    bounds = [(float(t[0]), float(t[1]), float(t[2])) for t in bounds_raw]
    shape = tuple(int(x) for x in meta["shape"])
    _, shape_check, axes = build_grid(bounds)
    if shape_check != shape:
        msg = f"shape в meta {shape} не совпадает с сеткой по bounds {shape_check}"
        raise ValueError(msg)

    j_vals: list[float] = []
    with csv_path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        expected = list(PARAM_LABELS) + ["J"]
        if reader.fieldnames != expected:
            msg = f"Неверные заголовки CSV: {reader.fieldnames}"
            raise ValueError(msg)
        for row in reader:
            j_vals.append(float(row["J"]))

    n_exp = int(np.prod(shape))
    if len(j_vals) != n_exp:
        msg = f"Число строк CSV {len(j_vals)} != произведение shape {n_exp}"
        raise ValueError(msg)

    tensor = np.asarray(j_vals, dtype=np.float64).reshape(shape, order="C")
    return Grid4D(axes=axes, values=tensor)


def _cli_landscape() -> None:
    parser = argparse.ArgumentParser(
        description="Сэмплинг J(θ) по 4D-сетке: сохраняет landscape.csv и meta.json в каталог -o",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Каталог для landscape.csv и meta.json",
    )
    parser.add_argument(
        "--param",
        action="append",
        nargs=3,
        type=float,
        metavar=("MIN", "MAX", "STEP"),
        required=True,
        help="Диапазон и шаг (повторить 4 раза: w_food, w_danger, w_space, w_wall)",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--field-height", type=int, default=10)
    parser.add_argument("--field-width", type=int, default=10)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=50_000)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    if len(args.param) != 4:
        parser.error("нужно ровно четыре блока --param MIN MAX STEP")

    bounds = [(float(a[0]), float(a[1]), float(a[2])) for a in args.param]
    grid, shape, _ = build_grid(bounds)
    n = int(grid.shape[0])
    if n > args.max_points:
        parser.error(f"узлов {n} > --max-points {args.max_points}")

    out = Path(args.output_dir)
    print(f"Сохранение в {out.resolve()} | узлов {n}, shape {shape}", flush=True)
    save_landscape_dir(
        out,
        bounds,
        max_steps=args.max_steps,
        field_size=(args.field_height, args.field_width),
        workers=args.workers,
        progress=not args.no_progress,
    )
    print(f"Готово: {out / CSV_FILENAME}, {out / META_FILENAME}", flush=True)


if __name__ == "__main__":
    try:
        _cli_landscape()
    except KeyboardInterrupt:
        sys.exit(130)
