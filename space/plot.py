from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from space.landscape import build_grid, evaluate_j_grid
from space.mpl_plots import plot_landscape_2d, plot_landscape_surface, positions_xy_j
from space.projection import orthonormal_projection_4_to_2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Статичный PNG: проекция весов в (u1,u2), Z=J (matplotlib)",
    )
    parser.add_argument(
        "--param",
        action="append",
        nargs=3,
        type=float,
        metavar=("MIN", "MAX", "STEP"),
        required=True,
        help="Диапазон и шаг (x4: w_food, w_danger, w_space, w_wall)",
    )
    parser.add_argument("-o", "--output", type=str, default="landscape.png", help="Файл PNG")
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument("--no-normalize-w", action="store_true")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--field-height", type=int, default=10)
    parser.add_argument("--field-width", type=int, default=10)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-points", type=int, default=50_000)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--box-aspect", type=str, choices=("equal", "data"), default="equal")
    parser.add_argument("--z-scale", type=str, choices=("linear", "log1p"), default="linear")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--2d", action="store_true", dest="plot_2d")
    args = parser.parse_args()

    if len(args.param) != 4:
        parser.error("нужно четыре блока --param")

    bounds = [(float(a[0]), float(a[1]), float(a[2])) for a in args.param]
    grid, shape, _ = build_grid(bounds)
    n = grid.shape[0]
    if n > args.max_points:
        parser.error(f"узлов {n} > --max-points {args.max_points}")

    field_size = (args.field_height, args.field_width)
    print(f"Сетка: {n} узлов, shape {shape}", flush=True)
    j = evaluate_j_grid(
        grid,
        max_steps=args.max_steps,
        field_size=field_size,
        workers=args.workers,
        progress=not args.no_progress,
    )

    rng = np.random.default_rng(args.projection_seed)
    q = orthonormal_projection_4_to_2(rng)
    pos = positions_xy_j(grid, j, q, normalize_weights=not args.no_normalize_w)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.plot_2d:
        plot_landscape_2d(pos, str(out), dpi=args.dpi, z_scale=args.z_scale)
    else:
        plot_landscape_surface(
            pos,
            str(out),
            dpi=args.dpi,
            box_aspect=args.box_aspect,
            z_scale=args.z_scale,
        )
    print(f"Сохранено: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
