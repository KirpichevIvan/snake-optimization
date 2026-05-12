from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation

from space.projection import orthonormal_projection_4_to_2


def _minmax_rows_weights(w: np.ndarray) -> np.ndarray:
    lo = w.min(axis=0)
    hi = w.max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    return (w - lo) / span


def project_weights_to_xy(
    weights: np.ndarray,
    q: np.ndarray,
    *,
    normalize_weights: bool,
) -> np.ndarray:
    w = weights.astype(np.float64)
    if normalize_weights:
        w = _minmax_rows_weights(w)
    return w @ q


def positions_xy_j(
    weights: np.ndarray,
    j: np.ndarray,
    q: np.ndarray,
    *,
    normalize_weights: bool,
) -> np.ndarray:
    xy = project_weights_to_xy(weights, q, normalize_weights=normalize_weights)
    return np.column_stack([xy[:, 0], xy[:, 1], j.astype(np.float64)])


def _apply_z_scale(z_raw: np.ndarray, z_scale: str) -> tuple[np.ndarray, str, str]:
    if z_scale == "linear":
        z = z_raw.astype(np.float64)
        return z, "J", "J"
    if z_scale == "log1p":
        z_min = float(z_raw.min())
        z = np.log1p(z_raw.astype(np.float64) - z_min + 1e-9)
        lab = "log1p(J - Jmin + eps)"
        return z, lab, lab
    msg = f"неизвестный z_scale: {z_scale}"
    raise ValueError(msg)


def _triangulation_xy(x: np.ndarray, y: np.ndarray) -> Triangulation:
    try:
        return Triangulation(x, y)
    except (RuntimeError, ValueError):
        rng = np.random.default_rng(0)
        eps = 1e-9 * (abs(x).max() + abs(y).max() + 1.0)
        return Triangulation(
            x + rng.standard_normal(x.shape) * eps,
            y + rng.standard_normal(y.shape) * eps,
        )


def plot_landscape_2d(
    pos: np.ndarray,
    out_path: str,
    *,
    title: str | None = None,
    dpi: int = 150,
    z_scale: str = "linear",
    contour_levels: int = 64,
) -> None:
    x = pos[:, 0].astype(np.float64)
    y = pos[:, 1].astype(np.float64)
    z_raw = pos[:, 2].astype(np.float64)
    z, _z_axis_label, cbar_label = _apply_z_scale(z_raw, z_scale)
    tri = _triangulation_xy(x, y)

    fig, ax = plt.subplots(figsize=(10, 8))
    tcf = ax.tricontourf(tri, z, levels=contour_levels, cmap="viridis")
    fig.colorbar(tcf, ax=ax, label=cbar_label)
    ax.set_xlabel("u1 (proj. w)")
    ax.set_ylabel("u2 (proj. w)")
    ax.set_aspect("equal", adjustable="box")

    if title:
        ax.set_title(title)
    else:
        ax.set_title("2D: (w1..w4) -> (u1,u2), J as color")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_landscape_surface(
    pos: np.ndarray,
    out_path: str,
    *,
    title: str | None = None,
    dpi: int = 150,
    z_scale: str = "linear",
    box_aspect: str = "equal",
) -> None:
    x = pos[:, 0].astype(np.float64)
    y = pos[:, 1].astype(np.float64)
    z_raw = pos[:, 2].astype(np.float64)
    z, z_label, cbar_label = _apply_z_scale(z_raw, z_scale)
    tri = _triangulation_xy(x, y)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(
        tri,
        z,
        cmap="viridis",
        linewidth=0,
        edgecolor="none",
        antialiased=True,
        shade=True,
    )
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08, label=cbar_label)

    ax.set_xlabel("u1 (proj. w)")
    ax.set_ylabel("u2 (proj. w)")
    ax.set_zlabel(z_label)

    if box_aspect == "equal":
        ax.set_box_aspect((1.0, 1.0, 1.0))
    elif box_aspect == "data":
        xr = float(x.max() - x.min()) or 1.0
        yr = float(y.max() - y.min()) or 1.0
        zr = float(z.max() - z.min()) or 1.0
        ax.set_box_aspect([xr, yr, zr])
    else:
        msg = f"неизвестный box_aspect: {box_aspect}"
        raise ValueError(msg)

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Surface: (w1..w4) -> (u1,u2), Z = J")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
