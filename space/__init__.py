"""Пакет space: сэмплинг ландшафта (landscape) и визуализация (visualizer / plot)."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from space.grid_model import PARAM_LABELS, Grid4D, slice_j_for_ix_iy
from space.mpl_plots import plot_landscape_2d, plot_landscape_surface, positions_xy_j, project_weights_to_xy
from space.projection import orthonormal_projection_4_to_2

__all__ = [
    "CSV_FILENAME",
    "META_FILENAME",
    "PARAM_LABELS",
    "Grid4D",
    "build_grid",
    "compute_grid4d",
    "evaluate_j_grid",
    "load_landscape_dir",
    "orthonormal_projection_4_to_2",
    "plot_landscape_2d",
    "plot_landscape_surface",
    "positions_xy_j",
    "project_weights_to_xy",
    "save_landscape_dir",
    "slice_j_for_ix_iy",
]

_LANDSCAPE_NAMES = frozenset(
    {
        "CSV_FILENAME",
        "META_FILENAME",
        "build_grid",
        "compute_grid4d",
        "evaluate_j_grid",
        "load_landscape_dir",
        "save_landscape_dir",
    }
)


def __getattr__(name: str) -> Any:
    if name in _LANDSCAPE_NAMES:
        mod = import_module("space.landscape")
        return getattr(mod, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
