from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PARAM_LABELS = ("w_food", "w_danger", "w_space", "w_wall")


@dataclass(frozen=True)
class Grid4D:
    axes: list[np.ndarray]
    values: np.ndarray  # (n0, n1, n2, n3), J

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (
            int(self.values.shape[0]),
            int(self.values.shape[1]),
            int(self.values.shape[2]),
            int(self.values.shape[3]),
        )


def slice_j_for_ix_iy(
    values: np.ndarray,
    *,
    ix: int,
    iy: int,
    indices: list[int],
) -> np.ndarray:
    """
    Срез ``values`` по осям ``ix``, ``iy`` (полные диапазоны), остальные из ``indices``.
    ``Z`` формы ``(len(axis[iy]), len(axis[ix]))`` для 3D-поверхности: X — ``ix``, Y — ``iy``, Z — J.
    """
    d0, d1 = sorted([ix, iy])
    idx: list[slice | int] = [0] * 4
    for d in range(4):
        if d == ix or d == iy:
            idx[d] = slice(None)
        else:
            idx[d] = int(indices[d])
    sub = values[tuple(idx)]
    if ix == d0 and iy == d1:
        return np.asarray(sub.T, dtype=np.float64)
    if ix == d1 and iy == d0:
        return np.asarray(sub, dtype=np.float64)
    msg = "slice_j_for_ix_iy: internal error"
    raise RuntimeError(msg)
