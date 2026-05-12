from __future__ import annotations

import numpy as np


def orthonormal_projection_4_to_2(rng: np.random.Generator) -> np.ndarray:
    """
    Случайная ортонормированная проекция R^4 -> R^2 (матрица 4x2).
    Для строки весов ``w`` формы (4,) координаты на плоскости: ``w @ Q`` формы (2,).
    """
    a = rng.standard_normal(size=(4, 2))
    q, _ = np.linalg.qr(a)
    return q[:, :2].astype(np.float64)
