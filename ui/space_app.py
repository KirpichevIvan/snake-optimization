"""
Визуализация сохранённого ландшафта (Streamlit + Plotly, 3D Surface).

Данные: каталог с ``landscape.csv`` и ``meta.json`` (см. ``python -m space.landscape``).

Запуск:
  set SNAKE_LANDSCAPE_DIR=<каталог>
  uv run streamlit run ui/space_app.py

или:
  uv run python -m space.visualizer <каталог>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from space.grid_model import PARAM_LABELS, Grid4D, slice_j_for_ix_iy
from space.landscape import load_landscape_dir


def _fixed_dims(ix: int, iy: int) -> tuple[int, int]:
    fix = [d for d in range(4) if d not in (ix, iy)]
    return fix[0], fix[1]


def _slice_index_control(
    data: Grid4D,
    label: str,
    axis_index: int,
    axis_len: int,
    current: int,
    *,
    widget_key: str,
) -> int:
    """Индекс по оси для среза; при одном узле слайдер Streamlit создать нельзя (min=max)."""
    if axis_len <= 1:
        v = float(np.asarray(data.axes[axis_index], dtype=np.float64)[0])
        st.caption(f"{label}: зафиксирован в данных · **{v:.6g}** (один узел)")
        return 0
    return int(
        st.slider(
            f"{label} (узел)",
            min_value=0,
            max_value=axis_len - 1,
            value=int(np.clip(current, 0, axis_len - 1)),
            key=widget_key,
        )
    )


def _plot_surface_3d(data: Grid4D, ix: int, iy: int, idx: list[int]) -> go.Figure:
    Z = slice_j_for_ix_iy(data.values, ix=ix, iy=iy, indices=idx)
    xv = np.asarray(data.axes[ix], dtype=np.float64)
    yv = np.asarray(data.axes[iy], dtype=np.float64)
    fig = go.Figure(
        data=[
            go.Surface(
                x=xv,
                y=yv,
                z=Z,
                colorscale="Viridis",
                colorbar=dict(title="J"),
                showscale=True,
            )
        ]
    )
    f0, f1 = _fixed_dims(ix, iy)
    subtitle = f"fixed: {PARAM_LABELS[f0]}={float(data.axes[f0][idx[f0]]):.4g}, {PARAM_LABELS[f1]}={float(data.axes[f1][idx[f1]]):.4g}"
    fig.update_layout(
        title=dict(text=f"J (3D) · {subtitle}", font=dict(size=14)),
        scene=dict(
            xaxis_title=PARAM_LABELS[ix],
            yaxis_title=PARAM_LABELS[iy],
            zaxis_title="J",
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=700,
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="J — ландшафт", layout="wide")
    st.title("Визуализация ландшафта J(θ)")
    st.caption("Загрузите каталог с `landscape.csv` и `meta.json` (команда `python -m space.landscape -o ...`).")

    default_dir = os.environ.get("SNAKE_LANDSCAPE_DIR", "").strip()
    with st.sidebar:
        st.header("Данные")
        path_str = st.text_input(
            "Каталог с landscape.csv и meta.json",
            value=default_dir,
            placeholder="C:\\path\\to\\run_001",
        )
        load_btn = st.button("Загрузить", type="primary")

    if load_btn and path_str.strip():
        p = Path(path_str.strip()).expanduser().resolve()
        try:
            data = load_landscape_dir(p)
        except (OSError, ValueError, FileNotFoundError) as e:
            st.error(str(e))
            return
        st.session_state["grid4d"] = data
        st.session_state["slice_idx"] = [max(0, s // 2) for s in data.shape]
        st.session_state["loaded_from"] = str(p)
        st.success(f"Загружено: {p} · shape {data.shape}")

    data = st.session_state.get("grid4d")
    if not isinstance(data, Grid4D):
        st.info("Укажите каталог в боковой панели и нажмите **Загрузить**.")
        return

    src = st.session_state.get("loaded_from", "")
    if src:
        st.sidebar.caption(f"Источник: `{src}`")

    idx = list(st.session_state.get("slice_idx", [max(0, s // 2) for s in data.shape]))

    colx, coly = st.columns(2)
    with colx:
        ix = int(
            st.selectbox(
                "Параметр по оси X",
                options=list(range(4)),
                format_func=lambda i: PARAM_LABELS[i],
                key="surf_ix",
            )
        )
    remaining_y = [j for j in range(4) if j != ix]
    with coly:
        iy = int(
            st.selectbox(
                "Параметр по оси Y",
                options=remaining_y,
                index=0,
                format_func=lambda j: PARAM_LABELS[j],
                key="surf_iy",
            )
        )

    f0, f1 = _fixed_dims(ix, iy)
    s0 = _slice_index_control(
        data,
        PARAM_LABELS[f0],
        f0,
        data.shape[f0],
        idx[f0],
        widget_key=f"sl_{ix}_{iy}_{f0}",
    )
    s1 = _slice_index_control(
        data,
        PARAM_LABELS[f1],
        f1,
        data.shape[f1],
        idx[f1],
        widget_key=f"sl_{ix}_{iy}_{f1}",
    )
    idx[f0] = int(s0)
    idx[f1] = int(s1)
    st.session_state["slice_idx"] = idx

    fig = _plot_surface_3d(data, ix=ix, iy=iy, idx=idx)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
