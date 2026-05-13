"""
Один прогон змейки в реальном времени по заданным θ (Streamlit).

Запуск из корня репозитория:
  uv run streamlit run ui/app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from game.models import GameStatus

from ui.viz import field_to_image, iter_episode, status_ru

DEFAULT_THETA = (6.1540, -0.0650, -6.1440, 0.0720)


def main() -> None:
    st.set_page_config(page_title="Snake — визуализация θ", layout="centered")
    st.title("Один прогон в реальном времени")
    st.caption("Только визуализация: поле обновляется после каждого шага с задержкой.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        w0 = st.number_input("w₁ (еда)", value=float(DEFAULT_THETA[0]), format="%.6f")
    with c2:
        w1 = st.number_input("w₂ (опасность)", value=float(DEFAULT_THETA[1]), format="%.6f")
    with c3:
        w2 = st.number_input("w₃ (пространство)", value=float(DEFAULT_THETA[2]), format="%.6f")
    with c4:
        w3 = st.number_input("w₄ (стена)", value=float(DEFAULT_THETA[3]), format="%.6f")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        fh = st.number_input("Высота поля", min_value=3, max_value=40, value=10, step=1)
    with r2:
        fw = st.number_input("Ширина поля", min_value=3, max_value=40, value=10, step=1)
    with r3:
        max_steps = st.number_input(
            "Макс. шагов",
            min_value=10,
            max_value=2000,
            value=400,
            step=10,
            help="Ограничение длины прогона для отзывчивости интерфейса.",
        )
    with r4:
        seed = st.number_input("Зерно", min_value=0, max_value=2**31 - 1, value=42, step=1)

    r5, r6 = st.columns(2)
    with r5:
        delay_ms = st.slider("Пауза между кадрами, мс", min_value=5, max_value=400, value=60, step=5)
    with r6:
        cell_px = st.slider("Размер клетки, px", min_value=14, max_value=36, value=22, step=2)

    theta = (float(w0), float(w1), float(w2), float(w3))
    delay_sec = delay_ms / 1000.0

    st.markdown(
        "**Цвета:** тёмный фон — пусто, зелёный — тело, светлый — голова, красный — яблоко."
    )

    if st.button("▶ Запустить прогон", type="primary"):
        mx = int(max_steps)
        prog_ph = st.empty()
        prog_ph.progress(0, text="Генерация партии (симуляция)…")

        def on_sim(cur: int, mx_: int) -> None:
            if mx_ <= 0:
                prog_ph.progress(0.0, text="Симуляция…")
                return
            prog_ph.progress(min(1.0, cur / mx_), text=f"Симуляция: ход {cur} / до {mx_}")

        ph = st.empty()
        cap = st.empty()
        frames = 0
        last = None
        for fr in iter_episode(
            theta,
            field_height=int(fh),
            field_width=int(fw),
            max_steps=mx,
            seed=int(seed),
            progress_callback=on_sim,
        ):
            last = fr
            ph.image(
                field_to_image(fr.field, cell_px=int(cell_px)),
                use_container_width=False,
            )
            cap.caption(
                f"Кадр {frames + 1} · внутр. шаг {fr.step} · яблок {fr.score} · **{status_ru(fr)}**"
            )
            frames += 1
            if fr.status is not GameStatus.IN_PROGRESS:
                break
            if frames > 1:
                prog_ph.progress(1.0, text="Показ кадров…")
            time.sleep(delay_sec)

        prog_ph.empty()

        if last is not None:
            st.success(
                f"Готово: **{status_ru(last)}**, яблок **{last.score}**, показано кадров: **{frames}**."
            )


main()
