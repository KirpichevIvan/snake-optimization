"""
Генетический алгоритм: гиперпараметры, прогресс обучения, остановка, просмотр лучшей партии (Streamlit).

Запуск из корня репозитория::

    uv run streamlit run ui/genetic_app.py
"""
from __future__ import annotations

import queue
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from game.models import GameStatus

from genetic import GeneticHyperparams, GeneticLearn
from ui.viz import field_to_image, iter_episode, status_ru

_VIZ_KEYS = ("ga_viz_w0", "ga_viz_w1", "ga_viz_w2", "ga_viz_w3")

GA_THREAD = "ga_train_thread"
GA_STOP = "ga_train_stop_event"
GA_QUEUE = "ga_train_queue"
GA_HOLDER = "ga_train_result_holder"
GA_CHART = "ga_chart_rows"


def _init_viz_weights(defaults: tuple[float, float, float, float]) -> None:
    for k, v in zip(_VIZ_KEYS, defaults, strict=True):
        if k not in st.session_state:
            st.session_state[k] = float(v)


def _viz_theta_from_session() -> tuple[float, float, float, float]:
    return tuple(float(st.session_state[k]) for k in _VIZ_KEYS)


def _drain_ga_queue() -> None:
    q = st.session_state.get(GA_QUEUE)
    if q is None:
        return
    rows: list[dict[str, float | int]] = st.session_state.setdefault(GA_CHART, [])
    while True:
        try:
            kind, payload = q.get_nowait()
        except queue.Empty:
            break
        if kind == "progress" and isinstance(payload, dict):
            st.session_state["ga_last_progress"] = payload
            if payload.get("phase") == "generation_done":
                rows.append(
                    {
                        "generation": int(payload["generation"]),
                        "j_min": float(payload["j_min"]),
                        "j_mean": float(payload["j_mean"]),
                        "j_max": float(payload["j_max"]),
                        "best_j_gen": float(payload["best_j_generation"]),
                        "best_j_ever": float(payload["best_j_ever"]),
                    }
                )


def _render_progress_from_session(prog_ph: Any) -> None:
    p = st.session_state.get("ga_last_progress")
    if not isinstance(p, dict):
        return
    if p.get("phase") == "snake_evaluated":
        g = int(p["generation"]) - 1
        done = g + float(p["snake_index"]) / float(p["snakes_total"])
        frac = min(1.0, done / float(p["generations_total"]))
        prog_ph.progress(
            frac,
            text=(
                f"Поколение {p['generation']}/{p['generations_total']} · "
                f"особь {p['snake_index']}/{p['snakes_total']}"
            ),
        )
    elif p.get("phase") == "generation_done":
        done_g = float(p["generation"])
        prog_ph.progress(
            min(1.0, done_g / float(p["generations_total"])),
            text=f"Поколение {p['generation']}/{p['generations_total']} (сводка)",
        )


def main() -> None:
    st.set_page_config(page_title="Snake — генетический алгоритм", layout="wide")
    st.title("Генетическое обучение весов политики")
    st.caption(
        "Настройте гиперпараметры, запустите обучение (фоновый поток). Можно **остановить** — "
        "сохранятся `results.csv`, `best.json` и обновлённый `meta.json`."
    )

    _init_viz_weights((0.0, 0.0, 0.0, 0.0))

    th: threading.Thread | None = st.session_state.get(GA_THREAD)
    if th is not None and th.is_alive():
        _drain_ga_queue()
        st.warning("Идёт обучение…")
        c_prog, c_stop = st.columns([4, 1])
        with c_prog:
            prog_live = st.progress(0.0, text="…")
            _render_progress_from_session(prog_live)
        with c_stop:
            if st.button("Остановить обучение", type="secondary", key="ga_btn_stop"):
                ev = st.session_state.get(GA_STOP)
                if isinstance(ev, threading.Event):
                    ev.set()
        m_live = st.empty()
        p = st.session_state.get("ga_last_progress")
        if isinstance(p, dict) and p.get("phase") == "generation_done":
            m_live.markdown(
                f"**Поколение {p['generation']}/{p['generations_total']}** · "
                f"J: min `{p['j_min']:.4f}` · mean `{p['j_mean']:.4f}` · max `{p['j_max']:.4f}` · "
                f"лучший за поколение `{p['best_j_generation']:.4f}` · "
                f"**лучший за прогон `{p['best_j_ever']:.4f}`**"
            )
        # Редкий опрос: иначе st.rerun() в tight loop грузит одно ядро на 100% и мешает UI (просмотр и т.д.).
        time.sleep(0.45)
        st.rerun()

    if th is not None and not th.is_alive():
        _drain_ga_queue()
        th.join(timeout=2.0)
        holder = st.session_state.get(GA_HOLDER, [])
        out_dir_s = st.session_state.pop("ga_out_dir", "")
        st.session_state[GA_THREAD] = None
        st.session_state[GA_STOP] = None
        st.session_state[GA_QUEUE] = None
        st.session_state[GA_HOLDER] = None

        if holder and holder[0] == "ok":
            best_theta, best_j, aborted = holder[1], holder[2], holder[3]
            for k, v in zip(_VIZ_KEYS, best_theta, strict=True):
                st.session_state[k] = float(v)
            st.session_state["ga_last_best_j"] = float(best_j)
            st.session_state["ga_last_out_dir"] = out_dir_s
            msg = (
                f"Обучение **остановлено вручную** (сохранено). Лучшее **J = {best_j:.6f}**, `{best_theta}`. "
                if aborted
                else f"Обучение **завершено**. Лучшее **J = {best_j:.6f}**, `{best_theta}`. "
            )
            st.success(msg + f"Каталог: `{out_dir_s}`")
        elif holder and holder[0] == "err":
            err = holder[1]
            if isinstance(err, BaseException):
                st.exception(err)
            else:
                st.error(f"Ошибка обучения: {err!r}")

    with st.sidebar:
        st.header("Гиперпараметры ГА")
        training_seed = st.number_input("training_seed", min_value=0, max_value=2**31 - 1, value=42, step=1)
        population_size = st.number_input("population_size", min_value=2, max_value=500, value=24, step=1)
        generations = st.number_input("generations", min_value=1, max_value=500, value=8, step=1)
        max_steps = st.number_input("max_steps (rollout)", min_value=1, max_value=5000, value=200, step=1)
        crossover_prob = st.slider("crossover_prob", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
        mutation_prob = st.slider("mutation_prob", min_value=0.0, max_value=1.0, value=0.15, step=0.05)
        mutation_sigma = st.number_input("mutation_sigma", min_value=0.001, max_value=5.0, value=0.25, step=0.01)
        elite_count = st.number_input("elite_count", min_value=0, max_value=population_size - 1, value=2, step=1)
        tournament_size = st.number_input("tournament_size", min_value=1, max_value=population_size, value=3, step=1)

        st.subheader("Старт поколения 0")
        use_initial_center = st.checkbox(
            "Задать центр initial_weights (иначе random_theta)",
            value=False,
        )
        iw0 = iw1 = iw2 = iw3 = 0.0
        if use_initial_center:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                iw0 = st.number_input("w_food (центр)", value=0.0, format="%.6f")
            with c2:
                iw1 = st.number_input("w_danger (центр)", value=0.0, format="%.6f")
            with c3:
                iw2 = st.number_input("w_space (центр)", value=0.0, format="%.6f")
            with c4:
                iw3 = st.number_input("w_wall (центр)", value=0.0, format="%.6f")
        use_per_axis_spread = st.checkbox(
            "Разный initial_spread по осям (initial_spread_per_weight)",
            value=False,
            disabled=not use_initial_center,
        )
        initial_spread = 0.35
        isp0 = isp1 = isp2 = isp3 = 0.2
        if use_initial_center and not use_per_axis_spread:
            initial_spread = st.number_input("initial_spread (общая σ)", min_value=1e-6, max_value=5.0, value=0.35, step=0.05)
        if use_initial_center and use_per_axis_spread:
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                isp0 = st.number_input("σ w_food", min_value=1e-6, max_value=5.0, value=0.2, step=0.05)
            with s2:
                isp1 = st.number_input("σ w_danger", min_value=1e-6, max_value=5.0, value=0.2, step=0.05)
            with s3:
                isp2 = st.number_input("σ w_space", min_value=1e-6, max_value=5.0, value=0.2, step=0.05)
            with s4:
                isp3 = st.number_input("σ w_wall", min_value=1e-6, max_value=5.0, value=0.2, step=0.05)

        st.subheader("Rollout / поле")
        snake_game_seed = st.number_input(
            "snake_game_seed",
            min_value=0,
            max_value=2**31 - 1,
            value=42,
            step=1,
            help="Фиксированный сид для каждого simulate (все поколения и особи).",
        )
        field_height = st.number_input("field_height", min_value=3, max_value=40, value=10, step=1)
        field_width = st.number_input("field_width", min_value=3, max_value=40, value=10, step=1)
        rollout_workers = st.number_input(
            "rollout_workers",
            min_value=0,
            max_value=64,
            value=0,
            step=1,
            help="0 — авто (пул процессов по числу ядер и размеру популяции); 1 — последовательно с прерыванием между особями; ≥2 — верхняя граница процессов.",
        )

        st.subheader("Сохранение")
        results_dir_str = st.text_input(
            "results_dir (пусто = временный каталог)",
            value="",
            placeholder="C:\\path\\to\\run или пусто",
        )

    col_left, col_right = st.columns([1, 1])

    with col_right:
        st.subheader("Кривая по поколениям")
        st.caption("mean / max / лучший J за прогон (обновляется по мере обучения).")
        chart_ph = st.empty()
        rows = st.session_state.get(GA_CHART, [])
        if rows:
            df = pd.DataFrame(rows)
            chart_ph.line_chart(
                df.set_index("generation")[["j_mean", "j_max", "best_j_ever"]],
                height=260,
            )

    with col_left:
        st.subheader("Обучение")
        _t = st.session_state.get(GA_THREAD)
        busy = isinstance(_t, threading.Thread) and _t.is_alive()
        if st.button("Запустить обучение", type="primary", disabled=busy):
            initial_weights: tuple[float, float, float, float] | None = None
            initial_spread_per_weight: tuple[float, float, float, float] | None = None
            if use_initial_center:
                initial_weights = (float(iw0), float(iw1), float(iw2), float(iw3))
                if use_per_axis_spread:
                    initial_spread_per_weight = (float(isp0), float(isp1), float(isp2), float(isp3))

            hp = GeneticHyperparams(
                training_seed=int(training_seed),
                population_size=int(population_size),
                generations=int(generations),
                max_steps=int(max_steps),
                crossover_prob=float(crossover_prob),
                mutation_prob=float(mutation_prob),
                mutation_sigma=float(mutation_sigma),
                elite_count=int(elite_count),
                tournament_size=int(tournament_size),
                initial_weights=initial_weights,
                initial_spread=float(initial_spread),
                initial_spread_per_weight=initial_spread_per_weight,
                rollout_workers=int(rollout_workers),
            )

            if results_dir_str.strip():
                out_dir = Path(results_dir_str.strip()).expanduser().resolve()
            else:
                out_dir = Path(tempfile.mkdtemp(prefix="streamlit_ga_"))

            learn = GeneticLearn(
                hp,
                field_size=(int(field_height), int(field_width)),
                snake_game_seed=int(snake_game_seed),
                results_dir=out_dir,
                verbose=False,
            )

            stop_ev = threading.Event()
            q: queue.Queue[tuple[str, object]] = queue.Queue()
            holder: list[object] = []

            def worker() -> None:
                try:
                    r = learn.run(
                        progress_callback=lambda info: q.put(("progress", info)),
                        interrupt_check=stop_ev.is_set,
                    )
                    holder[:] = [("ok", r[0], r[1], r[2])]
                except Exception as e:
                    holder[:] = [("err", e)]
                finally:
                    q.put(("finished", None))

            t = threading.Thread(target=worker, daemon=True)
            st.session_state[GA_CHART] = []
            st.session_state["ga_last_progress"] = {}
            st.session_state[GA_STOP] = stop_ev
            st.session_state[GA_QUEUE] = q
            st.session_state[GA_HOLDER] = holder
            st.session_state[GA_THREAD] = t
            st.session_state["ga_out_dir"] = str(out_dir)
            st.session_state["ga_viz_max_steps"] = int(max_steps)
            st.session_state["ga_viz_fh"] = int(field_height)
            st.session_state["ga_viz_fw"] = int(field_width)
            st.session_state["ga_viz_seed"] = int(snake_game_seed)
            t.start()
            st.rerun()

    st.divider()
    st.subheader("Просмотр лучшей попытки (как в `ui/app.py`)")
    st.caption(
        "После обучения веса подставляются автоматически. Для воспроизводимости с игрой при обучении "
        "используйте тот же **snake_game_seed** и размер поля."
    )

    v1, v2, v3, v4 = st.columns(4)
    with v1:
        st.number_input("w_food", format="%.6f", key="ga_viz_w0")
    with v2:
        st.number_input("w_danger", format="%.6f", key="ga_viz_w1")
    with v3:
        st.number_input("w_space", format="%.6f", key="ga_viz_w2")
    with v4:
        st.number_input("w_wall", format="%.6f", key="ga_viz_w3")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        v_fh = st.number_input(
            "Высота поля (визуализация)",
            min_value=3,
            max_value=40,
            value=int(st.session_state.get("ga_viz_fh", 10)),
            step=1,
            key="ga_viz_fh",
        )
    with r2:
        v_fw = st.number_input(
            "Ширина поля (визуализация)",
            min_value=3,
            max_value=40,
            value=int(st.session_state.get("ga_viz_fw", 10)),
            step=1,
            key="ga_viz_fw",
        )
    with r3:
        v_max_steps = st.number_input(
            "Макс. шагов (визуализация)",
            min_value=10,
            max_value=2000,
            value=int(st.session_state.get("ga_viz_max_steps", 400)),
            step=10,
            key="ga_viz_max_steps",
        )
    with r4:
        v_seed = st.number_input(
            "Зерно (визуализация)",
            min_value=0,
            max_value=2**31 - 1,
            value=int(st.session_state.get("ga_viz_seed", 42)),
            step=1,
            key="ga_viz_seed",
        )

    r5, r6 = st.columns(2)
    with r5:
        delay_ms = st.slider("Пауза между кадрами, мс", min_value=5, max_value=400, value=60, step=5, key="ga_delay_ms")
    with r6:
        cell_px = st.slider("Размер клетки, px", min_value=14, max_value=36, value=22, step=2, key="ga_cell_px")

    theta = _viz_theta_from_session()
    delay_sec = delay_ms / 1000.0

    st.markdown(
        "**Цвета:** тёмный фон — пусто, зелёный — тело, светлый — голова, красный — яблоко."
    )

    if st.button("▶ Воспроизвести партию", type="primary", key="ga_play_btn"):
        mx = int(v_max_steps)
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
            field_height=int(v_fh),
            field_width=int(v_fw),
            max_steps=mx,
            seed=int(v_seed),
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
                f"Готово: **{status_ru(last)}**, яблок **{last.score}**, кадров: **{frames}**."
            )

    if st.session_state.get("ga_last_best_j") is not None:
        st.info(
            f"Последнее обучение: лучшее J = **{st.session_state['ga_last_best_j']:.6f}**, "
            f"каталог: `{st.session_state.get('ga_last_out_dir', '')}`"
        )


if __name__ == "__main__":
    main()
