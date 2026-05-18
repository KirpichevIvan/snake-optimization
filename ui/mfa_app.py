"""
Modified Firefly Algorithm: гиперпараметры, прогресс обучения, остановка, просмотр лучшей партии (Streamlit).

Запуск из корня репозитория::

    uv run streamlit run ui/mfa_app.py
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

from swarm import MfaHyperparams, MfaLearn
from ui.viz import field_to_image, iter_episode, status_ru

_VIZ_KEYS = ("mfa_viz_w0", "mfa_viz_w1", "mfa_viz_w2", "mfa_viz_w3")

MFA_THREAD = "mfa_train_thread"
MFA_STOP = "mfa_train_stop_event"
MFA_QUEUE = "mfa_train_queue"
MFA_HOLDER = "mfa_train_result_holder"
MFA_CHART = "mfa_chart_rows"

_DEFAULT_HP = MfaHyperparams(training_seed=42)


def _init_viz_weights(defaults: tuple[float, float, float, float]) -> None:
    for k, v in zip(_VIZ_KEYS, defaults, strict=True):
        if k not in st.session_state:
            st.session_state[k] = float(v)


def _init_viz_controls() -> None:
    """Initialize replay controls once; widgets with these keys must not also pass value=."""
    defaults = {
        "mfa_viz_fh": 10,
        "mfa_viz_fw": 10,
        "mfa_viz_max_steps": 400,
        "mfa_viz_seed": 42,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _viz_theta_from_session() -> tuple[float, float, float, float]:
    return tuple(float(st.session_state[k]) for k in _VIZ_KEYS)


def _drain_mfa_queue() -> None:
    q = st.session_state.get(MFA_QUEUE)
    if q is None:
        return
    rows: list[dict[str, float | int]] = st.session_state.setdefault(MFA_CHART, [])
    while True:
        try:
            kind, payload = q.get_nowait()
        except queue.Empty:
            break
        if kind == "progress" and isinstance(payload, dict):
            st.session_state["mfa_last_progress"] = payload
            if payload.get("phase") == "generation_done":
                # Это именно те ключи, которые передает MfaLearn
                rows.append(
                    {
                        "generation": int(payload["generation"]),
                        "j_min": float(payload["j_min"]),
                        "j_mean": float(payload["j_mean"]),
                        "j_max": float(payload["j_max"]),
                        "best_j_ever": float(payload["best_j_ever"]),
                    }
                )


def _render_progress_from_session(prog_ph: Any) -> None:
    p = st.session_state.get("mfa_last_progress")
    if not isinstance(p, dict):
        return
    if p.get("phase") == "generation_started":
        g = int(p["generation"]) - 1
        frac = min(1.0, g / float(p["generations_total"]))
        prog_ph.progress(
            frac,
            text=(
                f"Поколение {p['generation']}/{p['generations_total']} стартовало · "
                f"workers={p.get('rollout_workers', '?')}"
            ),
        )
    elif p.get("phase") == "snake_evaluated":
        g = int(p["generation"]) - 1
        done = g + float(p["snake_index"]) / float(p["snakes_total"])
        frac = min(1.0, done / float(p["generations_total"]))
        elapsed = float(p.get("elapsed_sec", 0.0))
        prog_ph.progress(
            frac,
            text=(
                f"Поколение {p['generation']}/{p['generations_total']} · "
                f"готово rollout {p['snake_index']}/{p['snakes_total']} · {elapsed:.1f} сек"
            ),
        )
    elif p.get("phase") == "generation_done":
        done_g = float(p["generation"])
        elapsed = float(p.get("elapsed_sec", 0.0))
        prog_ph.progress(
            min(1.0, done_g / float(p["generations_total"])),
            text=f"Поколение {p['generation']}/{p['generations_total']} завершено за {elapsed:.1f} сек",
        )


def main() -> None:
    st.set_page_config(page_title="Snake — MFA", layout="wide")
    st.title("MFA-обучение весов политики")
    st.caption(
        "Настройте гиперпараметры, запустите обучение (фоновый поток). Можно **остановить** — "
        "сохранятся `results.csv`, `best.json` и обновлённый `meta.json`."
    )

    _init_viz_weights((0.0, 0.0, 0.0, 0.0))
    _init_viz_controls()

    th: threading.Thread | None = st.session_state.get(MFA_THREAD)
    if th is not None and th.is_alive():
        _drain_mfa_queue()
        st.warning("Идёт обучение…")
        c_prog, c_stop = st.columns([4, 1])
        with c_prog:
            prog_live = st.progress(0.0, text="…")
            _render_progress_from_session(prog_live)
        with c_stop:
            if st.button("Остановить обучение", type="secondary", key="mfa_btn_stop"):
                ev = st.session_state.get(MFA_STOP)
                if isinstance(ev, threading.Event):
                    ev.set()
        active_cfg = st.session_state.get("mfa_active_config")
        if isinstance(active_cfg, dict):
            st.caption(
                "Активный запуск: "
                f"population={active_cfg.get('population_size')}, generations={active_cfg.get('generations')}, "
                f"max_steps={active_cfg.get('max_steps')}, workers={active_cfg.get('rollout_workers')}, "
                f"field={active_cfg.get('field_size')}"
            )
        m_live = st.empty()
        p = st.session_state.get("mfa_last_progress")
        if isinstance(p, dict) and p.get("phase") == "snake_evaluated":
            elapsed = float(p.get("elapsed_sec", 0.0))
            m_live.markdown(
                f"**Поколение {p['generation']}/{p['generations_total']}** · "
                f"готово rollout `{p['snake_index']}/{p['snakes_total']}` · "
                f"последний rollout: `{p.get('steps', '?')}` шагов · elapsed `{elapsed:.1f}` сек"
            )
        elif isinstance(p, dict) and p.get("phase") == "generation_done":
            elapsed = float(p.get("elapsed_sec", 0.0))
            m_live.markdown(
                f"**Поколение {p['generation']}/{p['generations_total']}** · "
                f"J: min `{p['j_min']:.4f}` · mean `{p['j_mean']:.4f}` · max `{p['j_max']:.4f}` · "
                f"лучший за поколение `{p['best_j_generation']:.4f}` · "
                f"**лучший за прогон `{p['best_j_ever']:.4f}`** · elapsed `{elapsed:.1f}` сек"
            )
        # Редкий опрос: иначе st.rerun() в tight loop грузит одно ядро на 100% и мешает UI (просмотр и т.д.).
        time.sleep(0.45)
        st.rerun()

    if th is not None and not th.is_alive():
        _drain_mfa_queue()
        th.join(timeout=2.0)
        holder = st.session_state.get(MFA_HOLDER, [])
        out_dir_s = st.session_state.pop("mfa_out_dir", "")
        st.session_state[MFA_THREAD] = None
        st.session_state[MFA_STOP] = None
        st.session_state[MFA_QUEUE] = None
        st.session_state[MFA_HOLDER] = None

        if holder and holder[0] == "ok":
            best_theta, best_j, aborted = holder[1], holder[2], holder[3]
            for k, v in zip(_VIZ_KEYS, best_theta, strict=True):
                st.session_state[k] = float(v)
            st.session_state["mfa_last_best_j"] = float(best_j)
            st.session_state["mfa_last_out_dir"] = out_dir_s
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

    _t = st.session_state.get(MFA_THREAD)
    busy = isinstance(_t, threading.Thread) and _t.is_alive()

    with st.sidebar:
        st.header("Гиперпараметры MFA")
        st.caption("Параметры применяются только после нажатия кнопки запуска — изменение полей само по себе не перезапускает обучение.")
        with st.form("mfa_training_config"):
            training_seed = st.number_input("training_seed", min_value=0, max_value=2**31 - 1, value=42, step=1)
            population_size = st.number_input("population_size", min_value=2, max_value=500, value=_DEFAULT_HP.population_size, step=1)
            generations = st.number_input("generations", min_value=1, max_value=500, value=_DEFAULT_HP.generations, step=1)
            max_steps = st.number_input("max_steps (rollout)", min_value=1, max_value=5000, value=_DEFAULT_HP.max_steps, step=1)

            st.subheader("Параметры светлячков")
            beta0 = st.number_input(
                "beta0",
                min_value=1e-9,
                max_value=0.999999,
                value=_DEFAULT_HP.beta0,
                step=0.05,
                format="%.6f",
                help="Базовая привлекательность, 0 < beta0 < 1.",
            )
            gamma = st.number_input(
                "gamma",
                min_value=0.0,
                max_value=10.0,
                value=_DEFAULT_HP.gamma,
                step=0.05,
                format="%.4f",
                help="Коэффициент поглощения света в exp(-gamma * r^2).",
            )
            alpha0 = st.number_input(
                "alpha0",
                min_value=0.0,
                max_value=10.0,
                value=_DEFAULT_HP.alpha0,
                step=0.1,
                format="%.4f",
                help="Базовый случайный шаг; дальше уменьшается по формуле MFA.",
            )

            st.subheader("Opposition-based chaotic шаг")
            p_chaos = st.slider(
                "p",
                min_value=0.0,
                max_value=1.0,
                value=_DEFAULT_HP.p,
                step=0.05,
                help="Доля/вероятность для числа худших светлячков q.",
            )
            c_chaos = st.number_input(
                "c",
                min_value=0.0,
                max_value=4.0,
                value=_DEFAULT_HP.c,
                step=0.1,
                format="%.6f",
                help="Константа логистического отображения; c=4 соответствует хаосу.",
            )
            mu0 = st.number_input(
                "mu0",
                min_value=1e-9,
                max_value=0.999999,
                value=_DEFAULT_HP.mu0,
                step=0.001,
                format="%.6f",
                help="Начальное mu: в (0,1), не 0.25, 0.5, 0.75.",
            )

            st.subheader("Rollout / поле")
            stretch = st.checkbox(
                "Вытягивание (stretch)",
                value=False,
                help=(
                    "Если после max_steps игра ещё идёт — добавлять блоки шагов, пока за блок съедено ≥1 яблоко "
                    "(иначе стоп, чтобы не крутить зацикленную змейку)."
                ),
            )
            stretch_chunk = st.number_input(
                "stretch_chunk",
                min_value=1,
                max_value=2000,
                value=_DEFAULT_HP.stretch_chunk,
                step=10,
                disabled=not stretch,
                help="Размер одного блока дополнительных шагов.",
            )
            snake_game_seed = st.number_input(
                "snake_game_seed",
                min_value=0,
                max_value=2**31 - 1,
                value=42,
                step=1,
                help="Сид rollout: на все поколения, если не включена рандомизация ниже.",
            )
            randomize_game_seed_per_generation = st.checkbox(
                "Новый snake_game_seed каждое поколение",
                value=False,
                help=(
                    "На каждое поколение свой сид игры (все особи поколения — один сид). "
                    "Значения детерминированы от training_seed."
                ),
            )
            field_height = st.number_input("field_height", min_value=3, max_value=40, value=10, step=1)
            field_width = st.number_input("field_width", min_value=3, max_value=40, value=10, step=1)
            rollout_workers = st.number_input(
                "rollout_workers",
                min_value=0,
                max_value=64,
                value=1,
                step=1,
                help="Для Streamlit безопаснее 1. 0/≥2 запускают ProcessPool и лучше подходят для CLI/длинных прогонов.",
            )

            st.info("Если кажется, что UI завис на первых поколениях, сначала оставьте rollout_workers=1 и уменьшите population_size/max_steps. Параллельный ProcessPool в Streamlit может стартовать заметно дольше.")

            st.subheader("Сохранение")
            results_dir_str = st.text_input(
                "results_dir (пусто = временный каталог)",
                value="",
                placeholder=r"C:\path\to\run или пусто",
            )
            start_requested = st.form_submit_button("Запустить обучение", type="primary", disabled=busy)

    col_left, col_right = st.columns([1, 1])

    with col_right:
        st.subheader("Кривая по поколениям")
        chart_ph = st.empty()
        rows = st.session_state.get(MFA_CHART, [])
        if rows:
            df = pd.DataFrame(rows)
            # Убедитесь, что имена столбцов совпадают с ключами в словаре выше
            chart_ph.line_chart(
                df.set_index("generation")[["j_mean", "j_max", "best_j_ever"]],
                height=260,
            )

    with col_left:
        st.subheader("Обучение")
        st.caption("Единая точка запуска MFA — форма в боковой панели. Текущий запуск получает снимок всех её значений.")
        if start_requested and not busy:
            hp = MfaHyperparams(
                training_seed=int(training_seed),
                population_size=int(population_size),
                generations=int(generations),
                max_steps=int(max_steps),
                beta0=float(beta0),
                gamma=float(gamma),
                alpha0=float(alpha0),
                p=float(p_chaos),
                c=float(c_chaos),
                mu0=float(mu0),
                rollout_workers=int(rollout_workers),
                stretch=bool(stretch),
                stretch_chunk=max(1, int(stretch_chunk)),
                randomize_game_seed_per_generation=bool(randomize_game_seed_per_generation),
            )
            st.session_state["mfa_active_config"] = {
                "training_seed": int(training_seed),
                "snake_game_seed": int(snake_game_seed),
                "population_size": int(population_size),
                "generations": int(generations),
                "max_steps": int(max_steps),
                "field_size": (int(field_height), int(field_width)),
                "rollout_workers": int(rollout_workers),
                "beta0": float(beta0),
                "gamma": float(gamma),
                "alpha0": float(alpha0),
                "p": float(p_chaos),
                "c": float(c_chaos),
                "mu0": float(mu0),
            }

            if results_dir_str.strip():
                out_dir = Path(results_dir_str.strip()).expanduser().resolve()
            else:
                out_dir = Path(tempfile.mkdtemp(prefix="streamlit_mfa_"))

            learn = MfaLearn(
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
            st.session_state[MFA_CHART] = []
            st.session_state["mfa_last_progress"] = {}
            st.session_state[MFA_STOP] = stop_ev
            st.session_state[MFA_QUEUE] = q
            st.session_state[MFA_HOLDER] = holder
            st.session_state[MFA_THREAD] = t
            st.session_state["mfa_out_dir"] = str(out_dir)
            st.session_state["mfa_viz_max_steps"] = int(max_steps)
            st.session_state["mfa_viz_fh"] = int(field_height)
            st.session_state["mfa_viz_fw"] = int(field_width)
            st.session_state["mfa_viz_seed"] = int(snake_game_seed)
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
        st.number_input("w_food", format="%.6f", key="mfa_viz_w0")
    with v2:
        st.number_input("w_danger", format="%.6f", key="mfa_viz_w1")
    with v3:
        st.number_input("w_space", format="%.6f", key="mfa_viz_w2")
    with v4:
        st.number_input("w_wall", format="%.6f", key="mfa_viz_w3")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        v_fh = st.number_input(
            "Высота поля (визуализация)",
            min_value=3,
            max_value=40,
            step=1,
            key="mfa_viz_fh",
        )
    with r2:
        v_fw = st.number_input(
            "Ширина поля (визуализация)",
            min_value=3,
            max_value=40,
            step=1,
            key="mfa_viz_fw",
        )
    with r3:
        v_max_steps = st.number_input(
            "Макс. шагов (визуализация)",
            min_value=10,
            max_value=2000,
            step=10,
            key="mfa_viz_max_steps",
        )
    with r4:
        v_seed = st.number_input(
            "Зерно (визуализация)",
            min_value=0,
            max_value=2**31 - 1,
            step=1,
            key="mfa_viz_seed",
        )

    r5, r6 = st.columns(2)
    with r5:
        delay_ms = st.slider("Пауза между кадрами, мс", min_value=5, max_value=400, value=60, step=5, key="mfa_delay_ms")
    with r6:
        cell_px = st.slider("Размер клетки, px", min_value=14, max_value=36, value=22, step=2, key="mfa_cell_px")

    theta = _viz_theta_from_session()
    delay_sec = delay_ms / 1000.0

    st.markdown(
        "**Цвета:** тёмный фон — пусто, зелёный — тело, светлый — голова, красный — яблоко."
    )

    if st.button("▶ Воспроизвести партию", type="primary", key="mfa_play_btn"):
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

    if st.session_state.get("mfa_last_best_j") is not None:
        st.info(
            f"Последнее обучение: лучшее J = **{st.session_state['mfa_last_best_j']:.6f}**, "
            f"каталог: `{st.session_state.get('mfa_last_out_dir', '')}`"
        )


if __name__ == "__main__":
    main()