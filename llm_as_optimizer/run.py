from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO, cast

from llm_as_optimizer.llm_client import ask_llm
from llm_as_optimizer.prompts import build_llm_user_payload, user_message_content
from player.policy import Theta, random_theta
from player.rollout import simulate_packed

type ResultRow = tuple[Theta, float]

GAME_STEPS = 1000
OPT_ITERATIONS = 50
TOP_K = 5
WORST_K = 5
NUM_LLM_CANDIDATES = 10
# Сколько последних средних J смотреть на «плато» (узкий разброс → режим plateau_break)
PLATEAU_WINDOW = 4
# если среднее J по популяции почти не меняется столько итераций — усиливаем разведку в промпте
PLATEAU_SPREAD_MAX = 0.2
# Сколько полных партий на один набор четырёх чисел (оценка — среднее J по партиям)
ROLLOUTS_PER_THETA = 100
FIELD_SIZE = (10, 10)
# 0 = число ядер; партии считаются параллельно в отдельных процессах (ускорение CPU)
DEFAULT_WORKERS = 0
# Траектория оптимизации в промпт LLM: сырой буфер (все оценки по итерациям) и сколько точек отдавать после сортировки
TRAJECTORY_MAX_RAW = 320
TRAJECTORY_MAX_SEND = 72
# Таймаут запроса к LLM (сек.); можно переопределить --llm-timeout или LLM_TIMEOUT_SEC в .env
DEFAULT_LLM_TIMEOUT_SEC = 240.0


class _TeeTextIO:
    """Пишет в несколько потоков (терминал + файл); flush после каждой записи."""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        for stream in self._streams:
            stream.write(s)
            stream.flush()
        return len(s)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return self._streams[0].isatty()

    def fileno(self) -> int:
        return self._streams[0].fileno()


def _theta_key(t: Theta) -> tuple[float, ...]:
    return tuple(round(x, 6) for x in t)


def _fmt_theta(t: Theta, *, decimals: int = 4) -> str:
    inner = ", ".join(f"{x:.{decimals}f}" for x in t)
    return f"[{inner}]"


def _rollout_seed(base_seed: int, iteration: int, theta_index: int, rollout_index: int) -> int:
    x = base_seed ^ (iteration * 1_009_663) ^ (theta_index * 100_003) ^ (rollout_index * 917_521)
    return x & 0x7FFFFFFF


def init_population(n: int, rng: random.Random) -> list[Theta]:
    """Случайные наборы только здесь — для стартовой популяции."""
    return [random_theta(rng) for _ in range(n)]


def _resolve_workers(workers: int) -> int:
    if workers <= 0:
        return max(1, os.cpu_count() or 4)
    return workers


def evaluate_population(
    population: list[Theta],
    *,
    max_steps: int,
    rollouts_per_theta: int,
    base_seed: int,
    iteration: int,
    workers: int,
    progress_context: tuple[int, int] | None = None,
) -> list[ResultRow]:
    n = len(population)
    tasks: list[tuple[Theta, int, tuple[int, int], int]] = []
    for i in range(n):
        theta = population[i]
        for k in range(rollouts_per_theta):
            tasks.append((theta, max_steps, FIELD_SIZE, _rollout_seed(base_seed, iteration, i, k)))

    w = _resolve_workers(workers)
    total_tasks = len(tasks)
    show_progress = progress_context is not None and total_tasks >= 800
    prog_it, prog_tot = progress_context if progress_context else (0, 0)
    prog_step = max(2000, total_tasks // 6) if show_progress else 0

    if w <= 1:
        flat = []
        for idx, t in enumerate(tasks, start=1):
            flat.append(simulate_packed(t))
            if show_progress and prog_step and (idx % prog_step == 0 or idx == total_tasks):
                print(
                    f"  [iter {prog_it}/{prog_tot}] симуляции: {idx}/{total_tasks} партий",
                    flush=True,
                )
    else:
        chunksize = max(1, len(tasks) // (w * 8))
        flat = []
        with ProcessPoolExecutor(max_workers=w) as pool:
            # ProcessPoolExecutor не имеет imap (это multiprocessing.Pool) — батчи + map
            if show_progress:
                batch_size = max(800, total_tasks // 8)
                done = 0
                for start in range(0, total_tasks, batch_size):
                    end = min(start + batch_size, total_tasks)
                    part = tasks[start:end]
                    part_cs = max(1, len(part) // (w * 4))
                    flat.extend(pool.map(simulate_packed, part, chunksize=part_cs))
                    done = end
                    print(
                        f"  [iter {prog_it}/{prog_tot}] симуляции: {done}/{total_tasks} партий",
                        flush=True,
                    )
            else:
                flat = list(pool.map(simulate_packed, tasks, chunksize=chunksize))

    results: list[ResultRow] = []
    for i, theta in enumerate(population):
        lo = i * rollouts_per_theta
        hi = lo + rollouts_per_theta
        chunk = flat[lo:hi]
        avg = sum(chunk) / rollouts_per_theta
        results.append((theta, avg))
    return results


def _normalize_candidate(row: object) -> Theta | None:
    if not isinstance(row, list) or len(row) != 4:
        return None
    try:
        return cast(
            Theta,
            tuple(float(x) for x in row),
        )
    except (TypeError, ValueError):
        return None


def _dedupe_thetas_ordered(rows: list[Theta], max_n: int) -> list[Theta]:
    """Уникальные θ по округлённому ключу, порядок сохраняем, не больше max_n."""
    seen: set[tuple[float, ...]] = set()
    out: list[Theta] = []
    for t in rows:
        k = _theta_key(t)
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
        if len(out) >= max_n:
            break
    return out


def parse_llm_step(parsed: object, *, max_keep: int) -> tuple[list[Theta], str]:
    """
    Разбор ответа модели: векторы θ и поле hypothesis_note (одно предложение с гипотезой шага).
    Голый JSON-массив — только кандидаты, заметки нет.
    """
    note = ""
    raw: object | None = None
    if isinstance(parsed, dict):
        hn = parsed.get("hypothesis_note")
        if isinstance(hn, str):
            note = hn.strip()
        elif hn is not None:
            note = str(hn).strip()
        raw = parsed.get("candidates")
    elif isinstance(parsed, list):
        raw = parsed
    else:
        return [], note
    if not isinstance(raw, list):
        return [], note
    out: list[Theta] = []
    for item in raw:
        t = _normalize_candidate(item)
        if t is not None:
            out.append(t)
    return _dedupe_thetas_ordered(out, max_keep), note


def build_next_population(
    last_results: list[ResultRow],
    llm_thetas: list[Theta],
    pop_size: int,
) -> list[Theta]:
    """
    Сначала наборы из ответа модели (без повторов), затем лучшие из прошлой проверки
    по убыванию оценки, пока не наберётся нужный размер. Новые случайные числа не генерируются.
    """
    seen: set[tuple[float, ...]] = set()
    out: list[Theta] = []
    for t in llm_thetas:
        k = _theta_key(t)
        if k in seen:
            continue
        out.append(t)
        seen.add(k)
        if len(out) >= pop_size:
            return out[:pop_size]
    ranked = sorted(last_results, key=lambda x: -x[1])
    for t, _s in ranked:
        k = _theta_key(t)
        if k in seen:
            continue
        out.append(t)
        seen.add(k)
        if len(out) >= pop_size:
            return out[:pop_size]
    j = 0
    while len(out) < pop_size and ranked:
        t = ranked[j % len(ranked)][0]
        out.append(t)
        j += 1
    return out[:pop_size]


def llm_optimizer(
    *,
    seed: int,
    iterations: int = OPT_ITERATIONS,
    pop_size: int | None = None,
    max_steps: int = GAME_STEPS,
    worst_k: int = WORST_K,
    rollouts_per_theta: int = ROLLOUTS_PER_THETA,
    workers: int = DEFAULT_WORKERS,
    n_llm_candidates: int = NUM_LLM_CANDIDATES,
    trajectory_max_raw: int = TRAJECTORY_MAX_RAW,
    trajectory_max_send: int = TRAJECTORY_MAX_SEND,
    llm_timeout_sec: float = DEFAULT_LLM_TIMEOUT_SEC,
) -> list[ResultRow]:
    if pop_size is None:
        pop_size = n_llm_candidates
    if pop_size < 1:
        raise ValueError("Размер набора θ (pop_size) должен быть >= 1.")
    rng = random.Random(seed)
    population = init_population(pop_size, rng)
    last_results: list[ResultRow] = []
    mean_history: deque[float] = deque(maxlen=PLATEAU_WINDOW)
    traj_buffer: list[tuple[int, Theta, float]] = []
    score_cache: dict[tuple[float, ...], float] = {}
    prev_llm_keys: set[tuple[float, ...]] = set()

    if pop_size < n_llm_candidates:
        print(
            f"ВНИМАНИЕ: pop_size={pop_size} < llm_candidates={n_llm_candidates} — в следующий набор "
            f"возьмётся только {pop_size} первых уникальных θ из ответа модели, остальные кандидаты отбрасываются.",
            flush=True,
        )

    for it in range(iterations):
        wn = _resolve_workers(workers)
        if it == 0:
            n_parties = pop_size * rollouts_per_theta
            print(
                f"[iter {it + 1}/{iterations}] полная оценка стартового набора θ: "
                f"{pop_size} θ × {rollouts_per_theta} = {n_parties} партий (workers={wn})",
                flush=True,
            )
            last_results = evaluate_population(
                population,
                max_steps=max_steps,
                rollouts_per_theta=rollouts_per_theta,
                base_seed=seed,
                iteration=it,
                workers=workers,
                progress_context=(it + 1, iterations),
            )
            for theta, score in last_results:
                score_cache[_theta_key(theta)] = score
        else:
            to_sim_keys: set[tuple[float, ...]] = set()
            to_sim: list[Theta] = []
            for t in population:
                k = _theta_key(t)
                if k in prev_llm_keys and k not in to_sim_keys:
                    to_sim_keys.add(k)
                    to_sim.append(t)
            n_parties = len(to_sim) * rollouts_per_theta
            if to_sim:
                print(
                    f"[iter {it + 1}/{iterations}] симуляция только для θ от модели: "
                    f"{len(to_sim)} уникальных векторов × {rollouts_per_theta} = {n_parties} партий "
                    f"(остальные {len(population) - len(to_sim)} θ набора — из кэша, workers={wn})",
                    flush=True,
                )
                sim_rows = evaluate_population(
                    to_sim,
                    max_steps=max_steps,
                    rollouts_per_theta=rollouts_per_theta,
                    base_seed=seed,
                    iteration=it,
                    workers=workers,
                    progress_context=(it + 1, iterations) if n_parties >= 800 else None,
                )
                for theta, score in sim_rows:
                    score_cache[_theta_key(theta)] = score
            else:
                print(
                    f"[iter {it + 1}/{iterations}] без новых θ от модели — все {len(population)} из кэша (0 партий)",
                    flush=True,
                )
            missing = [t for t in population if _theta_key(t) not in score_cache]
            if missing:
                msg = f"В кэше нет оценки для {len(missing)} θ; это ошибка логики (ключи не из прошлого шага)."
                raise RuntimeError(msg)
            last_results = [(t, score_cache[_theta_key(t)]) for t in population]

        mean_s = sum(s for _, s in last_results) / len(last_results)
        mean_history.append(mean_s)
        plateau_break = (
            len(mean_history) == PLATEAU_WINDOW
            and max(mean_history) - min(mean_history) <= PLATEAU_SPREAD_MAX
        )
        for theta, score in last_results:
            traj_buffer.append((it + 1, theta, score))
        if len(traj_buffer) > trajectory_max_raw:
            del traj_buffer[:-trajectory_max_raw]

        payload = build_llm_user_payload(
            last_results,
            top_k=TOP_K,
            worst_k=worst_k,
            iteration=it + 1,
            total_iterations=iterations,
            n_llm_candidates=n_llm_candidates,
            plateau_break=plateau_break,
            trajectory_entries=list(traj_buffer),
            trajectory_max_send=trajectory_max_send,
        )
        user_msg = user_message_content(payload, n_candidates=n_llm_candidates)
        llm_temp = 0.92 if plateau_break else 0.7
        hypothesis_note = ""
        print(f"  [iter {it + 1}/{iterations}] запрос к LLM (таймаут {llm_timeout_sec:.0f} с)...", flush=True)
        try:
            response = ask_llm(
                user_msg,
                num_candidates=n_llm_candidates,
                temperature=llm_temp,
                timeout_sec=llm_timeout_sec,
            )
            new_candidates, hypothesis_note = parse_llm_step(response, max_keep=n_llm_candidates)
        except (RuntimeError, json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[iter {it + 1}] ошибка ответа модели: {e}, следующий набор θ только из прошлой оценки")
            new_candidates = []
            hypothesis_note = ""
        except Exception as e:
            print(
                f"[iter {it + 1}] сбой LLM/API ({type(e).__name__}): {e}; "
                "следующий набор θ только из прошлой оценки (проверь сеть и --llm-timeout)",
            )
            new_candidates = []
            hypothesis_note = ""

        prev_llm_keys = {_theta_key(t) for t in new_candidates}
        population = build_next_population(last_results, new_candidates, pop_size)
        best_t, best_j = max(last_results, key=lambda x: x[1])
        worst_t, worst_j = min(last_results, key=lambda x: x[1])
        sched = payload.get("schedule")
        phase = sched.get("phase", "?") if isinstance(sched, dict) else "?"
        extra = (
            f" | плато: разброс mean(J) за {PLATEAU_WINDOW} ит. ≤ {PLATEAU_SPREAD_MAX} → plateau_break"
            if plateau_break
            else ""
        )
        print(f"[iter {it + 1}/{iterations}] phase={phase}{extra}")
        ot = payload.get("optimization_trajectory")
        if isinstance(ot, list):
            print(
                f"            траектория в промпт: {len(ot)} точек (буфер {len(traj_buffer)}, "
                f"J по возрастанию после прорежки)",
            )
        if hypothesis_note:
            print(f"            гипотеза (hypothesis_note): {hypothesis_note}")
        print(
            f"            поле {FIELD_SIZE[0]}×{FIELD_SIZE[1]}, до {max_steps} шагов за партию; "
            f"оценка варианта — среднее J по {rollouts_per_theta} независимым прогонам",
        )
        print(f"            среднее J по активным θ ({pop_size} вариантов): {mean_s:.6f}")
        print(f"            лучший вариант:  J={best_j:.6f}  θ={_fmt_theta(best_t)}")
        print(f"            худший вариант: J={worst_j:.6f}  θ={_fmt_theta(worst_t)}")
        if new_candidates:
            shown = ", ".join(_fmt_theta(t, decimals=3) for t in new_candidates)
            print(f"            ответ модели (уникальные θ, {len(new_candidates)}/{n_llm_candidates} шт.): {shown}")
        print()

    return last_results


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM black-box meta-optimizer для весов политики Snake")
    parser.add_argument("--iterations", type=int, default=OPT_ITERATIONS)
    parser.add_argument("--steps", type=int, default=GAME_STEPS)
    parser.add_argument("--seed", type=int, required=True, help="Зерно RNG для стартовой популяции и партий (обязательно)")
    parser.add_argument("--worst-k", type=int, default=WORST_K, help="Сколько худших (θ, score) отдавать в LLM")
    parser.add_argument(
        "--rollouts",
        type=int,
        default=ROLLOUTS_PER_THETA,
        help="Сколько партий на один набор весов (оценка = среднее J)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Параллельных процессов для партий (0 = число ядер, 1 = без параллелизма)",
    )
    parser.add_argument(
        "--llm-candidates",
        type=int,
        default=NUM_LLM_CANDIDATES,
        metavar="N",
        help="Сколько разных θ за запрос к модели и сколько векторов одновременно ведём в оптимизации",
    )
    parser.add_argument(
        "--trajectory-raw",
        type=int,
        default=TRAJECTORY_MAX_RAW,
        metavar="N",
        help="Макс. точек (θ, J) в буфере траектории по всем прошлым итерациям",
    )
    parser.add_argument(
        "--trajectory-send",
        type=int,
        default=TRAJECTORY_MAX_SEND,
        metavar="N",
        help="Сколько точек траектории класть в JSON после сортировки по score (возр.)",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=DEFAULT_LLM_TIMEOUT_SEC,
        metavar="SEC",
        help="Таймаут HTTP к API LLM (сек.); при 0 — из env LLM_TIMEOUT_SEC или 240",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        metavar="DIR",
        help="Каталог для лог-файла (имя: optimization_seed_<seed>_<timestamp>.log)",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"optimization_seed_{args.seed}_{ts}.log"
    log_fp = log_path.open("w", encoding="utf-8")
    orig_out = sys.stdout
    orig_err = sys.stderr
    try:
        sys.stdout = _TeeTextIO(orig_out, log_fp)
        sys.stderr = _TeeTextIO(orig_err, log_fp)
        print(f"[log] весь вывод дублируется в файл: {log_path.resolve()}", flush=True)
        _run_inner(args)
    finally:
        sys.stdout = orig_out
        sys.stderr = orig_err
        log_fp.close()


def _run_inner(args: argparse.Namespace) -> None:
    llm_to = float(os.environ.get("LLM_TIMEOUT_SEC", str(DEFAULT_LLM_TIMEOUT_SEC))) if args.llm_timeout <= 0 else args.llm_timeout
    n_llm = max(1, min(args.llm_candidates, 32))

    results = llm_optimizer(
        seed=args.seed,
        iterations=args.iterations,
        pop_size=None,
        max_steps=args.steps,
        worst_k=args.worst_k,
        rollouts_per_theta=args.rollouts,
        workers=args.workers,
        n_llm_candidates=n_llm,
        trajectory_max_raw=max(20, args.trajectory_raw),
        trajectory_max_send=max(8, min(args.trajectory_send, 200)),
        llm_timeout_sec=max(30.0, llm_to),
    )
    best_theta, best_score = max(results, key=lambda x: x[1])
    worst_theta, worst_score = min(results, key=lambda x: x[1])
    pop_mean = sum(s for _, s in results) / len(results) if results else 0.0
    print("--- итог последней итерации ---")
    print(
        f"Среднее J по активным θ: {pop_mean:.6f} "
        f"(у каждого варианта J — среднее за {args.rollouts} прогонов)",
    )
    print(f"Лучший вариант:  J={best_score:.6f}  θ={_fmt_theta(best_theta)}")
    print(f"Худший вариант:  J={worst_score:.6f}  θ={_fmt_theta(worst_theta)}")
    print("(порядок в θ: w_food, w_danger, w_space, w_wall)")


if __name__ == "__main__":
    main()
