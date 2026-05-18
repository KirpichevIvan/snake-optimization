from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from player.rollout import RolloutPacked, simulate_rollout, simulate_rollout_packed

Theta = tuple[float, float, float, float]
_WEIGHT_BOUNDS: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]] = (
    (-2.0, 4.0),
    (-4.0, 1.0),
    (-1.0, 3.0),
    (-3.0, 2.0),
)


def _fmt_theta(theta: Theta, *, decimals: int = 4) -> str:
    inner = ", ".join(f"{x:.{decimals}f}" for x in theta)
    return f"[{inner}]"


def _resolve_rollout_workers(requested: int, population_size: int) -> int:
    """Resolve rollout worker count: 0 means auto, 1 means in-process, 2+ caps the pool."""
    if population_size < 1:
        return 1
    if requested <= 0:
        cpu = os.cpu_count() or 4
        return max(1, min(int(cpu), int(population_size), 32))
    return max(1, min(int(requested), int(population_size)))


@dataclass(frozen=True)
class MfaHyperparams:
    """Гиперпараметры Modified Firefly Algorithm для поиска ``Theta``."""

    training_seed: int
    population_size: int = 50
    generations: int = 20
    max_steps: int = 1000
    beta0: float = 0.8
    gamma: float = 0.1
    alpha0: float = 2.0
    p: float = 0.1
    c: float = 4.0
    mu0: float = 0.123
    rollout_workers: int = 0
    stretch: bool = False
    stretch_chunk: int = 100
    randomize_game_seed_per_generation: bool = False


def _random_theta(rng: random.Random) -> Theta:
    """Случайная точка MFA в тех же диапазонах, что исторически использовались для политики."""
    return (
        rng.uniform(*_WEIGHT_BOUNDS[0]),
        rng.uniform(*_WEIGHT_BOUNDS[1]),
        rng.uniform(*_WEIGHT_BOUNDS[2]),
        rng.uniform(*_WEIGHT_BOUNDS[3]),
    )


def _clip_theta(theta: Theta) -> Theta:
    """Обрезает координаты ``theta`` по локальным границам поиска MFA."""
    return tuple(
        min(max(float(theta[i]), _WEIGHT_BOUNDS[i][0]), _WEIGHT_BOUNDS[i][1]) for i in range(4)
    )


def _distance_sq(a: Theta, b: Theta) -> float:
    """Квадрат евклидова расстояния между двумя светлячками в 4D-пространстве весов."""
    return sum((float(a[i]) - float(b[i])) ** 2 for i in range(4))


class MfaLearn:
    """
    Обучение весов политики Modified Firefly Algorithm (MFA).

    Интерфейс совместим с ``GeneticLearn.run``: тот же формат ``progress_callback``,
    аналогичные ``meta.json``, ``results.csv`` и ``best.json``.
    """

    def __init__(
        self,
        hyperparams: MfaHyperparams,
        *,
        field_size: tuple[int, int] = (10, 10),
        snake_game_seed: int,
        results_dir: str | Path,
        verbose: bool = True,
    ) -> None:
        self._hp = hyperparams
        self._field_size = field_size
        self._snake_game_seed = int(snake_game_seed)
        self._results_dir = Path(results_dir)
        self._verbose = verbose

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(msg, flush=True)

    def _rollout_pack(self, theta: Theta, rollout_seed: int) -> RolloutPacked:
        hp = self._hp
        if hp.stretch:
            return (
                theta,
                hp.max_steps,
                self._field_size,
                int(rollout_seed),
                True,
                hp.stretch_chunk,
            )
        return (theta, hp.max_steps, self._field_size, int(rollout_seed))

    def _rollout_seed_for_generation(self, _gen_index: int, rng: random.Random) -> int:
        hp = self._hp
        if hp.randomize_game_seed_per_generation:
            return int(rng.randrange(0, 0x8000_0000))
        return self._snake_game_seed

    @staticmethod
    def _alpha(iteration: int, alpha0: float) -> float:
        """Адаптивный шаг MFA: alpha(t) = (e - (1 + 1/t)^t) * alpha0."""
        if iteration <= 1:
            return float(alpha0)
        return float((math.e - (1.0 + 1.0 / iteration) ** iteration) * alpha0)

    @staticmethod
    def _q(iteration: int, generations: int, population_size: int, p: float) -> int:
        """Число худших светлячков для opposition-based chaotic шага."""
        q = math.floor(float(p) * population_size * (1.0 - iteration / generations)) + 1
        return min(max(1, int(q)), population_size)

    @staticmethod
    def _validate_mu0(mu0: float) -> None:
        forbidden = (0.0, 0.25, 0.5, 0.75, 1.0)
        if not 0.0 < float(mu0) < 1.0 or any(math.isclose(float(mu0), x) for x in forbidden):
            msg = "mu0 должен быть в (0, 1) и не равен 0, 0.25, 0.5, 0.75 или 1.0"
            raise ValueError(msg)

    def _move_population(
        self,
        population: list[Theta],
        fitness: list[float],
        *,
        iteration: int,
        mu: float,
        rng: random.Random,
    ) -> tuple[list[Theta], float, int, float]:
        """Один шаг MFA после оценки поколения: сортировка, движение хороших и хаос для худших."""
        hp = self._hp
        ranked = sorted(range(len(population)), key=lambda i: fitness[i], reverse=True)
        sorted_population = [population[i] for i in ranked]
        sorted_fitness = [fitness[i] for i in ranked]

        alpha = self._alpha(iteration, hp.alpha0)
        q = self._q(iteration, hp.generations, hp.population_size, hp.p)
        good_end = max(0, hp.population_size - q)

        moved: list[Theta] = [tuple(theta) for theta in sorted_population]

        for i in range(good_end):
            theta_i = moved[i]
            for j in range(hp.population_size):
                if sorted_fitness[j] <= sorted_fitness[i]:
                    continue
                theta_j = moved[j]
                r2 = _distance_sq(theta_i, theta_j)
                beta = hp.beta0 * math.exp(-hp.gamma * r2)
                theta_i = _clip_theta(
                    tuple(
                        theta_i[k]
                        + beta * (theta_j[k] - theta_i[k])
                        + alpha * (rng.random() - 0.5)
                        for k in range(4)
                    )
                )
            moved[i] = theta_i

        mu = hp.c * mu * (1.0 - mu)
        for poor in range(good_end, hp.population_size):
            moved[poor] = _clip_theta(
                tuple(mu * (_WEIGHT_BOUNDS[k][1] - _WEIGHT_BOUNDS[k][0]) + _WEIGHT_BOUNDS[k][0] for k in range(4))
            )

        return moved, mu, q, alpha

    def run(
        self,
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        interrupt_check: Callable[[], bool] | None = None,
    ) -> tuple[Theta, float, bool]:
        """
        Запускает MFA и возвращает ``(best_theta, best_j, aborted)``.

        ``progress_callback`` использует те же фазы, что и ``GeneticLearn``:
        ``snake_evaluated`` и ``generation_done``.
        """
        hp = self._hp
        if hp.population_size < 2:
            msg = "population_size должен быть не меньше 2"
            raise ValueError(msg)
        if hp.generations < 1:
            msg = "generations должен быть >= 1"
            raise ValueError(msg)
        if hp.max_steps < 1:
            msg = "max_steps должен быть >= 1"
            raise ValueError(msg)
        if not 0.0 < hp.beta0 < 1.0:
            msg = "beta0 должен быть в интервале (0, 1)"
            raise ValueError(msg)
        if hp.gamma < 0.0:
            msg = "gamma должен быть >= 0"
            raise ValueError(msg)
        if hp.alpha0 < 0.0:
            msg = "alpha0 должен быть >= 0"
            raise ValueError(msg)
        if not 0.0 <= hp.p <= 1.0:
            msg = "p должен быть в интервале [0, 1]"
            raise ValueError(msg)
        if hp.stretch_chunk < 1:
            msg = "stretch_chunk должен быть >= 1"
            raise ValueError(msg)
        self._validate_mu0(hp.mu0)

        self._results_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(hp.training_seed)
        mu = float(hp.mu0)

        stretch_note = (
            f"вытягивание: блоки по {hp.stretch_chunk} шагов, пока идёт игра и съедено ≥1 яблоко за блок"
            if hp.stretch
            else "вытягивание: выкл."
        )
        seed_note = (
            "сид игры: новый на каждое поколение (RNG от training_seed)"
            if hp.randomize_game_seed_per_generation
            else f"сид игры (rollout)={self._snake_game_seed} на все поколения"
        )
        self._log(
            "MFA обучение: поле "
            f"{self._field_size[0]}x{self._field_size[1]}, базово до {hp.max_steps} шагов за эпизод "
            f"({stretch_note}); популяция {hp.population_size}, поколений {hp.generations}; "
            f"{seed_note}; сид обучения={hp.training_seed}."
        )
        self._log(
            f"Параметры MFA: beta0={hp.beta0:.6g}, gamma={hp.gamma:.6g}, "
            f"alpha0={hp.alpha0:.6g}, p={hp.p:.6g}, c={hp.c:.6g}, mu0={hp.mu0:.6g}."
        )
        self._log(f"Каталог результатов: {self._results_dir.resolve()}")

        meta: dict[str, Any] = {
            "version": 1,
            "algorithm": "modified_firefly_algorithm",
            "snake_game_seed": self._snake_game_seed,
            "field_height": self._field_size[0],
            "field_width": self._field_size[1],
            "weight_bounds": [[float(lo), float(hi)] for lo, hi in _WEIGHT_BOUNDS],
            "mfa_hyperparams": asdict(hp),
        }
        meta_path = self._results_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self._log(f"Записан {meta_path.name} (гиперпараметры, границы и сиды).")

        csv_path = self._results_dir / "results.csv"
        n_workers = _resolve_rollout_workers(hp.rollout_workers, hp.population_size)
        self._log(
            f"Оценка rollout: processes={n_workers} "
            f"(1 — по одной особи в главном процессе с прерыванием между шагами; "
            f">1 — пул процессов, остановка только между поколениями)."
        )

        with csv_path.open("w", newline="", encoding="utf-8") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(
                [
                    "w_food",
                    "w_danger",
                    "w_space",
                    "w_wall",
                    "J",
                    "steps",
                    "snake_game_seed",
                    "gen_number",
                    "snake_number",
                ]
            )
            generation_seeds: list[int] = []

            self._log("Инициализация популяции MFA (случайные веса по локальным диапазонам _WEIGHT_BOUNDS).")
            population: list[Theta] = [_random_theta(rng) for _ in range(hp.population_size)]
            best_ever: Theta = population[0]
            best_j_ever = float("-inf")
            aborted = False

            if n_workers > 1:
                pool_cm = ProcessPoolExecutor(max_workers=n_workers)
            else:
                pool_cm = nullcontext()

            with pool_cm as pool:
                for gen in range(hp.generations):
                    iteration = gen + 1
                    if interrupt_check is not None and interrupt_check():
                        self._log("Остановка: запрос до начала оценки поколения.")
                        aborted = True
                        break

                    gen_rollout_seed = self._rollout_seed_for_generation(gen, rng)
                    generation_seeds.append(gen_rollout_seed)
                    self._log("")
                    self._log(
                        f"--- Поколение {iteration} / {hp.generations}: оценка пригодности (rollout), "
                        f"snake_game_seed={gen_rollout_seed} ---"
                    )

                    fitness: list[float] = []
                    steps_vals: list[int] = []
                    generation_started_at = time.perf_counter()
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "phase": "generation_started",
                                "generation": iteration,
                                "generations_total": hp.generations,
                                "snakes_total": hp.population_size,
                                "snake_game_seed": gen_rollout_seed,
                                "rollout_workers": n_workers,
                            }
                        )
                    if n_workers > 1 and pool is not None:
                        packs: list[RolloutPacked] = [
                            self._rollout_pack(theta, gen_rollout_seed) for theta in population
                        ]
                        futures = {
                            pool.submit(simulate_rollout_packed, pack): (snake_idx, population[snake_idx])
                            for snake_idx, pack in enumerate(packs)
                        }
                        results_by_index: list[tuple[float, int] | None] = [None] * len(population)
                        done_count = 0
                        progress_iter = as_completed(futures)
                        if self._verbose:
                            progress_iter = tqdm(
                                progress_iter,
                                total=len(futures),
                                desc=f"Поколение {iteration}/{hp.generations}",
                                unit="особь",
                            )
                        for fut in progress_iter:
                            snake_idx, _theta = futures[fut]
                            j, steps = fut.result()
                            results_by_index[snake_idx] = (float(j), int(steps))
                            done_count += 1
                            if progress_callback is not None:
                                progress_callback(
                                    {
                                        "phase": "snake_evaluated",
                                        "generation": iteration,
                                        "generations_total": hp.generations,
                                        "snake_index": done_count,
                                        "snake_number": snake_idx + 1,
                                        "snakes_total": hp.population_size,
                                        "steps": int(steps),
                                        "snake_game_seed": gen_rollout_seed,
                                        "elapsed_sec": time.perf_counter() - generation_started_at,
                                    }
                                )
                            if interrupt_check is not None and interrupt_check():
                                self._log(
                                    "Остановка: прервана параллельная оценка поколения "
                                    "(ожидание уже запущенных rollout до закрытия пула)."
                                )
                                aborted = True
                                for pending in futures:
                                    pending.cancel()
                                break
                        if aborted:
                            break
                        for snake_idx, result in enumerate(results_by_index):
                            if result is None:
                                continue
                            theta = population[snake_idx]
                            j, steps = result
                            fitness.append(float(j))
                            steps_vals.append(int(steps))
                            writer.writerow(
                                [
                                    f"{theta[0]:.10g}",
                                    f"{theta[1]:.10g}",
                                    f"{theta[2]:.10g}",
                                    f"{theta[3]:.10g}",
                                    f"{j:.10g}",
                                    int(steps),
                                    gen_rollout_seed,
                                    iteration,
                                    snake_idx + 1,
                                ]
                            )
                    else:
                        eval_iter = enumerate(population)
                        if self._verbose:
                            eval_iter = tqdm(
                                eval_iter,
                                total=len(population),
                                desc=f"Поколение {iteration}/{hp.generations}",
                                unit="особь",
                            )
                        for snake_idx, theta in eval_iter:
                            j, steps = simulate_rollout(
                                theta,
                                max_steps=hp.max_steps,
                                seed=gen_rollout_seed,
                                field_size=self._field_size,
                                stretch=hp.stretch,
                                stretch_chunk=hp.stretch_chunk,
                            )
                            fitness.append(float(j))
                            steps_vals.append(int(steps))
                            writer.writerow(
                                [
                                    f"{theta[0]:.10g}",
                                    f"{theta[1]:.10g}",
                                    f"{theta[2]:.10g}",
                                    f"{theta[3]:.10g}",
                                    f"{j:.10g}",
                                    int(steps),
                                    gen_rollout_seed,
                                    iteration,
                                    snake_idx + 1,
                                ]
                            )
                            if progress_callback is not None:
                                progress_callback(
                                    {
                                        "phase": "snake_evaluated",
                                        "generation": iteration,
                                        "generations_total": hp.generations,
                                        "snake_index": snake_idx + 1,
                                        "snakes_total": hp.population_size,
                                        "steps": int(steps),
                                        "snake_game_seed": gen_rollout_seed,
                                        "elapsed_sec": time.perf_counter() - generation_started_at,
                                    }
                                )
                            if interrupt_check is not None and interrupt_check():
                                self._log(
                                    "Остановка: прервана оценка поколения (частичное поколение сохранено в CSV)."
                                )
                                aborted = True
                                break
                    f_csv.flush()

                    if len(fitness) < hp.population_size:
                        for i, j in enumerate(fitness):
                            if j > best_j_ever:
                                best_j_ever = j
                                best_ever = population[i]
                        break

                    gen_best_i = max(range(len(fitness)), key=lambda i: fitness[i])
                    gen_best_j = fitness[gen_best_i]
                    steps_best = steps_vals[gen_best_i]
                    improved = gen_best_j > best_j_ever
                    if improved:
                        best_j_ever = gen_best_j
                        best_ever = population[gen_best_i]

                    j_min = min(fitness)
                    j_max = max(fitness)
                    j_mean = statistics.fmean(fitness)
                    steps_min = min(steps_vals)
                    steps_max = max(steps_vals)
                    steps_mean = statistics.fmean(steps_vals)

                    self._log(
                        f"Поколение {iteration}: J min={j_min:.4f}, mean={j_mean:.4f}, max={j_max:.4f}; "
                        f"шаги min={steps_min}, mean={steps_mean:.1f}, max={steps_max}; "
                        f"лучшая особь J={gen_best_j:.4f}, шагов={steps_best}, "
                        f"theta={_fmt_theta(population[gen_best_i])}."
                    )
                    if improved:
                        self._log("  -> новый лучший результат за весь прогон (best_ever обновлён).")
                    else:
                        self._log(f"  -> глобальный лучший J за прогон по-прежнему {best_j_ever:.4f}.")

                    if progress_callback is not None:
                        progress_callback(
                            {
                                "phase": "generation_done",
                                "generation": iteration,
                                "generations_total": hp.generations,
                                "j_min": j_min,
                                "j_mean": j_mean,
                                "j_max": j_max,
                                "best_j_generation": gen_best_j,
                                "best_theta_generation": tuple(population[gen_best_i]),
                                "best_j_ever": best_j_ever,
                                "best_theta_ever": tuple(best_ever),
                                "steps_min": steps_min,
                                "steps_mean": steps_mean,
                                "steps_max": steps_max,
                                "steps_best_generation": steps_best,
                                "snake_game_seed": gen_rollout_seed,
                                "elapsed_sec": time.perf_counter() - generation_started_at,
                            }
                        )

                    if iteration == hp.generations:
                        break

                    if interrupt_check is not None and interrupt_check():
                        self._log("Остановка: после полного поколения (движение MFA не выполняется).")
                        aborted = True
                        break

                    population, mu, q, alpha = self._move_population(
                        population,
                        fitness,
                        iteration=iteration,
                        mu=mu,
                        rng=rng,
                    )
                    self._log(
                        f"MFA движение: alpha={alpha:.6g}, q={q}, mu={mu:.6g}; "
                        f"следующее поколение собрано, снова {hp.population_size} особей."
                    )

        if best_j_ever == float("-inf"):
            best_j_ever, _ = simulate_rollout(
                best_ever,
                max_steps=hp.max_steps,
                seed=self._snake_game_seed,
                field_size=self._field_size,
                stretch=hp.stretch,
                stretch_chunk=hp.stretch_chunk,
            )

        meta_done = json.loads(meta_path.read_text(encoding="utf-8"))
        meta_done["run_aborted"] = bool(aborted)
        meta_done["final_mu"] = float(mu)
        if hp.randomize_game_seed_per_generation:
            meta_done["generation_rollout_seeds"] = generation_seeds
        meta_path.write_text(json.dumps(meta_done, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        best_payload = {
            "w_food": best_ever[0],
            "w_danger": best_ever[1],
            "w_space": best_ever[2],
            "w_wall": best_ever[3],
            "J": best_j_ever,
            "aborted": bool(aborted),
        }
        best_path = self._results_dir / "best.json"
        best_path.write_text(json.dumps(best_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        self._log("")
        self._log("--- MFA обучение завершено ---" + (" (остановлено)" if aborted else ""))
        self._log(f"Записан {csv_path.name} (история оценок по поколениям).")
        self._log(f"Записан {best_path.name}: лучшее J={best_j_ever:.6f}, theta={_fmt_theta(best_ever, decimals=6)}.")

        return best_ever, best_j_ever, aborted


__all__ = ["MfaHyperparams", "MfaLearn", "_clip_theta", "_distance_sq"]