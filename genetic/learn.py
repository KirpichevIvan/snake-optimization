from __future__ import annotations

import csv
import json
import os
import random
import statistics
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from player.policy import Theta, random_theta
from player.rollout import RolloutPacked, simulate_rollout, simulate_rollout_packed

# Границы как в ``random_theta`` (policy.py)
_WEIGHT_BOUNDS: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]] = (
    (-2.0, 4.0),
    (-4.0, 1.0),
    (-1.0, 3.0),
    (-3.0, 2.0),
)


def _clip_theta(t: Theta) -> Theta:
    return tuple(
        min(max(float(t[i]), _WEIGHT_BOUNDS[i][0]), _WEIGHT_BOUNDS[i][1]) for i in range(4)
    )


def _fmt_theta(theta: Theta, *, decimals: int = 4) -> str:
    inner = ", ".join(f"{x:.{decimals}f}" for x in theta)
    return f"[{inner}]"


def _resolve_rollout_workers(requested: int, population_size: int) -> int:
    """
    ``requested``: ``0`` — авто (число процессов не больше числа ядер и размера популяции),
    ``1`` — строго последовательный rollout по особям, ``>=2`` — верхняя граница пула.
    """
    if population_size < 1:
        return 1
    if requested <= 0:
        cpu = os.cpu_count() or 4
        return max(1, min(int(cpu), int(population_size), 32))
    return max(1, min(int(requested), int(population_size)))


def _initial_population(hp: GeneticHyperparams, rng: random.Random) -> list[Theta]:
    """Поколение 0: либо ``random_theta``, либо случайные возмущения вокруг ``initial_weights`` с клипом."""
    if hp.initial_weights is None:
        return [random_theta(rng) for _ in range(hp.population_size)]
    base = hp.initial_weights
    if hp.initial_spread_per_weight is not None:
        sigs = hp.initial_spread_per_weight
        if len(sigs) != 4:
            msg = "initial_spread_per_weight должен быть кортежем из четырёх положительных чисел"
            raise ValueError(msg)
        for i, s in enumerate(sigs):
            if float(s) <= 0.0:
                msg = f"initial_spread_per_weight[{i}] должен быть > 0"
                raise ValueError(msg)
        return [
            _clip_theta(
                tuple(base[i] + rng.gauss(0.0, float(sigs[i])) for i in range(4)),
            )
            for _ in range(hp.population_size)
        ]
    if hp.initial_spread <= 0.0:
        msg = "initial_spread должен быть > 0 при заданных initial_weights и без initial_spread_per_weight"
        raise ValueError(msg)
    sig = float(hp.initial_spread)
    return [
        _clip_theta(tuple(base[i] + rng.gauss(0.0, sig) for i in range(4)))
        for _ in range(hp.population_size)
    ]


@dataclass(frozen=True)
class GeneticHyperparams:
    """Гиперпараметры ГА. Сид поля игры задаётся отдельно в ``GeneticLearn`` (``snake_game_seed``)."""

    training_seed: int
    population_size: int = 50
    generations: int = 30
    max_steps: int = 200
    crossover_prob: float = 0.7
    mutation_prob: float = 0.15
    mutation_sigma: float = 0.25
    elite_count: int = 2
    tournament_size: int = 3
    initial_weights: tuple[float, float, float, float] | None = None
    initial_spread: float = 0.35
    initial_spread_per_weight: tuple[float, float, float, float] | None = None
    rollout_workers: int = 0
    stretch: bool = False
    stretch_chunk: int = 100
    randomize_game_seed_per_generation: bool = False


class GeneticLearn:
    """
    Обучение весов политики генетическим алгоритмом.

    - ``snake_game_seed``: сид rollout для всех особей поколения; при
      ``randomize_game_seed_per_generation`` — базовый сид в meta, на каждое поколение
      новое значение (детерминированно от ``training_seed``).
    - ``training_seed``: зерно для RNG операторов ГА (инициализация, отбор, кроссовер, мутация).
    - ``verbose``: подробный вывод в stdout и прогресс-бар tqdm при оценке популяции.
    - Поколение 0: см. ``GeneticHyperparams.initial_weights`` / ``initial_spread`` (или ``random_theta``).
    - ``GeneticHyperparams.rollout_workers``: при ``>1`` оценка поколения идёт в ``ProcessPoolExecutor``;
      ``interrupt_check`` учитывается только **между** поколениями. При ``1`` — последовательный rollout,
      прерывание возможно после каждой особи.
    """

    def __init__(
        self,
        hyperparams: GeneticHyperparams,
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

    def run(
        self,
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        interrupt_check: Callable[[], bool] | None = None,
    ) -> tuple[Theta, float, bool]:
        """
        ``progress_callback`` вызывается из того же потока во время обучения (удобно для Streamlit):

        - ``{"phase": "snake_evaluated", "generation", "generations_total", "snake_index", "snakes_total"}``
        - ``{"phase": "generation_done", "generation", "generations_total", "j_min", "j_mean", "j_max",
          "best_j_generation", "best_theta_generation", "best_j_ever", "best_theta_ever"}``

        ``interrupt_check``: если при вызове возвращает ``True``, обучение корректно останавливается
        (история в ``results.csv`` и ``best.json`` сохраняются; возможна неполная последняя строка поколения).
        Возвращает ``(best_theta, best_j, aborted)``.
        """
        hp = self._hp
        if hp.population_size < 2:
            msg = "population_size должен быть не меньше 2"
            raise ValueError(msg)
        if hp.elite_count < 0 or hp.elite_count >= hp.population_size:
            msg = "elite_count должен быть в [0, population_size)"
            raise ValueError(msg)
        if hp.tournament_size < 1:
            msg = "tournament_size должен быть >= 1"
            raise ValueError(msg)
        if hp.stretch_chunk < 1:
            msg = "stretch_chunk должен быть >= 1"
            raise ValueError(msg)

        self._results_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(hp.training_seed)

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
            "Генетическое обучение: поле "
            f"{self._field_size[0]}x{self._field_size[1]}, базово до {hp.max_steps} шагов за эпизод "
            f"({stretch_note}); "
            f"популяция {hp.population_size}, поколений {hp.generations}; "
            f"{seed_note}; сид обучения (ГА)={hp.training_seed}."
        )
        self._log(f"Каталог результатов: {self._results_dir.resolve()}")

        meta: dict[str, Any] = {
            "version": 1,
            "snake_game_seed": self._snake_game_seed,
            "field_height": self._field_size[0],
            "field_width": self._field_size[1],
            "genetic_hyperparams": asdict(hp),
        }
        meta_path = self._results_dir / "meta.json"
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self._log(f"Записан {meta_path.name} (гиперпараметры и сиды).")

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

            if hp.initial_weights is None:
                self._log("Инициализация популяции (случайные веса по диапазонам random_theta).")
            elif hp.initial_spread_per_weight is not None:
                self._log(
                    "Инициализация популяции: центр "
                    f"{_fmt_theta(hp.initial_weights)}, гауссов разброс по осям sigma="
                    f"{_fmt_theta(hp.initial_spread_per_weight)}."
                )
            else:
                self._log(
                    "Инициализация популяции: центр "
                    f"{_fmt_theta(hp.initial_weights)}, гауссов разброс sigma={hp.initial_spread:.4g} по всем осям."
                )
            population: list[Theta] = _initial_population(hp, rng)
            best_ever: Theta = population[0]
            best_j_ever = float("-inf")
            aborted = False

            if n_workers > 1:
                pool_cm = ProcessPoolExecutor(max_workers=n_workers)
            else:
                pool_cm = nullcontext()

            with pool_cm as pool:
                for gen in range(hp.generations):
                    if interrupt_check is not None and interrupt_check():
                        self._log("Остановка: запрос до начала оценки поколения.")
                        aborted = True
                        break

                    gen_rollout_seed = self._rollout_seed_for_generation(gen, rng)
                    generation_seeds.append(gen_rollout_seed)
                    self._log("")
                    self._log(
                        f"--- Поколение {gen + 1} / {hp.generations}: оценка пригодности (rollout), "
                        f"snake_game_seed={gen_rollout_seed} ---"
                    )

                    fitness: list[float] = []
                    steps_vals: list[int] = []
                    if n_workers > 1 and pool is not None:
                        packs: list[RolloutPacked] = [
                            self._rollout_pack(theta, gen_rollout_seed) for theta in population
                        ]
                        chunksize = max(1, len(packs) // (n_workers * 8))
                        it = pool.map(simulate_rollout_packed, packs, chunksize=chunksize)
                        if self._verbose:
                            it = tqdm(
                                it,
                                total=len(packs),
                                desc=f"Поколение {gen + 1}/{hp.generations}",
                                unit="особь",
                            )
                        rollout_results = list(it)
                        for snake_idx, (theta, (j, steps)) in enumerate(
                            zip(population, rollout_results, strict=True)
                        ):
                            fitness.append(j)
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
                                    gen + 1,
                                    snake_idx + 1,
                                ]
                            )
                            if progress_callback is not None:
                                progress_callback(
                                    {
                                        "phase": "snake_evaluated",
                                        "generation": gen + 1,
                                        "generations_total": hp.generations,
                                        "snake_index": snake_idx + 1,
                                        "snakes_total": hp.population_size,
                                        "steps": int(steps),
                                        "snake_game_seed": gen_rollout_seed,
                                    }
                                )
                    else:
                        eval_iter = enumerate(population)
                        if self._verbose:
                            eval_iter = tqdm(
                                eval_iter,
                                total=len(population),
                                desc=f"Поколение {gen + 1}/{hp.generations}",
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
                            fitness.append(j)
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
                                    gen + 1,
                                    snake_idx + 1,
                                ]
                            )
                            if progress_callback is not None:
                                progress_callback(
                                    {
                                        "phase": "snake_evaluated",
                                        "generation": gen + 1,
                                        "generations_total": hp.generations,
                                        "snake_index": snake_idx + 1,
                                        "snakes_total": hp.population_size,
                                        "steps": int(steps),
                                        "snake_game_seed": gen_rollout_seed,
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
                        for i in range(len(fitness)):
                            if fitness[i] > best_j_ever:
                                best_j_ever = fitness[i]
                                best_ever = population[i]
                        break

                    gen_best_i = max(range(len(fitness)), key=lambda i: fitness[i])
                    gen_best_j = fitness[gen_best_i]
                    j_min = min(fitness)
                    j_max = max(fitness)
                    j_mean = statistics.mean(fitness)
                    steps_min = min(steps_vals)
                    steps_mean = statistics.mean(steps_vals)
                    steps_max = max(steps_vals)
                    steps_best = steps_vals[gen_best_i]
                    improved = gen_best_j > best_j_ever
                    if gen_best_j > best_j_ever:
                        best_j_ever = gen_best_j
                        best_ever = population[gen_best_i]

                    self._log(
                        f"Поколение {gen + 1}: J min={j_min:.4f}, mean={j_mean:.4f}, max={j_max:.4f}; "
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
                                "generation": gen + 1,
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
                            }
                        )

                    if gen + 1 == hp.generations:
                        break

                    if interrupt_check is not None and interrupt_check():
                        self._log("Остановка: после полного поколения (скрещивание не выполняется).")
                        aborted = True
                        break

                    self._log(
                        f"Формирование поколения {gen + 2}: элита {hp.elite_count} особей; "
                        f"остальные {hp.population_size - hp.elite_count} - турнир "
                        f"(k={hp.tournament_size}), кроссовер p={hp.crossover_prob:.2f}, "
                        f"мутация p={hp.mutation_prob:.2f}, sigma={hp.mutation_sigma:.3f}."
                    )
                    ranked = sorted(range(len(population)), key=lambda i: fitness[i], reverse=True)
                    elites = [population[i] for i in ranked[: hp.elite_count]]

                    next_pop: list[Theta] = [tuple(t) for t in elites]
                    while len(next_pop) < hp.population_size:
                        p1 = self._tournament(population, fitness, hp.tournament_size, rng)
                        p2 = self._tournament(population, fitness, hp.tournament_size, rng)
                        if rng.random() < hp.crossover_prob:
                            child = self._crossover(p1, p2, rng)
                        else:
                            child = p1 if rng.random() < 0.5 else p2
                        child = self._mutate(child, hp.mutation_prob, hp.mutation_sigma, rng)
                        next_pop.append(child)
                    population = next_pop
                    self._log(f"Следующее поколение собрано, снова {hp.population_size} особей.")

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
        best_path.write_text(
            json.dumps(best_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        self._log("")
        self._log("--- Обучение завершено ---" + (" (остановлено)" if aborted else ""))
        self._log(f"Записан {csv_path.name} (история оценок по поколениям).")
        self._log(f"Записан {best_path.name}: лучшее J={best_j_ever:.6f}, theta={_fmt_theta(best_ever, decimals=6)}.")

        return best_ever, best_j_ever, aborted

    @staticmethod
    def _tournament(pop: list[Theta], fitness: list[float], k: int, rng: random.Random) -> Theta:
        n = len(pop)
        k_eff = min(max(1, k), n)
        contenders = rng.sample(range(n), k=k_eff)
        best_i = max(contenders, key=lambda i: fitness[i])
        return pop[best_i]

    @staticmethod
    def _crossover(p1: Theta, p2: Theta, rng: random.Random) -> Theta:
        return tuple(
            (a := rng.random()) * p1[i] + (1.0 - a) * p2[i] for i in range(4)
        )

    @staticmethod
    def _mutate(theta: Theta, mutation_prob: float, sigma: float, rng: random.Random) -> Theta:
        out: list[float] = []
        for i in range(4):
            x = float(theta[i])
            if rng.random() < mutation_prob:
                x += rng.gauss(0.0, sigma)
            out.append(x)
        return _clip_theta((out[0], out[1], out[2], out[3]))


__all__ = ["GeneticHyperparams", "GeneticLearn"]
