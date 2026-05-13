"""
Пример запуска генетического обучения весов политики (пакет ``genetic``).

Результаты: ``results.csv``, ``meta.json``, ``best.json`` в каталоге ``--output-dir``.

Запуск из корня проекта::

    uv run python run_genetic_optimization.py --training-seed 1 --snake-game-seed 42

Параллельные rollout (загрузка CPU), см. ``--rollout-workers``::

    uv run python run_genetic_optimization.py --training-seed 1 --rollout-workers 0

``0`` = авто (число процессов по ядрам и размеру популяции), ``1`` = только главный процесс.

См. также программный пример в ``main()`` ниже.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from genetic import GeneticHyperparams, GeneticLearn
from genetic.learn import _resolve_rollout_workers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Генетический алгоритм для весов политики Snake",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  uv run python run_genetic_optimization.py --rollout-workers 0\n"
            "  uv run python run_genetic_optimization.py --rollout-workers 16 --population-size 80\n"
            "  uv run python run_genetic_optimization.py --rollout_workers 1\n"
            "\n"
            "Синонимы: --rollout-workers и --rollout_workers."
        ),
    )
    parser.add_argument("--training-seed", type=int, default=42, help="Сид RNG операторов ГА")
    parser.add_argument("--snake-game-seed", type=int, default=42, help="Сид rollout (поле/игра), фиксирован на весь прогон")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        metavar="DIR",
        help="Каталог результатов (пусто: out/genetic_training_<training-seed>_<timestamp>)",
    )
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=200, help="Максимум шагов змейки за эпизод")
    parser.add_argument("--field-height", type=int, default=10)
    parser.add_argument("--field-width", type=int, default=10)
    parser.add_argument("--crossover-prob", type=float, default=0.70)
    parser.add_argument("--mutation-prob", type=float, default=0.15)
    parser.add_argument("--mutation-sigma", type=float, default=0.25)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument(
        "--rollout-workers",
        "--rollout_workers",
        type=int,
        default=0,
        metavar="N",
        dest="rollout_workers",
        help=(
            "Параллельные rollout: 0 = авто (ядра и размер популяции); 1 = только главный процесс; "
            "2+ = не больше N процессов (и не больше размера популяции)."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Без подробного лога и tqdm в GeneticLearn (краткий итог в конце скрипта всё равно печатается)",
    )
    args = parser.parse_args()

    if args.output_dir:
        out = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("out") / f"genetic_training_{args.training_seed}_{ts}"

    hp = GeneticHyperparams(
        training_seed=args.training_seed,
        population_size=max(2, args.population_size),
        generations=max(1, args.generations),
        max_steps=max(1, args.max_steps),
        crossover_prob=args.crossover_prob,
        mutation_prob=args.mutation_prob,
        mutation_sigma=args.mutation_sigma,
        elite_count=max(0, args.elite_count),
        tournament_size=max(1, args.tournament_size),
        rollout_workers=max(0, args.rollout_workers),
    )
    field_size = (args.field_height, args.field_width)

    eff_workers = _resolve_rollout_workers(hp.rollout_workers, hp.population_size)
    if not args.quiet:
        print(
            f"Параллельные rollout: эффективно {eff_workers} процесс(ов) "
            f"(параметр --rollout-workers={hp.rollout_workers}).",
            flush=True,
        )

    learn = GeneticLearn(
        hp,
        field_size=field_size,
        snake_game_seed=args.snake_game_seed,
        results_dir=out,
        verbose=not args.quiet,
    )
    best_theta, best_j, _aborted = learn.run()

    wf, wd, ws, ww = best_theta
    print(f"Каталог результатов: {out.resolve()}", flush=True)
    print(f"Лучшее J: {best_j:.6f}", flush=True)
    print(f"Лучшие веса: w_food={wf:.6f}, w_danger={wd:.6f}, w_space={ws:.6f}, w_wall={ww:.6f}", flush=True)


if __name__ == "__main__":
      # hp = GeneticHyperparams(training_seed=1)
      # GeneticLearn(
      #     hp,
      #     field_size=(10, 10),
      #     snake_game_seed=42,
      #     results_dir=Path("logs/genetic_example"),
      #     verbose=True,
      # ).run()
    main()
