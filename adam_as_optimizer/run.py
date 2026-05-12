from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO

from .optimizer import AdamState, adam_max_step, center_value_and_fd_grad
from player.policy import Theta, random_theta

GAME_STEPS = 1000
ADAM_ITERATIONS = 80
FIELD_SIZE = (10, 10)
DEFAULT_WORKERS = 0
ROLLOUTS_CENTER = 24
ROLLOUTS_FD_ARM = 12
FD_EPSILON = 0.08
LEARNING_RATE = 0.15
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8


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


def _fmt_theta(t: Theta, *, decimals: int = 4) -> str:
    inner = ", ".join(f"{x:.{decimals}f}" for x in t)
    return f"[{inner}]"


def _resolve_workers(workers: int) -> int:
    if workers <= 0:
        return max(1, os.cpu_count() or 4)
    return workers


@dataclass
class AdamStepLog:
    iteration: int
    mean_j: float
    grad: tuple[float, float, float, float]
    theta: Theta


def adam_player_optimize(
    *,
    seed: int,
    iterations: int,
    max_steps: int,
    field_size: tuple[int, int],
    rollouts_center: int,
    rollouts_fd_arm: int,
    fd_epsilon: float,
    learning_rate: float,
    beta1: float,
    beta2: float,
    adam_eps: float,
    workers: int,
) -> tuple[Theta, float, list[AdamStepLog]]:
    """
    Цикл Adam: градиент J(θ) центральными разностями, один пул симуляций на итерацию.
    Возвращает лучшие по среднему J θ за прогон, финальное θ и журнал шагов.
    """
    rng = random.Random(seed)
    theta: Theta = random_theta(rng)
    state = AdamState.zeros(4)
    best_theta = theta
    best_j = float("-inf")
    history: list[AdamStepLog] = []

    wn = _resolve_workers(workers)
    n_tasks = rollouts_center + 8 * rollouts_fd_arm
    print(
        f"Adam: поле {field_size[0]}x{field_size[1]}, до {max_steps} ходов за партию; "
        f"на итерацию {n_tasks} партий (центр {rollouts_center}, FD плечо {rollouts_fd_arm}x8); workers={wn}",
        flush=True,
    )

    for it in range(iterations):
        mean_j, grad = center_value_and_fd_grad(
            theta,
            eps=fd_epsilon,
            rollouts_per_arm=rollouts_fd_arm,
            rollouts_center=rollouts_center,
            max_steps=max_steps,
            field_size=field_size,
            base_seed=seed,
            adam_iter=it,
            workers=workers,
        )
        history.append(AdamStepLog(iteration=it, mean_j=mean_j, grad=grad, theta=theta))
        if mean_j > best_j:
            best_j = mean_j
            best_theta = theta

        gnorm = sum(g * g for g in grad) ** 0.5
        print(
            f"[iter {it + 1}/{iterations}] mean J={mean_j:.6f}  |grad|_2={gnorm:.6f}  w={_fmt_theta(theta)}",
            flush=True,
        )

        theta, state = adam_max_step(
            theta,
            grad,
            state,
            lr=learning_rate,
            beta1=beta1,
            beta2=beta2,
            eps=adam_eps,
        )

    return best_theta, best_j, history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adam по конечным разностям для весов политики Snake (максимизация J)",
    )
    parser.add_argument("--seed", type=int, required=True, help="Зерно RNG для стартового θ и партий")
    parser.add_argument("--iterations", type=int, default=ADAM_ITERATIONS, help="Число шагов Adam")
    parser.add_argument("--steps", type=int, default=GAME_STEPS, metavar="N", help="Макс. шагов за партию")
    parser.add_argument(
        "--field-height",
        type=int,
        default=FIELD_SIZE[0],
        metavar="H",
    )
    parser.add_argument(
        "--field-width",
        type=int,
        default=FIELD_SIZE[1],
        metavar="W",
    )
    parser.add_argument(
        "--rollouts-center",
        type=int,
        default=ROLLOUTS_CENTER,
        help="Партий для оценки J(θ) на текущем θ за итерацию",
    )
    parser.add_argument(
        "--rollouts-fd",
        type=int,
        default=ROLLOUTS_FD_ARM,
        metavar="K",
        help="Партий на каждое плечо (θ±εe_i) при оценке производной",
    )
    parser.add_argument("--fd-eps", type=float, default=FD_EPSILON, help="Шаг ε центральной разности")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate Adam")
    parser.add_argument("--beta1", type=float, default=ADAM_BETA1)
    parser.add_argument("--beta2", type=float, default=ADAM_BETA2)
    parser.add_argument("--adam-eps", type=float, default=ADAM_EPS, help="ε в знаменателе Adam")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Параллельных процессов для партий (0 = число ядер)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        metavar="DIR",
        help="Каталог для лог-файла (adam_optimization_seed_<seed>_<timestamp>.log)",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"adam_optimization_seed_{args.seed}_{ts}.log"
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
    field_size = (args.field_height, args.field_width)
    best_theta, best_j, _hist = adam_player_optimize(
        seed=args.seed,
        iterations=args.iterations,
        max_steps=args.steps,
        field_size=field_size,
        rollouts_center=max(1, args.rollouts_center),
        rollouts_fd_arm=max(1, args.rollouts_fd),
        fd_epsilon=max(1e-9, args.fd_eps),
        learning_rate=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        adam_eps=max(1e-12, args.adam_eps),
        workers=args.workers,
    )
    print("--- итог ---", flush=True)
    print(f"Лучшее среднее J по итерациям: {best_j:.6f}", flush=True)
    print(f"Лучшие веса: {_fmt_theta(best_theta)}", flush=True)
    print("(порядок: w_food, w_danger, w_space, w_wall)", flush=True)


if __name__ == "__main__":
    main()
