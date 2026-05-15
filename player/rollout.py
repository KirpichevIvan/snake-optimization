from __future__ import annotations

import random

from game import SnakeGame
from game.models import GameStatus

from player.player import Player

type Theta = tuple[float, float, float, float]

# J(θ) = яблоки за N шагов − λ·I(death); λ = 1 как в ТЗ
DEATH_PENALTY = 1.0


def score_objective(apples: int, died: bool) -> float:
    return float(apples) - (DEATH_PENALTY if died else 0.0)


RolloutPacked = (
    tuple[tuple[float, float, float, float], int, tuple[int, int], int]
    | tuple[tuple[float, float, float, float], int, tuple[int, int], int, bool, int]
)


def _unpack_rollout_packed(
    packed: RolloutPacked,
) -> tuple[Theta, int, tuple[int, int], int, bool, int]:
    if len(packed) == 4:
        theta, max_steps, field_size, seed = packed
        return theta, max_steps, field_size, seed, False, 100
    theta, max_steps, field_size, seed, stretch, stretch_chunk = packed
    return theta, max_steps, field_size, seed, bool(stretch), int(stretch_chunk)


def simulate_packed(packed: RolloutPacked) -> float:
    """Одна партия для пула процессов (см. ``RolloutPacked``)."""
    return simulate_rollout_packed(packed)[0]


def simulate_rollout_packed(packed: RolloutPacked) -> tuple[float, int]:
    """Как ``simulate_packed``, но возвращает ``(J, фактическое_число_шагов)``."""
    theta, max_steps, field_size, seed, stretch, stretch_chunk = _unpack_rollout_packed(packed)
    return simulate_rollout(
        theta,
        max_steps=max_steps,
        seed=seed,
        field_size=field_size,
        stretch=stretch,
        stretch_chunk=stretch_chunk,
    )


def simulate(
    theta: Theta,
    *,
    max_steps: int,
    seed: int,
    field_size: tuple[int, int] = (5, 5),
    stretch: bool = False,
    stretch_chunk: int = 100,
) -> float:
    """
    Один rollout: не более max_steps успешных step (плюс опциональное «вытягивание»).
    Возвращает скалярную цель J (чем выше, тем лучше).
    Зерно RNG обязательно — воспроизводимость политики и игры.
    """
    j, _steps = simulate_rollout(
        theta,
        max_steps=max_steps,
        seed=seed,
        field_size=field_size,
        stretch=stretch,
        stretch_chunk=stretch_chunk,
    )
    return j


def simulate_rollout(
    theta: Theta,
    *,
    max_steps: int,
    seed: int,
    field_size: tuple[int, int] = (5, 5),
    stretch: bool = False,
    stretch_chunk: int = 100,
) -> tuple[float, int]:
    """
    Один rollout с учётом вытягивания.

    При ``stretch=True``: если после ``max_steps`` игра ещё идёт, добавляются блоки по
    ``stretch_chunk`` шагов, пока за блок съедено хотя бы одно яблоко (иначе стоп — зацикливание).
    """
    if max_steps < 1:
        msg = "max_steps должен быть >= 1"
        raise ValueError(msg)
    if stretch_chunk < 1:
        msg = "stretch_chunk должен быть >= 1"
        raise ValueError(msg)

    rng = random.Random(seed)
    game = SnakeGame(rng, field_size)
    player = Player(game, theta, rng, max_steps=max_steps)

    steps_total = 0
    taken, _ = player.play_steps(max_steps)
    steps_total += taken

    if stretch and game.get_state().status is GameStatus.IN_PROGRESS:
        while game.get_state().status is GameStatus.IN_PROGRESS:
            taken, apples_in_chunk = player.play_steps(stretch_chunk)
            steps_total += taken
            if game.get_state().status is not GameStatus.IN_PROGRESS:
                break
            if apples_in_chunk < 1:
                break

    final = game.get_state()
    apples = final.score
    died = final.status is GameStatus.GAME_FAILED
    return score_objective(apples, died), steps_total
