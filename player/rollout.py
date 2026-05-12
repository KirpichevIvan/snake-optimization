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


def simulate_packed(
    packed: tuple[tuple[float, float, float, float], int, tuple[int, int], int],
) -> float:
    """Одна партия для пула процессов: (четыре веса, лимит шагов, размер поля, зерно)."""
    theta, max_steps, field_size, seed = packed
    return simulate(theta, max_steps=max_steps, field_size=field_size, seed=seed)


def simulate(
    theta: Theta,
    *,
    max_steps: int,
    seed: int,
    field_size: tuple[int, int] = (5, 5),
) -> float:
    """
    Один rollout: не более max_steps успешных step.
    Возвращает скалярную цель J (чем выше, тем лучше).
    Зерно RNG обязательно — воспроизводимость политики и игры.
    """
    rng = random.Random(seed)
    game = SnakeGame(rng, field_size)
    Player(game, theta, rng, max_steps=max_steps).play()

    final = game.get_state()
    apples = final.score
    died = final.status is GameStatus.GAME_FAILED
    return score_objective(apples, died)
