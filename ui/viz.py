"""Кадры одной партии и растровое поле для UI."""
from __future__ import annotations

import random
from collections.abc import Callable, Iterator

import numpy as np
from PIL import Image

from game import SnakeGame
from game.models import GameState, GameStatus

from player.player import Player

# 0 пусто, 1 тело, 2 голова, 3 яблоко (как в build_field_matrix)
_PALETTE = np.array(
    [
        [18, 55, 34],
        [56, 120, 92],
        [180, 235, 200],
        [220, 50, 50],
    ],
    dtype=np.uint8,
)


def field_to_image(field: tuple[tuple[int, ...], ...], *, cell_px: int = 22) -> Image.Image:
    """RGB-картинка поля с увеличением без сглаживания (пиксель-арт)."""
    mat = np.asarray(field, dtype=np.int_)
    rgb = _PALETTE[np.clip(mat, 0, 3)]
    h, w, _ = rgb.shape
    img = Image.fromarray(rgb, mode="RGB")
    return img.resize((w * cell_px, h * cell_px), resample=Image.Resampling.NEAREST)


def iter_episode(
    theta: tuple[float, float, float, float],
    *,
    field_height: int,
    field_width: int,
    max_steps: int,
    seed: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Iterator[GameState]:
    """Пошагово отдаёт состояние поля: старт, затем после каждого хода, не больше max_steps ходов."""
    rng = random.Random(seed & 0x7FFFFFFF)
    game = SnakeGame(rng, (field_height, field_width))
    player = Player(game, theta, rng, max_steps=max_steps)
    yield game.get_state()
    player.play(progress_callback=progress_callback)
    for st in game.get_history():
        yield st


def status_ru(st: GameState) -> str:
    if st.status is GameStatus.IN_PROGRESS:
        return "идёт игра"
    if st.status is GameStatus.GAME_FAILED:
        return "поражение"
    if st.status is GameStatus.GAME_COMPLETED:
        return "победа (поле заполнено)"
    return str(st.status.value)
