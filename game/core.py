from __future__ import annotations

import random
from typing import TYPE_CHECKING

from game.models import GameState, GameStatus, Move, build_field_matrix

if TYPE_CHECKING:
    from collections.abc import Sequence


class SnakeGame:
    """
    Змейка на прямоугольном поле без стен внутри (границы — край матрицы).
    Победа: змейка заняла все клетки. Поражение: выход за границу или наезд на тело.
    """

    def __init__(
        self,
        field_size: tuple[int, int],
        *,
        rng: random.Random | None = None,
    ) -> None:
        height, width = field_size
        if height < 1 or width < 1:
            msg = "field_size должен быть (height, width) с положительными целыми значениями"
            raise ValueError(msg)

        self._height = height
        self._width = width
        self._rng = rng if rng is not None else random.Random()

        start_r, start_c = height // 2, width // 2
        self._snake: list[tuple[int, int]] = [(start_r, start_c)]
        self._apple: tuple[int, int] | None = None
        self._step = 0
        self._score = 0
        self._status = GameStatus.IN_PROGRESS
        self._history: list[GameState] = []

        if len(self._snake) == height * width:
            self._status = GameStatus.GAME_COMPLETED
        elif height * width > 1:
            self._spawn_apple()

    @property
    def field_size(self) -> tuple[int, int]:
        return self._height, self._width

    def get_state(self) -> GameState:
        field = build_field_matrix(
            self._height,
            self._width,
            self._snake,
            self._apple,
            self._status,
        )
        return GameState(
            field=field,
            step=self._step,
            score=self._score,
            status=self._status,
        )

    def get_history(self) -> list[GameState]:
        """Состояния после каждого успешно применённого step (включая завершающий шаг)."""
        return list(self._history)

    def step(self, move: Move) -> GameState:
        if self._status is not GameStatus.IN_PROGRESS:
            msg = "Игра уже завершена, step недопустим"
            raise RuntimeError(msg)

        dr, dc = move.value
        hr, hc = self._snake[0]
        nr, nc = hr + dr, hc + dc

        if not (0 <= nr < self._height and 0 <= nc < self._width):
            self._status = GameStatus.GAME_FAILED
            state = self._finalize_state()
            self._history.append(state)
            return state

        tail = self._snake[-1]
        will_eat = self._apple is not None and (nr, nc) == self._apple

        # Наезд на тело: хвост освобождается, если не растём — в хвост можно зайти
        body_without_tail: Sequence[tuple[int, int]] = self._snake[:-1] if len(self._snake) > 1 else ()
        if (nr, nc) in body_without_tail:
            self._status = GameStatus.GAME_FAILED
            state = self._finalize_state()
            self._history.append(state)
            return state

        new_snake = [(nr, nc), *self._snake]
        if not will_eat:
            new_snake.pop()

        self._snake = new_snake
        if will_eat:
            self._score += 1
            self._apple = None

        total_cells = self._height * self._width
        if len(self._snake) == total_cells:
            self._status = GameStatus.GAME_COMPLETED
            self._apple = None
        elif will_eat:
            self._spawn_apple()

        self._step += 1
        state = self.get_state()
        self._history.append(state)
        return state

    def _finalize_state(self) -> GameState:
        """Состояние после поражения без изменения змейки на недопустимой клетке."""
        self._step += 1
        return self.get_state()

    def _spawn_apple(self) -> None:
        occupied = set(self._snake)
        candidates = [
            (r, c)
            for r in range(self._height)
            for c in range(self._width)
            if (r, c) not in occupied
        ]
        if not candidates:
            self._apple = None
            return
        self._apple = self._rng.choice(candidates)
