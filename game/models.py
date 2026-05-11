from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class GameStatus(Enum):
    """Состояние жизненного цикла игры."""

    IN_PROGRESS = "in_progress"
    GAME_COMPLETED = "game_completed"
    GAME_FAILED = "game_failed"


class Move(Enum):
    """Направление одного шага головы змейки (смещение по строке и столбцу)."""

    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)


@dataclass(frozen=True, slots=True)
class GameState:
    """Снимок игры: поле, шаг, счёт, статус."""

    field: tuple[tuple[int, ...], ...]
    step: int
    score: int
    status: GameStatus

    def field_matrix(self) -> list[list[int]]:
        """Копия поля в виде изменяемой матрицы (удобно для отладки)."""
        return [list(row) for row in self.field]


def build_field_matrix(
    height: int,
    width: int,
    snake: Sequence[tuple[int, int]],
    apple: tuple[int, int] | None,
    status: GameStatus,
) -> tuple[tuple[int, ...], ...]:
    """
    Собирает матрицу поля: 0 пусто, 1 тело, 2 голова, 3 яблоко.
    Яблоко не рисуется при победе (поле заполнено змейкой).
    """
    field = [[0] * width for _ in range(height)]
    if len(snake) >= 2:
        for r, c in snake[1:]:
            field[r][c] = 1
    hr, hc = snake[0]
    field[hr][hc] = 2
    if apple is not None and status is GameStatus.IN_PROGRESS:
        ar, ac = apple
        field[ar][ac] = 3
    return tuple(tuple(row) for row in field)
