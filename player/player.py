from __future__ import annotations

import random
from collections.abc import Sequence

from game import SnakeGame
from game.models import GameState, GameStatus

from player.policy import Theta, choose_move


class Player:
    """
    Политика по весам на переданной игре. RNG должен совпадать с тем, что передан в ``SnakeGame``,
    иначе яблоки и разрывы ничьих в ``choose_move`` будут несогласованы с игрой.
    """

    def __init__(
        self,
        game: SnakeGame,
        weights: Sequence[float],
        rng: random.Random,
        *,
        max_steps: int = 1_000_000,
    ) -> None:
        if game.rng is not rng:
            msg = "В SnakeGame должен быть передан тот же экземпляр random.Random (game.rng is rng)."
            raise ValueError(msg)
        w = tuple(float(x) for x in weights)
        if len(w) != 4:
            msg = "weights должен содержать ровно 4 числа (w_food, w_danger, w_space, w_wall)."
            raise ValueError(msg)
        self._game = game
        self._theta: Theta = (w[0], w[1], w[2], w[3])
        self._rng = rng
        self._max_steps = max_steps

    def play(self) -> list[GameState]:
        game = self._game
        for _ in range(self._max_steps):
            st = game.get_state()
            if st.status is not GameStatus.IN_PROGRESS:
                break
            mv = choose_move(st, self._theta, max_steps=self._max_steps, rng=self._rng)
            game.step(mv)
        return game.get_history()
