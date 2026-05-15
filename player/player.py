from __future__ import annotations

import random
from collections.abc import Callable, Sequence

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

    def play_steps(self, n_steps: int) -> tuple[int, int]:
        """
        До ``n_steps`` успешных ``step`` или до конца игры.

        Возвращает ``(число_шагов, яблок_съедено_за_этот_фрагмент)``.
        """
        if n_steps <= 0:
            return 0, 0
        game = self._game
        apples_before = game.get_state().score
        taken = 0
        for _ in range(n_steps):
            st = game.get_state()
            if st.status is not GameStatus.IN_PROGRESS:
                break
            mv = choose_move(st, self._theta, max_steps=self._max_steps, rng=self._rng)
            game.step(mv)
            taken += 1
        apples_after = game.get_state().score
        return taken, apples_after - apples_before

    def play(
        self,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[GameState]:
        game = self._game
        if progress_callback is not None:
            progress_callback(0, self._max_steps)
        for i in range(self._max_steps):
            st = game.get_state()
            if st.status is not GameStatus.IN_PROGRESS:
                break
            mv = choose_move(st, self._theta, max_steps=self._max_steps, rng=self._rng)
            game.step(mv)
            if progress_callback is not None:
                progress_callback(i + 1, self._max_steps)
        return game.get_history()
