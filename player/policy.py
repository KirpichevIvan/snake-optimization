import random

from game.models import GameState, GameStatus, Move

from player.features import move_features

type Theta = tuple[float, float, float, float]


def random_theta(rng: random.Random | None = None) -> Theta:
    r = rng if rng is not None else random.Random()
    return (
        r.uniform(-2.0, 4.0),
        r.uniform(-4.0, 1.0),
        r.uniform(-1.0, 3.0),
        r.uniform(-3.0, 2.0),
    )


def score_move(theta: Theta, feats: tuple[float, float, float, float]) -> float:
    wf, wd, ws, ww = theta
    ff, fd, fs, fw = feats
    return wf * ff + wd * fd + ws * fs + ww * fw


def choose_move(
    state: GameState,
    theta: Theta,
    *,
    max_steps: int,
    rng: random.Random | None = None,
) -> Move:
    """Жадный выбор хода по взвешенной сумме фич. При ничьей — rng.choice (или модуль random, если rng=None)."""
    if state.status is not GameStatus.IN_PROGRESS:
        return Move.RIGHT

    best: list[Move] = []
    best_s = float("-inf")
    for m in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
        feats = move_features(state, m, max_steps=max_steps)
        if feats is None:
            continue
        s = score_move(theta, feats)
        if s > best_s:
            best_s = s
            best = [m]
        elif s == best_s:
            best.append(m)
    if not best:
        return Move.RIGHT
    if len(best) == 1:
        return best[0]
    return rng.choice(best) if rng is not None else random.choice(best)
