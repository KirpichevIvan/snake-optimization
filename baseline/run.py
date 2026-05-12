"""
Baseline: змейка на каждом шаге выбирает **равновероятно** один из **безопасных**
ходов (не в стену и не в тело по тем же правилам, что и в игре). Зависимость
только от пакета `game`.

Запуск из корня репозитория:
  uv run python baseline/run.py --seed 42
  uv run python baseline/run.py --seed 42 --rollouts 5000 --workers 4

Весь вывод дублируется в файл ``baseline/logs/baseline_seed_<seed>_<timestamp>.log``
(каталог меняется флагом ``--log-dir``).
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import TextIO

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from game import SnakeGame
from game.models import GameState, GameStatus, Move

_BASELINE_DIR = Path(__file__).resolve().parent
_DEFAULT_LOG_DIR = _BASELINE_DIR / "logs"


class _TeeTextIO:
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


def _find_head(field: tuple[tuple[int, ...], ...]) -> tuple[int, int] | None:
    for r, row in enumerate(field):
        for c, v in enumerate(row):
            if v == 2:
                return (r, c)
    return None


def _snake_cells(field: tuple[tuple[int, ...], ...]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r, row in enumerate(field):
        for c, v in enumerate(row):
            if v in (1, 2):
                cells.add((r, c))
    return cells


def _recover_snake_path(field: tuple[tuple[int, ...], ...]) -> list[tuple[int, int]] | None:
    head = _find_head(field)
    if head is None:
        return None
    snake_set = _snake_cells(field)
    if not snake_set:
        return [head]
    if len(snake_set) == 1:
        return [head]

    def neighbors(p: tuple[int, int]) -> list[tuple[int, int]]:
        r, c = p
        out: list[tuple[int, int]] = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            q = (r + dr, c + dc)
            if q in snake_set:
                out.append(q)
        return out

    target = len(snake_set)

    def dfs(cur: tuple[int, int], visited: tuple[tuple[int, int], ...]) -> list[tuple[int, int]] | None:
        if len(visited) == target:
            return list(visited)
        for nxt in neighbors(cur):
            if nxt in visited:
                continue
            res = dfs(nxt, (*visited, nxt))
            if res is not None:
                return res
        return None

    return dfs(head, (head,))


def legal_moves(state: GameState) -> list[Move]:
    """Ходы, после которых не будет немедленного поражения (как в SnakeGame.step)."""
    if state.status is not GameStatus.IN_PROGRESS:
        return []
    path = _recover_snake_path(state.field)
    if path is None:
        return []
    h, w = len(state.field), len(state.field[0])
    hr, hc = path[0]
    tail = path[-1]
    body_core = set(path[1:-1]) if len(path) > 2 else set()
    field = state.field
    out: list[Move] = []
    for m in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
        dr, dc = m.value
        nr, nc = hr + dr, hc + dc
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        if (nr, nc) in body_core:
            continue
        if (nr, nc) == tail:
            out.append(m)
        elif field[nr][nc] == 1:
            continue
        else:
            out.append(m)
    return out


def run_episode(
    *,
    field_height: int,
    field_width: int,
    max_steps: int,
    rng: random.Random,
) -> int:
    game = SnakeGame(rng, (field_height, field_width))
    for _ in range(max_steps):
        st = game.get_state()
        if st.status is not GameStatus.IN_PROGRESS:
            break
        moves = legal_moves(st)
        if not moves:
            break
        game.step(rng.choice(moves))
    return game.get_state().score


def _episode_worker(args: tuple[int, int, int, int, int]) -> int:
    rollout_index, base_seed, height, width, max_steps = args
    rng = random.Random((base_seed ^ (rollout_index * 1_000_003)) & 0x7FFFFFFF)
    return run_episode(field_height=height, field_width=width, max_steps=max_steps, rng=rng)


def _run_inner(args: argparse.Namespace) -> None:
    rollouts = max(1, args.rollouts)
    max_steps = max(1, args.steps)
    h, w = args.height, args.width
    workers = args.workers if args.workers > 0 else os.cpu_count() or 4

    tasks = [(k, args.seed, h, w, max_steps) for k in range(rollouts)]

    if workers <= 1:
        scores = [_episode_worker(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            scores = list(pool.map(_episode_worker, tasks, chunksize=max(1, rollouts // (workers * 4))))

    mean_apples = sum(scores) / len(scores)
    print(f"Поле: {h}×{w}, до {max_steps} шагов за партию, партий: {rollouts} (workers={workers})")
    print(f"Политика: равновероятный выбор среди безопасных ходов (без наезда на стену и тело).")
    print(f"Среднее число съеденных яблок: {mean_apples:.6f}")
    print(f"Мин / макс за партию: {min(scores)} / {max(scores)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Baseline: случайный безопасный ход, среднее число яблок")
    p.add_argument("--rollouts", type=int, default=10000, help="Число партий")
    p.add_argument("--steps", type=int, default=1000, help="Макс. шагов за партию")
    p.add_argument("--height", type=int, default=10)
    p.add_argument("--width", type=int, default=10)
    p.add_argument("--seed", type=int, default=0, help="Базовое зерно для партий (в имени лог-файла)")
    p.add_argument("--workers", type=int, default=0, help="0 = число ядер CPU")
    p.add_argument(
        "--log-dir",
        type=str,
        default=str(_DEFAULT_LOG_DIR),
        metavar="DIR",
        help="Каталог для лога (по умолчанию baseline/logs)",
    )
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"baseline_seed_{args.seed}_{ts}.log"
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


if __name__ == "__main__":
    main()
