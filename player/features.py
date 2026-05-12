from collections import deque

from game.models import GameState, GameStatus, Move


def _find_head(field: tuple[tuple[int, ...], ...]) -> tuple[int, int] | None:
    for r, row in enumerate(field):
        for c, v in enumerate(row):
            if v == 2:
                return (r, c)
    return None


def _find_apple(field: tuple[tuple[int, ...], ...]) -> tuple[int, int] | None:
    for r, row in enumerate(field):
        for c, v in enumerate(row):
            if v == 3:
                return (r, c)
    return None


def _snake_cells(field: tuple[tuple[int, ...], ...]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r, row in enumerate(field):
        for c, v in enumerate(row):
            if v in (1, 2):
                cells.add((r, c))
    return cells


def recover_snake_path(field: tuple[tuple[int, ...], ...]) -> list[tuple[int, int]] | None:
    """
    Восстанавливает порядок [голова, ..., хвост] по полю (простой путь по клеткам 1 и 2).
    """
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


def _flood_space_from_head(
    height: int,
    width: int,
    start: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> int:
    """
    Связная область, достижимая с новой головы после хода.
    Заняты только клетки occupied (новая змейка); старое поле не используется —
    иначе освободившийся хвост ошибочно оставался бы непроходимым.
    Старт — голова (она в occupied); в очередь добавляем только соседей вне occupied.
    """
    h, w = height, width
    r0, c0 = start
    if not (0 <= r0 < h and 0 <= c0 < w):
        return 0
    q: deque[tuple[int, int]] = deque([start])
    seen: set[tuple[int, int]] = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in seen:
                continue
            if (nr, nc) in occupied:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return len(seen)


def move_features(
    state: GameState,
    move: Move,
    *,
    max_steps: int = 30,
) -> tuple[float, float, float, float] | None:
    """
    (food, danger, space, wall) — скаляры для линейной политики.
    Возвращает None, если ход заведомо недопустим (стена).
    """
    if state.status is not GameStatus.IN_PROGRESS:
        return None

    field = state.field
    h, w = len(field), len(field[0])
    path = recover_snake_path(field)
    if path is None:
        return None

    head_r, head_c = path[0]
    dr, dc = move.value
    nr, nc = head_r + dr, head_c + dc

    if not (0 <= nr < h and 0 <= nc < w):
        return None

    will_eat = field[nr][nc] == 3
    new_head = (nr, nc)
    tail = path[-1]
    body_core = set(path[1:-1]) if len(path) > 2 else set()
    if (nr, nc) in body_core:
        return None
    if (nr, nc) == tail:
        pass  # хвост освобождается, если не растём; при росте тоже ок
    elif field[nr][nc] == 1:
        return None

    apple = _find_apple(field)
    if apple is None:
        f_food = 0.0
    else:
        dist = abs(nr - apple[0]) + abs(nc - apple[1])
        f_food = float(-dist)

    # danger: штраф за близость к «чужому» телу после шага (кроме хвоста)
    danger_pen = 0.0
    for dr2, dc2 in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ar, ac = nr + dr2, nc + dc2
        if not (0 <= ar < h and 0 <= ac < w):
            danger_pen += 1.0
            continue
        if field[ar][ac] == 2:
            continue
        if field[ar][ac] == 1:
            if (ar, ac) == tail and not will_eat:
                continue
            danger_pen += 0.35
    f_danger = -danger_pen

    if will_eat:
        new_snake = [new_head, *path]
    else:
        new_snake = [new_head, *path[:-1]]

    occupied_after = set(new_snake)
    f_space = float(_flood_space_from_head(h, w, new_head, occupied_after))

    f_wall = float(min(nr, nc, h - 1 - nr, w - 1 - nc))

    # нормировка для численной устойчивости (не меняет оптимум при положительных w)
    f_food /= max(h + w, 1)
    f_space /= max(h * w, 1)

    return (f_food, f_danger, f_space, f_wall)
