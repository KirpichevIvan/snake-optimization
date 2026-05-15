# Snake optimization

Репозиторий объединяет **движок змейки** (пакет `game`) и **игрока с линейной политикой по фичам** (пакет `player`): на одних и тех же правилах можно вручную играть, прогонять ботов и оптимизировать веса θ (генетический алгоритм, другие оптимизаторы — см. каталоги `genetic/`, `adam_as_optimizer/` и т.д.).

- **Требования:** Python **≥ 3.12**
- **Зависимости:** задаются в `pyproject.toml` (Streamlit, NumPy, matplotlib и др.)

Установка и запуск удобно делать через [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

---

## Пакет `game`

Модуль описывает **чистую логику** «Змейки» на прямоугольном поле без внутренних стен (границы — край матрицы).

### Основные сущности

| Сущность | Назначение |
|----------|------------|
| `SnakeGame` | Контейнер симуляции: поле, змейка, яблоко, счётчик шагов, история успешных `step`. |
| `GameState` | Неизменяемый снимок: матрица поля `field`, номер шага `step`, съеденные яблоки `score`, статус `status`. |
| `GameStatus` | `IN_PROGRESS`, `GAME_COMPLETED` (поле заполнено), `GAME_FAILED` (стена или тело). |
| `Move` | Одно из направлений: `UP`, `DOWN`, `LEFT`, `RIGHT` (каждое — пара смещений по строке и столбцу). |

### Кодировка клеток в `field`

- `0` — пусто  
- `1` — тело  
- `2` — голова  
- `3` — яблоко  

Яблоко при победе на поле не рисуется (полностью занято змейкой). RNG передаётся в конструктор `SnakeGame`: от него зависят позиции яблок; **один и тот же объект `random.Random` должен использоваться игрой и игроком**, иначе политика и симулятор разойдутся.

### Главные операции API

- `SnakeGame(rng, field_size=(h, w))` — создание партии  
- `get_state() -> GameState` — текущее состояние  
- `step(move: Move) -> GameState` — один ход головой; при уже завершённой игре — `RuntimeError`  
- `get_history() -> list[GameState]` — состояния после каждого успешного `step`

Публичный импорт из пакета:

```python
from game import SnakeGame, GameState, GameStatus, Move
```

---

## Пакет `player`

Здесь **политика** с параметром **θ** — кортеж из четырёх вещественных чисел:

`(w₁, w₂, w₃, w₄)` = **еда, опасность, пространство, стена**.

Для каждого допустимого хода из текущего `GameState` считаются признаки `(food, danger, space, wall)` (модуль `player/features.py`). Оценка хода — линейная комбинация весами θ; среди допустимых ходов выбирается **максимум**, при равенстве — случайный выбор с помощью переданного `rng` (`player/policy.py`, функция `choose_move`).

### Основные сущности

| Сущность | Назначение |
|----------|------------|
| `Player` | Обёртка вокруг `SnakeGame` + θ + общий RNG; методы `play()`, `play_steps()`. |
| `Theta` | Тип-кортеж `tuple[float, float, float, float]`. |
| `choose_move` | Выбор следующего хода по θ и состоянию (жадная линейная политика). |
| `random_theta` | Случайные веса в заданных диапазонах (для инициализации поиска). |
| `simulate` | Один rollout: создание игры и прогон `Player.play()` до лимита шагов или конца партии; возвращает скаляр **J(θ)** = яблоки − штраф за смерть (`player/rollout.py`). |

Экспорт из пакета:

```python
from player import (
    Player,
    Theta,
    choose_move,
    random_theta,
    score_objective,
    simulate,
    simulate_packed,
)
```

---

## Пример: только `game` (ручные ходы)

Минимальный цикл из корня репозитория (поле Python должен видеть пакеты; при необходимости добавьте корень в `PYTHONPATH` или запускайте из корня после `uv run python`):

```python
import random

from game import SnakeGame, Move
from game.models import GameStatus

rng = random.Random(42)
game = SnakeGame(rng, field_size=(8, 8))

while game.get_state().status == GameStatus.IN_PROGRESS:
    st = game.get_state()
    print(st.field_matrix())
    cmd = input("w/a/s/d: ").strip()
    mv = {"w": Move.UP, "s": Move.DOWN, "a": Move.LEFT, "d": Move.RIGHT}[cmd]
    game.step(mv)

print("Финальный счёт:", game.get_state().score, "статус:", game.get_state().status)
```

Аналогичная идея реализована в `main.py`: консольный ввод WASD для одной партии на поле `10×10`.

---

## Пример: связка `game` + `player`

Один воспроизводимый прогон линейного агента:

```python
import random

from game import SnakeGame
from player import Player

theta = (6.1540, -0.0650, -6.1440, 0.0720)  # числа из умолчанию в Streamlit-приложении
seed = 42
rng = random.Random(seed)
game = SnakeGame(rng, field_size=(10, 10))
player = Player(game, theta, rng, max_steps=400)

history = player.play()
print("Яблок съедено:", game.get_state().score)
print("Шагов в истории успешных move:", len(history))
```

Пошагово «вне» класса `Player`:

```python
import random

from game import SnakeGame
from game.models import GameStatus
from player import choose_move

rng = random.Random(0)
theta = (1.0, -1.0, 1.0, 0.5)
game = SnakeGame(rng, (7, 7))

while game.get_state().status == GameStatus.IN_PROGRESS:
    st = game.get_state()
    move = choose_move(st, theta, max_steps=300, rng=rng)
    game.step(move)

print(game.get_state().score)
```

Цель оптимизации для одной партии (как используется в поиске весов):

```python
from player import simulate

j = simulate(
    (6.1540, -0.0650, -6.1440, 0.0720),
    max_steps=400,
    seed=42,
    field_size=(10, 10),
)
print("J(theta) =", j)
```

---

## Запуск `ui/app.py` (Streamlit)

Файл: **`ui/app.py`**. Это не консольный скрипт, а **одностраничное Streamlit-приложение**, которое **визуализирует одну партию** агента с заданными весами θ в почти реальном времени (пауза между кадрами настраивается).

### Команда запуска (из корня репозитория)

```bash
uv run streamlit run ui/app.py
```

### Что делает интерфейс

1. **Поля чисел w₁ … w₄** — компоненты θ (еда, опасность, пространство, стена). По умолчанию подставлены те же числа, что в коде приложения (`DEFAULT_THETA`).
2. **Высота / ширина поля** — размер решётки для `SnakeGame`.
3. **Макс. шагов** — предел для `Player` (защита от слишком долгих прогонков в браузере).
4. **Зерно** — фиксирует `random.Random` для игры и политики.
5. **Пауза между кадрами** и **размер клетки** — только отображение.
6. Кнопка **«Запустить прогон»** запускает симуляцию: показывает поле после старта и после каждого хода, подпись с номером кадра, шагом, счётом яблок и статусом игры до поражения, победы или исчерпания лимита шагов.

Внутри приложение использует `SnakeGame` + `Player` и итератор `iter_episode` из `ui/viz.py`: тот генерирует последовательность `GameState` для отрисовки.

---

## Дополнительно в репозитории

Краткая документация по отдельным подсистемам:

- `genetic/README.md` — генетическая оптимизация θ  
- `space/README.md` — визуализация и анализ ландшафта цели  

Скрипты верхнего уровня вроде `run_genetic_optimization.py` подключают эти модули к полному циклу обучения/логирования.
