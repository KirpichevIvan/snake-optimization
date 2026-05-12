# Пакет `space`

Сэмплинг ландшафта целевой функции **J(θ)** по четырёхмерной сетке весов политики и просмотр результатов в браузере.

## Зависимости

Из корня проекта (см. `pyproject.toml`): `numpy`, `tqdm`, для визуализатора — `streamlit`, `plotly`. Запуск из окружения проекта, например:

```bash
uv run python -m space.landscape --help
uv run python -m space.visualizer --help
```

## 1. Landscape — расчёт сетки

Модуль: `space.landscape`. Считает J для всех узлов сетки и сохраняет в каталог **`-o` / `--output-dir`** два файла:

| Файл | Содержимое |
|------|------------|
| `landscape.csv` | Колонки `w_food`, `w_danger`, `w_space`, `w_wall`, `J` (по одной строке на узел) |
| `meta.json` | Версия формата, `max_steps`, размер поля, `workers`, границы и шаги по осям, `shape`, число точек |

### Аргументы (важное)

- **`-o`, `--output-dir`** — каталог вывода (обязателен).
- **`--param MIN MAX STEP`** — ровно **четыре** раза, в порядке: **w_food**, **w_danger**, **w_space**, **w_wall**.
- **`--max-steps`** — лимит шагов змейки за эпизод (по умолчанию `200`).
- **`--field-height`, `--field-width`** — размер поля (по умолчанию `10` и `10`).
- **`--workers`** — число процессов (`0` → авто по CPU).
- **`--max-points`** — жёсткий лимит числа узлов сетки; при превышении программа завершится с ошибкой (по умолчанию `50000`). Для плотных сеток увеличьте значение.
- **`--no-progress`** — отключить прогресс-бар.

### Пример

Сетка от −2 до 2 с шагом 0.2 по каждому весу, до 200 шагов за эпизод, вывод в `zone4x4`:

```bash
uv run python -m space.landscape -o zone4x4 ^
  --param -2 2 0.2 --param -2 2 0.2 --param -2 2 0.2 --param -2 2 0.2 ^
  --max-steps 200
```

В bash / PowerShell без переносов строк:

```bash
uv run python -m space.landscape -o zone4x4 --param -2 2 0.2 --param -2 2 0.2 --param -2 2 0.2 --param -2 2 0.2 --max-steps 200
```

Если число узлов больше `--max-points`, добавьте, например: `--max-points 3000000`.

## 2. Visualizer — браузерный UI

Модуль: `space.visualizer`. Проверяет, что в каталоге есть `landscape.csv` и `meta.json`, подгружает данные, выставляет переменную **`SNAKE_LANDSCAPE_DIR`** и запускает **`streamlit run ui/space_app.py`**.

### Запуск

С явным путём к каталогу с результатами landscape:

```bash
uv run python -m space.visualizer zone4x4
```

или (абсолютный путь):

```bash
uv run python -m space.visualizer C:\path\to\zone4x4
```

Если аргумент каталога **не указан**, используется **`SNAKE_LANDSCAPE_DIR`** (если задана), иначе **текущая рабочая директория** — в ней должны лежать `landscape.csv` и `meta.json`.

В открывшемся приложении при необходимости укажите путь в боковой панели и нажмите **«Загрузить»** (по умолчанию подставляется значение из `SNAKE_LANDSCAPE_DIR`).

### Ручной запуск Streamlit

```bash
set SNAKE_LANDSCAPE_DIR=C:\path\to\zone4x4
uv run streamlit run ui/space_app.py
```

## Прочее

- Справка по подмодулям: `uv run python -m space`.
- Статичные PNG через matplotlib: `uv run python -m space.plot --help`.
