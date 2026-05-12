"""
Точка входа: Adam-оптимизатор весов политики (пакет adam_as_optimizer).

Весь stdout/stderr дублируется в файл ``logs/adam_optimization_seed_<seed>_<timestamp>.log``
(каталог задаётся ``--log-dir``). Запуск: ``uv run python run_adam_optimization.py --seed 42``.
"""

from adam_as_optimizer.run import main

if __name__ == "__main__":
    main()
