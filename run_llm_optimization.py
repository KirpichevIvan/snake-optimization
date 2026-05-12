"""
Точка входа: LLM-оптимизатор (пакет llm_as_optimizer).

Весь stdout/stderr пишется в файл `logs/optimization_seed_<seed>_<timestamp>.log`
(каталог задаётся флагом `--log-dir`). Запуск: `uv run python run_optimization.py --seed 42 ...`
"""

from llm_as_optimizer.run import main

if __name__ == "__main__":
    main()
