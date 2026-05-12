"""
Устаревший алиас: запускает ``space.visualizer`` (Streamlit).

Предпочтительно: ``python -m space.visualizer <каталог>``
или: ``uv run streamlit run ui/space_app.py`` с переменной ``SNAKE_LANDSCAPE_DIR``.
"""
from __future__ import annotations

from space.visualizer import main

if __name__ == "__main__":
    raise SystemExit(main())
