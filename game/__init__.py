"""Математическая модель игры «Змейка» без графического интерфейса."""

from game.core import SnakeGame
from game.models import GameState, GameStatus, Move

__all__ = ["SnakeGame", "GameState", "GameStatus", "Move"]
