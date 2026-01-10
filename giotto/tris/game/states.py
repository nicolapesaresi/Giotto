from enum import IntEnum

class States(IntEnum):
    """Render states for Tris Pygame."""
    MENU = 0
    GAME = 1
    GAMEOVER = 2
    CLOSE = 3