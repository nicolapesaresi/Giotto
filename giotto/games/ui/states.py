from enum import IntEnum


class States(IntEnum):
    """Enumeration of possible UI states."""

    MENU = 0
    GAME = 1
    GAMEOVER = 2
    CLOSE = 3
    LAUNCHER = 4
