import giotto.games.settings.settings_global as settings_global
from giotto.games.ui.player_types import PlayerType

GAME_TITLE = "Giotto Connect4"
CAPTION = settings_global.CAPTION

WIDTH = settings_global.WIDTH
HEIGHT = settings_global.HEIGHT

FPS = settings_global.FPS

BACKGROUND_COLOR = settings_global.BACKGROUND_COLOR
GRID_COLOR = settings_global.GRID_COLOR
O_COLOR = settings_global.O_COLOR
X_COLOR = settings_global.X_COLOR

GRID_WIDTH = int(WIDTH * 0.01)
GRID_X0 = int(WIDTH * 0.15)  # topleft x
GRID_X1 = int(WIDTH * 0.25)  # first line x
GRID_X2 = int(WIDTH * 0.35)  # second line x
GRID_X3 = int(WIDTH * 0.45)
GRID_X4 = int(WIDTH * 0.55)
GRID_X5 = int(WIDTH * 0.65)
GRID_X6 = int(WIDTH * 0.75)
GRID_X7 = int(WIDTH * 0.85)  # bottom right x
GRID_Y0 = int(HEIGHT * 0.25)  # topleft y
GRID_Y1 = int(HEIGHT * 0.35)  # first row y
GRID_Y2 = int(HEIGHT * 0.45)  # second row y
GRID_Y3 = int(HEIGHT * 0.55)
GRID_Y4 = int(HEIGHT * 0.65)
GRID_Y5 = int(HEIGHT * 0.75)
GRID_Y6 = int(HEIGHT * 0.85)

cell_map = {
    1: (GRID_X0, GRID_Y0),
    2: (GRID_X1, GRID_Y0),
    3: (GRID_X2, GRID_Y0),
    4: (GRID_X3, GRID_Y0),
    5: (GRID_X4, GRID_Y0),
    6: (GRID_X5, GRID_Y0),
    7: (GRID_X6, GRID_Y0),
}
CELL_WIDTH = GRID_X2 - GRID_X1
CELL_HEIGHT = GRID_Y2 - GRID_Y1

# which player types are available for this game (order matters for UI cycling)
SUPPORTED_PLAYER_TYPES = [
    PlayerType.HUMAN,
    PlayerType.RANDOM,
    # PlayerType.MINIMAX, # too slow for connect4
    PlayerType.MCTS,
]
