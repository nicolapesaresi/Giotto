import giotto.games.settings.settings_global as settings_global
from giotto.games.ui.player_types import PlayerType

GAME_TITLE = "Giotto TicTacToe"
CAPTION = settings_global.CAPTION

WIDTH = settings_global.WIDTH
HEIGHT = settings_global.HEIGHT

FPS = settings_global.FPS

BACKGROUND_COLOR = settings_global.BACKGROUND_COLOR
GRID_COLOR = settings_global.GRID_COLOR
O_COLOR = settings_global.O_COLOR
X_COLOR = settings_global.X_COLOR

GRID_WIDTH = int(WIDTH * 0.01)
GRID_X0 = int(WIDTH * 0.2 ) # topleft x
GRID_X1 = int(WIDTH * 0.4 ) # first line x
GRID_X2 = int(WIDTH * 0.6 ) # second line x
GRID_X3 = int(WIDTH * 0.8 ) # bottom right x
GRID_Y0 = int(HEIGHT * 0.2)  # topleft y
GRID_Y1 = int(HEIGHT * 0.4)  # first row y
GRID_Y2 = int(HEIGHT * 0.6)  # second row y
GRID_Y3 = int(HEIGHT * 0.8)  # bottom right y

cell_map = {
    1 : (GRID_X0, GRID_Y0),
    2 : (GRID_X1, GRID_Y0),
    3 : (GRID_X2, GRID_Y0),
    4 : (GRID_X0, GRID_Y1),
    5 : (GRID_X1, GRID_Y1),
    6 : (GRID_X2, GRID_Y1),
    7 : (GRID_X0, GRID_Y2),
    8 : (GRID_X1, GRID_Y2),
    9 : (GRID_X2, GRID_Y2),
}
CELL_WIDTH = GRID_X2 - GRID_X1
CELL_HEIGHT = GRID_Y2 - GRID_Y1

# which player types are available for this game (order matters for UI cycling)
SUPPORTED_PLAYER_TYPES = [
    PlayerType.HUMAN,
    PlayerType.RANDOM,
    PlayerType.MINIMAX,
    PlayerType.MCTS,
]
