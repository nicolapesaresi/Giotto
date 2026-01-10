WIDTH = 800
HEIGHT = 600

FPS = 10

BACKGROUND_COLOR = "black"
GRID_COLOR = "white"
O_COLOR = "blue"
X_COLOR = "red"

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