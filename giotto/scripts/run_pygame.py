"""Launch pygame desktop version.
python ./giotto/scripts/run_pygame.py # runs launcher
python ./giotto/scripts/run_pygame.py -g tris # runs TicTacToe
python ./giotto/scripts/run_pygame.py -g connect4 # runs Connect4
"""

import argparse
from giotto.games.launcher import launch as launch_launcher
from giotto.games.tris import launch as launch_tris
from giotto.games.connect4 import launch as launch_connect4


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch GiottoAI Pygame Desktop')
    parser.add_argument('-g','--game', help='game to play [launcher, tris, connect4]', required=False)
    args = vars(parser.parse_args())

    # launch game
    if args["game"] and args["game"].lower() == "tris":
        launch_tris()
    elif args["game"] and args["game"].lower() == "connect4":
        launch_connect4()
    else:
        launch_launcher()
        