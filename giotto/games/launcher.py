"""Simple launcher to run Tris or Connect4 from one entry point."""
from giotto.games.tris import PygameRenderer as TrisGame
from giotto.games.connect4 import PygameConnect4 as Connect4Game


def run():
    print("Choose a game to run:")
    print("1) TicTacToe (Tris)")
    print("2) Connect4")
    choice = input("Enter 1 or 2: ").strip()
    if choice == "1":
        TrisGame().run()
    else:
        Connect4Game().run()


if __name__ == "__main__":
    run()
