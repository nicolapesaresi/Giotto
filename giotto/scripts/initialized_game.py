"""Play n games between agents.
Example command:
python ./giotto/scripts/initialized_game.py -g connect4 -p random -o random -m 4
"""

import argparse
from giotto.envs.tris import TrisEnv
from giotto.envs.connect4 import Connect4Env
from giotto.agents.human import HumanAgent
from giotto.agents.random import RandomAgent
from giotto.agents.mcts import MCTSAgent
from giotto.agents.minimax import MinimaxAgent
from giotto.utils.text_play import initialized_game


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Play an initialized game against an agent"
    )
    parser.add_argument(
        "-g", "--game", help="game to play [tris, connect4]", required=True
    )
    parser.add_argument(
        "-p",
        "--player",
        help="first player [human, random, mcts, minimax]",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--opp",
        help="second player [human, random, mcts, minimax]",
        required=True,
    )
    parser.add_argument(
        "-m", "--move", help="first move [default random]", required=False
    )
    args = vars(parser.parse_args())

    # game
    if args["game"].lower() == "tris":
        env = TrisEnv()
    elif args["game"].lower() == "connect4":
        env = Connect4Env()
    else:
        raise ValueError(f"{args["game"]} not a valid game")

    # number of games
    if args["move"]:
        move = int(args["move"])
    else:
        move = None

    # hero
    if args["player"].lower() == "human":
        player = HumanAgent(name="Human_opp")
    elif args["player"].lower() == "random":
        player = RandomAgent()
    elif args["player"].lower() == "mcts":
        player = MCTSAgent()
    elif args["player"].lower() == "minimax":
        player = MinimaxAgent()
    else:
        raise ValueError(f"{args["player"]} not a valid opponent")

    # opponent
    if args["opp"].lower() == "human":
        opp = HumanAgent(name="Human_opp")
    elif args["opp"].lower() == "random":
        opp = RandomAgent()
    elif args["opp"].lower() == "mcts":
        opp = MCTSAgent()
    elif args["opp"].lower() == "minimax":
        opp = MinimaxAgent()
    else:
        raise ValueError(f"{args["opp"]} not a valid opponent")

    agents = [player, opp]
    if agents[0].name == agents[1].name:
        agents[0].name += "_1"
        agents[1].name += "_2"

    initialized_game(env, agents, starter=0, first_move=move, render=True)
