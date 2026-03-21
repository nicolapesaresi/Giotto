"""Play n games between agents.
Example command:
python ./giotto/scripts/play_n_games.py -g connect4 -p random -o random
"""  # noqa: D415

import argparse

from giotto.agents.alphazero import AlphaZeroAgent
from giotto.agents.bitbully import BitBullyAgent
from giotto.agents.giotto import GiottoAgent
from giotto.agents.human import HumanAgent
from giotto.agents.mcts import MCTSAgent
from giotto.agents.minimax import MinimaxAgent
from giotto.agents.random import RandomAgent
from giotto.envs.connect4 import Connect4Env
from giotto.envs.tris import TrisEnv
from giotto.utils.text_play import play_n_games

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play one game against an agent")
    parser.add_argument("-g", "--game", help="game to play [tris, connect4]", required=True)
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
    parser.add_argument("-n", "--number", help="number of games to play [default 100]", required=False)
    args = vars(parser.parse_args())

    # game
    if args["game"].lower() == "tris":
        env = TrisEnv()
    elif args["game"].lower() == "connect4":
        env = Connect4Env()
    else:
        raise ValueError(f"{args['game']} not a valid game")

    # number of games
    if args["number"]:
        n_games = int(args["number"])
    else:
        n_games = 100

    # hero
    if args["player"].lower() == "human":
        player = HumanAgent(name="Human_opp")
    elif args["player"].lower() == "random":
        player = RandomAgent()
    elif args["player"].lower() == "mcts":
        player = MCTSAgent()
    elif args["player"].lower() == "minimax":
        player = MinimaxAgent()
    elif args["player"].lower() == "giotto":
        player = GiottoAgent(game=args["game"])
    elif args["player"].lower() == "alphazero":
        player = AlphaZeroAgent(simulations=800, cpuct=3.5, game=args["game"])
    elif args["player"].lower() == "bitbully":
        player = BitBullyAgent()
    else:
        raise ValueError(f"{args['player']} not a valid opponent")

    # opponent
    if args["opp"].lower() == "human":
        opp = HumanAgent(name="Human_opp")
    elif args["opp"].lower() == "random":
        opp = RandomAgent()
    elif args["opp"].lower() == "mcts":
        opp = MCTSAgent(simulations=800, cpuct=3.5)
    elif args["opp"].lower() == "minimax":
        opp = MinimaxAgent()
    elif args["opp"].lower() == "giotto":
        opp = GiottoAgent(game=args["game"])
    elif args["opp"].lower() == "alphazero":
        opp = AlphaZeroAgent(simulations=800, cpuct=3.5, game=args["game"])
    elif args["opp"].lower() == "bitbully":
        opp = BitBullyAgent()
    else:
        raise ValueError(f"{args['opp']} not a valid opponent")

    agents = [player, opp]

    if agents[0].name == agents[1].name:
        agents[0].name += "_1"
        agents[1].name += "_2"

    play_n_games(n_games, env, agents, invert_starts=False)
