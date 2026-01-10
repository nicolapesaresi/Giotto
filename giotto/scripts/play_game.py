"""Play a game against an agent.
Example command:
python ./giotto/scripts/play_game.py -g connect4 -o random
"""

import argparse
from giotto.envs.tris import TrisEnv
from giotto.envs.connect4 import Connect4Env
from giotto.agents.human import HumanAgent
from giotto.agents.random import RandomAgent
from giotto.agents.mcts import MCTSAgent
from giotto.agents.minimax import MinimaxAgent
from giotto.utils.text_play import play_game


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play one game against an agent')
    parser.add_argument('-g','--game', help='game to play [tris, connect4]', required=True)
    parser.add_argument('-o','--opp', help='Opponent to play against [human, random, mcts, minimax]', required=True)
    args = vars(parser.parse_args())

    # game
    if args["game"].lower() == "tris":
        env = TrisEnv()
    elif args["game"].lower() == "connect4":
        env = Connect4Env()
    else:
        raise ValueError(f"{args["game"]} not a valid game")

    # opponent
    if args["opp"].lower() == "human":
        opp = HumanAgent()
    elif args["opp"].lower() == "random":
        opp = RandomAgent()
    elif args["opp"].lower() == "mcts":
        opp = MCTSAgent()
    elif args["opp"].lower() == "minimax":
        opp = MinimaxAgent()
    else:
        raise ValueError(f"{args["opp"]} not a valid opponent")
    
    agents = [HumanAgent(), opp]
    if agents[0].name == agents[1].name:
        agents[0].name += "_1"
        agents[1].name += "_2"
        
    play_game(env, agents, starter=0, render=True)