from enum import Enum


class PlayerType(Enum):
    HUMAN = "Human"
    RANDOM = "Random"
    MINIMAX = "Minimax"
    MCTS = "MCTS"


PLAYER_ORDER = [
    PlayerType.HUMAN,
    PlayerType.RANDOM,
    PlayerType.MINIMAX,
    PlayerType.MCTS,
]
