"""Default agent settings when launching pygame."""

from enum import Enum
from giotto.agents.human import HumanAgent
from giotto.agents.random import RandomAgent
from giotto.agents.minimax import MinimaxAgent
from giotto.agents.mcts import MCTSAgent
from giotto.agents.giotto import GiottoAgent


class PlayerType(Enum):
    HUMAN = "Human"
    RANDOM = "Random"
    MINIMAX = "Minimax"
    MCTS = "MCTS"
    GIOTTO_TRIS = "Giotto_tris"
    GIOTTO_C4 = "Giotto_c4"


PLAYER_ORDER = [
    PlayerType.HUMAN,
    PlayerType.RANDOM,
    PlayerType.MINIMAX,
    PlayerType.MCTS,
    PlayerType.GIOTTO_TRIS,
    PlayerType.GIOTTO_C4,
]

AGENT_CLASS_MAP = {
    PlayerType.HUMAN: HumanAgent,
    PlayerType.RANDOM: RandomAgent,
    PlayerType.MINIMAX: MinimaxAgent,
    PlayerType.MCTS: lambda: MCTSAgent(simulations=1000, cpuct=1.4),
    PlayerType.GIOTTO_TRIS: lambda: GiottoAgent(game="tris", simulations=1000),
    PlayerType.GIOTTO_C4: lambda: GiottoAgent(game="connect4", simulations=2000),
}
