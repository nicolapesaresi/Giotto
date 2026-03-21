"""Default agent settings when launching pygame."""

from enum import Enum

from giotto.agents.alphazero import AlphaZeroAgent
from giotto.agents.human import HumanAgent
from giotto.agents.mcts import MCTSAgent
from giotto.agents.minimax import MinimaxAgent
from giotto.agents.random import RandomAgent


class PlayerType(Enum):
    """Types of players/agents available."""

    HUMAN = "Human"
    RANDOM = "Random"
    MINIMAX = "Minimax"
    MCTS = "MCTS"
    GIOTTO_TRIS = "Giottino"
    GIOTTO_C4 = "Giotto"


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
    PlayerType.MCTS: lambda: MCTSAgent(simulations=800, cpuct=3.5),
    PlayerType.GIOTTO_TRIS: lambda: AlphaZeroAgent(name="Giottino", game="tris", simulations=100, cpuct=1.5),
    PlayerType.GIOTTO_C4: lambda: AlphaZeroAgent(name="Giotto", game="connect4", simulations=800, cpuct=3.5),
}
