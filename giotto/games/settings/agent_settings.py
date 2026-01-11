"""Default agent settings when launching pygame."""

from giotto.games.ui.player_types import PlayerType
from giotto.agents.human import HumanAgent
from giotto.agents.random import RandomAgent
from giotto.agents.minimax import MinimaxAgent
from giotto.agents.mcts import MCTSAgent

AGENT_CLASS_MAP = {
    PlayerType.HUMAN: HumanAgent,
    PlayerType.RANDOM: RandomAgent,
    PlayerType.MINIMAX: MinimaxAgent,
    PlayerType.MCTS: lambda: MCTSAgent(simulations=10000, cpuct=1.4),
}
