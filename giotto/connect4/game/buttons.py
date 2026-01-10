import pygame
import giotto.connect4.game.settings as settings
from enum import Enum

class PlayerType(Enum):
    HUMAN = "Human"
    RANDOM = "Random"
    # GIOTTO = "Giotto"
    MINIMAX = "Minimax"
    MCTS = "MCTS"

PLAYER_ORDER = [
    PlayerType.HUMAN,
    PlayerType.RANDOM,
    # PlayerType.GIOTTO,
    PlayerType.MINIMAX,
    PlayerType.MCTS,
]

class Button(pygame.sprite.Sprite):
    def __init__(
        self,
        text: str,
        pos: tuple[int, int],
        size: tuple[int, int] = (300, 60),
        bg_color=(40, 40, 40),
        text_color=(255, 255, 255),
    ):
        super().__init__()

        self.font = pygame.font.Font(None, settings.HEIGHT // 20)
        self.bg_color = bg_color
        self.text_color = text_color
        self.size = size

        self.image = pygame.Surface(size)
        self.rect = self.image.get_rect(center=pos)

        self.set_text(text)

    def set_text(self, text: str):
        self.text = text
        self.image.fill(self.bg_color)
        surf = self.font.render(text, True, self.text_color)
        rect = surf.get_rect(center=(self.size[0] // 2, self.size[1] // 2))
        self.image.blit(surf, rect)

    def clicked(self, mouse_pos: tuple[int, int]) -> bool:
        return self.rect.collidepoint(mouse_pos)


class PlayerSelectButton(Button):
    def __init__(self, label, pos, initial_type):
        self.label = label  # "X" or "O"
        self.player_type = initial_type
        super().__init__(self._label(), pos)

    def _label(self) -> str:
        return f"{self.label.upper()} player: {self.player_type.value}"

    def next_option(self):
        idx = PLAYER_ORDER.index(self.player_type)
        self.player_type = PLAYER_ORDER[(idx + 1) % len(PLAYER_ORDER)]
        self.set_text(self._label())
