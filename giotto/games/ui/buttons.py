import pygame
import giotto.games.settings.settings_global as settings_global
from giotto.games.ui.player_types import PLAYER_ORDER


class Button(pygame.sprite.Sprite):
    """Generic button sprite"""

    def __init__(
        self,
        text: str,
        pos: tuple[int, int],
        size: tuple[int, int] = (
            settings_global.WIDTH * 0.375,
            settings_global.HEIGHT * 0.1,
        ),
        bg_color=(40, 40, 40),
        text_color=(255, 255, 255),
        settings_module=None,
    ):
        """Instantiates button sprite.
        Args:
            text: button text.
            pos: center position of the button.
            size: size of the button.
            bg_color: background color of the button.
            text_color: text color of the button.
            settings_module: settings module for the game.
        """
        super().__init__()

        self.font = pygame.font.Font(None, settings_module.HEIGHT // 20)
        self.bg_color = bg_color
        self.text_color = text_color
        self.size = size
        self.settings_module = settings_module

        self.image = pygame.Surface(size)
        self.rect = self.image.get_rect(center=pos)

        self.set_text(text)

    def set_text(self, text: str):
        """Sets button text.
        Args:
            text: button text.
        """
        self.text = text
        self.image.fill(self.bg_color)
        surf = self.font.render(text, True, self.text_color)
        rect = surf.get_rect(center=(self.size[0] // 2, self.size[1] // 2))
        self.image.blit(surf, rect)

    def clicked(self, mouse_pos: tuple[int, int]) -> bool:
        """Checks if button was clicked."""
        return self.rect.collidepoint(mouse_pos)


class PlayerSelectButton(Button):
    def __init__(self, label, pos, initial_type, settings_module=None):
        """Instantiates player select button.
        Args:
            label: "X" or "O".
            pos: center position of the button.
            initial_type: initial player type.
            settings_module: settings module for the game.
        """
        self.label = label
        self.player_type = initial_type
        super().__init__(self._label(), pos, settings_module=settings_module)

    def _label(self) -> str:
        """Returns button label."""
        return f"{self.label.upper()} player: {self.player_type.value}"

    def next_option(self):
        """Cycles to next player type option."""
        order = getattr(self.settings_module, "SUPPORTED_PLAYER_TYPES", PLAYER_ORDER)
        idx = order.index(self.player_type)
        self.player_type = order[(idx + 1) % len(order)]
        self.set_text(self._label())
