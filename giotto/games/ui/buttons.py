import pygame
from giotto.games.ui.player_types import PlayerType, PLAYER_ORDER

class Button(pygame.sprite.Sprite):
    def __init__(
        self,
        text: str,
        pos: tuple[int, int],
        size: tuple[int, int] = (300, 60),
        bg_color=(40, 40, 40),
        text_color=(255, 255, 255),
        settings_module=None,
    ):
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
        self.text = text
        self.image.fill(self.bg_color)
        surf = self.font.render(text, True, self.text_color)
        rect = surf.get_rect(center=(self.size[0] // 2, self.size[1] // 2))
        self.image.blit(surf, rect)

    def clicked(self, mouse_pos: tuple[int, int]) -> bool:
        return self.rect.collidepoint(mouse_pos)


class PlayerSelectButton(Button):
    def __init__(self, label, pos, initial_type, settings_module=None):
        self.label = label
        self.player_type = initial_type
        super().__init__(self._label(), pos, settings_module=settings_module)

    def _label(self) -> str:
        return f"{self.label.upper()} player: {self.player_type.value}"

    def next_option(self):
        order = getattr(self.settings_module, 'SUPPORTED_PLAYER_TYPES', PLAYER_ORDER)
        idx = order.index(self.player_type)
        self.player_type = order[(idx + 1) % len(order)]
        self.set_text(self._label())
