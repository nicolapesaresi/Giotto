import pygame
import giotto.tris.game.settings as settings


class TextSprite(pygame.sprite.Sprite):
    """Generic text sprite object."""

    def __init__(
        self,
        text: str,
        font: pygame.font.Font,
        color: tuple[int, int, int],
        pos: tuple[int, int],
    ):
        super().__init__()
        self.font = font
        self.color = color
        self.text = text

        self.image = self.font.render(self.text, True, self.color)
        self.rect = self.image.get_rect(topleft=pos)

    def update_text(self, new_text: str | int):
        """Updates sprite text."""
        if isinstance(new_text, int):
            new_text = str(new_text)
        self.text = new_text
        self.image = self.font.render(self.text, True, self.color)
        self.rect = self.image.get_rect(topleft=self.rect.topleft)

class XOSprite(TextSprite):
    """O/X-element for the playing board."""

    def __init__(self, sign: str, cell:int):
        """Instantiates sprite.
        Args:
            sign: 'x' or 'o'.
            cell: position on the playing board (1-9).
        """
        assert sign in ("o","x"), "Invalid sign received"
        if sign == "o":
            color = "blue"
        else:
            color = "red"
        font = pygame.font.Font(None, size=settings.HEIGHT // 5)
        pos = settings.cell_map[cell]
        super().__init__(sign, font, color, pos)
        # center text
        self.rect.centerx = pos[0] + settings.CELL_WIDTH / 2
        self.rect.centery = pos[1] + settings.CELL_HEIGHT / 2

class TurnSprite(pygame.sprite.Sprite):
    """Current turn indicator text with colored sign."""

    def __init__(self, sign: str, turn: int):
        super().__init__()
        assert sign in ("o", "x"), "Invalid sign received"

        self.text_color = (255, 255, 255)  # white
        self.sign_color = (0, 0, 255) if sign == "o" else (255, 0, 0)

        font_size = settings.HEIGHT // 15
        self.font = pygame.font.Font(None, font_size)

        before = f"Turn {turn} - "
        after = " to move"

        surf_before = self.font.render(before, True, self.text_color)
        surf_sign = self.font.render(sign.upper(), True, self.sign_color)
        surf_after = self.font.render(after, True, self.text_color)

        width = surf_before.get_width() + surf_sign.get_width() + surf_after.get_width()
        height = max(surf_before.get_height(), surf_sign.get_height(), surf_after.get_height())
        self.image = pygame.Surface((width, height), pygame.SRCALPHA)  # allow transparency

        self.image.blit(surf_before, (0, 0))
        self.image.blit(surf_sign, (surf_before.get_width(), 0))
        self.image.blit(surf_after, (surf_before.get_width() + surf_sign.get_width(), 0))

        # Correct rect
        self.rect = self.image.get_rect()
        self.rect.centerx = settings.WIDTH // 2
        self.rect.y = int(settings.HEIGHT * 0.05)

class ResultSprite(pygame.sprite.Sprite):
    """Result text with colored sign."""

    def __init__(self, result: str | int):
        super().__init__()
        assert result in ("o", "x", -1), f"Invalid sign received: {result}"

        self.text_color = (255, 255, 255)  # white
        self.sign_color = (255, 0, 0) if result == "x" else (0, 0, 255)

        font_size = settings.HEIGHT // 15
        self.font = pygame.font.Font(None, font_size)

        before = "Game over - "
        if result == -1:
            # draw: no colored sign
            text = before + "draw"
            self.image = self.font.render(text, True, self.text_color)
        else:
            # winner: include colored sign
            after = " wins"
            surf_before = self.font.render(before, True, self.text_color)
            surf_sign = self.font.render(result.upper(), True, self.sign_color)
            surf_after = self.font.render(after, True, self.text_color)

            width = surf_before.get_width() + surf_sign.get_width() + surf_after.get_width()
            height = max(surf_before.get_height(), surf_sign.get_height(), surf_after.get_height())
            self.image = pygame.Surface((width, height), pygame.SRCALPHA)

            self.image.blit(surf_before, (0, 0))
            self.image.blit(surf_sign, (surf_before.get_width(), 0))
            self.image.blit(surf_after, (surf_before.get_width() + surf_sign.get_width(), 0))

        # Correct rect
        self.rect = self.image.get_rect()
        self.rect.centerx = settings.WIDTH // 2
        self.rect.y = int(settings.HEIGHT * 0.05)


class TitleText(TextSprite):
    """Main menu title text sprite."""

    def __init__(self):
        text = "Giotto TicTacToe"
        font = pygame.font.Font(None, size=settings.HEIGHT // 5)
        color = "yellow"
        pos = (settings.WIDTH // 2, settings.HEIGHT * 0.4)
        super().__init__(text, font, color, pos)
        # center text
        self.rect.centerx = settings.WIDTH // 2


class PressToPlayText(TextSprite):
    """Text that indiucates how to start a game."""

    def __init__(self):
        text = "Press enter to start a game"
        font = pygame.font.SysFont(None, settings.HEIGHT // 20, italic=True)
        color = "white"
        pos = (settings.WIDTH // 2, settings.HEIGHT * 0.55)
        super().__init__(text, font, color, pos)
        # center text
        self.rect.centerx = settings.WIDTH // 2


class PressToGoToMenuText(TextSprite):
    """Text that indicates how to go back to main menu."""

    def __init__(self):
        text = "Press enter to retry"
        font = pygame.font.SysFont(None, settings.HEIGHT // 20, italic=True)
        color = "white"
        pos = (settings.WIDTH // 2, settings.HEIGHT * 0.9)
        super().__init__(text, font, color, pos)
        # center text
        self.rect.centerx = settings.WIDTH // 2
