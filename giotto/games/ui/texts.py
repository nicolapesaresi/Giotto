import pygame


class TextSprite(pygame.sprite.Sprite):
    def __init__(self, text: str, font: pygame.font.Font, color: tuple, pos: tuple[int, int]):
        super().__init__()
        self.font = font
        self.color = color
        self.text = text
        self.image = self.font.render(self.text, True, self.color)
        self.rect = self.image.get_rect(topleft=pos)

    def update_text(self, new_text: str | int):
        if isinstance(new_text, int):
            new_text = str(new_text)
        self.text = new_text
        self.image = self.font.render(self.text, True, self.color)
        self.rect = self.image.get_rect(topleft=self.rect.topleft)


class XOSprite(TextSprite):
    """Generic X/O sprite that supports both single-cell (tris) and row/col (connect4)
    usage. If `pos_arg` is a single int, it's treated as cell id mapped via
    `settings.cell_map`. If two ints are provided, they are treated as (row, col)
    and converted to pixel coordinates using `settings` grid offsets and cell sizes.
    """

    def __init__(self, sign: str, *pos_args, settings_module=None):
        assert sign in ("o", "x"), "Invalid sign received"
        color = settings_module.O_COLOR if sign == "o" else settings_module.X_COLOR
        # font size may be tuned by caller via settings
        font = pygame.font.Font(None, settings_module.HEIGHT // 5)

        if len(pos_args) == 1:
            # cell id (tris)
            cell = pos_args[0]
            pos = settings_module.cell_map[cell]
        elif len(pos_args) == 2:
            # row, col (connect4) - convert to pixel
            row, col = pos_args
            pos = (settings_module.GRID_X0 + col * settings_module.CELL_WIDTH, settings_module.GRID_Y0 + row * settings_module.CELL_HEIGHT)
        else:
            raise ValueError("Invalid position args for XOSprite")

        super().__init__(sign, font, color, pos)
        self.rect.centerx = pos[0] + settings_module.CELL_WIDTH / 2
        self.rect.centery = pos[1] + settings_module.CELL_HEIGHT / 2


class TurnSprite(pygame.sprite.Sprite):
    def __init__(self, sign: str, turn: int, settings_module):
        super().__init__()
        assert sign in ("o", "x"), "Invalid sign received"

        self.text_color = (255, 255, 255)
        self.sign_color = settings_module.O_COLOR if sign == "o" else settings_module.X_COLOR

        font_size = settings_module.HEIGHT // 15
        self.font = pygame.font.Font(None, font_size)

        before = f"Turn {turn} - "
        after = " to move"

        surf_before = self.font.render(before, True, self.text_color)
        surf_sign = self.font.render(sign.upper(), True, self.sign_color)
        surf_after = self.font.render(after, True, self.text_color)

        width = surf_before.get_width() + surf_sign.get_width() + surf_after.get_width()
        height = max(surf_before.get_height(), surf_sign.get_height(), surf_after.get_height())
        self.image = pygame.Surface((width, height), pygame.SRCALPHA)

        self.image.blit(surf_before, (0, 0))
        self.image.blit(surf_sign, (surf_before.get_width(), 0))
        self.image.blit(surf_after, (surf_before.get_width() + surf_sign.get_width(), 0))

        self.rect = self.image.get_rect()
        self.rect.centerx = settings_module.WIDTH // 2
        self.rect.y = int(settings_module.HEIGHT * 0.05)


class ResultSprite(pygame.sprite.Sprite):
    def __init__(self, result: str | int, settings_module):
        super().__init__()
        assert result in ("o", "x", -1), f"Invalid sign received: {result}"

        self.text_color = (255, 255, 255)
        self.sign_color = settings_module.O_COLOR if result == "o" else settings_module.X_COLOR

        font_size = settings_module.HEIGHT // 15
        self.font = pygame.font.Font(None, font_size)

        before = "Game over - "
        if result == -1:
            text = before + "draw"
            self.image = self.font.render(text, True, self.text_color)
        else:
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

        self.rect = self.image.get_rect()
        self.rect.centerx = settings_module.WIDTH // 2
        self.rect.y = int(settings_module.HEIGHT * 0.05)


class TitleText(TextSprite):
    def __init__(self, settings_module):
        text = getattr(settings_module, "GAME_TITLE", "Giotto Game")
        font = pygame.font.Font(None, size=settings_module.HEIGHT // 5)
        color = "yellow"
        pos = (settings_module.WIDTH // 2, settings_module.HEIGHT * 0.3)
        super().__init__(text, font, color, pos)
        self.rect.centerx = settings_module.WIDTH // 2


class PressToPlayText(TextSprite):
    def __init__(self, settings_module):
        text = "Press enter to start a game"
        font = pygame.font.SysFont(None, settings_module.HEIGHT // 20, italic=True)
        color = "white"
        pos = (settings_module.WIDTH // 2, settings_module.HEIGHT * 0.5)
        super().__init__(text, font, color, pos)
        self.rect.centerx = settings_module.WIDTH // 2


class PressToGoToMenuText(TextSprite):
    def __init__(self, settings_module):
        text = "Press enter to retry"
        font = pygame.font.SysFont(None, settings_module.HEIGHT // 20, italic=True)
        color = "white"
        pos = (settings_module.WIDTH // 2, settings_module.HEIGHT * 0.9)
        super().__init__(text, font, color, pos)
        self.rect.centerx = settings_module.WIDTH // 2
