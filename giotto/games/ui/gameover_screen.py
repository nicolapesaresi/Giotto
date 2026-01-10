import pygame
from pygame.sprite import Group
from giotto.games.ui.texts import ResultSprite, PressToGoToMenuText
from giotto.games.ui.states import States

class GameOver:
    """Handles game over screen."""
    def __init__(self, result: int | str, settings_module):
        """Instantiates game over screen.
        Args:
            result: game result to display.
            settings_module: settings configuration for the chosen game.
        """
        self.settings = settings_module
        self.game_over_texts = Group()
        self.game_over_texts.add(ResultSprite(result, self.settings), PressToGoToMenuText(self.settings))

    def draw(self, screen: pygame.Surface):
        """Draw elements on the screen."""
        self.game_over_texts.draw(screen)

    def handle_events(self) -> object:
        """Handles events on game over screen."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return States.CLOSE
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                return States.MENU
        return States.GAMEOVER
