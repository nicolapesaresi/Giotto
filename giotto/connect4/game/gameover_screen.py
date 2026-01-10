import pygame
from pygame.sprite import Group
from giotto.connect4.game.texts import ResultSprite, PressToGoToMenuText
from giotto.connect4.game.states import States


class GameOver:
    """Handles game over screen."""

    def __init__(self, result: int|str):
        """Instantiates game over screen elements.
        Args:
            score: final score of the game.
        """
        self.game_over_texts = Group()
        self.game_over_texts.add(
            ResultSprite(result), PressToGoToMenuText()
        )

    def draw(self, screen: pygame.Surface):
        """Draw the game over screen. Must be called after PygameRenderer.draw_screen() to have background.
        Args:
            screen: surface to draw on.
        """
        self.game_over_texts.draw(screen)

    def handle_events(self) -> States:
        """Handles main menù events.
        Returns:
            render_state: outcome of event handling.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return States.CLOSE
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                return States.MENU
        return States.GAMEOVER