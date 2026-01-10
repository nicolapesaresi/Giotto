import pygame
from pygame.sprite import Group
from giotto.games.ui.texts import ResultSprite, PressToGoToMenuText


class GameOver:
    def __init__(self, result: int | str, settings_module, states_class):
        self.settings = settings_module
        self.states = states_class
        self.game_over_texts = Group()
        self.game_over_texts.add(ResultSprite(result, self.settings), PressToGoToMenuText(self.settings))

    def draw(self, screen: pygame.Surface):
        self.game_over_texts.draw(screen)

    def handle_events(self) -> object:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return self.states.CLOSE
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                return self.states.MENU
        return self.states.GAMEOVER
