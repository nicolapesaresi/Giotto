import pygame
from pygame.sprite import Group
from giotto.tris.game.texts import TitleText, PressToPlayText
from giotto.tris.game.states import States
import giotto.tris.game.settings as settings
from giotto.tris.game.buttons import PlayerSelectButton


class MainMenu:
    def __init__(self, selections: dict):
        self.main_menu_texts = Group()
        self.buttons = Group()

        self.main_menu_texts.add(
            TitleText(),
            PressToPlayText(),
        )

        self.x_button = PlayerSelectButton(
            "X",
            pos=(settings.WIDTH // 2, int(settings.HEIGHT * 0.65)),
            initial_type=selections["x_player"],
        )
        self.o_button = PlayerSelectButton(
            "O",
            pos=(settings.WIDTH // 2, int(settings.HEIGHT * 0.75)),
            initial_type=selections["o_player"],
        )

        self.buttons.add(self.x_button, self.o_button)

    def draw(self, screen: pygame.Surface):
        self.main_menu_texts.draw(screen)
        self.buttons.draw(screen)

    def handle_events(self) -> States | tuple[States, dict]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return States.CLOSE

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for button in self.buttons:
                    if button.clicked(event.pos):
                        button.next_option()

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                # return selections when starting game
                return (
                    States.GAME,
                    {
                        "o_player": self.o_button.player_type,
                        "x_player": self.x_button.player_type,
                    },
                )

        return States.MENU
