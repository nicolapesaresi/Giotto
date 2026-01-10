import pygame
from pygame.sprite import Group
from giotto.games.ui.texts import TitleText, PressToPlayText
from giotto.games.ui.buttons import PlayerSelectButton, Button


class MainMenu:
    def __init__(self, selections: dict, settings_module, states_class):
        self.settings = settings_module
        self.states = states_class
        self.main_menu_texts = Group()
        self.buttons = Group()

        self.main_menu_texts.add(
            TitleText(self.settings),
            PressToPlayText(self.settings),
        )

        self.back_button = Button(
            "Back to launcher",
            pos=(self.settings.WIDTH // 2, int(self.settings.HEIGHT * 0.15)),
            size=(250, 60),
            settings_module=self.settings,
        )

        self.x_button = PlayerSelectButton(
            "X",
            pos=(self.settings.WIDTH // 2, int(self.settings.HEIGHT * 0.7)),
            initial_type=selections["x_player"],
            settings_module=self.settings,
        )
        self.o_button = PlayerSelectButton(
            "O",
            pos=(self.settings.WIDTH // 2, int(self.settings.HEIGHT * 0.82)),
            initial_type=selections["o_player"],
            settings_module=self.settings,
        )

        self.buttons.add(self.x_button, self.o_button, self.back_button)

    def draw(self, screen: pygame.Surface):
        self.main_menu_texts.draw(screen)
        self.buttons.draw(screen)

    def handle_events(self) -> object:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return self.states.CLOSE

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for button in self.buttons:
                    if button.clicked(event.pos):
                        if isinstance(button, PlayerSelectButton):
                            button.next_option()
                        else:
                            return self.states.LAUNCHER

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                return (
                    self.states.GAME,
                    {
                        "o_player": self.o_button.player_type,
                        "x_player": self.x_button.player_type,
                    },
                )

        return self.states.MENU
