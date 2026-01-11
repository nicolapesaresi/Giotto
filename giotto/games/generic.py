import pygame
import sys
from types import ModuleType
from giotto.games.ui.states import States
from giotto.envs.generic import GenericEnv
from giotto.games.ui.main_menu import MainMenu
from giotto.games.ui.gameover_screen import GameOver
from giotto.agents.human import HumanAgent


class GenericGame:
    """Generic class for pygame rendering of the envs."""

    def __init__(
        self,
        env: GenericEnv,
        settings_module: ModuleType,
        player_types: dict,
        screen: pygame.Surface | None = None,
        start_at_menu: bool = True,
        from_launcher: bool = False,
    ):
        """Instantiates generic pygame renderer.
        Args:
            env: environment for the chosen game.
            settings_module: settings configuration for the chosen game.
            player_types: supported player types for the chosen game.
            screen: None for desktop mode, pygame screen for browser.
            start_at_menu: whether pygame starts form main menu or runs directly.
            from_launcher: whether game was started from launcher.
        """
        self.settings = settings_module
        self.player_types = player_types  # supported agents for the game
        self.from_launcher = from_launcher

        if screen is None:
            self.setup_pygame()
        else:
            self.screen = screen
        self.fps = self.settings.FPS

        self.env = env
        self.env.reset(1)  # X always starts

        self.menu_selections = {
            "o_player": list(self.player_types.keys())[0],
            "x_player": list(self.player_types.keys())[0],
        }
        self.set_agents(self.menu_selections)

        self.initial_state = States.MENU if start_at_menu else States.GAME
        self.menu = None
        self.gameover = None

    def setup_pygame(self):
        """Sets up pygame and screen for desktop mode."""
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.settings.WIDTH, self.settings.HEIGHT)
        )
        pygame.display.set_caption(self.settings.CAPTION)

    def set_agents(self, agents: dict):
        """Sets up agents that are going to play the game."""
        self.agents = {
            0: self.player_types[agents["o_player"]](),
            1: self.player_types[agents["x_player"]](),
        }

    def main_menu(self) -> States:
        """Renders main menu of the game."""
        if not self.menu:
            self.menu = MainMenu(
                self.menu_selections, self.settings, self.from_launcher
            )
        self.menu.draw(self.screen)
        outcome = self.menu.handle_events()
        if isinstance(outcome, tuple):
            render_state, agents = outcome
            self.menu_selections = agents
            self.set_agents(agents)
        else:
            render_state = outcome

        if render_state != States.MENU and render_state:
            self.menu = None
            if render_state not in (States.CLOSE, States.LAUNCHER):
                self.draw_screen()
                self.draw_text()
                pygame.display.flip()
        return render_state

    def run(self):
        """Runs env loop in pygame desktop mode."""
        self.clock = pygame.time.Clock()
        self.load_gamescreen()

        render_state = self.initial_state
        running = True
        while running:
            action = None
            if render_state == States.CLOSE:
                self.close()
            elif render_state == States.MENU:
                render_state = self.main_menu()

            elif render_state == States.LAUNCHER:
                running = False
                break

            elif render_state == States.GAMEOVER:
                render_state = self.gameover_screen()

            elif render_state == States.GAME:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.close()

                    if self.is_human_turn():
                        action = self.check_move_click(event)

                if action is None and not self.is_human_turn():
                    pygame.time.delay(200)
                    action = self.agents[self.env.current_player].select_action(
                        self.env
                    )

                if action is not None:
                    self.env.step(action)

                if render_state not in (States.CLOSE, States.LAUNCHER):
                    self.draw_screen()
                    self.draw_text()

                if self.env.done:
                    render_state = States.GAMEOVER

            pygame.display.flip()
            self.clock.tick(self.fps)
        if render_state == States.CLOSE:
            self.close()

    def gameover_screen(self) -> States:
        """Renders game over screen."""
        if not self.gameover:
            winner = self.env.info.get("winner")
            if winner == -1:
                result = -1
            elif winner is None:
                result = -1
            else:
                result = getattr(self.env, "signs", lambda: [])[winner]
            self.gameover = GameOver(result, self.settings)
        self.draw_screen()
        self.gameover.draw(self.screen)
        render_state = self.gameover.handle_events()

        if render_state != States.GAMEOVER:
            self.gameover = None
            self.env.reset(1)
            self.load_gamescreen()
        return render_state

    @staticmethod
    def close():
        """Closes pygame."""
        pygame.quit()
        sys.exit()

    def is_human_turn(self) -> bool:
        """Check if it's turn of human player, to expect input."""
        return isinstance(self.agents[self.env.current_player], HumanAgent)

    def load_gamescreen(self):
        """Prints background of gamescreen."""
        self.screen.fill(self.settings.BACKGROUND_COLOR)

    # -------------
    # game specific methods, to be implemented in the child classes
    # -------------
    def draw_screen(self):
        """Draws game elements on the screen."""
        raise NotImplementedError()

    def draw_text(self):
        """Draws game texts on the screen."""
        raise NotImplementedError()

    def check_move_click(self, event):
        """Checks human input for move."""
        raise NotImplementedError()
