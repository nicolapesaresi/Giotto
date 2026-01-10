import pygame
import sys
from pygame.sprite import Group
import giotto.connect4.game.settings as settings
from giotto.connect4.game.states import States
from giotto.connect4.envs.connect4 import Connect4Env
from giotto.connect4.game.texts import XOSprite, TurnSprite, ResultSprite
from giotto.connect4.game.gameover_screen import GameOver
from giotto.connect4.game.main_menu import MainMenu
from giotto.connect4.game.buttons import PlayerType
from giotto.agents.human import HumanAgent
from giotto.agents.random import RandomAgent
from giotto.agents.minimax import MinimaxAgent
from giotto.agents.mcts import MCTSAgent

# importing js only in browser mode
try:
    from js import window
    BROWSER_MODE = True
except ImportError:
    BROWSER_MODE = False


class PygameConnect4:
    """Pygame renderer for Connect4."""

    def __init__(
        self,
        screen: pygame.Surface | None = None,
        start_at_menu: bool = True,
    ):
        """Initializes pygame renderer.
        Args:
            screen: None for desktop mode, pygame screen for browser.
            seed: seed to be passed to the env for random cow generation.
            menu: whether pygame starts form main menu or runs directly.
        """
        if screen is None:
            self.setup_pygame()
        else:
            self.screen = screen
        self.fps = settings.FPS

        self.env = Connect4Env()
        self.env.reset(1)
        self.player_types = {
            PlayerType.HUMAN : HumanAgent,
            PlayerType.RANDOM : RandomAgent,
            # PlayerType.GIOTTO : GiottoAgent,
            PlayerType.MINIMAX : MinimaxAgent,
            PlayerType.MCTS : lambda:MCTSAgent(simulations=10000, cpuct=1.4),
        }
        self.menu_selections = {
            "o_player": PlayerType.HUMAN,
            "x_player": PlayerType.HUMAN,
        }
        self.set_agents(self.menu_selections)

        if start_at_menu:
            self.initial_state = States.MENU
        else:
            self.initial_state = States.GAME
        self.menu = None
        self.gameover = None
        # self.run()

    def setup_pygame(self):
        """Sets up pygame and screen for desktop mode."""
        pygame.init()
        self.screen = pygame.display.set_mode((settings.WIDTH, settings.HEIGHT))
        pygame.display.set_caption("Play Giotto at Connect4")

    def main_menu(self):
        """Renders main menu of the game."""
        if not self.menu:
            self.menu = MainMenu(self.menu_selections)
        self.menu.draw(self.screen)
        outcome = self.menu.handle_events()
        if isinstance(outcome, tuple):
            render_state, agents = outcome
            self.menu_selections = agents
            self.set_agents(agents)
        else:
            render_state = outcome

        # if leaving menu, destroy menu to recreate next time
        if render_state != States.MENU:
            self.menu = None
            # draw empty grid
            self.draw_screen()
            self.draw_text()
            pygame.display.flip()
        return render_state
    
    def set_agents(self, agents):
        """Sets agents that will play the game."""
        self.agents = {
            0: self.player_types[agents["o_player"]](),  # O
            1: self.player_types[agents["x_player"]](),  # X
        }

    def check_move_click(self, event) -> int|None:
        """Check events for click in one on the cells.
        Returns:
            id [1-9] of the clicked cell, or None if no click is detected.
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            col = None
            if settings.GRID_X0 < x < settings.GRID_X1 and settings.GRID_Y0 < y < settings.GRID_Y6:
                col = 1
            elif settings.GRID_X1 < x < settings.GRID_X2 and settings.GRID_Y0 < y < settings.GRID_Y6:
                col = 2
            elif settings.GRID_X2 < x < settings.GRID_X3 and settings.GRID_Y0 < y < settings.GRID_Y6:
                col = 3
            elif settings.GRID_X3 < x < settings.GRID_X4 and settings.GRID_Y0 < y < settings.GRID_Y6:
                col = 4
            elif settings.GRID_X4 < x < settings.GRID_X5 and settings.GRID_Y0 < y < settings.GRID_Y6:
                col = 5
            elif settings.GRID_X5 < x < settings.GRID_X6 and settings.GRID_Y0 < y < settings.GRID_Y6:
                col = 6
            elif settings.GRID_X6 < x < settings.GRID_X7 and settings.GRID_Y0 < y < settings.GRID_Y6:
                col = 7
            # return click col if it's a valid action
            if col is not None and col in self.env.get_valid_actions():
                return col
        return None

    def load_gamescreen(self):
        """Loads main game elements, paddle and cliffs groups."""
        self.screen.fill(settings.BACKGROUND_COLOR)

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

            elif render_state == States.GAMEOVER:
                render_state = self.gameover_screen()

            elif render_state == States.GAME:
                # main game logic
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        render_state = States.CLOSE

                    # player input and env update
                    if isinstance(self.agents[self.env.current_player], HumanAgent):
                        action = self.check_move_click(event)
                # AI move
                if action is None and not isinstance(self.agents[self.env.current_player], HumanAgent):
                    pygame.time.delay(100)
                    action = self.agents[self.env.current_player].select_action(self.env)

                if action is not None:
                    self.env.step(
                        action
                    )

                # draw new screen
                self.draw_screen()
                self.draw_text()

                # check game finished
                if self.env.done:
                    render_state = States.GAMEOVER
            pygame.display.flip()
            self.clock.tick(self.fps)
        self.close()


    def draw_screen(self):
        """Draws the updated screen."""
        self.load_gamescreen()
        # grid
        pygame.draw.line(self.screen, settings.GRID_COLOR, (settings.GRID_X0, settings.GRID_Y0), (settings.GRID_X0, settings.GRID_Y6), settings.GRID_WIDTH)
        pygame.draw.line(self.screen, settings.GRID_COLOR, (settings.GRID_X1, settings.GRID_Y0), (settings.GRID_X1, settings.GRID_Y6), settings.GRID_WIDTH)
        pygame.draw.line(self.screen, settings.GRID_COLOR, (settings.GRID_X2, settings.GRID_Y0), (settings.GRID_X2, settings.GRID_Y6), settings.GRID_WIDTH)
        pygame.draw.line(self.screen, settings.GRID_COLOR, (settings.GRID_X3, settings.GRID_Y0), (settings.GRID_X3, settings.GRID_Y6), settings.GRID_WIDTH)
        pygame.draw.line(self.screen, settings.GRID_COLOR, (settings.GRID_X4, settings.GRID_Y0), (settings.GRID_X4, settings.GRID_Y6), settings.GRID_WIDTH)
        pygame.draw.line(self.screen, settings.GRID_COLOR, (settings.GRID_X5, settings.GRID_Y0), (settings.GRID_X5, settings.GRID_Y6), settings.GRID_WIDTH)
        pygame.draw.line(self.screen, settings.GRID_COLOR, (settings.GRID_X6, settings.GRID_Y0), (settings.GRID_X6, settings.GRID_Y6), settings.GRID_WIDTH)
        pygame.draw.line(self.screen, settings.GRID_COLOR, (settings.GRID_X7, settings.GRID_Y0), (settings.GRID_X7, settings.GRID_Y6), settings.GRID_WIDTH)
        pygame.draw.line(self.screen, settings.GRID_COLOR, (settings.GRID_X0, settings.GRID_Y1), (settings.GRID_X7, settings.GRID_Y1), settings.GRID_WIDTH)
        pygame.draw.line(self.screen, settings.GRID_COLOR, (settings.GRID_X0, settings.GRID_Y2), (settings.GRID_X7, settings.GRID_Y2), settings.GRID_WIDTH)
        pygame.draw.line(self.screen, settings.GRID_COLOR, (settings.GRID_X0, settings.GRID_Y3), (settings.GRID_X7, settings.GRID_Y3), settings.GRID_WIDTH)
        pygame.draw.line(self.screen, settings.GRID_COLOR, (settings.GRID_X0, settings.GRID_Y4), (settings.GRID_X7, settings.GRID_Y4), settings.GRID_WIDTH)
        pygame.draw.line(self.screen, settings.GRID_COLOR, (settings.GRID_X0, settings.GRID_Y5), (settings.GRID_X7, settings.GRID_Y5), settings.GRID_WIDTH)
        pygame.draw.line(self.screen, settings.GRID_COLOR, (settings.GRID_X0, settings.GRID_Y6), (settings.GRID_X7, settings.GRID_Y6), settings.GRID_WIDTH)
        # player moves
        moves = Group()
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                player_id = self.env.board[row, col]
                if player_id != -1:
                    sign = self.env.signs[player_id]

                    draw_row = self.env.rows - 1 - row
                    moves.add(XOSprite(sign, draw_row, col))

        moves.draw(self.screen)

    def draw_text(self):
        """Draws turn indicator text."""
        texts = Group()
        texts.add(TurnSprite(self.env.signs[self.env.current_player], self.env.turn_counter + 1))
        texts.draw(self.screen)

    def gameover_screen(self):
        """Renders game over screen."""
        if not self.gameover:
            winner = self.env.info["winner"]
            if winner == -1:
                result = -1
            else:
                result = self.env.signs[winner]
            self.gameover = GameOver(result)
        self.draw_screen()
        self.gameover.draw(self.screen)
        render_state = self.gameover.handle_events()

        # if leaving menu, destroy gameover screen to recreate next time
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