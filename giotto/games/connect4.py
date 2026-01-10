import pygame
from pygame.sprite import Group
import giotto.games.settings.settings_connect4 as settings_connect4
from giotto.games.settings.agent_settings import AGENT_CLASS_MAP
from giotto.envs.connect4 import Connect4Env
from giotto.games.ui.texts import XOSprite, TurnSprite
from giotto.games.generic import GenericGame


class PygameConnect4(GenericGame):
    """Pygame renderer for Connect4."""

    def __init__(self, screen: pygame.Surface | None = None, start_at_menu: bool = True):
        """Instantiates pygame renderer for Connect4.
        Args:
            screen: None for desktop mode, pygame screen for browser.
            start_at_menu: whether pygame starts form main menu or runs directly.
        """
        env = Connect4Env()
        player_types = {pt: AGENT_CLASS_MAP[pt] for pt in settings_connect4.SUPPORTED_PLAYER_TYPES}
        super().__init__(env, settings_connect4, player_types, screen, start_at_menu)

    def check_move_click(self, event) -> int | None:
        """Checks human input for move."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            col = None
            if settings_connect4.GRID_X0 < x < settings_connect4.GRID_X1 and settings_connect4.GRID_Y0 < y < settings_connect4.GRID_Y6:
                col = 1
            elif settings_connect4.GRID_X1 < x < settings_connect4.GRID_X2 and settings_connect4.GRID_Y0 < y < settings_connect4.GRID_Y6:
                col = 2
            elif settings_connect4.GRID_X2 < x < settings_connect4.GRID_X3 and settings_connect4.GRID_Y0 < y < settings_connect4.GRID_Y6:
                col = 3
            elif settings_connect4.GRID_X3 < x < settings_connect4.GRID_X4 and settings_connect4.GRID_Y0 < y < settings_connect4.GRID_Y6:
                col = 4
            elif settings_connect4.GRID_X4 < x < settings_connect4.GRID_X5 and settings_connect4.GRID_Y0 < y < settings_connect4.GRID_Y6:
                col = 5
            elif settings_connect4.GRID_X5 < x < settings_connect4.GRID_X6 and settings_connect4.GRID_Y0 < y < settings_connect4.GRID_Y6:
                col = 6
            elif settings_connect4.GRID_X6 < x < settings_connect4.GRID_X7 and settings_connect4.GRID_Y0 < y < settings_connect4.GRID_Y6:
                col = 7
            if col is not None and col in self.env.get_valid_actions():
                return col
        return None

    def draw_screen(self):
        """Draws game elements on the screen."""
        self.load_gamescreen()
        # grid
        pygame.draw.line(self.screen, settings_connect4.GRID_COLOR, (settings_connect4.GRID_X0, settings_connect4.GRID_Y0), (settings_connect4.GRID_X0, settings_connect4.GRID_Y6), settings_connect4.GRID_WIDTH)
        pygame.draw.line(self.screen, settings_connect4.GRID_COLOR, (settings_connect4.GRID_X1, settings_connect4.GRID_Y0), (settings_connect4.GRID_X1, settings_connect4.GRID_Y6), settings_connect4.GRID_WIDTH)
        pygame.draw.line(self.screen, settings_connect4.GRID_COLOR, (settings_connect4.GRID_X2, settings_connect4.GRID_Y0), (settings_connect4.GRID_X2, settings_connect4.GRID_Y6), settings_connect4.GRID_WIDTH)
        pygame.draw.line(self.screen, settings_connect4.GRID_COLOR, (settings_connect4.GRID_X3, settings_connect4.GRID_Y0), (settings_connect4.GRID_X3, settings_connect4.GRID_Y6), settings_connect4.GRID_WIDTH)
        pygame.draw.line(self.screen, settings_connect4.GRID_COLOR, (settings_connect4.GRID_X4, settings_connect4.GRID_Y0), (settings_connect4.GRID_X4, settings_connect4.GRID_Y6), settings_connect4.GRID_WIDTH)
        pygame.draw.line(self.screen, settings_connect4.GRID_COLOR, (settings_connect4.GRID_X5, settings_connect4.GRID_Y0), (settings_connect4.GRID_X5, settings_connect4.GRID_Y6), settings_connect4.GRID_WIDTH)
        pygame.draw.line(self.screen, settings_connect4.GRID_COLOR, (settings_connect4.GRID_X6, settings_connect4.GRID_Y0), (settings_connect4.GRID_X6, settings_connect4.GRID_Y6), settings_connect4.GRID_WIDTH)
        pygame.draw.line(self.screen, settings_connect4.GRID_COLOR, (settings_connect4.GRID_X7, settings_connect4.GRID_Y0), (settings_connect4.GRID_X7, settings_connect4.GRID_Y6), settings_connect4.GRID_WIDTH)
        pygame.draw.line(self.screen, settings_connect4.GRID_COLOR, (settings_connect4.GRID_X0, settings_connect4.GRID_Y1), (settings_connect4.GRID_X7, settings_connect4.GRID_Y1), settings_connect4.GRID_WIDTH)
        pygame.draw.line(self.screen, settings_connect4.GRID_COLOR, (settings_connect4.GRID_X0, settings_connect4.GRID_Y2), (settings_connect4.GRID_X7, settings_connect4.GRID_Y2), settings_connect4.GRID_WIDTH)
        pygame.draw.line(self.screen, settings_connect4.GRID_COLOR, (settings_connect4.GRID_X0, settings_connect4.GRID_Y3), (settings_connect4.GRID_X7, settings_connect4.GRID_Y3), settings_connect4.GRID_WIDTH)
        pygame.draw.line(self.screen, settings_connect4.GRID_COLOR, (settings_connect4.GRID_X0, settings_connect4.GRID_Y4), (settings_connect4.GRID_X7, settings_connect4.GRID_Y4), settings_connect4.GRID_WIDTH)
        pygame.draw.line(self.screen, settings_connect4.GRID_COLOR, (settings_connect4.GRID_X0, settings_connect4.GRID_Y5), (settings_connect4.GRID_X7, settings_connect4.GRID_Y5), settings_connect4.GRID_WIDTH)
        pygame.draw.line(self.screen, settings_connect4.GRID_COLOR, (settings_connect4.GRID_X0, settings_connect4.GRID_Y6), (settings_connect4.GRID_X7, settings_connect4.GRID_Y6), settings_connect4.GRID_WIDTH)
        # player moves
        moves = Group()
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                player_id = self.env.board[row, col]
                if player_id != -1:
                    sign = self.env.signs[player_id]

                    draw_row = self.env.rows - 1 - row
                    moves.add(XOSprite(sign, draw_row, col, settings_module=settings_connect4))

        moves.draw(self.screen)

    def draw_text(self):
        """Draws game texts on the screen."""
        texts = Group()
        texts.add(TurnSprite(self.env.signs[self.env.current_player], self.env.turn_counter + 1, settings_connect4))
        texts.draw(self.screen)