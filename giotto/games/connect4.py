import pygame
from pygame.sprite import Group

from giotto.envs.connect4 import Connect4Env
from giotto.games.generic import GenericGame
from giotto.games.settings import settings_connect4
from giotto.games.settings.agent_settings import AGENT_CLASS_MAP
from giotto.games.ui.texts import TurnSprite, XOSprite


class PygameConnect4(GenericGame):
    """Pygame renderer for Connect4."""

    def __init__(
        self,
        screen: pygame.Surface | None = None,
        start_at_menu: bool = True,
        from_launcher: bool = False,
    ):
        """Instantiates pygame renderer for Connect4.

        Args:
            screen: None for desktop mode, pygame screen for browser.
            start_at_menu: whether pygame starts form main menu or runs directly.
            from_launcher: whether game was started from launcher.
        """
        env = Connect4Env()
        player_types = {pt: AGENT_CLASS_MAP[pt] for pt in settings_connect4.SUPPORTED_PLAYER_TYPES}
        super().__init__(env, settings_connect4, player_types, screen, start_at_menu, from_launcher)

    def check_move_click(self, event) -> int | None:
        """Checks human input for move."""
        # mouse
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos
        # touchscreen
        elif event.type == pygame.FINGERDOWN:
            # finger events give normalized coordinates (0..1)
            w, h = self.screen.get_size()
            pos = (event.x * w, event.y * h)
        else:
            return None
        for col, rect in self.get_grid_rects().items():
            if rect.collidepoint(pos) and col in self.env.get_valid_actions():
                return col
        return None

    def draw_screen(self):
        """Draws game elements on the screen."""
        self.load_gamescreen()
        # grid
        pygame.draw.line(
            self.screen,
            settings_connect4.GRID_COLOR,
            (settings_connect4.GRID_X0, settings_connect4.GRID_Y0),
            (settings_connect4.GRID_X0, settings_connect4.GRID_Y6),
            settings_connect4.GRID_WIDTH,
        )
        pygame.draw.line(
            self.screen,
            settings_connect4.GRID_COLOR,
            (settings_connect4.GRID_X1, settings_connect4.GRID_Y0),
            (settings_connect4.GRID_X1, settings_connect4.GRID_Y6),
            settings_connect4.GRID_WIDTH,
        )
        pygame.draw.line(
            self.screen,
            settings_connect4.GRID_COLOR,
            (settings_connect4.GRID_X2, settings_connect4.GRID_Y0),
            (settings_connect4.GRID_X2, settings_connect4.GRID_Y6),
            settings_connect4.GRID_WIDTH,
        )
        pygame.draw.line(
            self.screen,
            settings_connect4.GRID_COLOR,
            (settings_connect4.GRID_X3, settings_connect4.GRID_Y0),
            (settings_connect4.GRID_X3, settings_connect4.GRID_Y6),
            settings_connect4.GRID_WIDTH,
        )
        pygame.draw.line(
            self.screen,
            settings_connect4.GRID_COLOR,
            (settings_connect4.GRID_X4, settings_connect4.GRID_Y0),
            (settings_connect4.GRID_X4, settings_connect4.GRID_Y6),
            settings_connect4.GRID_WIDTH,
        )
        pygame.draw.line(
            self.screen,
            settings_connect4.GRID_COLOR,
            (settings_connect4.GRID_X5, settings_connect4.GRID_Y0),
            (settings_connect4.GRID_X5, settings_connect4.GRID_Y6),
            settings_connect4.GRID_WIDTH,
        )
        pygame.draw.line(
            self.screen,
            settings_connect4.GRID_COLOR,
            (settings_connect4.GRID_X6, settings_connect4.GRID_Y0),
            (settings_connect4.GRID_X6, settings_connect4.GRID_Y6),
            settings_connect4.GRID_WIDTH,
        )
        pygame.draw.line(
            self.screen,
            settings_connect4.GRID_COLOR,
            (settings_connect4.GRID_X7, settings_connect4.GRID_Y0),
            (settings_connect4.GRID_X7, settings_connect4.GRID_Y6),
            settings_connect4.GRID_WIDTH,
        )
        pygame.draw.line(
            self.screen,
            settings_connect4.GRID_COLOR,
            (settings_connect4.GRID_X0, settings_connect4.GRID_Y1),
            (settings_connect4.GRID_X7, settings_connect4.GRID_Y1),
            settings_connect4.GRID_WIDTH,
        )
        pygame.draw.line(
            self.screen,
            settings_connect4.GRID_COLOR,
            (settings_connect4.GRID_X0, settings_connect4.GRID_Y2),
            (settings_connect4.GRID_X7, settings_connect4.GRID_Y2),
            settings_connect4.GRID_WIDTH,
        )
        pygame.draw.line(
            self.screen,
            settings_connect4.GRID_COLOR,
            (settings_connect4.GRID_X0, settings_connect4.GRID_Y3),
            (settings_connect4.GRID_X7, settings_connect4.GRID_Y3),
            settings_connect4.GRID_WIDTH,
        )
        pygame.draw.line(
            self.screen,
            settings_connect4.GRID_COLOR,
            (settings_connect4.GRID_X0, settings_connect4.GRID_Y4),
            (settings_connect4.GRID_X7, settings_connect4.GRID_Y4),
            settings_connect4.GRID_WIDTH,
        )
        pygame.draw.line(
            self.screen,
            settings_connect4.GRID_COLOR,
            (settings_connect4.GRID_X0, settings_connect4.GRID_Y5),
            (settings_connect4.GRID_X7, settings_connect4.GRID_Y5),
            settings_connect4.GRID_WIDTH,
        )
        pygame.draw.line(
            self.screen,
            settings_connect4.GRID_COLOR,
            (settings_connect4.GRID_X0, settings_connect4.GRID_Y6),
            (settings_connect4.GRID_X7, settings_connect4.GRID_Y6),
            settings_connect4.GRID_WIDTH,
        )
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

        # last move indicator: subtle outline on the most recently played cell
        if self.env.info["moves"]:
            last_col = int(self.env.info["moves"][-1]) - 1  # 0-indexed
            col_vals = self.env.board[:, last_col]
            last_row = max(i for i, v in enumerate(col_vals) if v != -1)
            draw_row = self.env.rows - 1 - last_row
            cell_rect = pygame.Rect(
                settings_connect4.GRID_X0 + last_col * settings_connect4.CELL_WIDTH,
                settings_connect4.GRID_Y0 + draw_row * settings_connect4.CELL_HEIGHT,
                settings_connect4.CELL_WIDTH,
                settings_connect4.CELL_HEIGHT,
            )
            pygame.draw.rect(self.screen, (160, 160, 160), cell_rect, 2)

        # winning line
        winner = self.env.info.get("winner")
        if self.env.done and winner not in (None, -1):
            winning_cells = self.env.get_winning_cells(winner)
            if winning_cells:
                color = (255, 220, 0)

                def _center(row, col):
                    draw_row = self.env.rows - 1 - row
                    return (
                        settings_connect4.GRID_X0
                        + col * settings_connect4.CELL_WIDTH
                        + settings_connect4.CELL_WIDTH // 2,
                        settings_connect4.GRID_Y0
                        + draw_row * settings_connect4.CELL_HEIGHT
                        + settings_connect4.CELL_HEIGHT // 2,
                    )

                start = _center(*winning_cells[0])
                end = _center(*winning_cells[-1])
                pygame.draw.line(self.screen, color, start, end, 5)

    def draw_text(self):
        """Draws game texts on the screen."""
        texts = Group()
        texts.add(
            TurnSprite(
                self.env.signs[self.env.current_player],
                self.env.turn_counter + 1,
                self.agents[self.env.current_player].name,
                settings_connect4,
            )
        )
        texts.draw(self.screen)

    @staticmethod
    def get_grid_rects() -> dict[int, pygame.Rect]:
        """Retrievs grid rects for clicking on the board."""
        rects = {}
        col_coords = [
            settings_connect4.GRID_X0,
            settings_connect4.GRID_X1,
            settings_connect4.GRID_X2,
            settings_connect4.GRID_X3,
            settings_connect4.GRID_X4,
            settings_connect4.GRID_X5,
            settings_connect4.GRID_X6,
            settings_connect4.GRID_X7,
        ]
        y0 = settings_connect4.GRID_Y0
        y1 = settings_connect4.GRID_Y6

        for i in range(7):
            rects[i + 1] = pygame.Rect(col_coords[i], y0, col_coords[i + 1] - col_coords[i], y1 - y0)
        return rects


# -----------
def launch():
    """Directly launches Connect4 game."""
    PygameConnect4().run()


if __name__ == "__main__":
    launch()
