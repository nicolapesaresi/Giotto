import pygame
from pygame.sprite import Group

import giotto.games.settings.settings_tris as settings
from giotto.envs.tris import TrisEnv
from giotto.games.generic import GenericGame
from giotto.games.settings.agent_settings import AGENT_CLASS_MAP
from giotto.games.ui.texts import TurnSprite, XOSprite


class PygameTris(GenericGame):
    """Pygame renderer for Tris (TicTacToe), subclassing shared base."""

    def __init__(
        self,
        screen: pygame.Surface | None = None,
        start_at_menu: bool = True,
        from_launcher: bool = False,
    ):
        """Instantiates pygame renderer for TicTacToe.

        Args:
            screen: None for desktop mode, pygame screen for browser.
            start_at_menu: whether pygame starts form main menu or runs directly.
            from_launcher: whether game was started from launcher.
        """
        env = TrisEnv()
        player_types = {pt: AGENT_CLASS_MAP[pt] for pt in settings.SUPPORTED_PLAYER_TYPES}
        super().__init__(env, settings, player_types, screen, start_at_menu, from_launcher)

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
        for cell, rect in self.get_grid_rects().items():
            if rect.collidepoint(pos) and cell in self.env.get_valid_actions():
                return cell
        return None

    def draw_screen(self):
        """Draws game elements on the screen."""
        self.load_gamescreen()
        # grid
        pygame.draw.line(
            self.screen,
            settings.GRID_COLOR,
            (settings.GRID_X1, settings.GRID_Y0),
            (settings.GRID_X1, settings.GRID_Y3),
            settings.GRID_WIDTH,
        )
        pygame.draw.line(
            self.screen,
            settings.GRID_COLOR,
            (settings.GRID_X2, settings.GRID_Y0),
            (settings.GRID_X2, settings.GRID_Y3),
            settings.GRID_WIDTH,
        )
        pygame.draw.line(
            self.screen,
            settings.GRID_COLOR,
            (settings.GRID_X0, settings.GRID_Y1),
            (settings.GRID_X3, settings.GRID_Y1),
            settings.GRID_WIDTH,
        )
        pygame.draw.line(
            self.screen,
            settings.GRID_COLOR,
            (settings.GRID_X0, settings.GRID_Y2),
            (settings.GRID_X3, settings.GRID_Y2),
            settings.GRID_WIDTH,
        )
        # player moves
        moves = Group()
        for cell in range(1, 10):
            cell_player_id = self.env.board[self.env.decode_action(cell)]
            if cell_player_id != -1:
                sign = self.env.signs[cell_player_id]
                moves.add(XOSprite(sign, cell, settings_module=settings))
        moves.draw(self.screen)

        # winning line
        winner = self.env.info.get("winner")
        if self.env.done and winner not in (None, -1):
            winning_cells = self.env.get_winning_cells(winner)
            if winning_cells:
                color = (255, 220, 0)

                def _center(row, col):
                    return (
                        settings.GRID_X0 + col * settings.CELL_WIDTH + settings.CELL_WIDTH // 2,
                        settings.GRID_Y0 + row * settings.CELL_HEIGHT + settings.CELL_HEIGHT // 2,
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
                settings,
            )
        )
        texts.draw(self.screen)

    @staticmethod
    def get_grid_rects() -> dict[int, pygame.Rect]:
        """Retrievs grid rects for clicking on the board."""
        return {
            1: pygame.Rect(
                settings.GRID_X0,
                settings.GRID_Y0,
                settings.GRID_X1 - settings.GRID_X0,
                settings.GRID_Y1 - settings.GRID_Y0,
            ),
            2: pygame.Rect(
                settings.GRID_X1,
                settings.GRID_Y0,
                settings.GRID_X2 - settings.GRID_X1,
                settings.GRID_Y1 - settings.GRID_Y0,
            ),
            3: pygame.Rect(
                settings.GRID_X2,
                settings.GRID_Y0,
                settings.GRID_X3 - settings.GRID_X2,
                settings.GRID_Y1 - settings.GRID_Y0,
            ),
            4: pygame.Rect(
                settings.GRID_X0,
                settings.GRID_Y1,
                settings.GRID_X1 - settings.GRID_X0,
                settings.GRID_Y2 - settings.GRID_Y1,
            ),
            5: pygame.Rect(
                settings.GRID_X1,
                settings.GRID_Y1,
                settings.GRID_X2 - settings.GRID_X1,
                settings.GRID_Y2 - settings.GRID_Y1,
            ),
            6: pygame.Rect(
                settings.GRID_X2,
                settings.GRID_Y1,
                settings.GRID_X3 - settings.GRID_X2,
                settings.GRID_Y2 - settings.GRID_Y1,
            ),
            7: pygame.Rect(
                settings.GRID_X0,
                settings.GRID_Y2,
                settings.GRID_X1 - settings.GRID_X0,
                settings.GRID_Y3 - settings.GRID_Y2,
            ),
            8: pygame.Rect(
                settings.GRID_X1,
                settings.GRID_Y2,
                settings.GRID_X2 - settings.GRID_X1,
                settings.GRID_Y3 - settings.GRID_Y2,
            ),
            9: pygame.Rect(
                settings.GRID_X2,
                settings.GRID_Y2,
                settings.GRID_X3 - settings.GRID_X2,
                settings.GRID_Y3 - settings.GRID_Y2,
            ),
        }


# -----------
def launch():
    """Directly launches Tris game."""
    PygameTris().run()


if __name__ == "__main__":
    launch()
