import pygame
from pygame.sprite import Group
import giotto.games.settings.settings_tris as settings
from giotto.games.settings.agent_settings import AGENT_CLASS_MAP
from giotto.envs.tris import TrisEnv
from giotto.games.ui.texts import XOSprite, TurnSprite
from giotto.games.generic import GenericGame


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
        player_types = {
            pt: AGENT_CLASS_MAP[pt] for pt in settings.SUPPORTED_PLAYER_TYPES
        }
        super().__init__(
            env, settings, player_types, screen, start_at_menu, from_launcher
        )

    def check_move_click(self, event) -> int | None:
        """Checks human input for move."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            cell = None
            if (
                settings.GRID_X0 < x < settings.GRID_X1
                and settings.GRID_Y0 < y < settings.GRID_Y1
            ):
                cell = 1
            elif (
                settings.GRID_X1 < x < settings.GRID_X2
                and settings.GRID_Y0 < y < settings.GRID_Y1
            ):
                cell = 2
            elif (
                settings.GRID_X2 < x < settings.GRID_X3
                and settings.GRID_Y0 < y < settings.GRID_Y1
            ):
                cell = 3
            elif (
                settings.GRID_X0 < x < settings.GRID_X1
                and settings.GRID_Y1 < y < settings.GRID_Y2
            ):
                cell = 4
            elif (
                settings.GRID_X1 < x < settings.GRID_X2
                and settings.GRID_Y1 < y < settings.GRID_Y2
            ):
                cell = 5
            elif (
                settings.GRID_X2 < x < settings.GRID_X3
                and settings.GRID_Y1 < y < settings.GRID_Y2
            ):
                cell = 6
            elif (
                settings.GRID_X0 < x < settings.GRID_X1
                and settings.GRID_Y2 < y < settings.GRID_Y3
            ):
                cell = 7
            elif (
                settings.GRID_X1 < x < settings.GRID_X2
                and settings.GRID_Y2 < y < settings.GRID_Y3
            ):
                cell = 8
            elif (
                settings.GRID_X2 < x < settings.GRID_X3
                and settings.GRID_Y2 < y < settings.GRID_Y3
            ):
                cell = 9
            if cell is not None and cell in self.env.get_valid_actions():
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

    def draw_text(self):
        """Draws game texts on the screen."""
        texts = Group()
        texts.add(
            TurnSprite(
                self.env.signs[self.env.current_player],
                self.env.turn_counter + 1,
                settings,
            )
        )
        texts.draw(self.screen)


# -----------
def launch():
    PygameTris().run()


if __name__ == "__main__":
    launch()
