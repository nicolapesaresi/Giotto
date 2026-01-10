import pygame
import giotto.games.settings.settings_global as settings
from giotto.games.tris import PygameRenderer as TrisGame
from giotto.games.connect4 import PygameConnect4 as Connect4Game
from giotto.games.ui.buttons import Button


class PygameLauncher:
    """Pygame launcher menu to select between TicTacToe and Connect4."""
    def __init__(self, screen: pygame.Surface | None = None):
        """Instantiates launcher.
        Args:
            screen: None for desktop mode, pygame screen for browser.
        """
        if screen is None:
            self.setup_pygame()
        else:
            self.screen = screen

        # title and prompt fonts
        self.title_font = pygame.font.Font(None, settings.HEIGHT // 6)
        self.prompt_font = pygame.font.Font(None, settings.HEIGHT // 20)

        # buttons
        cx = settings.WIDTH // 2
        self.tictactoe_btn = Button("Tic Tac Toe", pos=(cx, int(settings.HEIGHT * 0.55)), settings_module=settings)
        self.connect4_btn = Button("Connect4", pos=(cx, int(settings.HEIGHT * 0.67)), settings_module=settings)
        self.buttons = [self.tictactoe_btn, self.connect4_btn]

    def setup_pygame(self):
        """Sets up pygame and screen for desktop mode."""
        pygame.init()
        self.screen = pygame.display.set_mode((settings.WIDTH, settings.HEIGHT))
        pygame.display.set_caption(settings.CAPTION)

    def draw(self):
        """Draw elements on the screen."""
        self.screen.fill(settings.BACKGROUND_COLOR)

        title_surf = self.title_font.render("Giotto AI", True, "yellow")
        title_rect = title_surf.get_rect(center=(settings.WIDTH // 2, int(settings.HEIGHT * 0.25)))
        self.screen.blit(title_surf, title_rect)

        prompt_surf = self.prompt_font.render("Select game:", True, (255, 255, 255))
        prompt_rect = prompt_surf.get_rect(center=(settings.WIDTH // 2, int(settings.HEIGHT * 0.42)))
        self.screen.blit(prompt_surf, prompt_rect)

        for btn in self.buttons:
            self.screen.blit(btn.image, btn.rect)

    def run(self):
        """Runs launcher menù."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pos = event.pos
                    if self.tictactoe_btn.clicked(pos):
                        game = TrisGame(screen=self.screen, start_at_menu=True)
                        game.run()
                    elif self.connect4_btn.clicked(pos):
                        game = Connect4Game(screen=self.screen, start_at_menu=True)
                        game.run()
            self.draw()
            pygame.display.flip()
        pygame.quit()


# -----------
def launch():
    PygameLauncher().run()

if __name__ == "__main__":
    launch()
