import asyncio  # noqa: I001
import numpy  # needed here, otherwise pygbag breaks  # noqa: F401
import pygame
from enum import IntEnum

import giotto.games.settings.settings_global as settings
from giotto.games.connect4 import PygameConnect4
from giotto.games.launcher import PygameLauncher
from giotto.games.tris import PygameTris
from giotto.games.ui.states import States

# torch is not used in browser mode, numpy version of neural network is used instead

pygame.init()
screen = pygame.display.set_mode((settings.WIDTH, settings.HEIGHT))


class GlobalStates(IntEnum):
    """Manages main loop states."""

    LAUNCHER = 0
    GAME = 1


async def main():
    """Main loop for browser mode."""
    global_state = GlobalStates.LAUNCHER
    launcher = PygameLauncher(screen)

    # main combined loop
    while True:
        # launcher loop
        if global_state == GlobalStates.LAUNCHER:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pos = event.pos
                    if launcher.tictactoe_btn.clicked(pos):
                        game = PygameTris(
                            screen=launcher.screen,
                            start_at_menu=True,
                            from_launcher=True,
                        )
                        global_state = GlobalStates.GAME
                        game.clock = pygame.time.Clock()
                        render_state = game.initial_state

                    elif launcher.connect4_btn.clicked(pos):
                        game = PygameConnect4(
                            screen=launcher.screen,
                            start_at_menu=True,
                            from_launcher=True,
                        )
                        global_state = GlobalStates.GAME
                        game.clock = pygame.time.Clock()
                        render_state = game.initial_state
            launcher.draw()
        # game loop
        elif global_state == GlobalStates.GAME:
            action = None
            game.load_gamescreen()
            if render_state == States.CLOSE:
                game.close()
            elif render_state == States.MENU:
                render_state = game.main_menu()

            elif render_state == States.LAUNCHER:
                global_state = GlobalStates.LAUNCHER
                continue

            elif render_state == States.GAMEOVER:
                render_state = game.gameover_screen()

            elif render_state == States.GAME:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        game.close()

                    if game.is_human_turn():
                        action = game.check_move_click(event)

                if action is None and not game.is_human_turn():
                    pygame.time.delay(200)
                    action = game.agents[game.env.current_player].select_action(game.env)

                if action is not None:
                    game.env.step(action)

                if render_state not in (States.CLOSE, States.LAUNCHER):
                    game.draw_screen()
                    game.draw_text()

                if game.env.done:
                    render_state = States.GAMEOVER
            game.clock.tick()

        pygame.display.flip()
        await asyncio.sleep(0)


asyncio.run(main())
