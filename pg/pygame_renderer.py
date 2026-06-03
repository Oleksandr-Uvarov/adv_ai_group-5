import pygame
from pathlib import Path

_SPRITES_DIR = Path(__file__).parent / "sprites"

class Renderer:
    TILE_SIZE = 32

    def __init__(self, grid_size=10):
        pygame.init()
        self.grid_size = grid_size
        size = grid_size * self.TILE_SIZE
        self.screen = pygame.display.set_mode((size, size))
        pygame.display.set_caption("RL Roguelike")
        self._load_sprites()

    def _load_sprites(self):
        self.sprites = {}
        for name in ("floor", "wall", "player", "exit", "enemy", "freeze", "key", "guard"):
            img = pygame.image.load(_SPRITES_DIR / f"{name}.png").convert_alpha()
            self.sprites[name] = pygame.transform.scale(img, (self.TILE_SIZE, self.TILE_SIZE))

    def draw(self, game):
        ts = self.TILE_SIZE

        for r in range(game.grid_size):
            for c in range(game.grid_size):
                sprite = "wall" if game.grid[r, c] == 1 else "floor"
                self.screen.blit(self.sprites[sprite], (c * ts, r * ts))

        def blit(name, pos):
            if pos is not None:
                self.screen.blit(self.sprites[name], (pos[1] * ts, pos[0] * ts))


        blit("exit", game.exit_pos)
        blit("freeze", game.freeze_pos)
        blit("key", game.key_pos)
        for enemy_pos in game.melee_poses:
            blit("enemy", enemy_pos)
        blit("guard", game.guard_pos)
        blit("player", game.player_pos)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit


    def close(self):
        pygame.quit()
