import pygame
from pathlib import Path

_SPRITES_DIR = Path(__file__).parent / "sprites"

class Renderer:
    TILE_SIZE = 32
    SHOT_FRAME_MS = 60  # delay between projectile animation frames

    def __init__(self, grid_size=10):
        pygame.init()
        self.grid_size = grid_size
        size = grid_size * self.TILE_SIZE
        self.screen = pygame.display.set_mode((size, size))
        pygame.display.set_caption("RL Roguelike")
        self._load_sprites()

    def _load_sprites(self):
        self.sprites = {}
        for name in ("floor", "wall", "player", "exit", "enemy", "freeze", "key", "guard", "fireball"):
            img = pygame.image.load(_SPRITES_DIR / f"{name}.png").convert_alpha()
            self.sprites[name] = pygame.transform.scale(img, (self.TILE_SIZE, self.TILE_SIZE))

    def draw(self, game):
        # Shooting is hitscan in the game logic, but for display we animate the
        # projectile one tile per frame with everything else frozen.
        shot = getattr(game, "last_shot", None)
        if shot is not None and shot["path"]:
            self._animate_shot(game, shot)

        self._draw_scene(game)
        pygame.display.flip()
        self._pump_events()

    def _animate_shot(self, game, shot):
        hit = shot["hit"]
        for tile in shot["path"]:
            # The game state has already removed the killed target, so we redraw
            # it ourselves until the projectile actually reaches its tile.
            extra = (shot["hit_sprite"], hit) if (hit is not None and tile != hit) else None
            self._draw_scene(game, extra=extra, fireball=tile)
            pygame.display.flip()
            self._pump_events()
            pygame.time.delay(self.SHOT_FRAME_MS)

    def _draw_scene(self, game, extra=None, fireball=None):
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
        if extra is not None:
            blit(extra[0], extra[1])
        blit("player", game.player_pos)
        if fireball is not None:
            blit("fireball", fireball)

    def _pump_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit

    def close(self):
        pygame.quit()