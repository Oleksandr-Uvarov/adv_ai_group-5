import sys
import os
import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game_engine import Game

class GameEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, grid_size=10, render_mode=None):
        super().__init__()
        self.game = Game(grid_size)
        self.grid_size = grid_size
        self.render_mode = render_mode
        self._renderer = None

        # What the agent sees: a grid_size x grid_size grid for every
        # channel (exit, enemy, powerup, etc.)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(10, grid_size, grid_size),  # channel per entity
            dtype=np.float32
        )

        # What the agent can do: 4 directions to go to and 4 directions to shoot to
        self.action_space = spaces.Discrete(8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
        obs = self.game.reset()
        return obs, {}  # Gymnasium expects (obs, info)

    def step(self, action):
        obs, reward, terminated, truncated = self.game.step(int(action))

        return obs, reward, terminated, truncated, {}  # Gymnasium expects 5 values

    def render(self):
        if self.render_mode == "human":
            if self._renderer is None:
                from pg.pygame_renderer import Renderer
                self._renderer = Renderer(self.grid_size)
            self._renderer.draw(self.game)
        else:
            g = self.game
            grid_display = [['#' if g.grid[r, c] == 1 else '.' for c in range(g.grid_size)] for r in range(g.grid_size)]
            grid_display[g.exit_pos[0]][g.exit_pos[1]] = 'X'
            for enemy_pos in g.melee_poses:
                if enemy_pos is not None:
                    grid_display[enemy_pos[0]][enemy_pos[1]] = 'E'
            grid_display[g.player_pos[0]][g.player_pos[1]] = '@'
            if g.freeze_pos is not None:
                grid_display[g.freeze_pos[0]][g.freeze_pos[1]] = 'F'
            if g.key_pos is not None:
                grid_display[g.key_pos[0]][g.key_pos[1]] = 'K'
            if g.guard_pos is not None:
                grid_display[g.guard_pos[0]][g.guard_pos[1]] = 'G'
            for row in grid_display:
                print(' '.join(row))
            print()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
