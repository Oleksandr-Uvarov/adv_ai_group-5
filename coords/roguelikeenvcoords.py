from game_engine import Game
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RoguelikeEnvCoords(gym.Env):
    def __init__(self, grid_size=10):
        super().__init__()
        self.game = Game(grid_size)
        self.grid_size = grid_size

        # 6 numbers: player_r, player_c, exit_r, exit_c, enemy_r, enemy_c
        # all normalized to [0, 1] so the network doesn't have to care about grid size
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(4)

    def _get_coords_obs(self):
        g = self.grid_size - 1  # for normalization
        pr, pc = self.game.player_pos
        er, ec = self.game.exit_pos
        # nr, nc = self.game.enemy_pos
        # return np.array([pr/g, pc/g, er/g, ec/g, nr/g, nc/g], dtype=np.float32)
        return np.array([pr/g, pc/g, er/g, ec/g], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        return self._get_coords_obs(), {}

    def step(self, action):
        _, reward, terminated, truncated = self.game.step(int(action.item()))
        return self._get_coords_obs(), reward, terminated, truncated, {}

    def render(self):
        symbols = {0: ".", 255: "#", 64: "@", 128: "X", 192: "E"}
        state = self.game._get_state()[0]
        for row in state:
            print(" ".join(symbols.get(cell, "?") for cell in row))
        print()