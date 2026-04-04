import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game_engine import Game

class GameEnv(gym.Env):
    def __init__(self, grid_size=10):
        super().__init__()
        self.game = Game(grid_size)
        self.grid_size = grid_size

        # What the agent sees: a grid_size x grid_size grid, values 0-3
        self.observation_space = spaces.Box(
            low=0, high=4,
            shape=(grid_size, grid_size),
            dtype=np.int32
        )

        # What the agent can do: 4 directions
        self.action_space = spaces.Discrete(4)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.game.reset()
        return obs, {}  # Gymnasium expects (obs, info)

    def step(self, action):
        obs, reward, done = self.game.step(int(action))
        truncated = False  # you can use this for time limits later
        return obs, reward, done, truncated, {}  # Gymnasium expects 5 values

    def render(self):
        # Simple terminal render for now
        symbols = {0: ".", 1: "#", 2: "@", 3: "X", 4: "E"}
        state = self.game._get_state()
        for row in state:
            print(" ".join(symbols[cell] for cell in row))
        print()