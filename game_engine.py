import numpy as np
import random

# Tile types
FLOOR = 0
WALL  = 1

class Game:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()
        self.done = False

    def reset(self):
        """Reset to a fresh episode. Returns the initial state."""
        # Simple map: walls on border, floor inside
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.grid[0, :]  = WALL
        self.grid[-1, :] = WALL
        self.grid[:, 0]  = WALL
        self.grid[:, -1] = WALL

        # Place player and exit randomly on floor tiles
        floor_cells = self._floor_cells()
        positions = random.sample(floor_cells, 3)
        self.player_pos = list(positions[0])
        self.exit_pos   = list(positions[1])
        self.enemy_pos  = list(positions[2])
        self.done = False
        self.steps = 0
        return self._get_state()

    def step(self, action):
        """
        Actions: 0=up, 1=down, 2=left, 3=right
        Returns: state, reward, done
        """
        if self.done:
            raise RuntimeError("Episode is over. Call reset().")

        terminated = False
        truncated = False

        old_dist = abs(self.player_pos[0] - self.exit_pos[0]) + \
                    abs(self.player_pos[1] - self.exit_pos[1])

        # Movement deltas
        deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = deltas[action]
        new_r = self.player_pos[0] + dr
        new_c = self.player_pos[1] + dc

        # Don't move into walls
        if self.grid[new_r, new_c] != WALL:
            self.player_pos = [new_r, new_c]

        new_dist = abs(self.player_pos[0] - self.exit_pos[0]) + \
                    abs(self.player_pos[1] - self.exit_pos[1])

        reward = (old_dist - new_dist) * 0.1

        # Check win condition
        if self.player_pos == self.exit_pos:
            speed_bonus = (200 - self.steps) / 200
            reward = 1.0 + speed_bonus
            terminated = True
            self.done = True

        self._move_enemy()

        if self.enemy_pos == self.player_pos:
            reward = -1.0
            terminated = True
            self.done = True

        self.steps += 1
        if self.steps >= 200:  # step limit
            # print("Limit of steps reached!")
            truncated = True
            self.done = True


        return self._get_state(), reward, terminated, truncated

    def _move_enemy(self):
        """Moves enemy one step closer to the player"""
        r, c = self.enemy_pos
        pr, pc = self.player_pos

        options = []
        if pr < r: options.append((-1, 0))
        if pr > r: options.append((1, 0))
        if pc < c: options.append((0, -1))
        if pc > c: options.append((0, 1))

        for dr, dc in options:
            new_r, new_c = r + dr, c + dc
            if self.grid[new_r, new_c] != WALL:
                self.enemy_pos = [new_r, new_c]
                break

    def _get_state(self):
        """Returns a copy of the grid with player and exit marked."""
        state = self.grid.copy()
        state[self.player_pos[0], self.player_pos[1]] = 2  # player = 2
        state[self.exit_pos[0], self.exit_pos[1]]     = 3  # exit   = 3
        # state[self.enemy_pos[0], self.enemy_pos[1]]   = 4

        return state

    def _floor_cells(self):
        cells = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r, c] == FLOOR:
                    cells.append((r, c))
        return cells