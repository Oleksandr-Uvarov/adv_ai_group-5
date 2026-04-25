import numpy as np
import random
from collections import deque

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
        while True:
            self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
            self.grid[0, :]  = WALL
            self.grid[-1, :] = WALL
            self.grid[:, 0]  = WALL
            self.grid[:, -1] = WALL

            for r in range(1, self.grid_size - 1):
                for c in range(1, self.grid_size - 1):
                    if random.random() < 0.2:
                        self.grid[r, c] = WALL

            # Place player and exit randomly on floor tiles
            floor_cells = self._floor_cells()
            if len(floor_cells) < 3:
                continue


            positions = random.sample(floor_cells, 3)
            self.player_pos = list(positions[0])
            self.exit_pos   = list(positions[1])
            self.enemy_pos  = list(positions[2])

            if self._is_reachable(self.player_pos, self.exit_pos):
                break

        self.done = False
        self.steps = 0
        return self._get_state()

    def _is_reachable(self, start, goal):
        """BFS to check if goal is reachable from the start."""
        start = tuple(start)
        goal = tuple(goal)
        visited = {start}
        queue = deque([start])

        while queue:
            r, c = queue.popleft()
            if (r,c) == goal:
                return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited and self.grid[nr, nc] != WALL:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return False

    def step(self, action):
        """
        Actions: 0=up, 1=down, 2=left, 3=right
        Returns: state, reward, done
        """
        if self.done:
            raise RuntimeError("Episode is over. Call reset().")

        terminated = False
        truncated = False

        old_exit_dist = abs(self.player_pos[0] - self.exit_pos[0]) + \
                    abs(self.player_pos[1] - self.exit_pos[1])

        old_enemy_dist = abs(self.player_pos[0] - self.enemy_pos[0]) + \
                    abs(self.player_pos[1] - self.enemy_pos[1])

        # Movement deltas
        deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = deltas[action]
        new_r = self.player_pos[0] + dr
        new_c = self.player_pos[1] + dc

        # Don't move into walls
        if self.grid[new_r, new_c] != WALL:
            self.player_pos = [new_r, new_c]

        new_exit_dist = abs(self.player_pos[0] - self.exit_pos[0]) + \
                    abs(self.player_pos[1] - self.exit_pos[1])

        reward = (old_exit_dist - new_exit_dist) * 0.1

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

        new_enemy_dist = abs(self.player_pos[0] - self.enemy_pos[0]) + \
                    abs(self.player_pos[1] - self.enemy_pos[1])

        reward -= (old_enemy_dist - new_enemy_dist) * 0.05

        self.steps += 1
        if self.steps >= 200:  # step limit
            # print("Limit of steps reached!")
            truncated = True
            self.done = True


        return self._get_state(), reward, terminated, truncated

    def _move_enemy(self):
        """Moves enemy one step closer to the player using BFS."""
        start = tuple(self.enemy_pos)
        goal  = tuple(self.player_pos)

        if start == goal:
            return

        # BFS: visited maps each cell to the cell it was reached from
        visited = {start: None}
        queue = deque([start])

        while queue:
            current = queue.popleft()
            if current == goal:
                break
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = current[0] + dr, current[1] + dc
                neighbor = (nr, nc)
                if neighbor not in visited and self.grid[nr, nc] != WALL:
                    visited[neighbor] = current
                    queue.append(neighbor)

        if goal not in visited:
            return  # player unreachable, stay put

        # Trace back from goal to find the first step
        current = goal
        while visited[current] != start:
            current = visited[current]

        self.enemy_pos = list(current)

    def _get_state(self):
        """Returns a copy of the grid with player and exit marked."""
        walls = (self.grid == WALL).astype(np.float32)
        player = (np.zeros_like(self.grid, dtype=np.float32))
        exit_ = (np.zeros_like(self.grid, dtype=np.float32))
        enemy = (np.zeros_like(self.grid, dtype=np.float32))

        player[self.player_pos[0], self.player_pos[1]] = 1.0
        exit_[self.exit_pos[0], self.exit_pos[1]] = 1.0
        enemy[self.enemy_pos[0], self.enemy_pos[1]] = 1.0

        return np.stack([walls, player, exit_, enemy], axis=0)
        # return np.stack([walls, player, exit_], axis=0)

    def _floor_cells(self):
        cells = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r, c] == FLOOR:
                    cells.append((r, c))
        return cells