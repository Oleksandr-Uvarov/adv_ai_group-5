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
        self.step_limit = 125

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

            if self._is_reachable(self.player_pos, self.exit_pos) and self._is_reachable(self.player_pos, self.enemy_pos):
                break

        self.done = False
        self.steps = 0
        return self._get_state()

    def _is_reachable(self, start, goal):
        """BFS to check if goal is reachable from the start."""
        start = tuple(start)
        goal = tuple(goal)
        if start == goal:
            return True
        visited = {start}
        queue = deque([start])

        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (r, c) == goal:
                    return True
                if (nr, nc) not in visited and self.grid[nr, nc] != WALL:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return False


    def _bfs_distance(self, start, goal):
        """Utility method to get a BFS distance between a start and a goal
            defined as an integer."""
        start = tuple(start)
        goal = tuple(goal)
        if start == goal:
            return 0

        visited = {start: 0}
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited and self.grid[nr, nc] != WALL:
                    visited[(nr, nc)] = visited[(r, c)] + 1
                    if (nr, nc) == goal:
                        return visited[(nr, nc)]
                    queue.append((nr, nc))
        return float("inf") # if exit is unreachable (blocked by walls, which shouldn't normally happen)

    def step(self, action):
        """
        Actions: 0=up, 1=down, 2=left, 3=right
        Returns: state, reward, done
        """
        if self.done:
            raise RuntimeError("Episode is over. Call reset().")

        terminated = False
        truncated = False

        old_exit_dist = self._bfs_distance(self.player_pos, self.exit_pos)
        old_enemy_dist = self._bfs_distance(self.player_pos, self.enemy_pos)

        # Movement deltas
        deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = deltas[action]
        new_r = self.player_pos[0] + dr
        new_c = self.player_pos[1] + dc

        # Don't move into walls
        if self.grid[new_r, new_c] != WALL:
            self.player_pos = [new_r, new_c]

        new_exit_dist = self._bfs_distance(self.player_pos, self.exit_pos)

        reward = (old_exit_dist - new_exit_dist) * 0.1

        # Check win condition
        if self.player_pos == self.exit_pos:
            speed_bonus = max(0.0, (75 - self.steps) / 75)
            reward = 1.0 + speed_bonus
            terminated = True
            self.done = True

        if not terminated:
            self._move_enemy()

        if self.enemy_pos == self.player_pos:
            reward = -1.0
            terminated = True
            self.done = True

        new_enemy_dist = self._bfs_distance(self.player_pos, self.enemy_pos)

        reward -= (old_enemy_dist - new_enemy_dist) * 0.05

        self.steps += 1
        if self.steps >= self.step_limit:  # step limit
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

        goal = (self.exit_pos[0], self.exit_pos[1])
        return np.stack([walls, player, exit_, enemy, self._distance_map(goal)], axis=0)


    def _distance_map(self, goal):
        dist = np.full((self.grid_size, self.grid_size), -1.0, dtype=np.float32)
        goal = tuple(goal)
        queue = deque([goal])
        dist[goal[0], goal[1]] = 0.0

        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if dist[nr, nc] < 0 and self.grid[nr, nc] != WALL:
                    dist[nr, nc] = dist[r, c] + 1
                    queue.append((nr, nc))
        max_d = dist.max()
        if max_d > 0:
            dist = dist / max_d
        return dist

    def _floor_cells(self):
        cells = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r, c] == FLOOR:
                    cells.append((r, c))
        return cells