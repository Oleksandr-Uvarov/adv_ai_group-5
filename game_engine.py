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
        self.step_limit = 200

    def reset(self):
        """Reset to a fresh episode. Returns the initial state."""
        # Simple map: walls on border, floor inside
        while True:
            # Initializing an empty grid
            self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
            # Setting walls on all sides
            self.grid[0, :]  = WALL
            self.grid[-1, :] = WALL
            self.grid[:, 0]  = WALL
            self.grid[:, -1] = WALL

            # for every row and column, add a wall with a 20% chance
            # to add some random patterns into the game
            for r in range(1, self.grid_size - 1):
                for c in range(1, self.grid_size - 1):
                    if random.random() < 0.2:
                        self.grid[r, c] = WALL

            # place remaining stuff on remaining empty (floor) cells
            floor_cells = self._floor_cells()
            if len(floor_cells) < 4:
                continue

            positions = random.sample(floor_cells, 4)
            self.player_pos = list(positions[0])
            self.exit_pos   = list(positions[1])
            self.enemy_pos  = list(positions[2])
            self.freeze_pos = list(positions[3])
            self.freeze_ticks = 0

            if (self._is_reachable(self.player_pos, self.exit_pos)
                    and self._is_reachable(self.player_pos, self.enemy_pos)
                    and self._is_reachable(self.player_pos, self.freeze_pos)
                    and self._bfs_distance(self.player_pos, self.enemy_pos) >= 4
                    and self._bfs_distance(self.player_pos, self.exit_pos) >= 4):
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

    def _is_reachable_unidirectional(self, start, goal, direction):
        start = tuple(start)
        goal = tuple(goal)

        direction_values = {"left": (0, -1), "right": (0, 1), "up": (-1, 0), "down": (1, 0)}
        dr, dc = direction_values[direction]
        if start == goal:
            return True

        visited = {start}
        queue = deque([start])

        while queue:
            r, c = queue.popleft()
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

    def _move_agent(self, action, terminated):
        old_exit_dist = self._bfs_distance(self.player_pos, self.exit_pos)
        # Movement deltas
        deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = deltas[action]
        new_r = self.player_pos[0] + dr
        new_c = self.player_pos[1] + dc

        # Don't move into walls
        if self.grid[new_r, new_c] != WALL:
            self.player_pos = [new_r, new_c]

        new_exit_dist = self._bfs_distance(self.player_pos, self.exit_pos)

        # if old exit distance is greater than new exist distance,
        # then reward is positive.
        reward = (old_exit_dist - new_exit_dist) * 0.1

        # Check win condition
        if self.player_pos == self.exit_pos:
            speed_bonus = max(0.0, (50 - self.steps) / 50)
            reward = 1.0 + speed_bonus
            terminated = True
            self.done = True

        if self.player_pos == self.freeze_pos:
            reward += 0.15
            self._pickup_freeze_powerup()

        return reward, terminated



    def step(self, action):
        """
        Actions: 0=up, 1=down, 2=left, 3=right, 4=shoot left, 5=shoot right, 6=shoot up, shoot down
        Returns: state, reward, done
        """
        if self.done:
            raise RuntimeError("Episode is over. Call reset().")

        terminated = False
        truncated = False
        # checking for terminated status because if player has reached the exit
        # then the enemy's last step isn't relevant
        if not terminated:
            self._next_enemy_action()

        old_enemy_dist = self._bfs_distance(self.player_pos, self.enemy_pos) if self.enemy_pos is not None else float("inf")

        if action < 4:
            reward, terminated = self._move_agent(action, terminated)
        else:
            reward = self._shoot(action)

        if not terminated and self.enemy_pos is not None:
            terminated, reward = self._next_enemy_action(terminated, reward, old_enemy_dist)



        self.steps += 1
        if self.steps >= self.step_limit and not terminated:  # step limit
            truncated = True
            self.done = True


        return self._get_state(), reward, terminated, truncated

    def _move_melee_enemy(self):
        """Moves melee enemy one step closer to the player using BFS."""
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

        # Trace back from goal to find the first step
        current = goal
        while visited[current] != start:
            current = visited[current]

        self.enemy_pos = list(current)

    def _next_enemy_action(self, terminated, reward, old_enemy_dist):
        if self.freeze_ticks > 0:
            self.freeze_ticks -= 1
            return terminated, reward

        self._move_melee_enemy()
        self._move_melee_enemy()

        if self.enemy_pos == self.player_pos:
            reward = -1.0
            terminated = True
            self.done = True
            return terminated, reward


        new_enemy_dist = self._bfs_distance(self.player_pos, self.enemy_pos)

        reward -= (old_enemy_dist - new_enemy_dist) * 0.05

        return terminated, reward


    def _pickup_freeze_powerup(self):
        self.freeze_ticks = 3
        self.freeze_pos = None


    def _shoot(self, action):
        if self.enemy_pos is None:
            return 0
        row_player_pos, col_player_pos = self.player_pos[0], self.player_pos[1]
        row_enemy_pos, col_enemy_pos = self.enemy_pos[0], self.enemy_pos[1]


        if action == 4 or action == 5:
            if row_player_pos != row_enemy_pos:
                return 0

        if action == 6 or action == 7:
            if col_player_pos != col_enemy_pos:
                return 0

        if ((action == 4 and col_enemy_pos < col_enemy_pos and self._is_reachable_unidirectional(self.player_pos, self.enemy_pos, "left"))
        or (action == 5 and col_enemy_pos > col_player_pos and self._is_reachable_unidirectional(self.player_pos, self.enemy_pos, "right"))):
            if abs(col_player_pos - col_enemy_pos) <= 3:
                self.enemy_pos = None
                return 0.3

        if ((action == 6 and row_enemy_pos < row_player_pos and self._is_reachable_unidirectional(self.player_pos, self.enemy_pos, "up"))
        or (action == 7 and row_enemy_pos > row_player_pos and self._is_reachable_unidirectional(self.player_pos, self.enemy_pos, "down"))):
            if abs(row_enemy_pos - row_player_pos) <= 3:
                self.enemy_pos = None
                return 0.3

        return 0



    def _distance_map(self, goal):
        """Distance map that is used as a separate channel
        in _get_state() to give the agent an overview of
        how close the goal is based on a given tile."""
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

    def _get_state(self):
        """Returns a copy of the grid with player and exit marked."""
        # creating grids filled with zeros
        walls = (self.grid == WALL).astype(np.float32)
        player = (np.zeros_like(self.grid, dtype=np.float32))
        exit_ = (np.zeros_like(self.grid, dtype=np.float32))
        enemy = (np.zeros_like(self.grid, dtype=np.float32))
        freeze = (np.zeros_like(self.grid, dtype=np.float32))
        # separate channel where every tile has the same value -
        # how many ticks of freeze status are left (normalized)
        freeze_status = (np.full((self.grid_size, self.grid_size), self.freeze_ticks / 3.0, dtype=np.float32))

        # setting every object's position to 1 in the respective grid
        player[self.player_pos[0], self.player_pos[1]] = 1.0
        exit_[self.exit_pos[0], self.exit_pos[1]] = 1.0
        if self.enemy_pos is not None:
            enemy[self.enemy_pos[0], self.enemy_pos[1]] = 1.0
        if self.freeze_pos is not None:
            freeze[self.freeze_pos[0], self.freeze_pos[1]] = 1.0

        goal = (self.exit_pos[0], self.exit_pos[1])
        return np.stack([walls, player, exit_, enemy, freeze, freeze_status, self._distance_map(goal)], axis=0)


    def _floor_cells(self):
        cells = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r, c] == FLOOR:
                    cells.append((r, c))
        return cells