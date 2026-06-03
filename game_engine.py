import numpy as np
import random
from collections import deque

# Tile types
FLOOR = 0
WALL  = 1

class Game:
    # Reward shaping coefficients. Kept here as a single source of truth so the
    # exact values used can be logged with every trained version (see
    # version_utils.env_signature / reward_coeffs).
    REWARD_GOAL_STEP = 0.1    # per-tile progress toward the current goal
    REWARD_ENEMY_STEP = 0.05  # per-tile change in distance to the nearest enemy
    REWARD_KILL = 0.3         # shooting an enemy (melee or guard)
    REWARD_KEY = 0.3          # picking up the key
    REWARD_FREEZE = 0.15      # picking up the freeze power-up
    REWARD_WIN = 1.0          # reaching the exit holding the key (+ speed bonus)
    REWARD_LOSE = -1.0        # caught by an enemy, or grabbing the key past the guard

    @classmethod
    def reward_coeffs(cls):
        """Reward coefficients as a plain dict, for logging/versioning."""
        return {
            "goal_step": cls.REWARD_GOAL_STEP,
            "enemy_step": cls.REWARD_ENEMY_STEP,
            "kill": cls.REWARD_KILL,
            "key": cls.REWARD_KEY,
            "freeze": cls.REWARD_FREEZE,
            "win": cls.REWARD_WIN,
            "lose": cls.REWARD_LOSE,
        }

    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()
        self.done = False
        self.step_limit = 100

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

            positions = random.sample(floor_cells, 6)
            self.player_pos = list(positions[0])
            self.exit_pos   = list(positions[1])
            enemy_pos  = list(positions[2])
            self.freeze_pos = list(positions[3])
            self.key_pos = list(positions[4])
            self.guard_pos = self._initialize_guard_pos()
            enemy_2_pos = list(positions[5])
            self.freeze_ticks = 0
            self.has_key = False
            self.melee_poses = [enemy_pos, enemy_2_pos]

            melee_valid = all(
                self._is_reachable(self.player_pos, pos) and self._bfs_distance(self.player_pos, pos) > 4
                for pos in self.melee_poses
            )
            if not melee_valid:
                continue


            if (self._is_reachable(self.player_pos, self.exit_pos)
                    and self._is_reachable(self.player_pos, self.freeze_pos)
                    and self._is_reachable(self.player_pos, self.key_pos)
                    and self._is_guard_reachable()
                    and self._bfs_distance(self.player_pos, self.key_pos) >= 2
                    and self._bfs_distance(self.player_pos, self.exit_pos) >= 4
                    and self.guard_pos is not None):
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

    def _is_reachable_unidirectional(self, start, goal, direction, check_for_key=False):
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
            if ((nr, nc) not in visited and self.grid[nr, nc] != WALL
                    and (check_for_key is False or (check_for_key and [nr, nc] != self.key_pos))):
                visited.add((nr, nc))
                queue.append((nr, nc))
        return False

    def _is_guard_reachable(self):
        return (
            self._is_reachable_unidirectional(self.player_pos, self.guard_pos, "left", True) or
            self._is_reachable_unidirectional(self.player_pos, self.guard_pos, "right", True) or
            self._is_reachable_unidirectional(self.player_pos, self.guard_pos, "up", True) or
            self._is_reachable_unidirectional(self.player_pos, self.guard_pos, "down", True)
        )

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

    def _initialize_guard_pos(self):
        r, c = self.key_pos
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if self.grid[nr, nc] == FLOOR:
                return [nr, nc]
        return None


    def _move_agent(self, action, terminated):
        if self.has_key:
            goal_pos = self.exit_pos
        else:
            goal_pos = self.key_pos

        old_goal_dist = self._bfs_distance(self.player_pos, goal_pos)
        # Movement deltas
        deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = deltas[action]
        new_r = self.player_pos[0] + dr
        new_c = self.player_pos[1] + dc

        # Don't move into walls
        if self.grid[new_r, new_c] != WALL:
            self.player_pos = [new_r, new_c]

        new_goal_dist = self._bfs_distance(self.player_pos, goal_pos)

        # if old exit distance is greater than new exit distance,
        # then reward is positive.
        reward = (old_goal_dist - new_goal_dist) * self.REWARD_GOAL_STEP

        if self.key_pos and self.player_pos == self.key_pos:
            if self.guard_pos is not None:
                reward = self.REWARD_LOSE
                terminated = True
                self.done = True
            else:
                reward += self.REWARD_KEY
                self.key_pos = None
                self.has_key = True

        # Check win condition
        elif self.player_pos == self.exit_pos and self.has_key:
            speed_bonus = max(0.0, (50 - self.steps) / 50)
            reward = self.REWARD_WIN + speed_bonus
            terminated = True
            self.done = True

        elif self.player_pos == self.freeze_pos:
            reward += self.REWARD_FREEZE
            self._pickup_freeze_powerup()
        return reward, terminated



    def step(self, action):
        """
        Actions: 0=up, 1=down, 2=left, 3=right, 4=shoot left, 5=shoot right, 6=shoot up, 7=shoot down
        Returns: state, reward, done
        """
        if self.done:
            raise RuntimeError("Episode is over. Call reset().")

        terminated = False
        truncated = False

        # Distance to the nearest enemy *before* the player acts, so the
        # enemy-proximity shaping can actually credit the player's own move.
        # Only movement actions are shaped this way; shooting is rewarded by
        # the kill bonus in _shoot instead.
        old_enemy_dist = self._nearest_enemy_distance() if action < 4 else None

        if action < 4:
            reward, terminated = self._move_agent(action, terminated)
        else:
            reward = self._shoot(action)

        if not terminated and any(melee_pos is not None for melee_pos in self.melee_poses):
            # checking for terminated status because if player has reached the exit
            # then the enemy's last step isn't relevant
            terminated, reward = self._next_enemy_action(terminated, reward, old_enemy_dist)


        self.steps += 1
        if self.steps >= self.step_limit and not terminated:  # step limit
            truncated = True
            self.done = True


        return self._get_state(), reward, terminated, truncated

    def _move_melee_enemies(self):
        """Moves melee enemies two steps closer to the player using BFS."""
        starts = [tuple(melee_pos) if melee_pos is not None else None for melee_pos in self.melee_poses]
        goal  = tuple(self.player_pos)

        for i in range(len(starts)):
            if starts[i] == goal or starts[i] is None:
                continue

            # BFS: visited maps each cell to the cell it was reached from
            visited = {starts[i]: None}
            queue = deque([starts[i]])

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
            while visited[current] != starts[i]:
                current = visited[current]

            starts[i] = list(current)

        return [list(pos) if pos is not None else None for pos in starts]

    def _nearest_enemy_distance(self):
        """BFS distance from the player to the closest living enemy,
        or None if every enemy has been killed."""
        dists = [
            self._bfs_distance(self.player_pos, melee_pos)
            for melee_pos in self.melee_poses if melee_pos is not None
        ]
        return min(dists) if dists else None

    def _next_enemy_action(self, terminated, reward, old_enemy_dist):
        if self.freeze_ticks > 0:
            self.freeze_ticks -= 1
            return terminated, reward

        self.melee_poses = self._move_melee_enemies()
        self.melee_poses = self._move_melee_enemies()

        for melee_pos in self.melee_poses:
            if melee_pos == self.player_pos:
                reward = self.REWARD_LOSE
                terminated = True
                self.done = True
                return terminated, reward

        # Reward opening up the gap to the nearest threat, penalise closing it.
        # Using the nearest enemy (not the sum over all enemies) keeps this term
        # on the same scale regardless of how many enemies are alive.
        new_enemy_dist = self._nearest_enemy_distance()
        if old_enemy_dist is not None and new_enemy_dist is not None:
            reward += (new_enemy_dist - old_enemy_dist) * self.REWARD_ENEMY_STEP

        return terminated, reward


    def _pickup_freeze_powerup(self):
        self.freeze_ticks = 3
        self.freeze_pos = None


    def _get_closest_unidirectional_enemy(self, enemies, direction):
        shortest_distance = 100000
        closest_enemy = None

        for enemy in enemies:
            if self._is_reachable_unidirectional(self.player_pos, enemy, direction):
                distance = self._bfs_distance(self.player_pos, enemy)
                if distance < shortest_distance:
                    shortest_distance = distance
                    closest_enemy = enemy

        return closest_enemy

    def _shoot(self, action):
        enemies = [*self.melee_poses, self.guard_pos]
        enemies = [enemy for enemy in enemies if enemy is not None]
        if len(enemies) == 0:
            return 0

        directions = {4: "left", 5: "right", 6: "up", 7: "down"}
        direction = directions[action]

        closest_enemy = self._get_closest_unidirectional_enemy(enemies, direction)

        if closest_enemy is None:
            return 0

        # at this point, bfs distance is just unidirectional distance because it was checked for it above
        if self._bfs_distance(self.player_pos, closest_enemy) <= 3:
            if list(closest_enemy) == self.guard_pos:
                self.guard_pos = None
            else:
                for i in range(len(self.melee_poses)):
                    if list(closest_enemy) == self.melee_poses[i]:
                        self.melee_poses[i] = None
            return self.REWARD_KILL

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
        freeze = (np.zeros_like(self.grid, dtype=np.float32))
        # separate channel where every tile has the same value -
        # how many ticks of freeze status are left (normalized)
        freeze_status = (np.full((self.grid_size, self.grid_size), self.freeze_ticks / 3.0, dtype=np.float32))
        key = (np.zeros_like(self. grid, dtype=np.float32))
        guard = (np.zeros_like(self. grid, dtype=np.float32))
        # All melee enemies share a single channel: they are the same entity
        # type, so the agent should react to any of them the same way. A
        # different type of enemy (e.g. an archer) would get its own channel.
        melee = np.zeros_like(self.grid, dtype=np.float32)


        # setting every object's position to 1 in the respective grid
        player[self.player_pos[0], self.player_pos[1]] = 1.0
        exit_[self.exit_pos[0], self.exit_pos[1]] = 1.0
        if self.freeze_pos is not None:
            freeze[self.freeze_pos[0], self.freeze_pos[1]] = 1.0
        if self.key_pos is not None:
            key[self.key_pos[0], self.key_pos[1]] = 1.0
        if self.guard_pos is not None:
            guard[self.guard_pos[0], self.guard_pos[1]] = 1.0
        for enemy_pos in self.melee_poses:
            if enemy_pos is not None:
                melee[enemy_pos[0], enemy_pos[1]] = 1.0

        goal = tuple(self.key_pos) if not self.has_key else tuple(self.exit_pos)

        return np.stack([walls, player, exit_, freeze,
                         freeze_status, self._distance_map(goal), key,
                         guard, melee],
                        axis=0)


    def _floor_cells(self):
        cells = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r, c] == FLOOR:
                    cells.append((r, c))
        return cells