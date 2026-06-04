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
    REWARD_LOSE = -1.0        # killed (HP hits 0), or grabbing the key past the guard
    REWARD_HIT = -0.3         # taking non-lethal damage from an enemy

    MAX_HP = 100
    MELEE_ENEMY_DAMAGE = 50
    GUARD_DAMAGE = 50
    WARLOCK_DAMAGE = 50
    WARLOCK_FIREBALL_RANGE = 3
    SPIKE_DAMAGE = 34

    FREEZE_TICKS = 2
    SHOOT_RANGE = 3           # how many tiles a shot reaches in a straight line
    N_SPIKES = 3
    TOTAL_MELEE_ENEMIES = 4

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
            "hit": cls.REWARD_HIT,
            "shoot_range": cls.SHOOT_RANGE,
            "freeze_ticks": cls.FREEZE_TICKS,
            "hp": cls.MAX_HP,
            "enemy_damage": cls.MELEE_ENEMY_DAMAGE,
            "guard_damage": cls.GUARD_DAMAGE,
            "warlock_damage": cls.WARLOCK_DAMAGE,
            "warlock_fireball_range": cls.WARLOCK_FIREBALL_RANGE,
            "n_spikes": cls.N_SPIKES,
            "spike_damage": cls.SPIKE_DAMAGE
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

            # place remaining stuff on remaining empty (floor) cells. We need one
            # cell each for the 6 fixed entities (player, exit, key, two melee
            # enemies, warlock) plus one per spike.
            floor_cells = self._floor_cells()
            n_sampled = 6 + self.N_SPIKES
            if len(floor_cells) < n_sampled:
                continue

            positions = random.sample(floor_cells, n_sampled)
            self.player_pos = list(positions[0])
            self.exit_pos   = list(positions[1])
            self.freeze_available = True
            self.key_pos = list(positions[2])
            self.guard_pos = self._initialize_guard_pos()
            enemy_pos  = list(positions[3])
            enemy_2_pos = list(positions[4])
            self.melee_poses = [enemy_pos, enemy_2_pos]
            self.total_melee_enemies = 2
            self.warlock_pos = list(positions[5])
            self.warlock_fireball_pos = None
            self.warlock_fireball_dir = None
            self.warlock_fireball_ticks = 0
            # Stored as lists (not the sampled tuples) so == comparisons with
            # player_pos work: tuple == list is always False in Python.
            self.spike_poses = [list(positions[6 + i]) for i in range(self.N_SPIKES)]
            self.spike_statuses = [False] * self.N_SPIKES

            self.freeze_ticks = 0
            self.has_key = False
            self.hp = self.MAX_HP

            melee_valid = all(
                self._is_reachable(self.player_pos, pos) and self._bfs_distance(self.player_pos, pos) > 4
                for pos in self.melee_poses
            )
            if not melee_valid:
                continue


            if (self._is_reachable(self.player_pos, self.exit_pos)
                    and self._is_reachable(self.player_pos, self.key_pos)
                    and self._is_reachable_unidirectional_any(self.guard_pos)
                    and self._bfs_distance(self.player_pos, self.key_pos) >= 2
                    and self._bfs_distance(self.player_pos, self.exit_pos) >= 4
                    and self.guard_pos is not None):
                break

        self.done = False
        self.steps = 0
        # Straight-line path of the most recent shot, for rendering only. Set by
        # _shoot, cleared every step. Has no effect on the agent's observation.
        self.last_shot = None
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

    def _is_reachable_unidirectional_any(self, entity_pos):
        """
        Making sure that the guard is reachable (not blocked by a wall or a key).
        What is meant by blocked by a key is that if the guard is surrounded from 3 sides and then by a key from one
        side, then to get the key, the player has to walk into the guard and get damaged; therefore, an check_for_key
        is set to True.
        :return:
        """
        return (
            self._is_reachable_unidirectional(self.player_pos, entity_pos, "left", True) or
            self._is_reachable_unidirectional(self.player_pos, entity_pos, "right", True) or
            self._is_reachable_unidirectional(self.player_pos, entity_pos, "up", True) or
            self._is_reachable_unidirectional(self.player_pos, entity_pos, "down", True)
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
                reward, terminated = self._damage_player(self.GUARD_DAMAGE)
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

        return reward, terminated

    def _check_for_spikes(self):
        reward, terminated = 0, False
        for i in range(self.N_SPIKES):
            if self.spike_statuses[i] and self.spike_poses[i] == self.player_pos:
                reward, terminated = self._damage_player(self.SPIKE_DAMAGE)

        return reward, terminated

    def _damage_player(self, damage):
        terminated = False
        self.hp -= damage
        # Use <= 0 (not == 0): a stronger hit, or two hits in one step, can take
        # HP below zero, and == 0 would miss those and leave the player "alive".
        if self.hp <= 0:
            self.hp = 0
            reward = self.REWARD_LOSE
            terminated = True
            self.done = True
        else:
            reward = self.REWARD_HIT

        return reward, terminated

    def step(self, action):
        """
        Actions: 0=up, 1=down, 2=left, 3=right, 4=shoot left, 5=shoot right, 6=shoot up, 7=shoot down, 8=freeze
        Returns: state, reward, done
        """
        if self.done:
            raise RuntimeError("Episode is over. Call reset().")

        reward = 0
        terminated = False
        truncated = False
        # Cleared each step; only a shoot action repopulates it (for rendering).
        self.last_shot = None

        # Exactly one spike is active this step (chosen here).
        self._spikes()
        pos_before = list(self.player_pos)

        # Spike check #1: the player is on the spike as it triggers. This covers
        # staying put on a spike that fires, and being caught while stepping off
        # it the same step.
        spike_reward, terminated = self._check_for_spikes()
        reward += spike_reward

        # Distance to the nearest enemy *before* the player acts, so the
        # enemy-proximity shaping can actually credit the player's own move.
        # Only movement actions are shaped this way; shooting is rewarded by
        # the kill bonus in _shoot instead.
        old_enemy_dist = self._nearest_enemy_distance() if action < 4 else None

        # A lethal spike at the start of the step skips the player's action.
        if not terminated:
            if action == 8:
                if self.freeze_available:
                    reward += self.REWARD_FREEZE
                    self._activate_freeze_powerup()
                # Freezing while the warlock is still alive provokes it: the
                # warlock damages the player. This is the intended cost of using
                # freeze before the warlock has been dealt with.
                if self.warlock_pos is not None:
                    hit_reward, terminated = self._damage_player(self.WARLOCK_DAMAGE)
                    reward += hit_reward
            elif action < 4:
                move_reward, terminated = self._move_agent(action, terminated)
                reward += move_reward
            else:
                reward += self._shoot(action)

            # Spike check #2: the player moved onto an active spike this step.
            # Gated on an actual tile change so a player who stayed put is never
            # charged twice for the same tile (check #1 already covered it).
            if not terminated and self.player_pos != pos_before:
                spike_reward, terminated = self._check_for_spikes()
                reward += spike_reward

        # Enemies (melee + warlock) act after the player, unless frozen. The
        # freeze tick is consumed here so it gates every enemy the same way.
        # terminated is checked first: if the player has just reached the exit,
        # the enemies' last move is irrelevant.
        if not terminated:
            if self.freeze_ticks > 0:
                self.freeze_ticks -= 1
            else:
                if any(melee_pos is not None for melee_pos in self.melee_poses):
                    reward, terminated = self._next_enemy_action(reward, terminated, old_enemy_dist)
                if not terminated:
                    reward, terminated = self._warlock_action(reward, terminated)


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

    def _next_enemy_action(self, reward, terminated, old_enemy_dist):
        # Freeze gating now lives in step() so it applies to the warlock too.
        self.melee_poses = self._move_melee_enemies()
        self.melee_poses = self._move_melee_enemies()

        # Reward opening up the gap to the nearest threat, penalise closing it.
        # Using the nearest enemy (not the sum over all enemies) keeps this term
        # on the same scale regardless of how many enemies are alive.
        new_enemy_dist = self._nearest_enemy_distance()
        if old_enemy_dist is not None and new_enemy_dist is not None:
            reward += (new_enemy_dist - old_enemy_dist) * self.REWARD_ENEMY_STEP

        for melee_pos in self.melee_poses:
            if melee_pos == self.player_pos:
                reward, terminated = self._damage_player(self.MELEE_ENEMY_DAMAGE)

        return reward, terminated

    def _warlock_action(self, reward, terminated):
        """Advance an in-flight fireball; if none is flying, a living warlock
        launches one straight toward the player. A fireball already in flight
        keeps travelling even if the warlock is killed before it lands, which is
        why despawning it lives in _advance_fireball (called as long as a
        fireball exists) rather than being tied to the warlock being alive."""
        if self.warlock_fireball_pos is not None:
            return self._advance_fireball(reward, terminated)

        if self.warlock_pos is not None:
            self._launch_fireball()
            if self.warlock_fireball_pos is not None:
                return self._advance_fireball(reward, terminated)

        return reward, terminated

    def _launch_fireball(self):
        """Spawn a fireball at the warlock heading toward the player whenever the
        player shares the warlock's row or column. The fireball travels straight
        and passes through walls, so no line-of-sight check is needed; it just
        needs the player to be axis-aligned to have a direction to fly in."""
        wr, wc = self.warlock_pos
        pr, pc = self.player_pos
        if wr == pr and wc != pc:
            direction = (0, 1) if pc > wc else (0, -1)
        elif wc == pc and wr != pr:
            direction = (1, 0) if pr > wr else (-1, 0)
        else:
            return  # player is not aligned with the warlock; nothing to fire at

        self.warlock_fireball_dir = direction
        self.warlock_fireball_pos = list(self.warlock_pos)
        self.warlock_fireball_ticks = self.WARLOCK_FIREBALL_RANGE

    def _advance_fireball(self, reward, terminated):
        """Move the fireball one tile along its fixed direction. It passes
        through walls and despawns only when it leaves the grid, hits the player
        (dealing damage), or uses up its range."""
        dr, dc = self.warlock_fireball_dir
        nr = self.warlock_fireball_pos[0] + dr
        nc = self.warlock_fireball_pos[1] + dc

        # No wall check: fireballs fly through walls. Only stop at the map edge
        # (so we never index out of bounds) or when the range is spent.
        in_bounds = 0 <= nr < self.grid_size and 0 <= nc < self.grid_size
        if not in_bounds or self.warlock_fireball_ticks == 0:
            self._clear_fireball()
            return reward, terminated

        self.warlock_fireball_pos = [nr, nc]
        self.warlock_fireball_ticks -= 1

        if self.warlock_fireball_pos == self.player_pos:
            reward, terminated = self._damage_player(self.WARLOCK_DAMAGE)
            self._clear_fireball()
        elif self.warlock_fireball_ticks == 0:
            self._clear_fireball()

        return reward, terminated

    def _clear_fireball(self):
        self.warlock_fireball_pos = None
        self.warlock_fireball_dir = None
        self.warlock_fireball_ticks = 0


    def _activate_freeze_powerup(self):
        self.freeze_ticks = self.FREEZE_TICKS
        self.freeze_available  = False


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
        directions = {4: "left", 5: "right", 6: "up", 7: "down"}
        deltas = {"left": (0, -1), "right": (0, 1), "up": (-1, 0), "down": (1, 0)}
        direction = directions[action]

        enemies = [*self.melee_poses, self.guard_pos, self.warlock_pos]
        enemies = [enemy for enemy in enemies if enemy is not None]

        closest_enemy = (
            self._get_closest_unidirectional_enemy(enemies, direction)
            if enemies else None
        )

        reward = 0
        hit = None
        hit_sprite = None
        # at this point, bfs distance is just unidirectional distance because it
        # was checked for it in _get_closest_unidirectional_enemy
        if (closest_enemy is not None
                and self._bfs_distance(self.player_pos, closest_enemy) <= self.SHOOT_RANGE):
            hit = list(closest_enemy)
            if list(closest_enemy) == self.guard_pos:
                self.guard_pos = None
                hit_sprite = "guard"
            elif list(closest_enemy) == self.warlock_pos:
                self.warlock_pos = None
                hit_sprite = "warlock"
            else:
                hit_sprite = "enemy"
                for i in range(len(self.melee_poses)):
                    if list(closest_enemy) == self.melee_poses[i]:
                        self.melee_poses[i] = None
            reward = self.REWARD_KILL

        # Record the projectile's path purely for rendering; the game logic above
        # is unchanged (still hitscan) and the observation never sees this.
        self.last_shot = self._build_shot_path(deltas[direction], hit, hit_sprite)
        return reward

    def _build_shot_path(self, delta, hit, hit_sprite):
        """Tiles a shot visibly travels through, in a straight line from the
        player up to SHOOT_RANGE (stopping at a wall or the target it hit).
        Used only by the renderer to animate the otherwise-instant shot."""
        dr, dc = delta
        r, c = self.player_pos
        path = []
        for _ in range(self.SHOOT_RANGE):
            r += dr
            c += dc
            if self.grid[r, c] == WALL:
                break
            path.append([r, c])
            if hit is not None and [r, c] == hit:
                break
        return {"path": path, "hit": hit, "hit_sprite": hit_sprite}



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

    def _spikes(self):
        # randint is inclusive on both ends, so the range is [0, N_SPIKES - 1].
        spike_n = random.randint(0, self.N_SPIKES - 1)
        for i in range(self.N_SPIKES):
            self.spike_statuses[i] = (i == spike_n)


    def _get_state(self):
        """Returns a copy of the grid with player and exit marked."""
        # creating grids filled with zeros
        walls = (self.grid == WALL).astype(np.float32)
        player = (np.zeros_like(self.grid, dtype=np.float32))
        exit_ = (np.zeros_like(self.grid, dtype=np.float32))
        freeze = (np.zeros_like(self.grid, dtype=np.float32))
        # separate channel where every tile has the same value -
        # how many ticks of freeze status are left (normalized)
        freeze_status = (np.full((self.grid_size, self.grid_size), self.freeze_ticks / self.FREEZE_TICKS, dtype=np.float32))
        key = (np.zeros_like(self. grid, dtype=np.float32))
        guard = (np.zeros_like(self. grid, dtype=np.float32))
        # All melee enemies share a single channel: they are the same entity
        # type, so the agent should react to any of them the same way. A
        # different type of enemy (e.g. an archer) would get its own channel.
        melee = np.zeros_like(self.grid, dtype=np.float32)
        warlock = np.zeros_like(self.grid, dtype=np.float32)
        warlock_fireball = np.zeros_like(self.grid, dtype=np.float32)
        # Constant plane carrying the player's current HP (normalized) so the
        # agent can actually observe how much damage it can still take, the same
        # way freeze_status exposes the freeze timer.
        hp = np.full((self.grid_size, self.grid_size), self.hp / self.MAX_HP, dtype=np.float32)
        spikes = np.zeros_like(self.grid, dtype=np.float32)


        # setting every object's position to 1 in the respective grid
        player[self.player_pos[0], self.player_pos[1]] = 1.0
        exit_[self.exit_pos[0], self.exit_pos[1]] = 1.0
        if self.freeze_available:
            freeze = (np.full((self.grid_size, self.grid_size), 1, dtype=np.float32))
        if self.key_pos is not None:
            key[self.key_pos[0], self.key_pos[1]] = 1.0
        if self.guard_pos is not None:
            guard[self.guard_pos[0], self.guard_pos[1]] = 1.0
        for enemy_pos in self.melee_poses:
            if enemy_pos is not None:
                melee[enemy_pos[0], enemy_pos[1]] = 1.0
        if self.warlock_pos is not None:
            warlock[self.warlock_pos[0], self.warlock_pos[1]] = 1.0
        if self.warlock_fireball_pos is not None:
            warlock_fireball[self.warlock_fireball_pos[0], self.warlock_fireball_pos[1]] = 1.0
        for i in range(len(self.spike_poses)):
            if self.spike_statuses[i]:
                spikes[self.spike_poses[i][0], self.spike_poses[i][1]] = 1.0
            else:
                spikes[self.spike_poses[i][0], self.spike_poses[i][1]] = 0.5

        goal = tuple(self.key_pos) if not self.has_key else tuple(self.exit_pos)

        return np.stack([walls, player, exit_, freeze,
                         freeze_status, self._distance_map(goal), key,
                         guard, melee, warlock, warlock_fireball, hp, spikes],
                        axis=0)


    def _floor_cells(self):
        cells = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r, c] == FLOOR:
                    cells.append((r, c))
        return cells