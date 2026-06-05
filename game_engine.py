import numpy as np
import random

from pathfinding import (
    FLOOR, WALL, floor_cells, wall_neighbor_count, adjacent_floor_pos,
    is_reachable, is_reachable_avoiding, is_reachable_unidirectional,
    is_reachable_unidirectional_any, bfs_distance, distance_map,
)
from enemies import EnemyMixin


class Game(EnemyMixin):
    # Enemy behaviour (melee + warlock + fireball) lives in EnemyMixin
    # (enemies.py); all the grid/BFS maths lives in pathfinding.py. This file
    # keeps the map, the player, rewards and the observation.

    # Reward shaping coefficients. Kept here as a single source of truth so the
    # exact values used can be logged with every trained version (see
    # version_utils.env_signature / reward_coeffs).
    REWARD_GOAL_STEP = 0.1    # per-tile progress toward the current goal
    REWARD_KILL = 0.3         # shooting an enemy (melee, guard or warlock)
    REWARD_KEY = 0.3          # picking up the key
    REWARD_FREEZE = 0.3       # using the freeze power-up safely (warlock dead)
    REWARD_ALL_CLEARED = 0.5  # one-time: every enemy defeated, the exit opens
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
    # The player faces TOTAL_MELEE_ENEMIES melee enemies over an episode, but only
    # MAX_ACTIVE_MELEE are ever on the board at once: kill one and another walks in
    # from the reserve until all TOTAL_MELEE_ENEMIES have been defeated.
    TOTAL_MELEE_ENEMIES = 6
    MAX_ACTIVE_MELEE = 3

    @classmethod
    def reward_coeffs(cls):
        """Reward coefficients as a plain dict, for logging/versioning."""
        return {
            "goal_step": cls.REWARD_GOAL_STEP,
            "kill": cls.REWARD_KILL,
            "key": cls.REWARD_KEY,
            "freeze": cls.REWARD_FREEZE,
            "all_cleared": cls.REWARD_ALL_CLEARED,
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
            "spike_damage": cls.SPIKE_DAMAGE,
            "total_melee_enemies": cls.TOTAL_MELEE_ENEMIES,
            "max_active_melee": cls.MAX_ACTIVE_MELEE
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

            # Directly sample the cells for the player, exit, key, warlock and the
            # MAX_ACTIVE_MELEE melee enemies on the board at the start, then seat
            # the guard next to the key and drop the spikes on the remaining
            # floor. Every later placement avoids the cells already taken, so no
            # two entities ever share a tile. We need those sampled cells plus one
            # free cell per spike.
            n_sampled = 4 + self.MAX_ACTIVE_MELEE  # player, exit, key, warlock + melee
            cells = floor_cells(self.grid)
            if len(cells) < n_sampled + self.N_SPIKES:
                continue

            positions = random.sample(cells, n_sampled)
            self.player_pos  = list(positions[0])
            self.exit_pos    = list(positions[1])
            self.key_pos     = list(positions[2])
            self.warlock_pos = list(positions[3])
            self.melee_poses = [list(p) for p in positions[4:]]
            # The remaining enemies wait off-board and spawn in as the active ones
            # are killed (see EnemyMixin._respawn_melee).
            self.melee_reserve = self.TOTAL_MELEE_ENEMIES - self.MAX_ACTIVE_MELEE
            occupied = set(positions)

            self.freeze_available = True
            # The guard is anchored next to the key (the objective it protects),
            # skipping any neighbour already occupied; it comes back None if every
            # free side is walled in or taken (handled by the reachability check
            # below). The warlock used to be anchored to the exit the same way, but
            # it now hunts the player (see EnemyMixin._move_warlock), so it is just
            # one of the sampled entities above.
            self.guard_pos = adjacent_floor_pos(self.grid, self.key_pos, occupied)
            if self.guard_pos is not None:
                occupied.add(tuple(self.guard_pos))
            self.warlock_fireball_pos = None
            self.warlock_fireball_dir = None
            self.warlock_fireball_ticks = 0

            # Spikes are always-on floor hazards. Keep them off cells walled in on
            # two or more sides so the player can always route around them, and
            # (in the validity check below) guarantee a spike-free path to the key
            # and the exit. Stored as lists (not the sampled tuples) so ==
            # comparisons with player_pos work: tuple == list is always False.
            free = [cell for cell in cells
                    if cell not in occupied and wall_neighbor_count(self.grid, cell) <= 1]
            if len(free) < self.N_SPIKES:
                continue
            self.spike_poses = [list(cell) for cell in random.sample(free, self.N_SPIKES)]
            self.spike_statuses = [True] * self.N_SPIKES

            self.freeze_ticks = 0
            self.has_key = False
            self.won = False
            # Tracks whether the one-time "all enemies cleared" bonus has been paid.
            self.enemies_cleared_awarded = False
            self.hp = self.MAX_HP

            melee_valid = all(
                is_reachable(self.grid, self.player_pos, pos)
                and bfs_distance(self.grid, self.player_pos, pos) > 4
                for pos in self.melee_poses
            )
            if not melee_valid:
                continue

            # guard_pos is tested first so the unidirectional helper never receives
            # None (adjacent_floor_pos returns None when the key is walled in on
            # every side, and the check would crash on tuple(None)). The guard
            # needs the key-aware unidirectional check (killable without stepping
            # onto the key, which would wake it). The warlock only needs plain
            # reachability. Finally, the key and exit must each be reachable along
            # a route that never steps on a spike - the spikes must never wall off
            # an objective.
            spikes_set = {tuple(s) for s in self.spike_poses}
            if (self.guard_pos is not None
                    and is_reachable(self.grid, self.player_pos, self.exit_pos)
                    and is_reachable(self.grid, self.player_pos, self.key_pos)
                    and is_reachable(self.grid, self.player_pos, self.warlock_pos)
                    and is_reachable_unidirectional_any(
                        self.grid, self.player_pos, self.guard_pos, forbidden=self.key_pos)
                    and is_reachable_avoiding(self.grid, self.player_pos, self.key_pos, spikes_set)
                    and is_reachable_avoiding(self.grid, self.key_pos, self.exit_pos, spikes_set)
                    and bfs_distance(self.grid, self.player_pos, self.key_pos) >= 2
                    and bfs_distance(self.grid, self.player_pos, self.exit_pos) >= 4):
                break

        self.done = False
        self.steps = 0
        # Straight-line path of the most recent shot, for rendering only. Set by
        # _shoot, cleared every step. Has no effect on the agent's observation.
        self.last_shot = None
        return self._get_state()

    def _move_agent(self, action, terminated):
        if self.has_key:
            goal_pos = self.exit_pos
        else:
            goal_pos = self.key_pos

        old_goal_dist = bfs_distance(self.grid, self.player_pos, goal_pos)
        # Movement deltas
        deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = deltas[action]
        new_r = self.player_pos[0] + dr
        new_c = self.player_pos[1] + dc

        # Don't move into walls
        if self.grid[new_r, new_c] != WALL:
            self.player_pos = [new_r, new_c]

        new_goal_dist = bfs_distance(self.grid, self.player_pos, goal_pos)

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

        # The win condition (reaching the exit with the key) is no longer checked
        # here: the exit only opens once every enemy is dead, so it is evaluated in
        # step() after combat resolves (see the _all_enemies_defeated check).
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

        pos_before = list(self.player_pos)

        # Spike check #1: the player is on the spike as it triggers. This covers
        # staying put on a spike that fires, and being caught while stepping off
        # it the same step.
        spike_reward, terminated = self._check_for_spikes()
        reward += spike_reward

        # A lethal spike at the start of the step skips the player's action.
        if not terminated:
            if action == 8:
                if self.freeze_available:
                    self._activate_freeze_powerup()
                    if self.warlock_pos is not None:
                        # Using freeze while the warlock still lives provokes it:
                        # the warlock punishes the player and the freeze earns no
                        # reward. Kill the warlock first and freezing becomes both
                        # safe and rewarded - that is the behaviour we want.
                        hit_reward, terminated = self._damage_player(self.WARLOCK_DAMAGE)
                        reward += hit_reward
                    else:
                        reward += self.REWARD_FREEZE
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
        # terminated is checked first: if the player has just died, the enemies'
        # last move is irrelevant.
        if not terminated:
            if self.freeze_ticks > 0:
                self.freeze_ticks -= 1
            else:
                if any(melee_pos is not None for melee_pos in self.melee_poses):
                    reward, terminated = self._next_enemy_action(reward, terminated)
                if not terminated:
                    reward, terminated = self._warlock_action(reward, terminated)
            # Fill any slot left empty by a kill with a fresh enemy from the
            # reserve. Runs regardless of freeze, since the player can still shoot
            # an enemy on a frozen step.
            if not terminated:
                self._respawn_melee()
            # One-time milestone: the moment the board is finally clear (and the
            # exit opens), reward it to bridge the gap between the last kill and
            # the eventual win.
            if (not terminated and not self.enemies_cleared_awarded
                    and self._all_enemies_defeated()):
                reward += self.REWARD_ALL_CLEARED
                self.enemies_cleared_awarded = True

        # The exit only opens once every enemy is gone. Reaching it with the key
        # wins only when no enemy remains and the reserve is empty. Checked here,
        # after combat, so killing the final enemy while already standing on the
        # exit still counts as a win this step.
        if (not terminated and self.has_key
                and self.player_pos == self.exit_pos
                and self._all_enemies_defeated()):
            speed_bonus = max(0.0, (50 - self.steps) / 50)
            reward += self.REWARD_WIN + speed_bonus
            terminated = True
            self.won = True
            self.done = True

        self.steps += 1
        if self.steps >= self.step_limit and not terminated:  # step limit
            truncated = True
            self.done = True

        return self._get_state(), reward, terminated, truncated

    def _activate_freeze_powerup(self):
        self.freeze_ticks = self.FREEZE_TICKS
        self.freeze_available  = False

    def _get_closest_unidirectional_enemy(self, enemies, direction):
        shortest_distance = 100000
        closest_enemy = None

        for enemy in enemies:
            if is_reachable_unidirectional(self.grid, self.player_pos, enemy, direction):
                distance = bfs_distance(self.grid, self.player_pos, enemy)
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
                and bfs_distance(self.grid, self.player_pos, closest_enemy) <= self.SHOOT_RANGE):
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
        # The exit reads 1.0 once it is open (every enemy cleared) and 0.5 while it
        # is still locked, so the agent can see when finishing is actually possible.
        exit_[self.exit_pos[0], self.exit_pos[1]] = 1.0 if self._all_enemies_defeated() else 0.5
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
        # Mark the fireball's whole danger corridor, not just its current tile
        # (see EnemyMixin._fireball_danger_tiles).
        for tile in self._fireball_danger_tiles():
            warlock_fireball[tile[0], tile[1]] = 1.0
        # Spikes are always-on hazards now, so every spike tile reads 1.0.
        for spike_pos in self.spike_poses:
            spikes[spike_pos[0], spike_pos[1]] = 1.0

        goal = tuple(self.key_pos) if not self.has_key else tuple(self.exit_pos)

        return np.stack([walls, player, exit_, freeze,
                         freeze_status, distance_map(self.grid, goal), key,
                         guard, melee, warlock, warlock_fireball, hp, spikes],
                        axis=0)
