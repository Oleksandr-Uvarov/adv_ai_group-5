import random

import numpy as np

from pathfinding import (
    WALL, floor_cells, wall_neighbor_count, adjacent_floor_pos,
    is_reachable, is_reachable_avoiding, is_reachable_unidirectional_any,
    bfs_distance,
)


class LevelMixin:
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
