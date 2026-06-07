import numpy as np

from pathfinding import WALL, distance_map

class ObservationMixin:
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
