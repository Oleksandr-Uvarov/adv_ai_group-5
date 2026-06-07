"""Enemy behaviour for :class:`game_engine.Game`, split out as a mixin.

``EnemyMixin`` carries everything the enemies do - the melee enemies (BFS pathing
two steps toward the player each turn) and the warlock (it stands watch beside the
exit and launches a fireball whenever the player lines up with it). ``Game``
inherits from it, so these methods run with full access to the game's state via
``self`` (``self.grid``, ``self.player_pos``, the entity lists, the tunable
constants, ``self._damage_player``, etc.). All map queries go through the pure
helpers in :mod:`pathfinding`.

Kept here, away from game_engine.py, so the engine file stays focused on the map,
the player, rewards and the observation."""

from pathfinding import bfs_distance, bfs_step_toward


class EnemyMixin:
    # ------------------------------------------------------------------- melee
    def _move_melee_enemies(self):
        """Move every living melee enemy one tile closer to the player along a
        shortest path. Enemies do not avoid one another (they may share a tile),
        matching the original behaviour; an enemy already on the player's tile or
        with no path to it simply holds position."""
        goal = tuple(self.player_pos)

        new_poses = []
        for m in self.melee_poses:
            if m is None:
                new_poses.append(None)
                continue
            start = tuple(m)
            if start == goal:
                new_poses.append(list(start))
                continue
            step = bfs_step_toward(self.grid, start, goal, set())
            new_poses.append(list(step) if step is not None else list(start))

        return new_poses

    def _nearest_enemy_distance(self):
        """BFS distance from the player to the closest living enemy,
        or None if every enemy has been killed."""
        dists = [
            bfs_distance(self.grid, self.player_pos, melee_pos)
            for melee_pos in self.melee_poses if melee_pos is not None
        ]
        return min(dists) if dists else None

    def _next_enemy_action(self, reward, terminated, old_enemy_dist):
        # Melee enemies move two tiles per step toward the player.
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

    # ----------------------------------------------------------------- warlock
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
        player shares the warlock's row or column AND is within range. The
        fireball travels straight and passes through walls, so no line-of-sight
        check is needed; it just needs the player to be axis-aligned to have a
        direction to fly in."""
        wr, wc = self.warlock_pos
        pr, pc = self.player_pos
        if wr == pr and wc != pc:
            direction = (0, 1) if pc > wc else (0, -1)
            distance = abs(pc - wc)
        elif wc == pc and wr != pr:
            direction = (1, 0) if pr > wr else (-1, 0)
            distance = abs(pr - wr)
        else:
            return  # player is not aligned with the warlock; nothing to fire at

        # Only fire a shot that can actually land. The fireball despawns after
        # WARLOCK_FIREBALL_RANGE tiles, so launching at a farther target just
        # puts a harmless fireball in the player's lane every step and teaches
        # the agent to fear shots that can never reach it.
        if distance > self.WARLOCK_FIREBALL_RANGE:
            return

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

    def _fireball_danger_tiles(self):
        """Tiles the in-flight fireball occupies now plus the ones it will still
        sweep through before its range runs out, clipped to the grid edge (it
        flies through walls, so walls are not excluded). The observation marks
        this whole corridor instead of the single current tile: from one tile the
        agent cannot tell which way the fireball is heading or how far it can
        still reach, so it learns to fear any fireball in its lane. The corridor
        shows exactly which tiles can still be hit - and, by omission, that tiles
        beyond the range are safe."""
        if self.warlock_fireball_pos is None:
            return []
        dr, dc = self.warlock_fireball_dir
        r, c = self.warlock_fireball_pos
        tiles = [[r, c]]
        for _ in range(self.warlock_fireball_ticks):
            r, c = r + dr, c + dc
            if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
                break
            tiles.append([r, c])
        return tiles
