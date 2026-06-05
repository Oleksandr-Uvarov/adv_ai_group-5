"""Enemy behaviour for :class:`game_engine.Game`, split out as a mixin.

``EnemyMixin`` carries everything the enemies do - the melee enemies (collision-
free pathing toward the player and spawning in from the reserve) and the warlock
(kiting movement plus its fireball). ``Game`` inherits from it, so these methods
run with full access to the game's state via ``self`` (``self.grid``,
``self.player_pos``, the entity lists, the tunable constants, ``self._damage_player``,
etc.). All map queries go through the pure helpers in :mod:`pathfinding`.

Kept here, away from game_engine.py, so the engine file stays focused on the map,
the player, rewards and the observation."""

import random

from pathfinding import (
    FLOOR, floor_cells, bfs_distance, bfs_step_toward, is_reachable,
)


class EnemyMixin:
    # ------------------------------------------------------------------ shared
    def _occupied_cells(self):
        """Set of ``(row, col)`` tuples currently holding an entity (player, exit,
        key, guard, warlock, any living melee enemy, any spike, the potion). Used
        to keep the warlock and freshly spawned enemies from landing on anything."""
        cells = {tuple(self.player_pos), tuple(self.exit_pos)}
        if self.key_pos is not None:
            cells.add(tuple(self.key_pos))
        if self.guard_pos is not None:
            cells.add(tuple(self.guard_pos))
        if self.warlock_pos is not None:
            cells.add(tuple(self.warlock_pos))
        if self.potion_pos is not None:
            cells.add(tuple(self.potion_pos))
        for m in self.melee_poses:
            if m is not None:
                cells.add(tuple(m))
        for s in self.spike_poses:
            cells.add(tuple(s))
        return cells

    def _all_enemies_defeated(self):
        """True once no enemy of any kind remains and none are waiting to spawn:
        every melee enemy killed and the reserve exhausted, the warlock dead and
        the guard dead. This is the condition that opens the exit."""
        return (self.guard_pos is None
                and self.warlock_pos is None
                and self.melee_reserve == 0
                and all(m is None for m in self.melee_poses))

    # ------------------------------------------------------------------- melee
    def _next_enemy_action(self, reward, terminated):
        """Move every living melee enemy one tile toward the player, then apply
        damage to any that reached the player's tile."""
        self.melee_poses = self._move_melee_enemies()

        for melee_pos in self.melee_poses:
            if melee_pos == self.player_pos:
                reward, terminated = self._damage_player(self.MELEE_ENEMY_DAMAGE)

        return reward, terminated

    def _move_melee_enemies(self):
        """Step each living melee enemy one tile closer to the player along a
        shortest path, never landing two enemies on the same tile. Enemies are
        moved one at a time against a shared ``blocked`` set: every other enemy's
        cell (current if it hasn't moved yet, new once it has), plus the guard and
        warlock, is impassable, so an enemy routes around its neighbours instead
        of stacking on them. The player's own tile is the one exception - reaching
        it is the attack, and several enemies may converge on the player at once.
        An enemy with no clear path this turn simply holds position."""
        goal = tuple(self.player_pos)

        blocked = {tuple(m) for m in self.melee_poses if m is not None}
        if self.guard_pos is not None:
            blocked.add(tuple(self.guard_pos))
        if self.warlock_pos is not None:
            blocked.add(tuple(self.warlock_pos))

        new_poses = []
        for m in self.melee_poses:
            if m is None:
                new_poses.append(None)
                continue
            start = tuple(m)
            if start == goal:
                new_poses.append(list(start))
                continue
            # This enemy's own cell is not an obstacle to itself.
            step = bfs_step_toward(self.grid, start, goal, blocked - {start})
            new_cell = step if step is not None else start
            blocked.discard(start)
            # The player's tile is shared freely (it's the attack); every other
            # destination becomes an obstacle for the enemies that move after.
            if new_cell != goal:
                blocked.add(new_cell)
            new_poses.append(list(new_cell))

        return new_poses

    def _respawn_melee(self):
        """Refill empty melee slots from the reserve so a new enemy walks in to
        replace each one the player kills, until the level's reserve is exhausted.
        New enemies spawn on a reachable floor cell well away from the player so
        they never pop into existence right on top of it."""
        for i in range(len(self.melee_poses)):
            if self.melee_poses[i] is None and self.melee_reserve > 0:
                cell = self._spawn_melee_cell()
                if cell is not None:
                    self.melee_poses[i] = cell
                    self.melee_reserve -= 1

    def _spawn_melee_cell(self):
        """A random unoccupied floor cell that is reachable from the player and at
        least 4 tiles away, or None if no such cell exists this step (in which
        case the slot stays empty and the spawn is retried next step)."""
        occupied = self._occupied_cells()
        candidates = [
            cell for cell in floor_cells(self.grid)
            if cell not in occupied
            and is_reachable(self.grid, self.player_pos, cell)
            and bfs_distance(self.grid, self.player_pos, cell) >= 4
        ]
        return list(random.choice(candidates)) if candidates else None

    # ----------------------------------------------------------------- warlock
    def _warlock_action(self, reward, terminated):
        """Advance an in-flight fireball; if none is flying, a living warlock
        repositions to hover at firing range and then launches one straight toward
        the player. A fireball already in flight keeps travelling even if the
        warlock is killed before it lands, which is why despawning it lives in
        _advance_fireball (called as long as a fireball exists) rather than being
        tied to the warlock being alive. While a fireball is flying the warlock
        holds still - that is the window in which the player can close in on it."""
        if self.warlock_fireball_pos is not None:
            return self._advance_fireball(reward, terminated)

        if self.warlock_pos is not None:
            self._move_warlock()
            self._launch_fireball()
            if self.warlock_fireball_pos is not None:
                return self._advance_fireball(reward, terminated)

        return reward, terminated

    def _move_warlock(self):
        """Kite the player: close the gap until WARLOCK_FIREBALL_RANGE tiles away,
        then stop closing and instead sidestep onto the player's row or column so
        it can line up a shot. It never voluntarily steps nearer than the distance
        it is already keeping (the player has to chase it down and shoot it), and
        it never shares a tile with another entity. Distance is Manhattan, which
        matches the fireball's straight-line, wall-piercing reach: when the
        warlock is on the player's row/column the Manhattan distance is exactly the
        number of tiles the fireball must cross."""
        pr, pc = self.player_pos
        wr, wc = self.warlock_pos
        occupied = self._occupied_cells()
        occupied.discard((wr, wc))  # staying put is always an option

        candidates = [(wr, wc)]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = wr + dr, wc + dc
            if (0 <= nr < self.grid_size and 0 <= nc < self.grid_size
                    and self.grid[nr, nc] == FLOOR and (nr, nc) not in occupied):
                candidates.append((nr, nc))

        cur_dist = abs(wr - pr) + abs(wc - pc)
        # Never close to nearer than we already are once inside the ring; from
        # farther out the floor is the ring radius, so any approaching move is fine.
        floor_dist = min(cur_dist, self.WARLOCK_FIREBALL_RANGE)

        def score(cell):
            r, c = cell
            d = abs(r - pr) + abs(c - pc)
            aligned = (r == pr or c == pc)
            # Top priority: stand where it can actually fire (aligned, in range).
            shoot = 100 if (aligned and 0 < d <= self.WARLOCK_FIREBALL_RANGE) else 0
            # Hug the range ring; hard-forbid diving nearer than floor_dist.
            ring = -abs(d - self.WARLOCK_FIREBALL_RANGE)
            keep = -1000 if d < floor_dist else 0
            return shoot + ring + keep

        self.warlock_pos = list(max(candidates, key=score))

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
