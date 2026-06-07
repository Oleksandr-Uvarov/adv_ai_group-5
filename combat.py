from pathfinding import WALL, bfs_distance, is_reachable_unidirectional


class CombatMixin:
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
