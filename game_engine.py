from pathfinding import WALL, bfs_distance
from config import GameConfig
from level import LevelMixin
from combat import CombatMixin
from observation import ObservationMixin
from enemies import EnemyMixin


class Game(GameConfig, LevelMixin, CombatMixin, ObservationMixin, EnemyMixin):
    # The game is assembled from focused mixins so this file can stay small and
    # be about one thing - the per-step game loop. Each mixin owns one concern:
    #   GameConfig (config.py)        - tunable constants + reward_coeffs
    #   LevelMixin (level.py)         - reset / map + entity generation
    #   CombatMixin (combat.py)       - the player's shooting
    #   ObservationMixin (observation.py) - building the agent's observation
    #   EnemyMixin (enemies.py)       - melee + warlock + fireball behaviour
    # All the grid/BFS maths lives in pathfinding.py. What is left here is the
    # step loop that ties them together, plus the player's own movement, spike
    # and damage handling.

    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()
        self.done = False
        self.step_limit = 100

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
