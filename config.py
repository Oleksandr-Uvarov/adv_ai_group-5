class GameConfig:
    # Reward shaping coefficients.
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
