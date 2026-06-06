"""Throwaway diagnostic. Delete when done."""
import numpy as np
from game_engine import Game


def test_win_mechanics():
    """Synthetic: clear everything, hold the key, stand next to the exit, step
    onto it. Confirms the level transition and the win flag actually fire."""
    g = Game(10)
    g.reset()
    # Force a fully-cleared board with the key in hand.
    g.guard_pos = None
    g.warlock_pos = None
    g.melee_poses = [None] * len(g.melee_poses)
    g.melee_reserve = 0
    g.key_pos = None
    g.has_key = True
    g.enemies_cleared_awarded = True
    assert g._all_enemies_defeated()
    # Put the player one tile left of the exit, on floor, then walk right onto it.
    er, ec = g.exit_pos
    g.player_pos = [er, ec - 1]
    g.grid[er, ec - 1] = 0  # ensure floor
    lvl_before = g.level
    obs, r, term, trunc = g.step(3)  # move right onto exit
    print(f"  level {lvl_before} -> {g.level}, reward={r:.3f}, term={term}, won={g.won}")
    return g.level > lvl_before or g.won


def rollout_instrumented(model_path, n_episodes=400, grid_size=10):
    from env import GameEnv
    from evaluate import load_model
    from pathfinding import bfs_distance

    env = GameEnv(grid_size=grid_size, render_mode=None)
    model = load_model(model_path, env)

    # Per "open window" stats: a window starts the step the exit opens on a level
    # and ends when that level ends (cleared, died, truncated, or - for the open
    # check - the level transitions). We track what the agent does once finishing
    # is actually possible.
    n_windows = 0           # times an exit opened
    win_had_key = 0         # ...with the key already in hand at opening
    win_resolved = 0        # ...that ended with the player stepping on the exit (level cleared)
    dist_at_open = []       # BFS player->exit at the moment it opened
    min_dist_after = []     # closest the player got to the exit during the window
    steps_left_at_open = [] # step budget remaining when it opened

    for _ in range(n_episodes):
        obs, _ = env.reset()
        g = env.game
        done = False
        # window state for the *current* level
        in_window = False
        window_level = None
        win_mindist = None
        cleared_this_window = False

        def open_now():
            return g._all_enemies_defeated()

        while not done:
            level_before = g.level
            was_open = open_now()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Detect a fresh opening on the current level.
            if not in_window and open_now() and g.level == level_before:
                in_window = True
                window_level = g.level
                n_windows += 1
                if g.has_key:
                    win_had_key += 1
                d = bfs_distance(g.grid, g.player_pos, g.exit_pos)
                dist_at_open.append(d)
                win_mindist = d
                steps_left_at_open.append(g.step_limit - g.steps)

            # Level advanced => the previous window resolved by reaching the exit.
            if in_window and g.level > window_level:
                win_resolved += 1
                if win_mindist is not None:
                    min_dist_after.append(0)
                in_window = False
                window_level = None
                win_mindist = None

            # Track how close it gets to the exit while the window is open.
            if in_window and open_now() and g.level == window_level:
                d = bfs_distance(g.grid, g.player_pos, g.exit_pos)
                if win_mindist is None or d < win_mindist:
                    win_mindist = d

        if in_window and win_mindist is not None:
            min_dist_after.append(win_mindist)

    def avg(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print(f"  episodes={n_episodes}")
    print(f"  exit openings (windows):     {n_windows}")
    print(f"  ...with key in hand:         {win_had_key}  ({100*win_had_key/max(1,n_windows):.0f}%)")
    print(f"  ...that reached the exit:    {win_resolved}  ({100*win_resolved/max(1,n_windows):.0f}%)")
    print(f"  avg BFS dist to exit at open:{avg(dist_at_open):.1f}")
    print(f"  avg step budget left at open:{avg(steps_left_at_open):.1f}")
    print(f"  avg CLOSEST it got to exit during window: {avg(min_dist_after):.2f}")


if __name__ == "__main__":
    print("[1] win mechanics test:")
    ok = test_win_mechanics()
    print("  PASS" if ok else "  FAIL - transition did not fire")

    print("\n[2] instrumented rollout of 10M model (version 1):")
    rollout_instrumented(
        "version_history/9_levels_and_potion/zips/9_levels_and_potion_ppo_1_20260606-010948-aa8349.zip"
    )
