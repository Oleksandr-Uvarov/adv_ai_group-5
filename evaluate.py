from stable_baselines3 import PPO
from env import GameEnv
from smallgridcnn import SmallGridCNN
import faulthandler
import time
faulthandler.enable()

STEP_DELAY = 1


def evaluate(model_path, pygame_overview=False, n_episodes=10000, grid_size=10):
    env = GameEnv(grid_size=grid_size, render_mode="human" if pygame_overview else None)

    policy_kwargs = dict(
        features_extractor_class=SmallGridCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = PPO.load(
        model_path,
        env=env,
        custom_objects={
            "device": "cpu",
            "policy_kwargs": policy_kwargs,
        }
    )

    n_won = 0
    n_lost = 0
    n_truncated = 0

    for i in range(n_episodes):
        obs, info = env.reset()
        if pygame_overview:
            env.render()

        for step in range(env.game.step_limit):
            if pygame_overview:
                time.sleep(STEP_DELAY)
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)

            if pygame_overview:
                env.render()

            if done or truncated:
                if truncated:
                    n_truncated += 1
                elif env.game.player_pos == env.game.exit_pos and env.game.has_key:
                    n_won += 1
                else:
                    n_lost += 1
                break

    env.close()
    return {
        "n_episodes": n_episodes,
        "n_won": n_won,
        "n_lost": n_lost,
        "n_truncated": n_truncated,
    }


if __name__ == "__main__":
    results = evaluate(
        "version_history/6_warlock_active_freeze_hp/zips/6_warlock_active_freeze_hp_ppo_1.zip",
        pygame_overview=True,
    )
    print(results["n_won"], results["n_lost"], results["n_truncated"])