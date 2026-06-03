from stable_baselines3 import PPO
from env import GameEnv
from smallgridcnn import SmallGridCNN
import faulthandler
import time
faulthandler.enable()

env = GameEnv(grid_size=10, render_mode="human")

STEP_DELAY = 1
EPISODE_PAUSE = 3

policy_kwargs = dict(
    features_extractor_class=SmallGridCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

model = PPO.load(
    # "version_history/1_pre_freeze/zips/roguelike_ppo_12.zip",
    "version_history/4_key_and_guard/zips/4_key_and_guard_ppo_1.zip",
    env=env,
    custom_objects={
        "device": "cpu",
        "policy_kwargs": policy_kwargs,
    }
)


n_won = 0
n_lost = 0
n_truncated = 0

for i in range(1000):
    obs, info = env.reset()
    env.render()

    for step in range(env.game.step_limit):
        time.sleep(STEP_DELAY)
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)

        env.render()

        # Classify the episode exactly once. A win is reaching the exit while
        # holding the key; anything else terminal (caught by an enemy, or
        # stepping on the key while the guard is alive) is a loss.
        if done or truncated:
            if truncated:
                n_truncated += 1
            elif env.game.player_pos == env.game.exit_pos and env.game.has_key:
                n_won += 1
            else:
                n_lost += 1
            break



print(n_won, n_lost, n_truncated)