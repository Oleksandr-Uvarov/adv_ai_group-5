from stable_baselines3 import PPO
from env import GameEnv
from smallgridcnn import SmallGridCNN
import faulthandler
faulthandler.enable()

env = GameEnv(grid_size=10)

policy_kwargs = dict(
    features_extractor_class=SmallGridCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

model = PPO.load(
    # "version_history/1_pre_freeze/zips/roguelike_ppo_12.zip",
    "versions/shoot_ppo_1.zip",
    env=env,
    custom_objects={
        "device": "cpu",
        "policy_kwargs": policy_kwargs,
    }
)


obs, info = env.reset()

n_won = 0
n_lost = 0
n_truncated = 0

for i in range(10000):
    obs, info = env.reset()

    for step in range(200):
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)

        # env.render()
        if done:
            if env.game.player_pos == env.game.enemy_pos:
                n_lost += 1
            else:
                n_won += 1
            # env.render()
            break

        # env.render()

        if step == 199:
            n_truncated += 1

print(n_won, n_lost, n_truncated)