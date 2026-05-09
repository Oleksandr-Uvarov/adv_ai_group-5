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
    "versions/key_and_guard_ppo_2.zip",
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

for i in range(10):
    obs, info = env.reset()

    print("attempt", i)
    for step in range(100):
        env.render()
        action, _ = model.predict(obs)
        print("action", action)
        obs, reward, done, truncated, info = env.step(action)

        if done:
            for enemy_pos in env.game.melee_poses:
                if enemy_pos == env.game.player_pos:
                    n_lost += 1
                    break
                else:
                    n_won += 1
                    break
            break

        if step == 99:
            n_truncated += 1

print(n_won, n_lost, n_truncated)