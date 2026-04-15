from stable_baselines3 import PPO
from env import GameEnv
from smallgridcnn import SmallGridCNN

env = GameEnv(grid_size=10)

policy_kwargs = dict(
    features_extractor_class=SmallGridCNN,
    features_extractor_kwargs=dict(features_dim=64),
)

model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=300_000)
model.save("roguelike_ppo")
print("Training done.")
