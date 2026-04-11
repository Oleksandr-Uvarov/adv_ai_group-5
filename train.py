from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from env import GameEnv

env = GameEnv(grid_size=10)
eval_callback = EvalCallback(env, eval_freq=10000, verbose=1)


# Sanity check
check_env(env)

model = PPO("MlpPolicy", env, verbose=1)
# model = PPO("CnnPolicy", env, verbose=1)
# model = PPO("MlpPolicy", env)
# model.learn(total_timesteps=200000)
model.learn(total_timesteps=300_000, callback=eval_callback)
model.save("roguelike.ppo")
print("Training done.")









import torch
import torch.nn as nn
# from stable_baselines3 import PPO
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from env import GameEnv
# import gymnasium as gym
#
# class SmallGridCNN(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
#         super().__init__(observation_space, features_dim)
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # small kernel
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Flatten(),
#         )
#         # compute output size dynamically
#         import torch
#         with torch.no_grad():
#             sample = torch.zeros(1, *observation_space.shape)
#             n_flatten = self.cnn(sample).shape[1]
#
#         self.linear = nn.Sequential(
#             nn.Linear(n_flatten, features_dim),
#             nn.ReLU()
#         )
#
#     def forward(self, observations):
#         return self.linear(self.cnn(observations))
#
#
# env = GameEnv(grid_size=10)
#
# policy_kwargs = dict(
#     features_extractor_class=SmallGridCNN,
#     features_extractor_kwargs=dict(features_dim=64),
# )
#
# model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
# model.learn(total_timesteps=500_000)
# model.save("roguelike_ppo")
# print("Training done.")
