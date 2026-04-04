from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env import GameEnv

env = GameEnv(grid_size=10)

# Sanity check
check_env(env)

# model = PPO("MlpPolicy", env, verbose=1)
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=200000)
model.save("roguelike.ppo")
print("Training done.")