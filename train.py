from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from env import GameEnv

env = GameEnv(grid_size=10)
eval_callback = EvalCallback(env, eval_freq=10000, verbose=1)


# Sanity check
check_env(env)

model = PPO("MlpPolicy", env, verbose=1)
# model = PPO("MlpPolicy", env)
# model.learn(total_timesteps=200000)
model.learn(total_timesteps=300_000, callback=eval_callback)
model.save("roguelike.ppo")
print("Training done.")