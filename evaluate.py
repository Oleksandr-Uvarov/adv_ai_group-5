from stable_baselines3 import PPO
from env import GameEnv

env = GameEnv(grid_size=10)
model = PPO.load("roguelike.ppo")

obs, info = env.reset()
env.render()

for _ in range(200):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
        break