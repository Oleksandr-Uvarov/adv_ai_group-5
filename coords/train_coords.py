from coords.roguelikeenvcoords import RoguelikeEnvCoords
from stable_baselines3 import PPO

env = RoguelikeEnvCoords(grid_size=10)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=300_000)
model.save("roguelike_coords")

