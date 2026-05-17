import sys
import os
import time

# Allow imports from the project root
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from stable_baselines3 import PPO
from env import GameEnv
from smallgridcnn import SmallGridCNN
from pygame_renderer import Renderer

STEP_DELAY = 0.15   # seconds between steps
EPISODE_PAUSE = 0.5 # pause after each episode ends

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

env = GameEnv(grid_size=10)
renderer = Renderer(grid_size=10)

model = PPO.load(
    os.path.join(_root, "version_history/4_key_and_guard/zips/4_key_and_guard_ppo_1.zip"),
    env=env,
    custom_objects={
        "device": "cpu",
        "policy_kwargs": dict(
            features_extractor_class=SmallGridCNN,
            features_extractor_kwargs=dict(features_dim=128),
        ),
    },
)


while True:
    obs, _ = env.reset()
    renderer.draw(env.game)

    for _ in range(env.game.step_limit):
        time.sleep(STEP_DELAY)
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        renderer.draw(env.game)
        if terminated or truncated:
            time.sleep(EPISODE_PAUSE)
            break
