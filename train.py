import sys

DIRECTORY = "4_key_and_guard"
total_timesteps = 1000
total_seconds = total_timesteps / 400
hours = int(total_seconds // 3600)
minutes = round((total_seconds % 3600) / 60, 2)

while True:
    user_input = input(f"The model will be saved to directory {DIRECTORY}.\n"
                       f" Do you want to proceed? Y/N: ")
    if user_input.upper() not in ("Y", "N"):
        print("Please enter 'Y' or 'N'")
        continue
    elif user_input.upper() == "N":
        sys.exit()
    else:
        while True:
            user_input = input(f"The model will be trained for {total_timesteps} timesteps.\n"
                               f"It will take approximately {hours} hours and {minutes} minutes.\n"
                               f" Do you want to proceed? Y/N: ")
            if user_input.upper() not in ("Y", "N"):
                print("Please enter 'Y' or 'N'")
                continue
            elif user_input.upper() == "N":
                sys.exit()
            else:
                break
        break

from env import GameEnv
from pathlib import Path
from stable_baselines3 import PPO
from smallgridcnn import SmallGridCNN
from stable_baselines3.common.env_util import make_vec_env


version_dir = Path("version_history") / DIRECTORY
zips_dir = version_dir / "zips"
tb_dir = version_dir / "tb_logs"

for d in (version_dir, zips_dir, tb_dir):
    d.mkdir(parents=True, exist_ok=True)

n = len(list(zips_dir.iterdir())) + 1
model_name = f"{DIRECTORY}_ppo"

# n_envs - number of environments run in parallel.
# 8 should multiply the FPS by 8, but in reality it's slower than that.
env = make_vec_env(lambda: GameEnv(grid_size=10), n_envs=8, seed=42)

policy_kwargs = dict(
    features_extractor_class=SmallGridCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

model = PPO("CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            learning_rate=3e-4,
            verbose=1,
            tensorboard_log=str(tb_dir),
            seed=42
        )

model.learn(total_timesteps=total_timesteps, tb_log_name=model_name)

model.save(str(zips_dir / f"{model_name}_{n}"))
print("Training done.")

env.close()