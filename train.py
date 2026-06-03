import sys
from datetime import datetime
from pathlib import Path

DIRECTORY = "5_test"
total_timesteps = 1000
total_seconds = total_timesteps / 400
hours = int(total_seconds // 3600)
minutes = round((total_seconds % 3600) / 60, 2)

_zips_check = Path("version_history") / DIRECTORY / "zips"
_n_pre = len(list(_zips_check.iterdir())) + 1 if _zips_check.exists() else 1


def _confirm(prompt):
    while True:
        answer = input(prompt + "\nDo you want to proceed? Y/N: ").upper()
        if answer == "Y":
            return
        if answer == "N":
            sys.exit()
        print("Please enter 'Y' or 'N'")


_confirm(f"The model will be saved to directory '{DIRECTORY}'.")
_confirm(f"The model will be trained for {total_timesteps} timesteps "
         f"(~{hours}h {minutes}m).")

developer_comment = ""
if _n_pre > 1:
    developer_comment = input("Developer comment for this version (press Enter to leave blank): ")

from env import GameEnv
from stable_baselines3 import PPO
from smallgridcnn import SmallGridCNN
from stable_baselines3.common.env_util import make_vec_env
from version_utils import write_version_file, env_signature


version_dir = Path("version_history") / DIRECTORY
zips_dir = version_dir / "zips"
tb_dir = version_dir / "tb_logs"
version_differences_dir = Path("version_differences") / DIRECTORY

for d in (version_dir, zips_dir, tb_dir, version_differences_dir):
    d.mkdir(parents=True, exist_ok=True)

n = len(list(zips_dir.iterdir())) + 1
model_name = f"{DIRECTORY}_ppo"

GRID_SIZE = 10
N_ENVS = 8
FEATURES_DIM = 128
PPO_POLICY = "CnnPolicy"
PPO_PARAMS = dict(
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    learning_rate=3e-4,
    verbose=1,
    seed=42,
)

policy_kwargs = dict(
    features_extractor_class=SmallGridCNN,
    features_extractor_kwargs=dict(features_dim=FEATURES_DIM),
)

# n_envs - number of environments run in parallel.
# 8 should multiply the FPS by 8, but in reality it's slower than that.
env = make_vec_env(lambda: GameEnv(grid_size=GRID_SIZE), n_envs=N_ENVS, seed=42)

model = PPO(PPO_POLICY,
            env,
            policy_kwargs=policy_kwargs,
            **PPO_PARAMS,
            tensorboard_log=str(tb_dir),
            )

start_dt = datetime.now()
model.learn(total_timesteps=total_timesteps, tb_log_name=model_name)
end_dt = datetime.now()
model.save(str(zips_dir / f"{model_name}_{n}"))
print("Training done.")
env.close()

version_file = write_version_file(
    n, version_differences_dir,
    features_dim=FEATURES_DIM,
    ppo_policy=PPO_POLICY,
    ppo_params=PPO_PARAMS,
    total_timesteps=total_timesteps,
    n_envs=N_ENVS,
    signature=env_signature(GameEnv(grid_size=GRID_SIZE)),
    developer_comment=developer_comment,
    started_at=start_dt.strftime("%Y-%m-%d %H:%M:%S"),
    ended_at=end_dt.strftime("%Y-%m-%d %H:%M:%S"),
    duration_seconds=(end_dt - start_dt).total_seconds(),
)
print(f"Version differences saved to {version_file}")