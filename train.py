import sys
import uuid
from datetime import datetime
from pathlib import Path

DIRECTORY = "9_levels_and_potion"
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

model_name = f"{DIRECTORY}_ppo"
# Unique per-run tag so concurrent runs never collide on any artifact (zip,
# tb_logs, version files) even if they land on the same sequential n. Timestamp
# for readability + a short random part to break same-second ties.
run_tag = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

GRID_SIZE = 10
N_ENVS = 8
FEATURES_DIM = 128
PPO_POLICY = "CnnPolicy"
PPO_PARAMS = dict(
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    learning_rate=3e-4,
    # Entropy bonus on the policy. SB3's default is 0.0, which lets the policy
    # converge to a near-deterministic optimum and stop exploring - exactly how
    # the agent got stuck refusing the exit (it stopped sampling the step onto
    # the exit, so it never re-collected the +3 that would correct that). A small
    # positive value keeps it exploring those rarely-taken actions. Raise toward
    # 0.02 if the refusal persists.
    ent_coef=0.01,
    verbose=1,
    seed=42,
)

policy_kwargs = dict(
    features_extractor_class=SmallGridCNN,
    features_extractor_kwargs=dict(features_dim=FEATURES_DIM),
)

# n_envs - number of environments run in parallel.
# 8 should multiply the FPS by 8, but in reality it's slower than that.
# randomize_start spreads training across all levels (with realistic carried
# HP/charges) so the warlock levels aren't starved; evaluation still uses the
# default GameEnv (always starts at level 1) to measure the true task.
env = make_vec_env(lambda: GameEnv(grid_size=GRID_SIZE, randomize_start=True),
                   n_envs=N_ENVS, seed=42)

model = PPO(PPO_POLICY,
            env,
            policy_kwargs=policy_kwargs,
            **PPO_PARAMS,
            tensorboard_log=str(tb_dir),
            )

start_dt = datetime.now()
model.learn(total_timesteps=total_timesteps, tb_log_name=f"{model_name}_{run_tag}")
end_dt = datetime.now()
# Pick the version number now, not at launch, so two runs into the same directory
# don't both grab the same n and overwrite each other: by save time the earlier
# run's zip already exists on disk, so we take the next free slot.
n = len(list(zips_dir.iterdir())) + 1
model_path = str(zips_dir / f"{model_name}_{n}_{run_tag}")
model.save(model_path)
print("Training done.")
env.close()

from evaluate import evaluate
print("Evaluating...")
eval_results = evaluate(model_path, n_episodes=1000, pygame_overview=False)
print(f"Evaluation: won={eval_results['n_won']} lost={eval_results['n_lost']} truncated={eval_results['n_truncated']}")

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
    eval_results=eval_results,
    run_tag=run_tag,
)
print(f"Version differences saved to {version_file}")