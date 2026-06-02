import sys
from pathlib import Path

DIRECTORY = "5_test"
total_timesteps = 1000
total_seconds = total_timesteps / 400
hours = int(total_seconds // 3600)
minutes = round((total_seconds % 3600) / 60, 2)

_zips_check = Path("version_history") / DIRECTORY / "zips"
_n_pre = len(list(_zips_check.iterdir())) + 1 if _zips_check.exists() else 1

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

developer_comment = ""
if _n_pre > 1:
    developer_comment = input("Developer comment for this version (press Enter to leave blank): ")

from env import GameEnv
from stable_baselines3 import PPO
from smallgridcnn import SmallGridCNN
from stable_baselines3.common.env_util import make_vec_env


version_dir = Path("version_history") / DIRECTORY
zips_dir = version_dir / "zips"
tb_dir = version_dir / "tb_logs"
version_differences_dir = Path("version_differences") / DIRECTORY

for d in (version_dir, zips_dir, tb_dir, version_differences_dir):
    d.mkdir(parents=True, exist_ok=True)

n = len(list(zips_dir.iterdir())) + 1
model_name = f"{DIRECTORY}_ppo"

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
_PARAM_DISPLAY = {"learning_rate": "3e-4"}

policy_kwargs = dict(
    features_extractor_class=SmallGridCNN,
    features_extractor_kwargs=dict(features_dim=FEATURES_DIM),
)

# n_envs - number of environments run in parallel.
# 8 should multiply the FPS by 8, but in reality it's slower than that.
env = make_vec_env(lambda: GameEnv(grid_size=10), n_envs=8, seed=42)

model = PPO(PPO_POLICY,
            env,
            policy_kwargs=policy_kwargs,
            **PPO_PARAMS,
            tensorboard_log=str(tb_dir),
        )

model.learn(total_timesteps=total_timesteps, tb_log_name=model_name)

model.save(str(zips_dir / f"{model_name}_{n}"))
print("Training done.")

env.close()


def _fmt(key, val):
    return _PARAM_DISPLAY.get(key, str(val))

def _format_params_text():
    lines = [
        "Parameters:",
        "",
        f"    features_dim (policy_kwargs): {FEATURES_DIM}",
        "",
        "    PPO:",
        "",
        f"        {PPO_POLICY}",
    ]
    for key, val in PPO_PARAMS.items():
        lines.append(f"        {key}={_fmt(key, val)}")
    return "\n".join(lines)

def _parse_params(params_section):
    params = {}
    in_ppo = False
    for line in params_section.splitlines():
        s = line.strip()
        if not s or s == "Parameters:":
            continue
        if s == "PPO:":
            in_ppo = True
            continue
        if not in_ppo:
            if ": " in s:
                k, _, v = s.partition(": ")
                params[k] = v
        else:
            if "=" in s:
                k, _, v = s.partition("=")
                params[f"PPO.{k}"] = v
            else:
                params["PPO.policy"] = s
    return params

def _current_params():
    return {
        "features_dim (policy_kwargs)": str(FEATURES_DIM),
        "PPO.policy": PPO_POLICY,
        **{f"PPO.{k}": _fmt(k, v) for k, v in PPO_PARAMS.items()},
    }

params_text = _format_params_text()

if n == 1:
    diff_content = params_text
else:
    prev_file = version_differences_dir / f"version_{n - 1}.txt"
    if prev_file.exists():
        prev_text = prev_file.read_text(encoding="utf-8")
        idx = prev_text.find("Parameters:")
        prev_params = _parse_params(prev_text[idx:]) if idx != -1 else {}
    else:
        prev_params = {}

    curr_params = _current_params()
    changes = []
    for key in sorted(set(prev_params) | set(curr_params)):
        pv, cv = prev_params.get(key), curr_params.get(key)
        if pv != cv:
            if pv is None:
                changes.append(f"    Added {key}: {cv}")
            elif cv is None:
                changes.append(f"    Removed {key}")
            else:
                changes.append(f"    {key}: {pv} → {cv}")

    change_block = "\n".join(changes) if changes else "    No parameters changed."
    diff_content = (
        f"What changed from the previous version:\n{change_block}\n\n"
        f"Developer comment: {developer_comment}\n\n"
        f"{params_text}"
    )

version_file = version_differences_dir / f"version_{n}.txt"
version_file.write_text(diff_content, encoding="utf-8")
print(f"Version differences saved to {version_file}")