import modal
from pathlib import Path

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("stable-baselines3[extra]", "torch", "numpy")
    .add_local_python_source("env", "game_engine", "smallgridcnn", "version_utils")
)

app = modal.App("rl-training", image=image)
volume = modal.Volume.from_name("rl-artifacts", create_if_missing=True)


@app.function(
    volumes={"/artifacts": volume},
    cpu=8,
    timeout=86400,
)
def train(directory: str, total_timesteps: int, developer_comment: str = ""):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from env import GameEnv
    from smallgridcnn import SmallGridCNN
    from version_utils import write_version_file

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
    PARAM_DISPLAY = {"learning_rate": "3e-4"}

    version_dir = Path("/artifacts/version_history") / directory
    zips_dir = version_dir / "zips"
    tb_dir = version_dir / "tb_logs"
    version_differences_dir = Path("/artifacts/version_differences") / directory

    for d in (version_dir, zips_dir, tb_dir, version_differences_dir):
        d.mkdir(parents=True, exist_ok=True)

    n = len(list(zips_dir.iterdir())) + 1
    model_name = f"{directory}_ppo"

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

    write_version_file(
        n, version_differences_dir,
        FEATURES_DIM, PPO_POLICY, PPO_PARAMS, PARAM_DISPLAY,
        developer_comment,
    )
    volume.commit()
    print(f"Saved: {model_name}_{n}")


@app.local_entrypoint()
def main(
    directory: str = "4_key_and_guard",
    timesteps: int = 1_000_000,
    comment: str = "",
):
    train.remote(directory, timesteps, comment)