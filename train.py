from stable_baselines3 import PPO
from env import GameEnv
from smallgridcnn import SmallGridCNN
from stable_baselines3.common.env_util import make_vec_env

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
            tensorboard_log="./tb_logs/",
            seed=42
        )

model.learn(total_timesteps=33_000_000)
model.save("versions/key_and_guard_ppo_1")
print("Training done.")

env.close()

