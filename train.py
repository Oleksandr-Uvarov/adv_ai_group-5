from stable_baselines3 import PPO
from env import GameEnv
from smallgridcnn import SmallGridCNN

env = GameEnv(grid_size=10)

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
            learning_rate=2e-4,
            verbose=1,
            tensorboard_log="./tb_logs/"
        )

model.learn(total_timesteps=1_000_000)
model.save("roguelike_ppo")
print("Training done.")
