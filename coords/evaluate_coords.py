from coords.roguelikeenvcoords import RoguelikeEnvCoords
from stable_baselines3 import PPO


env = RoguelikeEnvCoords(grid_size=10)
model = PPO.load("roguelike_coords")

obs, info = env.reset()
# env.render()

n_finished = 0
n_truncated = 0

for i in range(1000):
    obs, info = env.reset()

    for step in range(200):
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)

        # print(f"Step {step+1} | Action: {action} | Reward: {reward:.2f} | Done: {done}")

        if done:
            n_finished += 1
            print("Episode finished!")
            # env.render()
            break

        # env.render()

        if step == 199:
            n_truncated += 1

    # print(i)


print(n_finished, n_truncated)