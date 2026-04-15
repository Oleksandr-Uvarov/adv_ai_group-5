from env import GameEnv

env = GameEnv(grid_size=10)
obs, info = env.reset()

# env.render()

n_finished = 0
n_truncated = 0

for i in range(10000):
    obs, info = env.reset()
    for step in range(200):
        action = env.action_space.sample()  # random agent
        obs, reward, done, truncated, info = env.step(action)

        # print(f"Step {step+1} | Action: {action} | Reward: {reward:.2f} | Done: {done}")

        if done:
            n_finished += 1
            # print(n_finished)
            # print("Episode finished!")
            env.render()
            break

        if step == 199:
            n_truncated += 1

        # env.render()


print(n_finished, n_truncated)