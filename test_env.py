from env import GameEnv

env = GameEnv(grid_size=10)
obs, info = env.reset()

env.render()

for step in range(200):
    action = env.action_space.sample()  # random agent
    obs, reward, done, truncated, info = env.step(action)

    print(f"Step {step+1} | Action: {action} | Reward: {reward:.2f} | Done: {done}")

    if done:
        print("Episode finished!")
        env.render()
        break