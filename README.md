# Roguelike RL Agent

**Authors:** Oleksandr Uvarov and Enis Chavush

---

## Overview

The agent operates in a 10×10 grid world that is randomly generated each episode. It must reach the exit tile while evading a melee enemy that pursues it using BFS pathfinding. A freeze powerup is available that temporarily stops the enemy for 3 steps.

### Game Elements

| Symbol | Meaning |
|--------|---------|
| `@` | Player (agent) |
| `X` | Exit (goal) |
| `E` | Enemy |
| `F` | Freeze powerup |
| `#` | Wall |
| `.` | Floor |

---

## Environment

### Observation Space

The agent receives a `(7, 10, 10)` tensor - 7 channels, each a full grid view:

| Channel | Description |
|---------|-------------|
| 0 | Walls |
| 1 | Player position |
| 2 | Exit position |
| 3 | Enemy position |
| 4 | Freeze powerup position |
| 5 | Freeze status (uniform value = remaining ticks / 3) |
| 6 | BFS distance map to exit (normalised) |

### Action Space

4 discrete actions: `0` = up, `1` = down, `2` = left, `3` = right. Moving into a wall is a no-op.

### Reward Structure

| Event | Reward |
|-------|--------|
| Moving closer to the exit | `+0.1` per BFS step closer |
| Moving away from the exit | `−0.1` per BFS step further |
| Enemy moving closer to the player | `−0.05` per BFS step closer |
| Reaching the exit | `+1.0` + speed bonus up to `+1.0` (based on steps taken vs 75-step target) |
| Caught by the enemy | `−1.0` |

### Episode Termination

- **Win:** player reaches the exit tile
- **Loss:** enemy reaches the player
- **Truncation:** step limit of 125 reached without resolution

### Map Generation

Each episode generates a fresh map:
- Border walls surround the grid
- Interior cells have a 20% chance of becoming a wall
- Player, exit, enemy, and freeze powerup are placed on random floor cells
- BFS is applied before starting an episode: it is made sure that the enemy can reach the player, and the player can reach the exit and the freeze powerup

---

## Architecture

The policy uses a custom CNN feature extractor (`SmallGridCNN`) with three convolutional layers (7→32→64→64 channels, kernel size 3, padding 1), followed by a fully connected layer projecting to a 128-dimensional feature vector. This is fed into the standard PPO actor-critic heads from Stable Baselines3.

---

## Training

Training uses PPO from [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) with 8 parallel environments.

### Hyperparameters

| Parameter | Value     |
|-----------|-----------|
| `n_steps` | 2048      |
| `batch_size` | 256       |
| `n_epochs` | 10        |
| `learning_rate` | 3e-4      |
| `total_timesteps` | 6,000,000 |
| `n_envs` | 8         |

---

All the hyperparameters can of course be changed, but this is a good start.

## Project Structure

```
.
├── game_engine.py       # Core game logic (grid, entities, rewards, BFS)
├── env.py               # Gymnasium wrapper used by the PPO trainer
├── smallgridcnn.py      # Custom CNN feature extractor
├── train.py             # Training script
├── evaluate.py          # Evaluation script (runs 1000 episodes, reports win/loss/truncation)
├── tb_logs/             # TensorBoard logs — one folder per training run
├── versions/            # Saved model checkpoints (.zip)
└── version_history/     # Per-version changelogs and archived tb_logs + zips
```

---

## Setup

```bash
pip install stable-baselines3 gymnasium torch tensorboard
```

---

## Usage

**Train:**
```bash
python train.py
```
The trained model is saved to `versions/freeze_ppo_1.zip`.

**Evaluate:**
```bash
python evaluate.py
```
Runs 1000 episodes and prints `wins / losses / truncations`.

**Monitor training in TensorBoard:**
```bash
tensorboard --logdir tb_logs
```
Key metrics to watch:
- `rollout/ep_rew_mean` — average episode reward
- `rollout/ep_len_mean` — average episode length
- `train/clip_fraction` — fraction of PPO updates that were clipped (target: 0.05–0.15)
- `train/approx_kl` — KL divergence between old and new policy
- `fps` — environment timesteps per second

---

## Version History

See `version_history/` for a changelog of each model version, including the differences relative to the previous version, archived TensorBoard logs, and saved model zips.

To load a historical model's TensorBoard logs, copy its folder from `version_history/` into `tb_logs/` at the project root, then run `tensorboard --logdir tb_logs`.