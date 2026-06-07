# Roguelike RL Agent

**Authors:** Oleksandr Uvarov and Enis Chavush

A reinforcement-learning agent (PPO + a small CNN) that plays a turn-based
roguelike on a randomly generated 10×10 grid. In each episode the agent must
wipe out every enemy on the level, pick up the key (guarded), and reach the
exit — which opens **only once the board is clear**. Reaching the open exit
while holding the key wins the run.

The project was built up one mechanic at a time (see
[Development stages](#development-stages)); this document describes the **current**
stage, `7_spikes`.

---

## Table of contents

- [Game mechanics](#game-mechanics)
- [Environment](#environment)
  - [Observation space](#observation-space)
  - [Action space](#action-space)
  - [Reward structure](#reward-structure)
  - [Termination, truncation and the win condition](#termination-truncation-and-the-win-condition)
  - [Map generation and validity guarantees](#map-generation-and-validity-guarantees)
- [Model architecture](#model-architecture)
- [Training](#training)
- [Evaluation and replay](#evaluation-and-replay)
- [Visualisation](#visualisation)
- [Versioning: how and why](#versioning-how-and-why)
- [Development stages](#development-stages)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Usage cheat-sheet](#usage-cheat-sheet)

---

## Game mechanics

| Entity | Behaviour |
|--------|-----------|
| **Player** | The agent. Moves one orthogonal tile per turn, or shoots a hitscan beam in one of four directions, or triggers freeze. Starts the episode at `MAX_HP = 100`. |
| **Melee enemy** | Walks one tile toward the player every turn along a BFS shortest path (routing around other enemies); deals `50` damage on contact. Only `MAX_ACTIVE_MELEE = 3` are on the board at once — the rest wait in reserve and spawn in (≥4 tiles away) as active ones die, until all `TOTAL_MELEE_ENEMIES = 6` have appeared. |
| **Guard** | Stationary, always seated next to the key it protects. Stepping onto the key while the guard is alive deals `50` damage and does **not** pick up the key — the guard must be shot first. |
| **Warlock** | Ranged. *Kites* the player: closes to `WARLOCK_FIREBALL_RANGE = 3`, then sidesteps onto the player's row/column and launches a fireball. While a fireball is in flight the warlock holds still — that is the window to close in and shoot it. |
| **Fireball** | Travels straight, **passes through walls**, `50` damage, despawns after 3 tiles or on hit. Its observation channel marks the whole danger corridor it can still sweep (bright = where it is now, on toward where it is heading). |
| **Spikes** | Always-on floor hazards (`N_SPIKES = 3`), `34` damage. Placed so the player can always route around them and so they never wall off the key or exit. |
| **Key** | Must be picked up (guard permitting) before the exit can be used. |
| **Exit** | Opens **only once every enemy on the level is dead** (and the melee reserve is exhausted). Reaching it with the key then wins. |
| **Freeze** | A single consumable charge per episode (refilled when the episode resets): it freezes all enemies for `FREEZE_TICKS = 2` turns. Using it while a warlock is still alive provokes the warlock (damages the player, no reward); once the board is warlock-free it is safe and rewarded. |

### Constants (single source of truth: `Game` in `game_engine.py`)

| Constant | Value | | Constant | Value |
|----------|-------|-|----------|-------|
| `MAX_HP` | 100 | | `SHOOT_RANGE` | 3 |
| melee / guard / warlock damage | 50 each | | `WARLOCK_FIREBALL_RANGE` | 3 |
| `SPIKE_DAMAGE` | 34 | | `FREEZE_TICKS` | 2 |
| `N_SPIKES` | 3 | | `TOTAL_MELEE_ENEMIES` / `MAX_ACTIVE_MELEE` | 6 / 3 |
| `step_limit` | 100 | | grid size | 10×10 |

---

## Environment

`env.py` wraps `Game` (`game_engine.py`) as a Gymnasium `Env`. The observation
shape and channel count are derived from a real observation at construction time,
so adding/removing a channel never requires touching the env or the network.

### Observation space

A `(13, 10, 10)` float tensor — 13 full-grid channels, all in `[0, 1]`:

| # | Channel | Description |
|---|---------|-------------|
| 0 | walls | 1 on wall tiles |
| 1 | player | 1 on the player tile |
| 2 | exit | `0.5` while any enemy remains (locked), `1.0` once the board is clear (open) |
| 3 | freeze charge | `1` while the freeze charge is unused, `0` once spent |
| 4 | freeze status | uniform plane = freeze ticks left / `FREEZE_TICKS` |
| 5 | goal distance map | normalised BFS distance from every tile to the current goal (key, or exit once keyed) |
| 6 | key | 1 on the key tile |
| 7 | guard | 1 on the guard tile |
| 8 | melee | 1 on every living melee enemy (all share one channel — same entity type) |
| 9 | warlock | 1 on the warlock tile |
| 10 | warlock fireball | danger corridor: the fireball's current tile plus every tile it can still reach before its range runs out |
| 11 | HP | uniform plane = current HP / `MAX_HP` |
| 12 | spikes | 1 on every spike tile |

### Action space

9 discrete actions:

| Action | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|--------|---|---|---|---|---|---|---|---|---|
| Meaning | up | down | left | right | shoot left | shoot right | shoot up | shoot down | freeze |

Movement into a wall is a no-op. A shot is hitscan: it hits the nearest enemy in
that direction within `SHOOT_RANGE`. Freeze with the charge already spent is a no-op.

### Reward structure

All coefficients live in `Game` and are logged with every trained version via
`Game.reward_coeffs()`.

| Event | Reward | Notes |
|-------|--------|-------|
| Progress toward the goal | `±0.1` / tile | `REWARD_GOAL_STEP`, goal = key then exit |
| Shooting an enemy (melee/guard/warlock) | `+0.3` | `REWARD_KILL` |
| Picking up the key | `+0.3` | `REWARD_KEY` (guard must be dead) |
| Safe freeze (no warlock alive) | `+0.3` | `REWARD_FREEZE` |
| Board fully cleared | `+0.5` (once) | `REWARD_ALL_CLEARED` — bridges the gap to the exit |
| Winning (clear board + key + on exit) | `+1.0` + speed bonus | `REWARD_WIN` |
| Non-lethal hit | `−0.3` | `REWARD_HIT` |
| Death (HP → 0) | `−1.0` | `REWARD_LOSE` |

Speed bonus on a win: `max(0, (50 − steps) / 50)`.

Why `REWARD_ALL_CLEARED` exists: the exit is gated on every enemy being dead, so
until the board is clear the only spatial gradient (channel 5 + the goal reward)
points at the key/exit even though the exit cannot yet be used. Paying a one-time
bonus the moment the board clears bridges the gap between the last kill and the
eventual win.

### Termination, truncation and the win condition

- **Win:** the board is clear (every enemy dead, reserve exhausted), the player
  holds the key, and stands on the exit. `game.won = True`.
- **Loss:** HP hits 0 (enemy, fireball or spike), or being downed while grabbing
  the key past a living guard.
- **Truncation:** the step budget (`step_limit = 100`) is exhausted.

### Map generation and validity guarantees

Each episode regenerates the map: border walls, then each interior cell is a wall
with 20% probability. Entities are sampled onto distinct floor cells (player,
exit, key, warlock, the active melee enemies), the guard is seated next to the
key, and spikes are dropped only on cells walled in on ≤1 side. A map is
re-rolled until all of the following hold:

- exit, key and warlock are reachable from the player;
- the guard is reachable along a single axis **without** stepping on the key (which would wake it);
- there is a spike-free route to the key, and from the key to the exit;
- the key is ≥2 and the exit ≥4 tiles from the player, and every starting melee enemy is >4 tiles away.

All grid maths (BFS reachability, distance maps, single-step pathing) lives in the
pure, stateless helpers in `pathfinding.py`.

---

## Model architecture

`SmallGridCNN` (`smallgridcnn.py`) is a custom Stable-Baselines3 feature
extractor:

```
input (C×10×10)  ->  Conv2d(C→32, k3, p1) -> ReLU
                 ->  Conv2d(32→64, k3, p1) -> ReLU
                 ->  Conv2d(64→64, k3, p1) -> ReLU
                 ->  Flatten -> Linear(→128) -> ReLU
```

The input channel count `C` is read from the observation space (currently 13), so
the network adapts automatically when channels change. The 128-d feature vector
feeds the standard PPO actor-critic heads (`CnnPolicy`).

---

## Training

Two entry points share the same hyperparameters and the same versioning output:

- **`train.py`** — local. Interactively confirms the target directory and step
  count, asks for a developer comment, trains with 8 parallel envs, saves the
  model zip + TensorBoard logs, evaluates, and writes the version record. The
  `DIRECTORY` (`"7_spikes"`) and `total_timesteps` values near the top of the
  file are edited per run (`total_timesteps` ships set to a small smoke value —
  raise it before a real run).
- **`modal_train.py`** — the same pipeline on [Modal](https://modal.com) cloud
  workers, writing artifacts to a Modal volume. Git info is collected locally and
  passed in (the worker has no `.git`). The target stage is set with `--directory`.
  See `modal_commands.txt` for the exact invocations.

| Hyperparameter | Value |
|----------------|-------|
| policy | `CnnPolicy` + `SmallGridCNN` (features_dim 128) |
| `n_steps` | 2048 |
| `batch_size` | 256 |
| `n_epochs` | 10 |
| `learning_rate` | 3e-4 |
| `n_envs` | 8 |
| `seed` | 42 |

---

## Evaluation and replay

`evaluate.py` provides:

- **`evaluate(model_path, n_episodes)`** — headless tally of `won / lost / truncated`. This is what the training scripts call to fill the version record's `eval` block.
- **`main()` / `record_episodes()` / `watch_episodes()`** — records episodes headlessly, then replays the chosen category (won/lost/truncated) in a pygame window with an HP bar, last-move glyph, freeze indicator and play/step/skip controls.

Running `python evaluate.py` calls `main()` on the model path hard-coded at the
bottom of the file (edit it to point at a current zip). The model must match the
current observation's channel count.

---

## Visualisation

- `pg/pygame_renderer.py` — the `Renderer` that draws the grid, entities, spikes, fireball and the player's shot animation. Used by both the live env (`render_mode="human"`) and the replay viewer.
- `pg/visualize.py` — a standalone loop that loads a model and renders live episodes (edit the model path inside to point at a current zip).

---

## Versioning: how and why

Training is iterative: every meaningful change to rewards, mechanics or
hyperparameters produces a new, fully documented model so results stay
reproducible and comparable. The bookkeeping is handled by `version_utils.py` and
lives in **two parallel trees**, both partitioned by development stage:

```
version_history/<stage>/zips/      <- saved model .zip files (the artifacts)
version_history/<stage>/tb_logs/   <- TensorBoard logs, one folder per run
version_differences/<stage>/       <- per-version records (the metadata)
```

For each run, `write_version_file` emits two files into `version_differences/<stage>/`:

- **`version_{n}_{run_tag}.json`** — the machine-readable **source of truth**. It
  captures everything needed to reproduce and to judge compatibility: the full
  environment signature from `env_signature` (observation shape, grid size, entity
  counts, action dim and **all reward coefficients**), the PPO hyperparameters,
  training duration, git commit + dirty flag, library versions, the evaluation
  results, and the developer comment.
- **`version_{n}_{run_tag}.txt`** — the human-readable changelog. Its top block is
  an automatic **diff against version *n−1*** ("What changed from the previous
  version"), followed by a readable snapshot of the run.

Design points:

- **`run_tag`** (`YYYYMMDD-HHMMSS-<rand>`) is appended to every artifact and
  record so two runs that land on the same sequential number `n` never overwrite
  each other.
- **`obs_shape` + `reward_coeffs` are recorded** precisely so you can tell at a
  glance whether a saved zip is loadable into a given env — the network's input
  channel count must match `obs_shape[0]`. (This is why changing the observation,
  e.g. adding or removing a channel, makes older zips incompatible.)
- The version number `n` is chosen at **save time**, not launch time, so
  concurrent runs into the same stage take successive free slots instead of
  colliding.

To revisit a historical run's TensorBoard, copy its folder from
`version_history/<stage>/tb_logs/` into a local `tb_logs/` and run
`tensorboard --logdir tb_logs`. As an alternative, simply run the tb command with the path to that tb_logs instead of the plain "tb_logs" argument.

---

## Development stages

Each stage directory is a capability milestone; the agent was retrained as each
mechanic was layered on. In order:

| Stage | What it added |
|-------|---------------|
| `1_pre_freeze` | Base navigation + a single chasing melee enemy |
| `2_freeze` | The freeze power-up |
| `3_shoot` | Shooting enemies |
| `4_key_and_guard` | The key objective and the guard protecting it |
| `5_two_melee_enemies` | A second simultaneous melee enemy |
| `6_warlock_active_freeze_hp` | The ranged warlock, HP as an observation, freeze made costly while the warlock lives |
| `7_spikes` | **Current:** always-on spike hazards |

Two refinements landed on top of the current stage rather than as new directories:

- The melee pool was expanded from a fixed pair into a **reserve/respawn** system
  (`TOTAL_MELEE_ENEMIES = 6`, `MAX_ACTIVE_MELEE = 3`): only three enemies are on
  the board at once and a fresh one walks in from reserve as each is killed.
- A `levels_and_potion` extension (a would-be stage 8/9) was prototyped on a
  branch — multiple levels played back-to-back in one episode with a difficulty
  curriculum, HP and freeze carried across levels, healing potions, and an extra
  nearest-enemy combat-shaping reward and observation channel. It did not train
  reliably (the agent struggled to clear even the first level), so it was dropped
  and is **not** part of the main line.

`challenges.txt` records design problems encountered along the way (e.g. the agent
refusing to engage the warlock and stalling until truncation, and the unreliable
learning around on/off spikes that led to making spikes always-on).

---

## Project structure

```
.
├── game_engine.py      # Game: map, player, rewards, observation. Core logic.
├── enemies.py          # EnemyMixin: melee/guard/warlock movement, fireball, respawns.
├── pathfinding.py      # Pure grid helpers: BFS reachability, distance maps, stepping.
├── env.py              # Gymnasium wrapper around Game (used by PPO).
├── smallgridcnn.py     # Custom CNN feature extractor for the policy.
├── train.py            # Local training entry point (interactive).
├── modal_train.py      # Cloud (Modal) training entry point.
├── evaluate.py         # evaluate() tally + record/replay viewer.
├── version_utils.py    # env_signature + write_version_file (json source of truth + txt changelog + diff).
├── pg/
│   ├── pygame_renderer.py   # Renderer for grid/entities/fireball/shots.
│   └── visualize.py         # Standalone live-rendering loop for a saved model.
├── version_history/    # Per-stage saved models (zips/) and TensorBoard logs (tb_logs/).
├── version_differences/# Per-stage version records (.json source of truth + .txt changelog).
├── requirements.txt    # Python dependencies.
├── modal_commands.txt  # Reference Modal CLI commands.
└── challenges.txt      # Notes on design challenges encountered.
```

---

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# then:
pip install -r requirements.txt
```

Core dependencies: `stable-baselines3`, `gymnasium`, `torch`, `numpy`,
`tensorboard`, `pygame` (and `modal` for cloud training).

---

## Usage cheat-sheet

```bash
# Train locally (edit DIRECTORY / total_timesteps in train.py first)
python train.py

# Train on Modal (detached so closing the terminal won't kill it)
modal run --detach modal_train.py --directory "7_spikes" --timesteps 1000000 --comment "..."

# Headless win/lose/truncation tally + interactive replay viewer
# (edit the model path at the bottom of evaluate.py first)
python evaluate.py

# Watch a saved model play live (edit the path inside the file)
python pg/visualize.py

# TensorBoard
tensorboard --logdir tb_logs
```