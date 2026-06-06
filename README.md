# Roguelike RL Agent

**Authors:** Oleksandr Uvarov and Enis Chavush

A reinforcement-learning agent (PPO + a small CNN) that plays a turn-based,
multi-level roguelike on a randomly generated 10×10 grid. Across one episode the
agent plays several levels back-to-back: on each level it must wipe out every
enemy, pick up the key (guarded), and reach the exit, carrying its HP and freeze
charges forward. Clearing the final level wins the run.

The project was built up one mechanic at a time (see
[Development stages](#development-stages)); this document describes the **current**
stage, `9_levels_and_potion`.

---

## Table of contents

- [Game mechanics](#game-mechanics)
- [Levels and the curriculum](#levels-and-the-curriculum)
- [Environment](#environment)
  - [Observation space](#observation-space)
  - [Action space](#action-space)
  - [Reward structure](#reward-structure)
  - [Termination, truncation and level transitions](#termination-truncation-and-level-transitions)
  - [Map generation and validity guarantees](#map-generation-and-validity-guarantees)
- [Model architecture](#model-architecture)
- [Training](#training)
- [Evaluation and diagnostics](#evaluation-and-diagnostics)
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
| **Player** | The agent. Moves one orthogonal tile per turn, or shoots a hitscan beam in one of four directions, or triggers freeze. Starts each run at `MAX_HP = 100`. |
| **Melee enemy** | Walks one tile toward the player every turn along a BFS shortest path (routing around other enemies); deals `25` damage on contact. Only `max_active` are on the board at once — the rest wait in reserve and spawn in (≥4 tiles away) as active ones die. |
| **Guard** | Stationary, always seated next to the key it protects. Stepping onto the key while the guard is alive deals `25` damage and does **not** pick up the key — the guard must be shot first. |
| **Warlock** | Ranged. *Kites* the player: closes to `WARLOCK_FIREBALL_RANGE = 3`, then sidesteps onto the player's row/column and launches a fireball. While a fireball is in flight the warlock holds still — that is the window to close in and shoot it. |
| **Fireball** | Travels straight, **passes through walls**, `25` damage, despawns after 3 tiles or on hit. Its observation channel fades along its direction of travel (bright = where it is now, dim = where it is heading). |
| **Spikes** | Always-on floor hazards (`N_SPIKES = 3`), `25` damage. Placed so the player can always route around them and so they never wall off the key or exit. |
| **Key** | Must be picked up (guard permitting) before the exit can be used. |
| **Exit** | Opens **only once every enemy on the level is dead**. Reaching it with the key then clears the level. |
| **Potion** | Appears on every level after the first. Heals up to `MAX_HP`; reward is scaled by HP actually restored, so it is only worth detouring for when hurt. |
| **Freeze** | A consumable power-up: `FREEZE_CHARGES = 2` per run (**not** refilled between levels), each freezes all enemies for `FREEZE_TICKS = 2` turns. Using it while a warlock is still alive provokes the warlock (damages the player, no reward); once the board is warlock-free it is safe and rewarded. |

Key run-wide rule: **HP and freeze charges carry across levels.** A level resets
the map, enemies, key, spikes, potion and the per-level step budget, but not your
health or remaining freezes.

### Constants (single source of truth: `Game` in `game_engine.py`)

| Constant | Value | | Constant | Value |
|----------|-------|-|----------|-------|
| `MAX_HP` | 100 | | `SHOOT_RANGE` | 3 |
| `POTION_HEAL` | 50 | | `WARLOCK_FIREBALL_RANGE` | 3 |
| melee / guard / warlock / spike damage | 25 each | | `FREEZE_TICKS` | 2 |
| `N_SPIKES` | 3 | | `FREEZE_CHARGES` | 2 |
| `step_limit` (per level) | 100 | | grid size | 10×10 |

---

## Levels and the curriculum

Levels are defined in `Game.LEVELS` as `(total_melee, max_active_melee, has_warlock)`
triples and played back-to-back inside a single episode:

| Level | Melee (total / active) | Guard | Warlock | Potion |
|-------|------------------------|-------|---------|--------|
| 1 | 4 / 2 | ✓ | ✗ | ✗ |
| 2 | 5 / 2 | ✓ | ✓ | ✓ |
| 3 | 7 / 3 | ✓ | ✓ | ✓ |

This ordering is a **difficulty curriculum**, and it is deliberate. Earlier
single-level stages showed the agent *could* learn melee + guard + warlock at
once. But stacking three such levels into one episode broke it: the agent never
even cleared level 1 (it emptied the board in under a fifth of episodes and never
reached level 2). Making level 1 strictly easier than a setup that already worked
(melee + guard, **no warlock**) lets the agent get a first clear and bootstrap;
level 2 reintroduces the warlock, level 3 adds more melee. No mechanic is removed
from the game — only the *first* level's composition is gentler.

---

## Environment

`env.py` wraps `Game` (`game_engine.py`) as a Gymnasium `Env`. The observation
shape and channel count are derived from a real observation at construction time,
so adding/removing a channel never requires touching the env or the network.

### Observation space

A `(15, 10, 10)` float tensor — 15 full-grid channels, all in `[0, 1]`:

| # | Channel | Description |
|---|---------|-------------|
| 0 | walls | 1 on wall tiles |
| 1 | player | 1 on the player tile |
| 2 | exit | `0.33` while enemies remain, `0.66` when the board is clear but the key is still needed, `1.0` when finishable this step (board clear **and** key in hand) |
| 3 | freeze charges | uniform plane = charges left / `FREEZE_CHARGES` (ticks down over a run) |
| 4 | freeze status | uniform plane = freeze ticks left / `FREEZE_TICKS` |
| 5 | goal distance map | normalised BFS distance from every tile to the current goal (key, or exit once keyed) |
| 6 | key | 1 on the key tile |
| 7 | guard | 1 on the guard tile |
| 8 | melee | 1 on every living melee enemy (all share one channel — same entity type) |
| 9 | warlock | 1 on the warlock tile (all-zero on warlock-free levels) |
| 10 | warlock fireball | danger corridor, intensity fading along the fireball's travel direction |
| 11 | HP | uniform plane = current HP / `MAX_HP` |
| 12 | spikes | 1 on every spike tile |
| 13 | potion | 1 on the potion tile (all-zero when there is none) |
| 14 | **nearest-enemy distance map** | normalised multi-source BFS distance to the *closest* living enemy; all-`1.0` once the board is clear |

Channel 14 is the spatial counterpart to the enemy-approach reward (below): it
tells the agent *where the fight is*, since channel 5 only ever points at the
key/exit.

### Action space

9 discrete actions:

| Action | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|--------|---|---|---|---|---|---|---|---|---|
| Meaning | up | down | left | right | shoot left | shoot right | shoot up | shoot down | freeze |

Movement into a wall is a no-op. A shot is hitscan: it hits the nearest enemy in
that direction within `SHOOT_RANGE`. Freeze with no charges left is a no-op.

### Reward structure

All coefficients live in `Game` and are logged with every trained version via
`Game.reward_coeffs()`.

| Event | Reward | Notes |
|-------|--------|-------|
| Progress toward the goal | `±0.1` / tile | `REWARD_GOAL_STEP`, goal = key then exit |
| Progress toward the nearest enemy | `±0.05` / tile | `REWARD_ENEMY_STEP`, only while the board is not clear; switches off once cleared |
| Shooting an enemy (melee/guard/warlock) | `+0.3` | `REWARD_KILL` |
| Picking up the key | `+0.3` | `REWARD_KEY` (guard must be dead) |
| Safe freeze (no warlock alive) | `+0.3` | `REWARD_FREEZE` |
| Board fully cleared | `+0.5` (once/level) | `REWARD_ALL_CLEARED` — bridges the gap to the exit |
| Picking up a potion | up to `+0.3` | `REWARD_POTION`, scaled by HP actually restored |
| Clearing a non-final level | `+3.0` + speed bonus | `REWARD_LEVEL_CLEAR`, then drop into the next level |
| Winning (clearing the final level) | `+5.0` + speed bonus | `REWARD_WIN` |
| Non-lethal hit | `−0.3` | `REWARD_HIT` |
| Death (HP → 0) | `−1.0` | `REWARD_LOSE` |
| Living cost | `−0.01` / step | `REWARD_STEP_PENALTY`, so idling is never free |

Speed bonus on a clear/win: `max(0, (50 − steps_this_level) / 50)`.

Why `REWARD_LEVEL_CLEAR` / `REWARD_WIN` are large: at a cleared level the agent
weighs "head to the exit" against "idle until truncation (~0)", and the exit drops
it into a harder, riskier next level. A small clear reward makes finishing
negative expected value, so the agent learns to avoid the exit. Making the clear
reward dominate that discounted next-level risk flips finishing back to worth it.

Why `REWARD_ENEMY_STEP` exists: the only spatial gradient (channels 5 + the goal
reward) points at the key/exit, but the exit is *gated* on every enemy being dead.
Nothing pulled the agent toward the enemies it had to kill, so hunting the guard
and the kiting warlock was left to blind exploration and rarely happened. This
term supplies the missing "go and fight" gradient; it is kept below
`REWARD_GOAL_STEP`, is measured over a fixed enemy snapshot per move (so it never
rewards/penalises a kill), and disables itself once the board is clear.

### Termination, truncation and level transitions

- **Win:** clearing the final level (board clear + key + on exit). `game.won = True`.
- **Loss:** HP hits 0 (enemy, fireball or spike), or being downed while grabbing the key past the guard.
- **Truncation:** the per-level step budget (`step_limit = 100`) is exhausted.
- **Level transition:** clearing a non-final level immediately rebuilds the next level in place (HP and freeze charges persist, the step budget resets) without ending the episode.

### Map generation and validity guarantees

Each level regenerates the map: border walls, then each interior cell is a wall
with 20% probability. Entities are sampled onto distinct floor cells (player,
exit, key, warlock if the level has one, the active melee enemies), the guard is
seated next to the key, spikes are dropped only on cells walled in on ≤1 side, and
a reachable potion is placed on levels after the first. A level is re-rolled until
all of the following hold:

- exit, key and (if present) warlock are reachable from the player;
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

The input channel count `C` is read from the observation space (currently 15), so
the network adapts automatically when channels change. The 128-d feature vector
feeds the standard PPO actor-critic heads (`CnnPolicy`).

---

## Training

Two entry points share the same hyperparameters and the same versioning output:

- **`train.py`** — local. Interactively confirms the target directory and step
  count, asks for a developer comment, trains with 8 parallel envs, saves the
  model zip + TensorBoard logs, evaluates, and writes the version record. The
  `total_timesteps` value near the top of the file is edited per run (it ships set
  to a small smoke value — raise it before a real run).
- **`modal_train.py`** — the same pipeline on [Modal](https://modal.com) cloud
  workers, writing artifacts to a Modal volume. Git info is collected locally and
  passed in (the worker has no `.git`). See `modal_commands.txt` for the exact
  invocations.

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

## Evaluation and diagnostics

`evaluate.py` provides three things:

- **`evaluate(model_path, n_episodes)`** — headless tally of `won / lost / truncated`. This is what the training scripts call to fill the version record's `eval` block.
- **`diagnose(model_path, n_episodes)`** — a richer headless breakdown of *where* the agent stalls, because win/lose/truncated is too coarse to see progress. It reports:
  - the **max level reached** histogram (entering level *k* means level *k−1* was cleared, so this is the per-level clear rate);
  - how often it **emptied the board** and **picked up the key** at least once;
  - of truncated episodes, how many ended standing in a finishable state (board clear + key) — the only true "afraid of the exit" case.

  Run it with `python evaluate.py diagnose path\to\model.zip`. The model must
  match the current observation's channel count.
- **`main()` / `watch_episodes()`** — records episodes and replays the chosen
  category (won/lost/truncated) in a pygame window with an HP bar, last-move
  glyph, freeze indicator and play/step controls.

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
  counts, action dim, **all reward coefficients**, the level table), the PPO
  hyperparameters, training duration, git commit + dirty flag, library versions,
  the evaluation results, and the developer comment.
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
  e.g. adding the nearest-enemy channel, makes older zips incompatible.)
- The version number `n` is chosen at **save time**, not launch time, so
  concurrent runs into the same stage take successive free slots instead of
  colliding.

To revisit a historical run's TensorBoard, copy its folder from
`version_history/<stage>/tb_logs/` into a local `tb_logs/` and run
`tensorboard --logdir tb_logs`. As an alternative, simply run tb command with the path to tb_logs instead of plain "tb_logs" argument.

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
| `7_spikes` | Always-on spike hazards |
| `9_levels_and_potion` | **Current:** multiple levels per episode, carried-over HP/freeze, healing potions, the per-level difficulty curriculum, and the nearest-enemy combat shaping |

`challenges.txt` records design problems encountered along the way (e.g. the agent
refusing to engage the warlock and stalling until truncation — the motivation for
the combat-shaping reward and the nearest-enemy channel).

---

## Project structure

```
.
├── game_engine.py      # Game: map, player, levels, rewards, observation. Core logic.
├── enemies.py          # EnemyMixin: melee/guard/warlock movement, fireball, respawns.
├── pathfinding.py      # Pure grid helpers: BFS reachability, distance maps, stepping.
├── env.py              # Gymnasium wrapper around Game (used by PPO).
├── smallgridcnn.py     # Custom CNN feature extractor for the policy.
├── train.py            # Local training entry point (interactive).
├── modal_train.py      # Cloud (Modal) training entry point.
├── evaluate.py         # evaluate() tally, diagnose() progress breakdown, replay viewer.
├── version_utils.py    # env_signature + write_version_file (json source of truth + txt changelog + diff).
├── pg/
│   ├── pygame_renderer.py   # Renderer for grid/entities/fireball/shots.
│   └── visualize.py         # Standalone live-rendering loop for a saved model.
├── version_history/    # Per-stage saved models (zips/) and TensorBoard logs (tb_logs/).
├── version_differences/# Per-stage version records (.json source of truth + .txt changelog).
├── requirements.txt    # Python dependencies.
├── modal_commands.txt  # Reference Modal CLI commands.
├── challenges.txt      # Notes on design challenges encountered.
└── _diag.py            # Ad-hoc diagnostic script (throwaway).
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
# Train locally (edit total_timesteps in train.py first)
python train.py

# Train on Modal (detached so closing the terminal won't kill it)
modal run --detach modal_train.py --directory "9_levels_and_potion" --timesteps 1000000 --comment "..."

# Headless win/lose/truncation tally + interactive replay viewer
python evaluate.py

# Progress breakdown: where does the agent stall? (needs a current-obs model)
python evaluate.py diagnose version_history/9_levels_and_potion/zips/<model>.zip

# Watch a saved model play live (edit the path inside the file)
python pg/visualize.py

# TensorBoard
tensorboard --logdir tb_logs
```
