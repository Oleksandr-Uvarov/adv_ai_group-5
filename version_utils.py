import json
import subprocess
from datetime import datetime
from pathlib import Path


# --------------------------------------------------------------------------
# Collectors
# --------------------------------------------------------------------------

def read_git_info():
    """Current commit hash + whether the working tree has uncommitted changes.

    Returns ``{"commit": None, "dirty": None}`` if git isn't available (e.g. on
    a remote worker that only received a few source files). Call this *locally*
    and pass the result through to remote training so the commit is still
    recorded (see modal_train.py)."""
    def _run(args):
        try:
            return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def _format_duration(seconds):
    if seconds is None:
        return "unknown"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s"


def _lib_versions():
    versions = {}
    for name in ("stable_baselines3", "torch", "gymnasium", "numpy"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = None
    return versions


def env_signature(env):
    """Describe the environment an agent is trained on: observation shape, grid
    size, entity counts, action space and reward coefficients.

    Pass a freshly constructed ``GameEnv`` (not a vectorised wrapper) so the
    entity counts reflect a clean spawn. This is the information you need to know
    whether a saved checkpoint is even loadable into a given env (the channel
    count lives in ``obs_shape[0]``)."""
    from game_engine import Game

    game = env.game
    return {
        "obs_shape": list(env.observation_space.shape),
        "grid_size": game.grid_size,
        "n_melee": sum(1 for m in game.melee_poses if m is not None),
        "n_melee_total": game.TOTAL_MELEE_ENEMIES,
        "n_guard": int(game.guard_pos is not None),
        "n_warlock": int(game.warlock_pos is not None),
        "action_dim": int(env.action_space.n),
        "reward_coeffs": Game.reward_coeffs(),
    }


# --------------------------------------------------------------------------
# Diffing (operates on the machine-readable records, not on the text)
# --------------------------------------------------------------------------

def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _load_prev_record(version_differences_dir, n):
    """The version_{n-1} record to diff against. Matches both the legacy
    ``version_{n-1}.json`` and the suffixed ``version_{n-1}_{run_tag}.json``; the
    ``_`` before the tag keeps ``version_1_*`` from also matching ``version_10``.
    If concurrent runs produced several version n-1 files, diff against the most
    recent (lexicographically last = latest timestamp tag)."""
    version_differences_dir = Path(version_differences_dir)
    prev_n = n - 1
    candidates = sorted(
        list(version_differences_dir.glob(f"version_{prev_n}.json"))
        + list(version_differences_dir.glob(f"version_{prev_n}_*.json"))
    )
    if not candidates:
        return None
    try:
        return json.loads(candidates[-1].read_text(encoding="utf-8"))
    except Exception:
        return None


def _format_diff(prev_record, curr_record):
    if prev_record is None:
        return "    (no previous machine-readable version to compare against)"

    # Only diff the substantive fields - skip timestamps, git and lib versions.
    fields = ("training", "env")
    prev = _flatten({k: prev_record.get(k, {}) for k in fields})
    curr = _flatten({k: curr_record.get(k, {}) for k in fields})

    changes = []
    for key in sorted(set(prev) | set(curr)):
        pv, cv = prev.get(key), curr.get(key)
        if pv != cv:
            if key not in prev:
                changes.append(f"    Added {key}: {cv}")
            elif key not in curr:
                changes.append(f"    Removed {key} (was {pv})")
            else:
                changes.append(f"    {key}: {pv} -> {cv}")
    return "\n".join(changes) if changes else "    No tracked values changed."


# --------------------------------------------------------------------------
# Human-readable rendering
# --------------------------------------------------------------------------

def _format_eval(ev):
    if ev is None:
        return "    (not evaluated)"
    n = ev.get("n_episodes", 0)
    won, lost, trunc = ev.get("n_won", 0), ev.get("n_lost", 0), ev.get("n_truncated", 0)
    pct = lambda k: f"{k / n * 100:.1f}%" if n else "n/a"
    return (f"    episodes={n}   won={won} ({pct(won)})   "
            f"lost={lost} ({pct(lost)})   truncated={trunc} ({pct(trunc)})")


def _format_txt(record, diff_block):
    env = record["env"]
    tr = record["training"]
    rc = env["reward_coeffs"]
    git = record.get("git") or {}
    libs = record.get("libraries") or {}
    timing = record.get("timing") or {}

    commit = git.get("commit")
    git_short = commit[:10] if commit else "unknown"
    dirty = git.get("dirty")
    dirty_str = " (dirty)" if dirty else ("" if dirty is False else " (unknown)")

    ppo_lines = "\n".join(f"        {k}={v}" for k, v in tr["ppo_params"].items())
    libs_str = "  ".join(f"{k} {v}" for k, v in libs.items() if v)

    lines = [
        "What changed from the previous version:",
        diff_block,
        "",
        f"Developer comment: {record.get('developer_comment', '')}",
        "",
        "version recorded:",
        f"obs_shape: {tuple(env['obs_shape'])}   grid_size: {env['grid_size']}",
        f"enemies: {env['n_melee']} melee, {env['n_guard']}, warlock {env['n_warlock']}  action_dim: {env['action_dim']}",
        f"reward coeffs: goal {rc['goal_step']}, kill {rc['kill']}, "
        f"freeze {rc['freeze']}, enemy_step {rc['enemy_step']}",
        "",
        f"training: total_timesteps={tr['total_timesteps']}   n_envs={tr['n_envs']}",
        f"duration: {timing.get('duration', 'unknown')}   "
        f"(started {timing.get('started_at')}, ended {timing.get('ended_at')})",
        f"git: {git_short}{dirty_str}   recorded: {record['recorded_at']}",
        f"libraries: {libs_str}",
        "",
        "Evaluation:",
        _format_eval(record.get("eval")),
        "",
        "Parameters:",
        "",
        f"    features_dim (policy_kwargs): {tr['features_dim']}",
        "",
        "    PPO:",
        "",
        f"        {tr['policy']}",
        ppo_lines,
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def write_version_file(n, version_differences_dir, *,
                       features_dim, ppo_policy, ppo_params,
                       total_timesteps, n_envs, signature,
                       developer_comment="", git_info=None,
                       started_at=None, ended_at=None, duration_seconds=None,
                       eval_results=None, run_tag=None):
    """Write both a machine-readable ``version_{n}.json`` (source of truth) and a
    human-readable ``version_{n}.txt`` (changelog + snapshot) for a trained run.

    ``signature`` comes from :func:`env_signature`. ``git_info`` may be supplied
    by the caller (for remote runs that lack a git checkout); otherwise it's
    collected here. ``started_at``/``ended_at`` are timestamp strings and
    ``duration_seconds`` the wall-clock training time. ``run_tag``, when given, is
    appended to the filenames (``version_{n}_{run_tag}.json``) so two runs that
    land on the same ``n`` never overwrite each other's records; it is also stored
    inside the record so it can be matched back to the model zip. Returns the
    ``.txt`` path."""
    version_differences_dir = Path(version_differences_dir)
    stem = f"version_{n}" if not run_tag else f"version_{n}_{run_tag}"

    record = {
        "version": n,
        "run_tag": run_tag,
        "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "timing": {
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_seconds": duration_seconds,
            "duration": _format_duration(duration_seconds),
        },
        "git": git_info if git_info is not None else read_git_info(),
        "libraries": _lib_versions(),
        "training": {
            "total_timesteps": total_timesteps,
            "n_envs": n_envs,
            "features_dim": features_dim,
            "policy": ppo_policy,
            "ppo_params": dict(ppo_params),
        },
        "env": signature,
        "eval": eval_results,
        "developer_comment": developer_comment,
    }

    json_file = version_differences_dir / f"{stem}.json"
    json_file.write_text(json.dumps(record, indent=2), encoding="utf-8")

    prev_record = _load_prev_record(version_differences_dir, n)
    diff_block = _format_diff(prev_record, record)
    txt_file = version_differences_dir / f"{stem}.txt"
    txt_file.write_text(_format_txt(record, diff_block), encoding="utf-8")

    return txt_file
