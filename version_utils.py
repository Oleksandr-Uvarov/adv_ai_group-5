from pathlib import Path


def _fmt(key, val, param_display):
    return param_display.get(key, str(val))


def _format_params_text(features_dim, ppo_policy, ppo_params, param_display):
    lines = [
        "Parameters:",
        "",
        f"    features_dim (policy_kwargs): {features_dim}",
        "",
        "    PPO:",
        "",
        f"        {ppo_policy}",
    ]
    for key, val in ppo_params.items():
        lines.append(f"        {key}={_fmt(key, val, param_display)}")
    return "\n".join(lines)


def _parse_params(params_section):
    params = {}
    in_ppo = False
    for line in params_section.splitlines():
        s = line.strip()
        if not s or s == "Parameters:":
            continue
        if s == "PPO:":
            in_ppo = True
            continue
        if not in_ppo:
            if ": " in s:
                k, _, v = s.partition(": ")
                params[k] = v
        else:
            if "=" in s:
                k, _, v = s.partition("=")
                params[f"PPO.{k}"] = v
            else:
                params["PPO.policy"] = s
    return params


def _current_params(features_dim, ppo_policy, ppo_params, param_display):
    return {
        "features_dim (policy_kwargs)": str(features_dim),
        "PPO.policy": ppo_policy,
        **{f"PPO.{k}": _fmt(k, v, param_display) for k, v in ppo_params.items()},
    }


def write_version_file(n, version_differences_dir,
                       features_dim, ppo_policy, ppo_params, param_display,
                       developer_comment=""):
    params_text = _format_params_text(features_dim, ppo_policy, ppo_params, param_display)

    if n == 1:
        diff_content = params_text
    else:
        prev_file = Path(version_differences_dir) / f"version_{n - 1}.txt"
        if prev_file.exists():
            prev_text = prev_file.read_text(encoding="utf-8")
            idx = prev_text.find("Parameters:")
            prev_params = _parse_params(prev_text[idx:]) if idx != -1 else {}
        else:
            prev_params = {}

        curr_params = _current_params(features_dim, ppo_policy, ppo_params, param_display)
        changes = []
        for key in sorted(set(prev_params) | set(curr_params)):
            pv, cv = prev_params.get(key), curr_params.get(key)
            if pv != cv:
                if pv is None:
                    changes.append(f"    Added {key}: {cv}")
                elif cv is None:
                    changes.append(f"    Removed {key}")
                else:
                    changes.append(f"    {key}: {pv} → {cv}")

        change_block = "\n".join(changes) if changes else "    No parameters changed."
        diff_content = (
            f"What changed from the previous version:\n{change_block}\n\n"
            f"Developer comment: {developer_comment}\n\n"
            f"{params_text}"
        )

    version_file = Path(version_differences_dir) / f"version_{n}.txt"
    version_file.write_text(diff_content, encoding="utf-8")
    return version_file