#!/usr/bin/env python3
"""Print a compact summary of the Go2 course environment.

This script is useful in class because it shows:
- which environment is loaded
- actor vs critic observation keys
- action size
- control frequency
- reward terms that are changed by the curriculum
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from course_common import (
    DEFAULT_CONFIG_PATH,
    apply_stage_config,
    build_env_overrides,
    ensure_environment_available,
    get_ppo_config,
    lazy_import_stack,
    load_json,
    set_runtime_env,
    to_jsonable,
)


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the course config JSON.")
    parser.add_argument(
        "--stage-name",
        choices=["stage_1", "stage_2"],
        default="stage_2",
        help="Which stage config to inspect.",
    )
    parser.add_argument("--force-cpu", action="store_true", help="Force JAX onto CPU.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    config["runtime_overrides"] = {}
    if args.force_cpu:
        config["force_cpu"] = True
        config["runtime_overrides"]["force_cpu"] = True

    force_cpu = bool(config.get("force_cpu")) or bool(config.get("runtime_overrides", {}).get("force_cpu"))
    if force_cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
    set_runtime_env(force_cpu=force_cpu)

    # Import task metadata only after runtime flags are set.
    from go2_pg_env.joystick import ACTION_SIZE, ACTOR_OBS_SIZE, CRITIC_OBS_SIZE, observation_layout

    stack = lazy_import_stack()
    registry = stack["registry"]
    locomotion_params = stack["locomotion_params"]
    jax = stack["jax"]

    env_name = config["environment_name"]
    ensure_environment_available(registry, env_name)

    env_cfg = registry.get_default_config(env_name)
    ppo_cfg = get_ppo_config(locomotion_params, env_name, config["backend_impl"])
    apply_stage_config(env_cfg, ppo_cfg, config, args.stage_name)

    env = registry.load(env_name, config=env_cfg, config_overrides=build_env_overrides(config))
    state = env.reset(jax.random.PRNGKey(int(config["seed"])))

    obs_shapes = {key: list(np.asarray(value).shape) for key, value in state.obs.items()}
    summary = {
        "environment_name": env_name,
        "stage_name": args.stage_name,
        "backend_impl": config["backend_impl"],
        "control_dt": float(env.dt),
        "sim_dt": float(env_cfg.sim_dt),
        "episode_length": int(env_cfg.episode_length),
        "action_size": ACTION_SIZE,
        "actor_obs_size": ACTOR_OBS_SIZE,
        "critic_obs_size": CRITIC_OBS_SIZE,
        "observation_layout": observation_layout(),
        "obs_shapes_from_reset": obs_shapes,
        "command_range_min": list(map(float, env_cfg.command_config.min)),
        "command_range_max": list(map(float, env_cfg.command_config.max)),
        "command_keep_prob": list(map(float, env_cfg.command_config.b)),
        "reward_scales_subset": {
            "action_rate": float(env_cfg.reward_config.scales.action_rate),
            "energy": float(env_cfg.reward_config.scales.energy),
            "tracking_lin_vel": float(env_cfg.reward_config.scales.tracking_lin_vel),
            "tracking_ang_vel": float(env_cfg.reward_config.scales.tracking_ang_vel),
        },
        "ppo_network_factory": to_jsonable(ppo_cfg.network_factory),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
