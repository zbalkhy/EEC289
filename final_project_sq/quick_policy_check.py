#!/usr/bin/env python3
"""Run a very small sanity check on the environment or a saved checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from benchmark_specs import build_demo_segments, command_for_step
from course_common import (
    DEFAULT_CONFIG_PATH,
    apply_stage_config,
    build_env_overrides,
    ensure_environment_available,
    get_ppo_config,
    lazy_import_stack,
    load_json,
    set_runtime_env,
)
from test_policy import load_policy_with_workaround


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the course config JSON.")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Optional checkpoint to test.")
    parser.add_argument(
        "--stage-name",
        choices=["stage_1", "stage_2"],
        default="stage_2",
        help="Which stage config to use.",
    )
    parser.add_argument("--num-steps", type=int, default=100, help="Number of control steps to run.")
    parser.add_argument("--force-cpu", action="store_true", help="Force JAX onto CPU.")
    return parser.parse_args()


def _force_command(state, command, jax):
    state.info["command"] = jax.numpy.asarray(command, dtype=jax.numpy.float32)
    state.info["steps_until_next_cmd"] = np.int32(10**9)
    return state


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

    reset_fn = env.reset if force_cpu else jax.jit(env.reset)
    step_fn = env.step if force_cpu else jax.jit(env.step)

    rng = jax.random.PRNGKey(int(config["seed"]) + 999)
    state = reset_fn(rng)
    start_xy = np.asarray(state.data.qpos[:2], dtype=np.float32)

    if args.checkpoint_dir is None:
        print("Environment reset successful.")
        print(f"obs keys: {list(state.obs.keys())}")
        print(f"action size: {env.action_size}")
        print(f"dt: {env.dt}")
        return

    policy = load_policy_with_workaround(args.checkpoint_dir.resolve(), deterministic=True)
    if not force_cpu:
        policy = jax.jit(policy)

    segments = build_demo_segments(config)
    for step_idx in range(int(args.num_steps)):
        command = command_for_step(segments, step_idx, int(args.num_steps))
        state = _force_command(state, command, jax)
        rng, act_key = jax.random.split(rng)
        action, _ = policy(state.obs, act_key)
        state = step_fn(state, action)
        state = _force_command(state, command, jax)
        if bool(np.asarray(state.done)):
            print(f"terminated early at step {step_idx + 1}")
            break

    end_xy = np.asarray(state.data.qpos[:2], dtype=np.float32)
    print(f"start_xy={start_xy.tolist()}")
    print(f"end_xy={end_xy.tolist()}")
    print(f"displacement_xy={(end_xy - start_xy).tolist()}")
    print(f"base_height={float(np.asarray(state.data.qpos[2])):.4f}")


if __name__ == "__main__":
    main()
