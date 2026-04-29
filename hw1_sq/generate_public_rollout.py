#!/usr/bin/env python3
"""Generate a deterministic public-benchmark rollout bundle from a checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from benchmark_specs import command_for_step, public_command_episode_label, public_command_script
from course_common import (
    DEFAULT_CONFIG_PATH,
    apply_stage_config,
    build_env_overrides,
    ensure_environment_available,
    get_ppo_config,
    lazy_import_stack,
    load_json,
    save_json,
    set_runtime_env,
)
from test_policy import load_policy_with_workaround


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Path to a PPO checkpoint directory.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the course config JSON.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the rollout bundle.")
    parser.add_argument(
        "--stage-name",
        choices=["stage_1", "stage_2"],
        default="stage_2",
        help="Which stage config to use when building the eval environment.",
    )
    parser.add_argument("--num-episodes", type=int, default=4, help="Number of public benchmark episodes to run.")
    parser.add_argument(
        "--episode-length-steps",
        type=int,
        default=None,
        help="Optional manual override for benchmark episode length.",
    )
    parser.add_argument("--render-first-episode", action="store_true", help="Render the first benchmark episode.")
    parser.add_argument("--render-width", type=int, default=960, help="Rendered video width.")
    parser.add_argument("--render-height", type=int, default=540, help="Rendered video height.")
    parser.add_argument("--render-camera", type=str, default="track", help="Camera name used for MuJoCo rendering.")
    parser.add_argument("--force-cpu", action="store_true", help="Force JAX onto CPU.")
    return parser.parse_args()


def _force_command(state, command: np.ndarray, jax):
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

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stack = lazy_import_stack()
    registry = stack["registry"]
    locomotion_params = stack["locomotion_params"]
    jax = stack["jax"]
    media = stack["media"]

    env_name = config["environment_name"]
    ensure_environment_available(registry, env_name)

    env_cfg = registry.get_default_config(env_name)
    ppo_cfg = get_ppo_config(locomotion_params, env_name, config["backend_impl"])
    apply_stage_config(env_cfg, ppo_cfg, config, args.stage_name)

    episode_length = int(round(config["public_eval"]["episode_length_seconds"] / env_cfg.ctrl_dt))
    if args.episode_length_steps is not None:
        episode_length = int(args.episode_length_steps)
    env_cfg.episode_length = episode_length

    env = registry.load(env_name, config=env_cfg, config_overrides=build_env_overrides(config))
    policy = load_policy_with_workaround(args.checkpoint_dir.resolve(), deterministic=True)
    if not force_cpu:
        policy = jax.jit(policy)

    reset_fn = env.reset if force_cpu else jax.jit(env.reset)
    step_fn = env.step if force_cpu else jax.jit(env.step)

    episode_ids = []
    command_xy = []
    measured_xy = []
    command_yaw = []
    measured_yaw = []
    fell = []
    joint_torques = []
    joint_velocities = []
    foot_slip_speed = []
    first_episode_trajectory = []

    safe_ranges = config["public_eval"]["safe_command_ranges"]
    rng = jax.random.PRNGKey(int(config["seed"]) + 42)
    episode_lengths = []

    for episode_idx in range(int(args.num_episodes)):
        rng, reset_key = jax.random.split(rng)
        state = reset_fn(reset_key)
        commands = public_command_script(safe_ranges, episode_idx)
        state = _force_command(state, np.asarray(commands[0], dtype=np.float32), jax)

        if episode_idx == 0 and args.render_first_episode:
            first_episode_trajectory = [state]

        steps_in_this_episode = 0
        for step_idx in range(episode_length):
            command = command_for_step(commands, step_idx, episode_length)
            state = _force_command(state, command, jax)

            rng, act_key = jax.random.split(rng)
            action, _ = policy(state.obs, act_key)
            state = step_fn(state, action)
            state = _force_command(state, command, jax)

            episode_ids.append(episode_idx)
            command_xy.append(command[:2])
            measured_xy.append(np.asarray(env.get_local_linvel(state.data)[:2], dtype=np.float32))
            command_yaw.append(command[2])
            measured_yaw.append(np.asarray(env.get_gyro(state.data)[2], dtype=np.float32))
            joint_torques.append(np.asarray(state.data.actuator_force, dtype=np.float32))
            joint_velocities.append(np.asarray(state.data.qvel[6:], dtype=np.float32))
            feet_vel = np.asarray(state.data.sensordata[env._foot_linvel_sensor_adr], dtype=np.float32)
            foot_slip_speed.append(np.linalg.norm(feet_vel[:, :2], axis=-1).astype(np.float32))
            done = bool(np.asarray(state.done))
            fell.append(done)

            steps_in_this_episode += 1
            if episode_idx == 0 and args.render_first_episode:
                first_episode_trajectory.append(state)

            if done:
                break

        episode_lengths.append(steps_in_this_episode)

    rollout_npz = output_dir / "rollout_public_eval.npz"
    np.savez(
        rollout_npz,
        episode_id=np.asarray(episode_ids, dtype=np.int32),
        command_lin_vel_xy=np.asarray(command_xy, dtype=np.float32),
        measured_lin_vel_xy=np.asarray(measured_xy, dtype=np.float32),
        command_yaw_rate=np.asarray(command_yaw, dtype=np.float32),
        measured_yaw_rate=np.asarray(measured_yaw, dtype=np.float32),
        fell=np.asarray(fell, dtype=bool),
        joint_torques=np.asarray(joint_torques, dtype=np.float32),
        joint_velocities=np.asarray(joint_velocities, dtype=np.float32),
        foot_slip_speed=np.asarray(foot_slip_speed, dtype=np.float32),
    )

    summary = {
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "stage_name": args.stage_name,
        "num_episodes": int(args.num_episodes),
        "episode_length_steps": episode_length,
        "episode_lengths_realized": episode_lengths,
        "episode_labels": [public_command_episode_label(idx) for idx in range(int(args.num_episodes))],
        "rollout_npz": str(rollout_npz),
    }

    if args.render_first_episode and first_episode_trajectory:
        video_path = output_dir / "public_eval_episode0.mp4"
        frames = env.render(
            first_episode_trajectory,
            height=int(args.render_height),
            width=int(args.render_width),
            camera=args.render_camera,
        )
        media.write_video(video_path, frames, fps=int(round(1.0 / env.dt)))
        summary["video_path"] = str(video_path)

    save_json(output_dir / "rollout_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
