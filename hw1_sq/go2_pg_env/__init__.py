"""Register the local Go2 joystick task into MuJoCo Playground."""

from __future__ import annotations

import functools

from mujoco_playground._src import locomotion

from . import joystick
from . import randomize


ENV_NAME = "Go2JoystickFlatTerrain"


def register() -> str:
    """Register the local environment and return its public name."""
    locomotion._envs[ENV_NAME] = functools.partial(joystick.Joystick, task="flat_terrain")
    locomotion._cfgs[ENV_NAME] = joystick.default_config
    locomotion._randomizer[ENV_NAME] = randomize.domain_randomize
    return ENV_NAME
