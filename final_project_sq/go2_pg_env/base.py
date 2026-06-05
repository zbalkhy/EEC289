"""Base utilities for the local Go2 MuJoCo Playground environments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env

from . import constants as consts


class Go2Env(mjx_env.MjxEnv):
    """Base class shared by local Go2 tasks."""

    def __init__(
        self,
        xml_path: str,
        config: config_dict.ConfigDict,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)

        self._xml_path = str(Path(xml_path).resolve())
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = self._config.sim_dt
        self._mj_model.opt.ccd_iterations = 20

        # The policy outputs position offsets.
        # MuJoCo actuators convert them into torques through a PD-style actuator model.
        self._mj_model.dof_damping[6:] = config.Kd
        self._mj_model.actuator_gainprm[:, 0] = config.Kp
        self._mj_model.actuator_biasprm[:, 1] = -config.Kp

        self._mj_model.vis.global_.offwidth = 1920
        self._mj_model.vis.global_.offheight = 1080

        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._imu_site_id = self._mj_model.site("imu").id
        self._feet_floor_found_sensor = [
            self._mj_model.sensor(f"{foot}_floor_found").id for foot in consts.FEET_GEOMS
        ]

    def get_upvector(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.UPVECTOR_SENSOR)

    def get_gravity(self, data: mjx.Data) -> jax.Array:
        return data.site_xmat[self._imu_site_id].T @ jp.array([0.0, 0.0, -1.0])

    def get_global_linvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.GLOBAL_LINVEL_SENSOR)

    def get_global_angvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.GLOBAL_ANGVEL_SENSOR)

    def get_local_linvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.LOCAL_LINVEL_SENSOR)

    def get_accelerometer(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.ACCELEROMETER_SENSOR)

    def get_gyro(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.GYRO_SENSOR)

    def get_feet_pos(self, data: mjx.Data) -> jax.Array:
        return jp.vstack(
            [mjx_env.get_sensor_data(self.mj_model, data, sensor_name) for sensor_name in consts.FEET_POS_SENSOR]
        )

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
