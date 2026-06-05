"""Domain randomization for the local Go2 environment.

This mirrors the official Go1 randomization recipe closely. The key idea is:
make the simulator slightly wrong on purpose, so the policy learns robustness.
"""

from __future__ import annotations

import jax
import mujoco
from mujoco import mjx

from . import constants as consts


_MODEL = mujoco.MjModel.from_xml_path(consts.FEET_ONLY_FLAT_TERRAIN_XML.as_posix())
FLOOR_GEOM_ID = _MODEL.geom("floor").id
TORSO_BODY_ID = _MODEL.body(consts.ROOT_BODY).id


def domain_randomize(model: mjx.Model, rng: jax.Array):
    @jax.vmap
    def rand_dynamics(single_rng):
        # 1) Floor friction
        single_rng, key = jax.random.split(single_rng)
        geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
            jax.random.uniform(key, minval=0.4, maxval=1.0)
        )

        # 2) Joint-side friction loss
        single_rng, key = jax.random.split(single_rng)
        frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(
            key, shape=(12,), minval=0.9, maxval=1.1
        )
        dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

        # 3) Armature
        single_rng, key = jax.random.split(single_rng)
        armature = model.dof_armature[6:] * jax.random.uniform(
            key, shape=(12,), minval=1.0, maxval=1.05
        )
        dof_armature = model.dof_armature.at[6:].set(armature)

        # 4) Center of mass jitter on the torso
        single_rng, key = jax.random.split(single_rng)
        dpos = jax.random.uniform(key, (3,), minval=-0.05, maxval=0.05)
        body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(model.body_ipos[TORSO_BODY_ID] + dpos)

        # 5) Global link mass scale
        single_rng, key = jax.random.split(single_rng)
        dmass = jax.random.uniform(key, shape=(model.nbody,), minval=0.9, maxval=1.1)
        body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

        # 6) Extra torso mass offset
        single_rng, key = jax.random.split(single_rng)
        dmass = jax.random.uniform(key, minval=-1.0, maxval=1.0)
        body_mass = body_mass.at[TORSO_BODY_ID].set(body_mass[TORSO_BODY_ID] + dmass)

        # 7) Rest-pose jitter
        single_rng, key = jax.random.split(single_rng)
        qpos0 = model.qpos0
        qpos0 = qpos0.at[7:].set(
            qpos0[7:] + jax.random.uniform(key, shape=(12,), minval=-0.05, maxval=0.05)
        )

        return (
            geom_friction,
            body_ipos,
            body_mass,
            qpos0,
            dof_frictionloss,
            dof_armature,
        )

    friction, body_ipos, body_mass, qpos0, dof_frictionloss, dof_armature = rand_dynamics(rng)

    in_axes = jax.tree_util.tree_map(lambda _: None, model)
    in_axes = in_axes.tree_replace(
        {
            "geom_friction": 0,
            "body_ipos": 0,
            "body_mass": 0,
            "qpos0": 0,
            "dof_frictionloss": 0,
            "dof_armature": 0,
        }
    )

    randomized_model = model.tree_replace(
        {
            "geom_friction": friction,
            "body_ipos": body_ipos,
            "body_mass": body_mass,
            "qpos0": qpos0,
            "dof_frictionloss": dof_frictionloss,
            "dof_armature": dof_armature,
        }
    )
    return randomized_model, in_axes
