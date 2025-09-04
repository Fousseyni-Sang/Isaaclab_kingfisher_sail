from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import ObservationTermCfg
from omni.isaac.lab.sensors import Camera, ContactSensor, Imu, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def uniform_context(env: ManagerBasedRLEnv, n=200):
    """Return a context vector sampled every n iterations."""
    # allocate storage once
    if not hasattr(env, "context_vec"):
        env.context_vec = torch.zeros(env.num_envs, 1, device=env.device)
        env._last_context_update = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # check if episode_length_buf is available
    if not hasattr(env, "episode_length_buf"):
        return env.context_vec
    
    # update context every n steps
    needs_update = (env.episode_length_buf % n == 0) & (env._last_context_update != env.episode_length_buf)
    if needs_update.any():
        env.context_vec[needs_update] = torch.rand_like(env.context_vec[needs_update])
        env._last_context_update[needs_update] = env.episode_length_buf[needs_update]
        #print(f"conetxt updated: {env.context_vec[needs_update]}")

    return env.context_vec