# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Quacopter environment.
"""

import gymnasium as gym

from . import agents
from .bipedal_walker_ctx import ContexWrapperCfg
##
# Register Gym environments.
##

gym.register(
    id="Bipedal-WalkerCtx-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.bipedal_gym:BipedalWalkerCtxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ContexWrapperCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
