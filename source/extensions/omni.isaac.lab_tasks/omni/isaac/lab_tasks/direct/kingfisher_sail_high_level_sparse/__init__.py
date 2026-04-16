# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Quacopter environment.
"""

import gymnasium as gym

from . import agents
from .kingfisher_sail_env import KingfisherSailEnv, KingfisherSailEnvCfg

##
# Register Gym environments.
##


gym.register(
    id="Isaac-KingfisherSail-Direct-High-SP-v0",
    entry_point="omni.isaac.lab_tasks.direct.kingfisher_sail_high_level_sparse:KingfisherSailEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": KingfisherSailEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KingfisherPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_sac_ddpg_cfg.yaml",
    },
)
