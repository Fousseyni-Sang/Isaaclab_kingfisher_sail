# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Heron robot."""

from __future__ import annotations

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.assets import RigidObjectCfg, RigidObjectCollectionCfg, ArticulationCfg
from omni.isaac.lab.markers.visualization_markers import VisualizationMarkersCfg
# from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##/home/fousseyni/asv-saw

root_path = "/mnt/gpu_storage/zrr/fsangare/asv-sawasp-fousseyni"
TURTLEBOT3_BURGER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/mnt/ssd-storage/fsangare/asv-sawasp-fousseyni/Isaaclab_kingfisher_sail/Usd/turtlebot3_burger.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            rigid_body_enabled=True,
            max_linear_velocity=10000.0,
            max_angular_velocity=10000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.023),
        joint_pos={
            ".*": 0.0,
        },
    ),
    actuators={
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=["a__namespace_wheel_left_joint", "a__namespace_wheel_right_joint"],
            stiffness=0.0,
            damping=1e4,
        ),

    },
)
