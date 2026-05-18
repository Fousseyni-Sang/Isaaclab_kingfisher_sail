# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Heron robot."""

from __future__ import annotations

from omni.isaac.lab.actuators.actuator_cfg import DCMotorCfg
from omni.isaac.lab.actuators.actuator_cfg import DCMotorCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.assets import RigidObjectCfg, RigidObjectCollectionCfg, ArticulationCfg
from omni.isaac.lab.markers.visualization_markers import VisualizationMarkersCfg
# from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##/home/fousseyni/asv-saw

root_path = "/home/GTL/fsangare"
TURTLEBOT3_BURGER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/GTL/fsangare/Isaaclab_kingfisher_sail/Usd/turtlebot2_burger.usd", #turtlebot3_bg.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            rigid_body_enabled=True,
            max_linear_velocity=0.7,
            max_angular_velocity=180.0,
            max_depenetration_velocity=1.0,
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
    "left_motor": DCMotorCfg(
        joint_names_expr=["a__namespace_wheel_left_joint"],
        velocity_limit=20.0,      # rad/s
        effort_limit=1.0,         # Nm
        stiffness=0.0,           # Nm/rad
        damping=1e4,             # Nms/rad
        saturation_effort=0.6,    # peak torque
    ),
    "right_motor": DCMotorCfg(
        joint_names_expr=["a__namespace_wheel_right_joint"],
        velocity_limit=20.0,
        effort_limit=1.0,
        stiffness=0.0,           # Nm/rad
        damping=1e4,             # Nms/rad
        saturation_effort=0.6,    # peak torque
    ),
    }
)

effort_limit=2.0         # Nm
saturation_effort=2.5   # peak torque
damping = 10      # Nms/rad
wheel_speed_limit = 70.0 # rad/s, corresponds 
friction = 0.1       # Adding a small physical friction helps RL convergence
KOBUKI_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/GTL/fsangare/Isaaclab_kingfisher_sail/Usd/kobuki_standalone.usd", #turtlebot3_bg.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            rigid_body_enabled=True,
            max_linear_velocity=0.8,
            max_angular_velocity=180.0,
            max_depenetration_velocity=1.0,
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
        pos=(0.0, 0.0, 0.0352),
        joint_pos={
            ".*": 0.0,
        },
    ),

    actuators={ # specification https://iclebo-kobuki.readthedocs.io/en/latest/anatomy.html with gear_ratio 50:1
        "wheels": DCMotorCfg( # You can group them if they share the same config
            joint_names_expr=["wheel_.*_joint"],
            effort_limit=effort_limit,
            stiffness=0.0,       # Velocity control usually uses 0 stiffness
            damping=damping,
            saturation_effort=saturation_effort,
            friction=friction,       # Adding a small physical friction helps RL convergence
        ),
    }
)

KOBUKI_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/GTL/fsangare/Isaaclab_kingfisher_sail/Usd/kobuki_standalone.usd", #turtlebot3_bg.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            rigid_body_enabled=True,
            max_linear_velocity=0.8,
            max_angular_velocity=180.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0352),
        joint_pos={
            ".*": 0.0,
        },
    ),
    actuators={ # specification https://iclebo-kobuki.readthedocs.io/en/latest/anatomy.html with gear_ratio 50:1
        "wheels": ImplicitActuatorCfg( # You can group them if they share the same config
            joint_names_expr=["wheel_.*_joint"],
            effort_limit=effort_limit,
            stiffness=0.0,       # Velocity control usually uses 0 stiffness
            damping=damping,
        ),
    }
)

ROBOT_CARTER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/GTL/fsangare/Isaaclab_kingfisher_sail/Usd/robotcarter.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            rigid_body_enabled=True,
            max_linear_velocity=10000.0,
            max_angular_velocity=10000.0,
            max_depenetration_velocity=10.0,
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
            joint_names_expr=["left_wheel", "right_wheel"],
            stiffness=0.0,
            damping=1e4,
        ),

    },
)