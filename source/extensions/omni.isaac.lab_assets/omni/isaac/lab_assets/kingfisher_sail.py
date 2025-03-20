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
##/home/isaac_user/asv-saw

KINGFISHER_SAIL_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/isaac_user/asv-sawasp-fousseyni/Usd/kingfisher_sail.usd",
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
        pos=(0.0, 0.0, 0.12),
        joint_pos={
            ".*": 0.0,
        },
    ),
    actuators={
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=["thruster_0_joint", "thruster_1_joint"],
            stiffness=0.0,
            damping=0.0,
        ),

        "wing_actuator": ImplicitActuatorCfg(
            joint_names_expr=["wing_joint"],
            effort_limit=500.0,
            stiffness=1e20,
            damping=0.0,
        ),
    },
)
"""Configuration for the Heron robot."""

CONE_CFG = RigidObjectCfg(
        spawn=sim_utils.ConeCfg(
            radius=0.1,
            height=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=10000000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

CUBOID_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "cuboid": sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
        ),
    }
)

RED_ARROW_X_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "arrow": sim_utils.UsdFileCfg(
            usd_path=f"/home/isaac_user/asv-sawasp-fousseyni/IsaacLab/source/my_standalone/Usd/arrow.usd",
            scale=(0.1, 0.3, 0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        )
    }
)

BLUE_ARROW_X_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "arrow": sim_utils.UsdFileCfg(
            usd_path=f"/home/isaac_user/asv-sawasp-fousseyni/IsaacLab/source/my_standalone/Usd/arrow.usd",
            scale=(0.1, 0.3, 0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        )
    }
)

GREEN_ARROW_X_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "arrow": sim_utils.UsdFileCfg(
            usd_path=f"/home/isaac_user/asv-sawasp-fousseyni/IsaacLab/source/my_standalone/Usd/arrow.usd",
            scale=(0.2, 0.4, 0.2),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        )
    }
)

YELLOW_ARROW_X_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "arrow": sim_utils.UsdFileCfg(
            usd_path=f"/home/isaac_user/asv-sawasp-fousseyni/IsaacLab/source/my_standalone/Usd/arrow.usd",
            scale=(0.1, 0.3, 0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.8, 0.0)),
        )
    }
)
