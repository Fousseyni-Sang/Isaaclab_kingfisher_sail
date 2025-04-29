# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/02_scene/create_scene_heron.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

torch.set_printoptions(precision=2, sci_mode=False)

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.physics.hydrodynamics import Hydrodynamics, HydrodynamicsCfg
from omni.isaac.lab.physics.aerodynamics import Aerodynamics, AerodynamicsCfg
from omni.isaac.lab.physics.hydrostatics import Hydrostatics, HydrostaticsCfg
from omni.isaac.lab.actuator_force.actuator_force import PropellerActuator, PropellerActuatorCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass
import numpy as np
from omni.isaac.lab.utils.math import transform_points

from omni.isaac.lab_assets import KINGFISHER_SAIL_CFG  # isort:skip
from omni.isaac.lab_assets import (KINGFISHER_SAIL_CFG, RED_ARROW_X_MARKER_CFG, 
 BLUE_ARROW_X_MARKER_CFG, CUBOID_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, YELLOW_ARROW_X_MARKER_CFG)  # isort: skip
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip
from omni.isaac.lab.markers import VisualizationMarkers


@configclass
class HeronSceneCfg(InteractiveSceneCfg):
    """Configuration for a heron scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    kingfisher: ArticulationCfg = KINGFISHER_SAIL_CFG.copy()
    kingfisher.prim_path = "/World/Robot"

def vector_field_visualization(
        robot, 
        vis_enabled: bool,
        directions: torch.Tensor,
        magnitudes: torch.Tensor | None,
        marker_obj,
        marker_translations: np.ndarray,
        wrap_distance: float=4.,
        update_scale: float=0.1,
        z_clip: tuple[float, float] = (0.0, 4.0),
        body_ids: list[int] | None = None,
        num_envs=1,
        step_dt=0.02
    ):
        """
        Generic function for visualizing vector fields (wind, forces, etc.)

        Args:
            vis_enabled: Whether to enable the visualization.
            directions: A (N,) tensor of angles (radians) or 2D force vectors (N x 2).
            magnitudes: Optional (N,) tensor of speeds or force magnitudes.
            marker_obj: The marker object (e.g., blue_marker).
            marker_translations: N x 3 numpy array of marker positions (updated in-place).
            wrap_distance: Distance threshold to wrap markers around.
            update_scale: How much to move markers per timestep.
            z_clip: Min and max bounds for Z values.
            body_ids: Optional list of body IDs (used for static markers like forces).
        """
        if not vis_enabled or num_envs != 1:
            return
        
        # Determine orientation angles (alpha)
        if directions.ndim == 2:  # it's a 2D vector like (Fx, Fy)
            alpha = torch.atan2(directions[:, 1], directions[:, 0]).cpu().numpy()
            speeds = torch.linalg.norm(directions, dim=-1).cpu().numpy()
        else:  # it's an angle
            alpha = directions.cpu().numpy()
            speeds = magnitudes.cpu().numpy() if magnitudes is not None else np.ones_like(alpha)

        # Compute quaternions for Z-axis (yaw) rotation
        rotation = np.stack([
            np.cos(alpha / 2),
            np.zeros_like(alpha),
            np.zeros_like(alpha),
            np.sin(alpha / 2)
        ], axis=-1)

        # Apply the same rotation to all markers
        marker_orientations = np.tile(rotation, (marker_translations.shape[0]//5, 1))
        
        # Update marker positions (only for dynamic fields like wind)
        if body_ids is None:
            marker_translations[:, :2] += (speeds * step_dt * update_scale)

            current_pos = robot.data.root_pos_w[:, :2].clone().cpu().numpy()
            dist = np.abs(marker_translations[:, :2] - current_pos)

            for i in range(2):  # x and y
                marker_translations[:, i] = np.where(
                    dist[:, i] > wrap_distance,
                    current_pos[:, i] - wrap_distance / 2,
                    marker_translations[:, i]
                )

            marker_translations[:, 2] = np.clip(marker_translations[:, 2], *z_clip)
        else:
            body_pos = robot.data.body_link_pos_w[:, body_ids, :].squeeze(0)
            marker_translations[:] = body_pos.cpu().numpy()
            marker_translations[:, 2] += 0.15
            #print(marker_translations.shape)

        # Visualize
        marker_obj.visualize(translations=marker_translations, orientations=marker_orientations)

def force_visualization(robot, force_vis:bool, array_forces_2D: torch.Tensor, body_ids:list, marker_ob, offset=0., num_envs=1):
    """
        This function visualize the forces arrow applied on the bodies based on their direction
        
        Args: 
        force_vis: True means visualize the arrow, Force means not visualize them in the simulation
        array_forces_2D: concatenate of the 2D vector of the forces to be applied on the bodies
        body_ids: Ids of the bodies which will the forces be applied on.
        Please sort the body_ids in the order as the array_forces_2D is concatenated
        ex: [force_right_thruster, force_left_thruster, force-sail] ===> [right_hull_id, left_hull_id, sail_id]
        
    """

    if force_vis and num_envs==1:
    
        n = len(body_ids)
        alpha = torch.atan2(array_forces_2D[:, 1], array_forces_2D[:, 0]).cpu().numpy()

        # Compute quaternion for rotation along the wind direction (in the world frame)
        marker_orientations = np.zeros((n, 4))
        for i in range(len(alpha)):
            marker_orientations[i, :] = np.array([np.cos(alpha[i] / 2), 0, 0, np.sin(alpha[i] / 2)]) # # Rotate around Z-axis (yaw) 

        # Apply the same rotation to all markers
        body_pos = robot.data.body_link_pos_w[:, body_ids, :].squeeze(0)
        marker_translations = body_pos.cpu().numpy()
        marker_translations += offset
        #print(f"trans: {marker_translations} shape: {marker_translations.shape}")

        # Visualize markers in the world frame
        marker_ob.visualize(translations=marker_translations, orientations=marker_orientations)

def debug_vis_callback(robot, aerodynamics, blue_marker, red_marker, marker_translations, red_marker_translations, 
                       green_marker, yellow_marker, thruster_forces, num_envs=1):
    # update the markers
    
    # apparent wind visualization
    app_wind_world = transform_points(torch.cat((aerodynamics.apparent_wind_speed, 
    torch.zeros((num_envs, 1), device=robot.device)), dim=-1), quat=robot.data.root_link_state_w[:, 3:7], 
    pos=None)[:, :2]

    app_wind_speed_world = torch.norm(app_wind_world, dim=-1) # magnitude of app_wind
    app_wind_angle_world = torch.atan2(app_wind_world[:, 1], app_wind_world[:, 0])


    true_wind_world = transform_points(torch.cat((aerodynamics.true_wind_speed2D, 
    torch.zeros((num_envs, 1), device=robot.device)), dim=-1), quat=robot.data.root_link_state_w[:, 3:7], 
    pos=None)[:, :2]
    true_wind_speed_world = torch.norm(true_wind_world, dim=-1) # magnitude of app_wind
    true_wind_angle_world = torch.atan2(true_wind_world[:, 1], true_wind_world[:, 0])
    #print(f"true_transf: {true_wind_speed_world}: {true_wind_angle_world}: {(180/torch.pi)*robot.data.heading_w}\n")
    #wind_drag_world = transform_points(aerodynamics.wind_drag[:, :3], quat=robot.data.root_link_state_w[:, 3:7], pos=None)
    vector_field_visualization(
                    robot,
                    vis_enabled=True,
                    directions=app_wind_world,
                    magnitudes=None,
                    marker_obj=blue_marker,
                    marker_translations=marker_translations,
                    wrap_distance=2.5,
                    update_scale=0.1,
                    body_ids=[3]
                    )
    
    # true wind visualization
    vector_field_visualization(
                    robot,
                    vis_enabled=True,
                    directions=true_wind_world,
                    magnitudes=None,
                    marker_obj=red_marker,
                    marker_translations=red_marker_translations,
                    wrap_distance=2.0,
                    update_scale=0.05,
                    )
    #print(f"\napp_speed: {app_wind_world} app_angle: {app_wind_angle_world}")
    #print(f"true_speed: {aerodynamics.Uw*torch.cos(aerodynamics.Beta_w), aerodynamics.Uw*torch.sin(aerodynamics.Beta_w)} :{true_wind_world} true_angle: {true_wind_angle_world}\n")

    
    thrust_left = thruster_forces[..., :3]
    thrust_right = thruster_forces[..., 3:]
    """thrust_left[:, :1] = actions[:, :1]
    thrust_right[:, :1] = actions[:, 1:2]"""
    #print(f"{thrust_left.shape} -> {thrust_right.shape} -> {aerodynamic_force.squeeze(0)[:, :2].shape}")
    #array_forces_2D = torch.cat((thrust_left, thrust_right, aerodynamics.wind_lift[:, :2], aerodynamics.wind_drag[:, :2]), dim=0)
    #force_visualization(True, thrust_left.squeeze(0)[:, :2], body_ids=[1], marker_ob=green_marker)
    #force_visualization(True, thrust_right.squeeze(0)[:, :2], body_ids=[2], marker_ob=green_marker)
    wind_lift_world = transform_points(aerodynamics.wind_lift[:, :3], quat=robot.data.root_link_state_w[:, 3:7], pos=None)
    wind_drag_world = transform_points(aerodynamics.wind_drag[:, :3], quat=robot.data.root_link_state_w[:, 3:7], pos=None)

    force_visualization(robot, True, wind_drag_world[:, :2], body_ids=[3], marker_ob=green_marker, offset=0.15)
    force_visualization(robot, True, wind_lift_world[:, :2], body_ids=[3], marker_ob=yellow_marker, offset=0.15)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    robot = scene["kingfisher"]

    # Initialize the hydrodynamics and hydrostatics
    hydrostatics_cfg = HydrostaticsCfg()
    hydrostatics_cfg.mass = 35.0  # Kg considering added sensors
    hydrostatics_cfg.width = 1.0  # Kingfisher/Heron width 1.0m in Spec Sheet
    hydrostatics_cfg.length = 1.3  # Kingfisher/Heron length 1.3m in Spec Sheet
    hydrostatics_cfg.waterplane_area = 0.33  # 0.15 width * 1.1 length * 2 hulls
    hydrostatics_cfg.draught_offset = 0.21986  # Distance from base_link to bottom of the hull
    hydrostatics_cfg.max_draught = 0.20  # Kingfisher/Heron draught 120mm in Spec Sheet
    hydrostatics_cfg.average_hydrostatics_force = 275.0
    hydrostatics = Hydrostatics(num_envs=scene.num_envs, device=robot.device, cfg=hydrostatics_cfg)

    # Aerodynamics
    aerodynamics_cfg: AerodynamicsCfg = AerodynamicsCfg()
    aerodynamics_cfg.air_density = 1.225
    aerodynamics_cfg.wing_span = 1
    aerodynamics_cfg.wing_chord = 0.2
    aerodynamics_cfg.wind_direction = 90*torch.pi/180
    aerodynamics_cfg.wind_speed = 20
    aerodynamics_cfg.angle_of_attack = 20*torch.pi/180
    aerodynamics = Aerodynamics(num_envs=scene.num_envs, device=robot.device, cfg=aerodynamics_cfg)

    hydrodynamics_cfg= HydrodynamicsCfg()
    hydrodynamics_cfg.linear_damping = [0.0, 99.99, 99.99, 13.0, 13.0, 5.83]
    hydrodynamics_cfg.quadratic_damping = [17.257603, 99.99, 10.0, 5.0, 5.0, 17.33600724]
    hydrodynamics_cfg.use_drag_randomization = False
    hydrodynamics_cfg.linear_damping_rand = [0.1, 0.1, 0.0, 0.0, 0.0, 0.1]
    hydrodynamics_cfg.quadratic_damping_rand = [0.1, 0.1, 0.0, 0.0, 0.0, 0.1]
    hydrodynamics = Hydrodynamics(num_envs=scene.num_envs, device=robot.device, cfg=hydrodynamics_cfg)

    propeller_cfg = PropellerActuatorCfg()
    propeller_cfg.cmd_lower_range = -1.0
    propeller_cfg.cmd_upper_range = 1.0
    propeller_cfg.command_rate = (propeller_cfg.cmd_upper_range - propeller_cfg.cmd_lower_range) / 2.0
    propeller_cfg.forces_left = [
        -4.0,  # -1.0
        -4.0,  # -0.9
        -4.0,  # -0.8
        -4.0,  # -0.7
        -2.0,  # -0.6
        -1.0,  # -0.5
        0.0,  # -0.4
        0.0,  # -0.3
        0.0,  # -0.2
        0.0,  # -0.1
        0.0,  # 0.0
        0.0,  # 0.1
        0.0,  # 0.2
        0.5,  # 0.3
        1.5,  # 0.4
        4.75,  # 0.5
        8.25,  # 0.6
        16.0,  # 0.7
        19.5,  # 0.8
        19.5,  # 0.9
        19.5,  # 1.0
    ]
    propeller_cfg.forces_right = propeller_cfg.forces_left
    thruster_dynamics = PropellerActuator(num_envs=scene.num_envs, device=scene.device, dt=sim_dt, cfg=propeller_cfg)
    # thruster_dynamics = PropellerActuator(num_envs=scene.num_envs, device=robot.device, dt=sim_dt, cfg=thruster_cfg)
    thruster_forces = torch.zeros((scene.num_envs, 1, 6), device=robot.device, dtype=torch.float32)
    aerodynamic_force = torch.zeros(scene.num_envs, 1, 3, device=robot.device, dtype=torch.float32)
    sail_angle = torch.zeros(scene.num_envs, device=robot.device, dtype=torch.float32)

    # Blue markers used to visualize apparent wind
    blue_marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
    blue_marker_cfg.prim_path = "/Visuals/Command/appWind"

    # red markers used to visualize true wind
    red_marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
    red_marker_cfg.prim_path = "/Visuals/Command/trueWind"

    # Green markers used to visualize thruster forces
    green_marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
    green_marker_cfg.prim_path = "/Visuals/Command/force"

    # Green markers used to visualize thruster forces
    yellow_marker_cfg = YELLOW_ARROW_X_MARKER_CFG.copy()
    yellow_marker_cfg.prim_path = "/Visuals/Command/force_yellow"

    blue_marker = VisualizationMarkers(blue_marker_cfg)
    green_marker = VisualizationMarkers(green_marker_cfg)
    red_marker = VisualizationMarkers(red_marker_cfg)
    yellow_marker = VisualizationMarkers(yellow_marker_cfg)

    n_markers = 500
    max_width = 5
    d = 5
    env_origins = robot.data.default_root_state[:, :3]
    env_pos = env_origins[0, :2].cpu().numpy()

    # Generate random translations for markers within a specified range in the world frame
    marker_translations = np.random.uniform([-env_pos[0]-d, -env_pos[1]-d, 0], 
                    [env_pos[0]+d, env_pos[1]+d, 2], (n_markers, 3))  # Adjust bounds as needed
    
    red_marker_translations = np.random.uniform([-env_pos[0]-d, -env_pos[1]-d, 0], 
                    [env_pos[0]+d, env_pos[1]+d, 2], (n_markers//10, 3))
    
    green_marker_translations = np.zeros((3, 3)) # 3 arrows for 3 bodies, both hulls and sail

    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 3000 == 0:
            # reset counter
            print("==========")
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = robot.data.default_root_state.clone()

            root_state[:, :3] += scene.env_origins
            robot.write_root_state_to_sim(root_state)
            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()

            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")

        # get robot data
        robot_pos = robot.data.root_pos_w.clone()
        robot_quat = robot.data.root_quat_w.clone()
        robot_vel = robot.data.root_vel_w.clone()
        robot_vel_b = robot.data.root_lin_vel_b.clone().detach()

        # get hydrostatics data
        hydrostatic_force = hydrostatics.compute_archimedes_metacentric_local(robot_pos, robot_quat)
        hydrodynamic_force = hydrodynamics.ComputeHydrodynamicsEffects(robot_quat, robot_vel)
        aerodynamic_force[:, 0, :] = aerodynamics.compute_wind_effect(
            aerodynamics.Uw, aerodynamics.Beta_w, robot.data.heading_w,
            robot_vel_b[:, :2], 
        )

        combined_force = hydrostatic_force + hydrodynamic_force

        force = combined_force[:, :3]
        torque = combined_force[:, 3:]

        # Get the link ids
        base_link_id, _ = robot.find_bodies("base_link")
        left_thruster_id, _ = robot.find_bodies("thruster_left")
        right_thruster_id, _ = robot.find_bodies("thruster_right")
        sail_wing_id = robot.find_bodies("sailwing")[0]
        wing_joint_dof_id = robot.find_joints("wing_joint")[0]

        # Expand forces and toques to match the number of bodies
        # E.g, for a [num_envs, 6] force, expand to [num_envs, len(body_ids), 3]
        force = force.unsqueeze(1).expand(-1, len(base_link_id), -1)
        torque = torque.unsqueeze(1).expand(-1, len(base_link_id), -1)
        robot.set_external_force_and_torque(force, torque, body_ids=base_link_id)

        thrust_cmds = torch.tensor([0.0, 1.0], dtype=torch.float32, device=robot.device)
        # Expand the thrust commands in the first dimension to match the number of environments.
        thrust_cmds = thrust_cmds.unsqueeze(0).expand(scene.num_envs, -1)

        thruster_dynamics.set_target_cmd(thrust_cmds)
        thruster_forces[:, 0, :] = 0*thruster_dynamics.update_forces()

        torque = torch.zeros_like(torque)
        robot.set_external_force_and_torque(thruster_forces[..., :3], torque, body_ids=left_thruster_id)
        robot.set_external_force_and_torque(thruster_forces[..., 3:], torque, body_ids=right_thruster_id)

        robot.set_external_force_and_torque(
                aerodynamic_force, torque, body_ids=sail_wing_id
            )
        

        current_joint_pos = robot.data.joint_pos[:, wing_joint_dof_id].clone()
        sail_angle = aerodynamics.get_sail_angle(aerodynamics.apparent_wind_angle, 
                        aerodynamics.angle_of_attack).reshape(current_joint_pos.shape)
        
        joint_error = (sail_angle - current_joint_pos + torch.pi) %(2*torch.pi) - torch.pi
        joint_pos_target = (current_joint_pos + joint_error + 2*torch.pi) % (4*torch.pi) - 2*torch.pi
        # Set psoition of the sail joint
        robot.set_joint_position_target(target=joint_pos_target.reshape(scene.num_envs,-1), 
             joint_ids=wing_joint_dof_id
            ) 
        
        debug_vis_callback(robot=robot, aerodynamics=aerodynamics, blue_marker=blue_marker, red_marker=red_marker, 
                           marker_translations=marker_translations, red_marker_translations=red_marker_translations, 
                           yellow_marker=yellow_marker, thruster_forces=thruster_forces, green_marker=green_marker)
        scene.write_data_to_sim()

        sim.step()
        count += 1
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([4.0, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = HeronSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
