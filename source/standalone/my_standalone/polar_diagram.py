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
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
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
from omni.isaac.lab.physics.hydrostatics import Hydrostatics, HydrostaticsCfg
from omni.isaac.lab.physics.foil_dynamics import FoilDynamicsCfg, FoilDynamics
from omni.isaac.lab.actuator_force.actuator_force import PropellerActuator, PropellerActuatorCfg
from omni.isaac.lab.actuator_force.foil_actuator_force import FoilActuator, FoilActuatorCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass
import numpy as np
from omni.isaac.lab.utils.math import transform_points, subtract_frame_transforms
from omni.isaac.lab_tasks.utils.my_utils.boat_config import (rudder_keel_cfg, rudder_keel_sail_cfg, 
jellyfish_roboat_cfg, single_motor_sailboat_cfg, kingfisher_roboat_cfg, vap2_roboat_cfg, vap4_roboat_cfg, 
hydrodynamics_config, hydrostatics_config, sail_config, sail_actuator_config, propeller_config)
from omni.isaac.lab_tasks.utils.my_utils.ros2_Node import DynamicsRlAgentPublisher
from omni.isaac.lab_assets import KINGFISHER_SAIL_CFG  # isort:skip
from omni.isaac.lab_assets import (KINGFISHER_SAIL_CFG, RED_ARROW_X_MARKER_CFG, 
 BLUE_ARROW_X_MARKER_CFG, CUBOID_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, YELLOW_ARROW_X_MARKER_CFG)  # isort: skip
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip
from omni.isaac.lab.markers import VisualizationMarkers

# Create Publisher Node
import rclpy
import torch

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


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""

    # ---- Initialize ROS2 ----
    rclpy.init()
    # Create a publisher node
    sail_publisher = DynamicsRlAgentPublisher(num_agents=scene.num_envs)

    # Extract scene entities

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    robot = scene["kingfisher"]

    # Initialize the hydrodynamics and hydrostatics
    hydrostatics_cfg = hydrostatics_config()
    hydrostatics = Hydrostatics(num_envs=scene.num_envs, device=robot.device, cfg=hydrostatics_cfg)

    # sail_Aerodynamics
    sail_aerodynamics_cfg: FoilDynamicsCfg = sail_config()
    sail_aerodynamics = FoilDynamics(num_envs=scene.num_envs, device=robot.device, cfg=sail_aerodynamics_cfg)
    sail_aerodynamics.init_flow_vector(robot.data.root_lin_vel_b[:, 0:2], robot.data.heading_w)
    hydrodynamics_cfg= hydrodynamics_config()
    hydrodynamics = Hydrodynamics(num_envs=scene.num_envs, device=robot.device, cfg=hydrodynamics_cfg)
    sail_actuator_cfg: FoilActuatorCfg = sail_actuator_config()

    propeller_cfg = propeller_config()
    thruster_dynamics = PropellerActuator(num_envs=scene.num_envs, device=scene.device, dt=sim_dt, cfg=propeller_cfg)
    # thruster_dynamics = PropellerActuator(num_envs=scene.num_envs, device=robot.device, dt=sim_dt, cfg=thruster_cfg)
    thruster_forces = torch.zeros((scene.num_envs, 1, 6), device=robot.device, dtype=torch.float32)
    sail_aerodynamic_force_b = torch.zeros(scene.num_envs, 1, 6, device=robot.device, dtype=torch.float32)
    foil_angle = torch.zeros(scene.num_envs, device=robot.device, dtype=torch.float32)
    sail_actuator = FoilActuator(
            num_envs=scene.num_envs, dt=sim_dt, dynamics=sail_aerodynamics, cfg=sail_actuator_cfg
        )


    env_origins = robot.data.default_root_state[:, :3]

    aoa = 25*torch.pi/180
    target_angle_of_attack = aoa*torch.ones(scene.num_envs, device=robot.device, dtype=torch.float32)
    count = 0

    # Environment
    min_target_distance = 100.0 
    max_target_distance =  100.0
    min_target_bearing =  0*torch.pi/180
    max_target_bearing = 4*torch.pi/180 #torch.pi / 2

    _desired_pos_w = torch.zeros(scene.num_envs, 3, device=robot.device)
    desired_pos_b = torch.zeros(scene.num_envs, 3, device=robot.device)

    initial_distance = torch.zeros_like(_desired_pos_w[:, 0]).uniform_(
            min_target_distance, max_target_distance
        )
    distance = initial_distance
    previous_distance = initial_distance
    initial_bearing = torch.zeros_like(_desired_pos_w[:, 0]).uniform_(
            min_target_bearing, max_target_bearing
        )
    _desired_pos_w[:, 0] = torch.cos(initial_bearing) * initial_distance
    _desired_pos_w[:, 1] = torch.sin(initial_bearing) * initial_distance
    # Simulation loop
    previous_distance = distance.clone()
    
    max_speed = 1.5  # Maximum speed of the robot
    max_energy = 2
    energy_penalty_scale = -1  # Scale for energy penalty
    progress_reward_scale = 1.0  # Scale for progress reward
    time_penalty_scale = -0.1  # Scale for time penalty


    max_flow_speed = np.sqrt(sail_aerodynamics.cfg.flow_speed + max_speed)  # Maximum wind speed
    max_aero_force = sail_aerodynamics.get_max_aero_force(sail_aerodynamics.cfg.flow_speed)
    target_cmds = torch.tensor([[0.9]], device=robot.device).repeat(scene.num_envs, 1)

    # Get the link ids
    base_link_id, _ = robot.find_bodies("base_link")
    left_thruster_id, _ = robot.find_bodies("thruster_left")
    right_thruster_id, _ = robot.find_bodies("thruster_right")
    wing_joint_dof_id = robot.find_joints("wing_joint")[0]  
    while simulation_app.is_running():
        # Reset
        if count == 0:
            
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = robot.data.default_root_state.clone()

            root_state[:, :3] += scene.env_origins
            robot.write_root_state_to_sim(root_state)
            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()

            #joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            initial_robot_pos = robot.data.root_link_pos_w.clone()

            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
            previous_robot_pos = robot.data.root_link_pos_w.clone()

        # get robot data
        robot_pos = robot.data.root_pos_w.clone()
        robot_quat = robot.data.root_quat_w.clone()
        robot_vel = robot.data.root_vel_w.clone()
        robot_vel_b = robot.data.root_lin_vel_b.clone().detach()

        #sail_aerodynamics.angle_of_attack = angle_of_attack.clone().detach()
        
        hydrostatic_force = hydrostatics.compute_archimedes_metacentric_local(robot_pos, robot_quat)
        hydrodynamic_force = hydrodynamics.ComputeHydrodynamicsEffects(robot_quat, robot_vel)
        
        current_joint_pos = sail_actuator.dynamics.foil_angle.reshape(scene.num_envs, -1).clone()
        sail_actuator.update_joint_cmd(current_joint_pos, target_cmds)
        sail_actuator.update_forces(robot.data.heading_w, robot.data.root_lin_vel_b)
        sail_aerodynamic_force_b[:, 0, :] = sail_actuator.get_forces_and_torques()

        
        combined_force = hydrostatic_force + hydrodynamic_force
        #print(f"combined_force: {combined_force.shape}, reshape_aero_force: {reshape_aero_force.shape}")
        combined_force[:, :3] = combined_force[:, :3] + sail_aerodynamic_force_b[:, 0, :3]
        combined_force[:, 3:] = combined_force[:, 3:] + sail_aerodynamic_force_b[:, 0, 3:]

        force = combined_force[:, :3]
        torque = combined_force[:, 3:] 

        force = force.unsqueeze(1).expand(-1, len(base_link_id), -1)
        torque = torque.unsqueeze(1).expand(-1, len(base_link_id), -1)
        robot.set_external_force_and_torque(force, torque, body_ids=base_link_id)
        
        # compute the desired bearing and distance to the next waypoint
        desired_pos_b_3d, _ = subtract_frame_transforms(
            robot.data.root_link_state_w[:, :3], robot.data.root_link_state_w[:, 3:7], _desired_pos_w
        )
        
        desired_pos_b[:, :2] = desired_pos_b_3d[:, :2]
        distance = torch.linalg.norm(desired_pos_b, dim=1)
        bearing = torch.round(torch.atan2(desired_pos_b[:, 1], desired_pos_b[:, 0]), decimals=2)
        
        full_thrust = 1  # full thrust command 

        progress = progress_reward_scale*(1/sim_dt)*(previous_distance - distance) / max_speed

        current_robot_pos = robot.data.root_link_pos_w.clone()
        position_progress = current_robot_pos - previous_robot_pos

        robot_vel_b = robot.data.root_lin_vel_b.clone().detach()

        grad_direction = _desired_pos_w - current_robot_pos
        grad_direction = grad_direction / torch.linalg.norm(grad_direction, dim=-1, keepdim=True)
        projected_progress = torch.sum(position_progress*grad_direction, dim=-1)/torch.linalg.norm(grad_direction, dim=-1)
        projected_progress = progress_reward_scale*(projected_progress / max_speed)*(1/sim_dt)  # normalize by the initial distance

        # compute thruster commands based on the distance and bearing
        width = 0  # width of the robot
        
        thruster_cmd1 = round(torch.clip(full_thrust + width*bearing / torch.pi, min=-1, max=1.0).item(), 2)
        thruster_cmd2 = round(torch.clip(full_thrust - width*bearing / torch.pi, min=-1, max=1.0).item(), 2)

        thrust_cmds = torch.tensor([thruster_cmd1, thruster_cmd2], dtype=torch.float32, device=robot.device)
        # Expand the thrust commands in the first dimension to match the number of environments.
        thrust_cmds = thrust_cmds.unsqueeze(0).expand(scene.num_envs, -1)

        thruster_dynamics.set_target_cmd(thrust_cmds)
        thruster_forces[:, 0, :] = thruster_dynamics.update_forces()

        torque = torch.zeros_like(torque)
        robot.set_external_force_and_torque(thruster_forces[..., :3], torque, body_ids=left_thruster_id)
        robot.set_external_force_and_torque(thruster_forces[..., 3:], torque, body_ids=right_thruster_id)

        joint_pos_target = sail_actuator.get_joint_positions()
        robot.set_joint_position_target(target=joint_pos_target.reshape(scene.num_envs,-1), 
             joint_ids=wing_joint_dof_id
            ) 
        progress = progress.cpu().numpy().item()  # Convert to numpy for publishing

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
