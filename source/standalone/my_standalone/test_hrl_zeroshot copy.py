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
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")

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

from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.actuator_force.robot_actuator_system import RobotActuatorSystem, RobotActuatorSystemCfg
from omni.isaac.lab_tasks.utils.my_utils.boat_config import *
from omni.isaac.lab.physics.hydrodynamics import Hydrodynamics, HydrodynamicsCfg
from omni.isaac.lab.physics.hydrostatics import Hydrostatics, HydrostaticsCfg
from omni.isaac.lab_tasks.utils.my_utils.control_agent import get_control_agent
from omni.isaac.lab_tasks.utils.my_utils.ros2_Node import DynamicsRlAgentPublisher
from omni.isaac.lab_assets import KINGFISHER_SAIL_CFG  # isort:skip
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.lab.actuator_force.actuator_force import PropellerActuator, PropellerActuatorCfg

# Create Publisher Node
import rclpy
import torch

global decimation_counter
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
    # Extract scene entities

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    decimation = sim.cfg.render_interval
    step_dt = sim_dt*decimation

    robot = scene["kingfisher"]
    base_link = robot.find_bodies("base_link")[0]
    
    hl_agent, env = get_control_agent(num_envs=scene.num_envs, checkpoint_path=args_cli.checkpoint, 
                                 task_name="Isaac-KingfisherSail-Direct-High-v0", device=scene.device, 
                                 act_dim=3, obs_dim=9)
    
    spec_path = "outputs/ll/ll_model_010/spec.yaml"
    robot_actuator_system_cfg:RobotActuatorSystemCfg = make_boat_model(load_spec(spec_path))
    robot_actuator_system:RobotActuatorSystem = RobotActuatorSystem(num_envs=scene.num_envs, device=scene.device,
                                dt=step_dt, cfg=robot_actuator_system_cfg)
    
    roboat_hydrodyn_cfg = robot_actuator_system_cfg.hydrodynamics_cfg
    hydrodynamics_cfg: HydrodynamicsCfg = hydrodynamics_config()
    if roboat_hydrodyn_cfg is not None:
            hydrodynamics_cfg = roboat_hydrodyn_cfg

     
    hydrostatics_cfg: HydrostaticsCfg = hydrostatics_config()

    hydrostatics = Hydrostatics(num_envs=scene.num_envs, device=scene.device, cfg=hydrostatics_cfg)
    hydrodynamics = Hydrodynamics(num_envs=scene.num_envs, device=scene.device, cfg=hydrodynamics_cfg)

    # Forces
    hydrodynamic_force = torch.zeros(scene.num_envs, 1, 6, device=scene.device)
    hydrostatic_force = torch.zeros(scene.num_envs, 1, 6, device=scene.device)

    robot_system_forces = torch.zeros((scene.num_envs, 6), device=scene.device)
    
    # Normalized grids
    vx_norm_grid = torch.linspace(-1, 1, 10, device=scene.device)
    vy_norm_grid = torch.linspace(-1, 1, 10, device=scene.device)
    w_norm_grid  = torch.linspace(-1, 1, 10, device=scene.device)

    # Create full 3D mesh
    vx_mesh, vy_mesh, w_mesh = torch.meshgrid(
        vx_norm_grid,
        vy_norm_grid,
        w_norm_grid,
        indexing="ij"
    )

    # Flatten into (1000, 3)
    grid_pairs = torch.stack(
        [vx_mesh.flatten(), vy_mesh.flatten(), w_mesh.flatten()],
        dim=-1
    )

    ll_feasibility_map = torch.zeros((scene.num_envs, 10, 10, 10), device=scene.device)
    scale_vx_vy = 0.1
    max_target_lin_wrench =  1.5
    min_target_lin_wrench =  -0.*max_target_lin_wrench
    max_target_ang_wrench =  0.5
    min_target_ang_wrench = - max_target_ang_wrench 

    # Environment
    min_target_distance = 40.0 
    max_target_distance = 41.0
    min_target_bearing =  45*torch.pi/180 #-torch.pi / 2
    max_target_bearing = 120*torch.pi/180 #torch.pi / 2

    actions = torch.zeros((scene.num_envs, 3), device=scene.device)
    bearing = torch.zeros(scene.num_envs, device=scene.device)
    distance = torch.zeros(scene.num_envs, device=scene.device)
    desired_pos_w = torch.zeros(scene.num_envs, 3, device=scene.device)
    decimation_counter = 0
    sim_step_counter = 0
    initial_distance = torch.zeros(scene.num_envs, device=scene.device)
    initial_bearing = torch.zeros(scene.num_envs, device=scene.device)

    def get_obs() -> torch.Tensor:
        nonlocal distance
        nonlocal bearing

        desired_pos_b_3d, _ = subtract_frame_transforms(
        robot.data.root_link_state_w[:, :3], robot.data.root_link_state_w[:, 3:7], desired_pos_w
        )
        desired_pos_b = desired_pos_b_3d[:, :2]
        distance = torch.linalg.norm(desired_pos_b, dim=1)
        bearing = torch.atan2(desired_pos_b[:, 1], desired_pos_b[:, 0])

        obs = torch.cat([
            actions, # 3
            robot.data.root_lin_vel_b[:, :2], # 2
            robot.data.root_ang_vel_b[:, 2].unsqueeze(1),  # 1
            torch.cos(bearing).unsqueeze(1),  # 1
            torch.sin(bearing).unsqueeze(1),  # 1
            distance.unsqueeze(1),  # 1
        ], dim=-1)
        
        return obs
    
    def pre_physics(actions):
        nonlocal decimation_counter

        actions = actions.clamp(-1.0, 1.0)
        decimation_counter = 0

        return actions
    
    def step(actions:torch.Tensor):
        nonlocal sim_step_counter

        actions = pre_physics(actions)
        is_rendering = sim.has_gui() or sim.has_rtx_sensors()
        for _ in range(sim.cfg.render_interval):
            sim_step_counter += 1

            apply_action(actions)

            scene.write_data_to_sim()

            sim.step(render=False)

            if sim_step_counter % sim.cfg.render_interval == 0 and is_rendering:
                sim.render()

            scene.update(dt=scene.physics_dt)

        obs = get_obs()
        return obs
    
    def reset(env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == scene.num_envs:
            env_ids = torch.arange(scene.num_envs, device=scene.device)

        root_state = robot.data.default_root_state.clone()

        root_state[:, :3] += scene.env_origins
        robot.write_root_state_to_sim(root_state)
        # set joint positions with some noise
        joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()

        #joint_pos += torch.rand_like(joint_pos) * 0.1
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        initial_bearing[env_ids] = torch.zeros_like(desired_pos_w[env_ids, 0]).uniform_(
            min_target_bearing, max_target_bearing
        )
        initial_distance[env_ids] = torch.zeros_like(desired_pos_w[env_ids, 0]).uniform_(
            min_target_distance, max_target_distance
        )
        desired_pos_w[env_ids, 0] = torch.cos(initial_bearing[env_ids]) * initial_distance[env_ids]
        desired_pos_w[env_ids, 1] = torch.sin(initial_bearing[env_ids]) * initial_distance[env_ids]
        desired_pos_w[env_ids, 2] = 0.0  # only in 2D
        desired_pos_w[env_ids, :2] += scene.env_origins[env_ids, :2]
        
        robot_actuator_system.reset()
        obs = get_obs()

        return obs
    
    def apply_action(actions:torch.Tensor):
        nonlocal decimation_counter

        decimation_counter += 1

        robot_actuator_system.set_target_cmd(actions)
        robot_actuator_system.update(
        robot.data.heading_w,
        robot.data.root_lin_vel_b,
        )
        robot_system_forces = robot_actuator_system.get_forces()

        # 2) Compute hydro forces every sim step (using current state)
        robot_pos = robot.data.root_pos_w
        robot_quat = robot.data.root_quat_w
        robot_vel = robot.data.root_vel_w

        hydrostatic_force[:, 0, :] = hydrostatics.compute_archimedes_metacentric_local(
                robot_pos, robot_quat
            )
        hydrodynamic_force[:, 0, :] = hydrodynamics.ComputeHydrodynamicsEffects(
            robot_quat, robot_vel
        )
        
        combined = hydrostatic_force + hydrodynamic_force
        
        combined_thruster_forces = robot_system_forces[:, :3]
        combined_thruster_torques = robot_system_forces[:, 3:]

        combined[:, 0, :3] += combined_thruster_forces
        combined[:, 0, 3:] += combined_thruster_torques

        robot.set_external_force_and_torque(
            combined[..., :3], combined[..., 3:], body_ids=base_link
        )

        return 
    
    count = 0
    propeller_cfg = propeller_config()
    thruster_dynamics = PropellerActuator(num_envs=scene.num_envs, device=scene.device, dt=sim_dt, cfg=propeller_cfg)
    # thruster_dynamics = PropellerActuator(num_envs=scene.num_envs, device=robot.device, dt=sim_dt, cfg=thruster_cfg)
    thruster_forces = torch.zeros((scene.num_envs, 1, 6), device=robot.device, dtype=torch.float32)

    left_thruster_id, _ = robot.find_bodies("thruster_left")
    right_thruster_id, _ = robot.find_bodies("thruster_right")
    while simulation_app.is_running():
        # Reset
        if count==0 or count%100==0:
            """if count ==201:
                # reset the scene entities
                # root state
                # reset counter
                
                obs = reset(None)

            if count%10000==0:"""
            #obs = reset(None)
            # reset counter
            print("==========")
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
                
        """actions = hl_agent.get_action(obs=hl_agent.obs_to_torch(obs), is_deterministic=True, feas=ll_feasibility_map)
        obs = step(actions)

        if not torch.isfinite(obs).all():
            print("🔥 NaN/Inf in LL obs", obs)
            raise RuntimeError("Invalid HL observation")
        scene.write_data_to_sim()
        sim.step()
        count += 1"""
        # get robot data
        robot_pos = robot.data.root_pos_w.clone()
        robot_quat = robot.data.root_quat_w.clone()
        robot_vel = robot.data.root_vel_w.clone()
        robot_vel_b = robot.data.root_lin_vel_b.clone().detach()

        #sail_aerodynamics.angle_of_attack = angle_of_attack.clone().detach()
        count += 1
        hydrostatic_force = hydrostatics.compute_archimedes_metacentric_local(robot_pos, robot_quat)
        hydrodynamic_force = hydrodynamics.ComputeHydrodynamicsEffects(robot_quat, robot_vel)

        
        combined_force = hydrostatic_force + hydrodynamic_force

        force = combined_force[:, :3]
        torque = combined_force[:, 3:] 

        force = force.unsqueeze(1).expand(-1, len(base_link), -1)
        torque = torque.unsqueeze(1).expand(-1, len(base_link), -1)
        robot.set_external_force_and_torque(force, torque, body_ids=base_link)
        
        robot_vel_b = robot.data.root_lin_vel_b.clone().detach()
        # compute thruster commands based on the distance and bearing
        width = 0  # width of the robot
        full_thrust = 2
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
        #scene.update(sim_dt)

        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)

def main():
    """Main function."""
    # Load kit helper

    physics_dt = 1 / 60.0  # 60 Hz
    decimation = 30
    
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt=physics_dt,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
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
