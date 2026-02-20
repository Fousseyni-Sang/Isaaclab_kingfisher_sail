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
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.utils.math import transform_points

# Create Publisher Node
import rclpy
import torch
import pandas as pd
import numpy as np

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
    # ---- Initialize ROS2 ----
    rclpy.init()
    # Create a publisher node
    publisher = DynamicsRlAgentPublisher(num_agents=scene.num_envs)

    # --- Setup ---
    sim_dt = sim.get_physics_dt()
    decimation = sim.cfg.render_interval
    step_dt = sim_dt * decimation

    robot = scene["kingfisher"]
    base_link = robot.find_bodies("base_link")[0]

    # HL agent
    hl_agent, env = get_control_agent(
        num_envs=scene.num_envs,
        checkpoint_path=args_cli.checkpoint,
        task_name="Isaac-KingfisherSail-Direct-High-v0",
        device=scene.device,
        act_dim=3,
        obs_dim=9,
    )

    # Boat model + hydro
    spec_path = "outputs/ll/ll_model_010/spec.yaml"
    feas_path = "eval_ll/rl_games/model_010/feasibility_map.csv"

    robot_actuator_system_cfg: RobotActuatorSystemCfg = make_boat_model(load_spec(spec_path))
    robot_actuator_system: RobotActuatorSystem = RobotActuatorSystem(
        num_envs=scene.num_envs, device=scene.device, dt=step_dt, cfg=robot_actuator_system_cfg
    )

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

    # Env stuff
    actions = torch.zeros((scene.num_envs, 3), device=scene.device)
    bearing = torch.zeros(scene.num_envs, device=scene.device)
    distance = torch.zeros(scene.num_envs, device=scene.device)
    desired_pos_w = torch.zeros(scene.num_envs, 3, device=scene.device)
    initial_distance = torch.zeros(scene.num_envs, device=scene.device)
    initial_bearing = torch.zeros(scene.num_envs, device=scene.device)

    # Thrusters
    propeller_cfg = propeller_config()
    thruster_dynamics = PropellerActuator(num_envs=scene.num_envs, device=scene.device, dt=sim_dt, cfg=propeller_cfg)
    thruster_forces = torch.zeros((scene.num_envs, 1, 6), device=scene.device)


    if os.path.exists(feas_path):
        feas_map = pd.read_csv(feas_path)

    feas_np = feas_map.to_numpy(dtype=float).flatten() 
    feas_tensor = torch.tensor(feas_np, device=scene.device, dtype=torch.float32) 
    ll_feasibility_map = feas_tensor.unsqueeze(0).repeat(scene.num_envs, 1)
    ll_feasibility_map = ll_feasibility_map.reshape(scene.num_envs, 10, 10, 10)

    goal_pos_visualizer = None

    scale_vx_vy = 0.1
    max_target_lin_wrench =  1.5
    min_target_lin_wrench =  -0.3*max_target_lin_wrench
    max_target_ang_wrench =  0.5
    min_target_ang_wrench = - max_target_ang_wrench 

    # --- Helpers ---

    def feasibility_ok(vx, vy, w, feasibility_map):
        
        """
        vx, vy, w: tensors of shape (num_envs,) or scalars in physical units
        feasibility_map: (10, 10, 10) int tensor with 0/1 entries
        """

        # 2. Convert normalized → grid index [0..9]
        i = ((vx + 1) * 0.5 * 9).long().clamp(0, 9)
        j = ((vy + 1) * 0.5 * 9).long().clamp(0, 9)
        k = ((w  + 1) * 0.5 * 9).long().clamp(0, 9)

        # 3. Lookup feasibility
        feas = feasibility_map[i, j, k]

        return feas.bool()


    def set_debug_vis_impl(debug_vis: bool):
        nonlocal goal_pos_visualizer
        # create markers if necessary for the_robot_mass first tome
        if debug_vis:
            
            marker_cfg = CUBOID_MARKER_CFG.copy()
            marker_cfg.markers["cuboid"].size = (0.1, 0.1, 0.5)
            # -- goal pose
            marker_cfg.prim_path = "/Visuals/Command/goal_position"
            goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            goal_pos_visualizer.set_visibility(True)
        

    def get_obs() -> torch.Tensor:
        nonlocal distance, bearing

        desired_pos_b_3d, _ = subtract_frame_transforms(
            robot.data.root_link_state_w[:, :3],
            robot.data.root_link_state_w[:, 3:7],
            desired_pos_w,
        )
        desired_pos_b = desired_pos_b_3d[:, :2]
        distance = torch.linalg.norm(desired_pos_b, dim=1)
        bearing = torch.atan2(desired_pos_b[:, 1], desired_pos_b[:, 0])

        obs = torch.cat(
            [
                actions,                                   # 3
                robot.data.root_lin_vel_b[:, :2],         # 2
                robot.data.root_ang_vel_b[:, 2].unsqueeze(1),  # 1
                torch.cos(bearing).unsqueeze(1),          # 1
                torch.sin(bearing).unsqueeze(1),          # 1
                distance.unsqueeze(1),                    # 1
            ],
            dim=-1,
        )
        return obs

    def reset(env_ids: torch.Tensor | None=None):
        if env_ids is None or len(env_ids) == scene.num_envs:
            env_ids = torch.arange(scene.num_envs, device=scene.device)

        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += scene.env_origins
        # zero velocities
        root_state[:, 7:13] = 0.0
        robot.write_root_state_to_sim(root_state)

        joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        # target sampling
        min_target_distance = 30.0
        max_target_distance = 31.0
        min_target_bearing = 0 * torch.pi / 180
        max_target_bearing = 1 * torch.pi / 180

        initial_bearing[env_ids] = torch.zeros_like(desired_pos_w[env_ids, 0]).uniform_(
            min_target_bearing, max_target_bearing
        )
        initial_distance[env_ids] = torch.zeros_like(desired_pos_w[env_ids, 0]).uniform_(
            min_target_distance, max_target_distance
        )
        desired_pos_w[env_ids, 0] = torch.cos(initial_bearing[env_ids]) * initial_distance[env_ids]
        desired_pos_w[env_ids, 1] = torch.sin(initial_bearing[env_ids]) * initial_distance[env_ids]
        desired_pos_w[env_ids, 2] = 0.0
        desired_pos_w[env_ids, :2] += scene.env_origins[env_ids, :2]

        robot_actuator_system.reset()
        thruster_dynamics.reset()

        scene.reset()

        obs = get_obs()
        return obs

    def apply_forces(actions: torch.Tensor):
        # clamp actions
        actions_clamped = actions.clamp(-1.0, 1.0)

        # LL system
        robot_actuator_system.set_target_cmd(actions_clamped)
        robot_actuator_system.update(
            robot.data.heading_w,
            robot.data.root_lin_vel_b,
        )
        robot_system_forces = robot_actuator_system.get_forces()

        # hydro
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

        if not torch.isfinite(combined).all():
            print("🔥 NaN/Inf in combined forces, skipping force application")
            return

        robot.set_external_force_and_torque(
            combined[..., :3], combined[..., 3:], body_ids=base_link
        )

    def apply_velocities(actions: torch.Tensor):
        """
        Perfect low-level controller:
        - actions are normalized HL commands in [-1, 1]
        - actions[:,0] = vx_norm
        - actions[:,1] = vy_norm
        - actions[:,2] = w_norm
        """

        # 1. Clamp HL actions
        actions = actions.clamp(-1.0, 1.0)

        vx_norm = actions[:, 0]
        vy_norm = actions[:, 1]
        w_norm  = actions[:, 2]

        # === Correct scaling for asymmetric ranges === 
        Vx_min = min_target_lin_wrench 
        Vx_max = max_target_lin_wrench 

        Vy_min = -scale_vx_vy * max_target_lin_wrench 
        Vy_max = scale_vx_vy * max_target_lin_wrench 

        W_min = -max_target_ang_wrench 
        W_max = max_target_ang_wrench

        # 2. Convert normalized [-1,1] → physical velocities
        #    (You can choose any scaling you want here)
        vx =  actions[:, 0]*Vx_max if actions[:, 0]>0 else actions[:, 0].abs()*Vx_min
        vy = actions[:, 1]*Vy_max if actions[:, 1]>0 else actions[:, 1].abs()*Vx_min
        w  = actions[:, 2]*W_max if actions[:, 2]>0 else actions[:, 2].abs()*W_min

        # 3. Build velocity tensor for PhysX
        #    Shape: (num_envs, 6)
        #    Order: [vx, vy, vz, wx, wy, wz]
        print(f"\n[INFO] vx: {vx} vy: {vy} w:{w}")
        vel_w = transform_points(torch.stack([vx, vy, torch.zeros_like(vx)], dim=-1), 
            quat=robot.data.body_state_w[:, base_link, 3:7].reshape(scene.num_envs, -1), 
            pos=None)[:, :2]
        
        
        root_vel = torch.zeros(scene.num_envs, 6, device=scene.device)
        root_vel[:, 0:2] = vel_w
        root_vel[:, 5] = w

        # 4. Apply hydrostatic + hydrodynamic forces (optional but realistic)
        robot_pos  = robot.data.root_pos_w
        robot_quat = robot.data.root_quat_w
        robot_vel  = robot.data.root_vel_w

        hydrostatic_force[:, 0, :] = hydrostatics.compute_archimedes_metacentric_local(
            robot_pos, robot_quat
        )
        hydrodynamic_force[:, 0, :] = hydrodynamics.ComputeHydrodynamicsEffects(
            robot_quat, robot_vel
        )

        combined = hydrostatic_force + hydrodynamic_force

        if not torch.isfinite(combined).all():
            print("🔥 NaN/Inf in hydro forces — skipping")
            return

        robot.set_external_force_and_torque(
            combined[..., :3], combined[..., 3:], body_ids=base_link
        )

        # 5. Write perfect velocity to PhysX
        robot.write_root_com_velocity_to_sim(root_vel)


    # --- Main loop ---

    obs = reset(None)
    sim_step_counter = 0
    episode_length_buf = torch.zeros(scene.num_envs, device=scene.device, dtype=torch.long)
    max_episode_length = 1000
    set_debug_vis_impl(True)
    
    while simulation_app.is_running():
        # HL policy
        with torch.no_grad():
            hl_obs = hl_agent.obs_to_torch(obs)
            actions[:] = hl_agent.get_action(obs=hl_obs, is_deterministic=True, feas=ll_feasibility_map)

        target_actions = actions.clone()
        # physics decimation
        is_rendering = sim.has_gui() or sim.has_rtx_sensors()
        for _ in range(decimation):
            sim_step_counter += 1
            
            # LL system
            robot_actuator_system.set_target_cmd(target_actions)
            robot_actuator_system.update(
                robot.data.heading_w,
                robot.data.root_lin_vel_b,
            )
            actions = robot_actuator_system.thruster_actuator.current_cmds.clone()
            
            #apply_forces(actions)
            apply_velocities(actions)
            print(f"\n[INFO] target_actions: {target_actions} \n[INFO] actions: {actions} \n[INFO] robot_vel: {robot.data.root_lin_vel_b}")

            publisher.publish(
            robot_pos=robot.data.root_pos_w,
            robot_lin_vel_b=robot.data.root_lin_vel_b,
            goal_pos=desired_pos_w,  
            actions=actions,    
            robot_ang_vel_b=robot.data.root_ang_vel_b, 
            )

            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt=sim_dt)

            if sim_step_counter % decimation == 0 and is_rendering:
                sim.render()

        episode_length_buf += 1
        
        obs = get_obs()

        if not torch.isfinite(obs).all():
            print("🔥 NaN/Inf in obs, resetting envs")
            obs = reset(None)

        time_out = episode_length_buf >= max_episode_length - 1
        # Finish episode if the goal is reached
        done = torch.zeros_like(time_out)
        done[distance < 0.1] = True

        goal_pos_visualizer.visualize(desired_pos_w)

        dones = torch.logical_or(done, time_out)
        if torch.any(dones):
            obs = reset(None)
            episode_length_buf = 0


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
