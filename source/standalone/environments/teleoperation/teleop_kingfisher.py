# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import torch

import omni.log

from omni.isaac.lab.devices import Se3GamepadKingfisher, Se3Keyboard, Se3SpaceMouse
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.utils import parse_env_cfg


# Create Publisher Node
import rclpy

from omni.isaac.lab_tasks.utils.my_utils.ros2_Node import RewardWeightSubscriber, DynamicsRlAgentPublisher 

def pre_process_actions(delta_pose: torch.Tensor) -> torch.Tensor:
    """Pre-process actions for the environment."""
    thrust_right = delta_pose[:, 0:1]

    thrust_left = delta_pose[:, 2:3]

    if delta_pose[:, 5:6]>=0.05:
        sail_angle = delta_pose[:, 5:6] 
    else: 
        sail_angle = delta_pose[:, 1:2]
    

    actions = torch.cat((thrust_right, thrust_left, sail_angle), dim=1)
    #print(delta_pose)
    return actions


def main():

     # ---- Initialize ROS2 ----
    rclpy.init()
    ros_node = DynamicsRlAgentPublisher(args_cli.num_envs)

    slider_names = ['time', 'energy', 'goal', 'speed']  # Must match the names you use in the publisher
    slider_node = RewardWeightSubscriber(slider_names)
    #rclpy.spin(slider_node)

    """Running keyboard teleoperation with Isaac Lab manipulation environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # modify configuration
    env_cfg.episode_length_s = 1000000000000
    """env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # check environment name (for reach , we don't allow the gripper)
    if "Reach" in args_cli.task:
        omni.log.warn(
            f"The environment '{args_cli.task}' does not support gripper control. The device command will be ignored."
        )"""
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.unwrapped.max_available_episode_energy = 1000000000*torch.ones_like(env.unwrapped.energy)
    # create controller
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(
            pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.05 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(
            pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.005 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "gamepad":
        teleop_interface = Se3GamepadKingfisher(
            pos_sensitivity=args_cli.sensitivity, rot_sensitivity= args_cli.sensitivity
        )
        
    else:
        raise ValueError(f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse'.")
    # add teleoperation key for env reset
    teleop_interface.add_callback("L", env.reset) # It was previously "L" instead of B
    #teleop_interface.add_callback()
    # print helper for keyboard
    #print(teleop_interface)
    
    # reset environment
    env.reset()
    teleop_interface.reset()
    flow_direction = 1
    flow_speed = env.unwrapped._sail_aerodynamics.cfg.flow_speed
    is_flow_speed = True
    env.unwrapped.is_Training = False
    flow_modulo = env.unwrapped._sail_aerodynamics.cfg.flow_direction
    # simulate environment
    while simulation_app.is_running():

        # Process ROS messages (non-blocking)
        rclpy.spin_once(slider_node, timeout_sec=0.0)
        time_context = slider_node.reward_weights["time"]
        energy_context = slider_node.reward_weights["energy"]
        reset_env = energy_context > 1.5 or time_context > 1.5
        desired_speed = slider_node.reward_weights["speed"]
        env.unwrapped.energy_context[:] = energy_context
        #env.unwrapped.time_context[:] = time_context
        #print(time_context)
        # run everything in inference mode
        with torch.inference_mode():
            # get keyboard command
            delta_pose, list_bool = teleop_interface.advance()
            gripper_command, is_flow_speed = list_bool
             
            delta_pose = delta_pose.astype("float32")
            if gripper_command:
                env.reset()
                gripper_command = False

            # convert to torch
            delta_pose = torch.tensor(delta_pose, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
            # pre-process actions
            if is_flow_speed:
                flow_speed += 0.1*delta_pose[:, -1] if torch.abs(delta_pose[:, -1]) > 0.6 else 0
                env.unwrapped._sail_aerodynamics.update_flow(flow_speed=flow_speed)
            else: 
                flow_direction += 0.1*delta_pose[:, -1] if torch.abs(delta_pose[:, -1]) > 0.6 else 0
                flow_modulo = (flow_direction + torch.pi)%(2*torch.pi) - torch.pi
                env.unwrapped._sail_aerodynamics.update_flow(flow_direction=flow_modulo)
            
            #print(f"delta: {delta_pose} \tflow: {flow_speed} \tflow_direc: {flow_direction}\n")
            #flow_direction += 
            actions = pre_process_actions(delta_pose)
            #actions = actions[:, :2]
            #print(f"actions: {actions}")
            
            #print(actions.shape, actions)
            # apply actions
            teleop_interface.add_callback("L", env.reset)
            # env stepping
            obs, rew, dones, _, extras = env.step(actions)
            obs = obs['policy']

            if actions.shape[1]<3:
                actions = torch.cat([actions, torch.zeros((args_cli.num_envs, 3-actions.shape[1]), device=rew.device)], dim=-1)

            robot_pos = extras["info"].get("robot_pos_w", torch.zeros((args_cli.num_envs, 2), device=rew.device))[..., :2] # (N, 2) 
            lift_coeff = extras.get("info", {}).get("lift_coeff", torch.zeros((args_cli.num_envs,), device=rew.device)) # (N,) 
            drag_coeff = extras.get("info", {}).get("drag_coeff", torch.zeros((args_cli.num_envs,), device=rew.device)) # (N,) 
            energy = extras.get("info", {}).get("energy", torch.zeros((args_cli.num_envs,), device=rew.device)) # (N,) 
            episode_energy = extras.get("info", {}).get("episode_energy", torch.zeros((args_cli.num_envs,), device=rew.device)) # (N,) 
            max_available_energy = extras.get("info", {}).get("max_available_energy", torch.zeros((args_cli.num_envs,), device=rew.device)) # (N,) 
            ratio_energy_usage = extras.get("info", {}).get("ratio_energy_usage", torch.zeros((args_cli.num_envs,), device=rew.device)) # (N,) 
            rew_progress = extras.get("info", {}).get("reward_progress", torch.zeros((args_cli.num_envs,), device=rew.device)) # (N,) 
            rew_energy = extras.get("info", {}).get("reward_energy", torch.zeros((args_cli.num_envs,), device=rew.device)) # (N,) 
            rew_backward = extras.get("info", {}).get("reward_backward", torch.zeros((args_cli.num_envs,), device=rew.device)) # (N,) 
            rew_aero = extras.get("info", {}).get("reward_aero", torch.zeros((args_cli.num_envs,), device=rew.device)) # (N,) 
            rew_acord = extras.get("info", {}).get("reward_acord", torch.zeros((args_cli.num_envs,), device=rew.device)) # (N,) 
            bearing = extras.get("info", {}).get("bearing", torch.zeros((args_cli.num_envs,), device=rew.device)) # (N,) 
            distance = extras.get("info", {}).get("distance", torch.zeros((args_cli.num_envs,), device=rew.device)) # (N,) 
            aoa = extras.get("info", {}).get("aoa", torch.zeros((args_cli.num_envs, ), device=rew.device)) # (N,) 
            app_wind_angle = extras.get("info", {}).get("app_wind_angle", torch.zeros((args_cli.num_envs, ), device=rew.device)) # (N,) 
            true_wind_angle = extras.get("info", {}).get("true_wind_angle", torch.zeros((args_cli.num_envs, ), device=rew.device)) # (N,) 
            sail_angle = extras.get("info", {}).get("sail_angle", torch.zeros((args_cli.num_envs,), device=rew.device)) # (N,) 
            aero_force = extras.get("info", {}).get("aero_force", torch.zeros((args_cli.num_envs, 1, 6), device=rew.device)) # (N, 6) 
            max_aero_force = extras.get("info", {}).get("max_aero_force", torch.zeros((args_cli.num_envs,), device=rew.device)) # (N,) 
            thruster_force = extras.get("info", {}).get("thruster_force", torch.zeros((args_cli.num_envs, 1, 6), device=rew.device)) # (N, 6) 
            norm_error_lin = extras["info"].get("norm_error_lin", torch.zeros((args_cli.num_envs,2), device=rew.device)) # (N,) 
            norm_error_ang = extras["info"].get("norm_error_ang", torch.zeros((args_cli.num_envs,), device=rew.device)) # (N,) 
            goal_pos = extras["info"].get("goal_pos", torch.zeros((args_cli.num_envs, 2), device=rew.device)) # (N, 2) 
            norm_error_cat=torch.cat( [norm_error_lin.reshape(args_cli.num_envs, -1), norm_error_ang.reshape(args_cli.num_envs, -1)], dim=-1 )

            if reset_env:
                env.reset()

            ros_node.publish( 
                                obs=obs,
                                actions=actions, 
                                total_rew=rew, 
                                aero_force=aero_force, 
                                thruster_force=thruster_force,
                                lin_vel_b=extras.get("info", {}).get("lin_vel_b", torch.zeros((args_cli.num_envs, 3), device=rew.device)), 
                                angle_of_attack=aoa, 
                                app_flow_angle=app_wind_angle, 
                                true_flow_angle=true_wind_angle, 
                                sail_angle=sail_angle, 
                                heading_w=extras.get("info", {}).get("heading_w", torch.zeros((args_cli.num_envs,), device=rew.device)), 
                                lift_drag_ratio=extras.get("info", {}).get("lift_drag_ratio", torch.zeros((args_cli.num_envs,), device=rew.device)), 
                                robot_pos_w=robot_pos, 
                                energy=energy, 
                                episode_energy=episode_energy, 
                                lift_force_b=extras.get("info", {}).get("lift_force_b", torch.zeros((args_cli.num_envs,3), device=rew.device)), 
                                drag_force_b=extras.get("info", {}).get("drag_force_b", torch.zeros((args_cli.num_envs,3), device=rew.device)), 
                                lift_coeff=lift_coeff, 
                                drag_coeff=drag_coeff, 
                                rew_progress=rew_progress, 
                                rew_energy=rew_energy, 
                                rew_backward=rew_backward, 
                                desired_wrench_b=extras.get("info", {}).get("desired_wrench_b", torch.zeros((args_cli.num_envs, 6), device=rew.device)), 
                                ang_speed_b=extras.get("info", {}).get("ang_vel_b", torch.zeros((args_cli.num_envs, 3), device=rew.device)), 
                                norm_error_cat=norm_error_cat, 
                                rew_goal=extras.get("info", {}).get("reward_goal", torch.zeros((args_cli.num_envs,), device=rew.device)), 
                                rew_aero=rew_aero, 
                                distance=distance, 
                                ) 
            
    # Cleanup
    ros_node.destroy_node()
    slider_node.destroy_node()
    rclpy.shutdown()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
