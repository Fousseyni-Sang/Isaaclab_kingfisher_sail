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

from ros2_Node import RlAgentPublisher, RewardWeightSubscriber

def pre_process_actions(delta_pose: torch.Tensor) -> torch.Tensor:
    """Pre-process actions for the environment."""
    thrust_right = delta_pose[:, 0:1]

    thrust_left = delta_pose[:, 2:3]

    if delta_pose[:, 5:6]>=0.01:
        sail_angle = delta_pose[:, 5:6] 
    else: 
        sail_angle = delta_pose[:, 1:2]
    actions = torch.cat((thrust_right, thrust_left, sail_angle), dim=1)
    #print(delta_pose)
    return actions


def main():

     # ---- Initialize ROS2 ----
    rclpy.init()
    ros_node = RlAgentPublisher(args_cli.num_envs)

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
    wind_direction = 1
    wind_speed = env.unwrapped._aerodynamics.cfg.wind_speed
    is_wind_speed = True
    env.unwrapped.is_Training = False
    wind_modulo = env.unwrapped._aerodynamics.cfg.wind_direction
    # simulate environment
    while simulation_app.is_running():

        # Process ROS messages (non-blocking)
        rclpy.spin_once(slider_node, timeout_sec=0.0)
        time_context = slider_node.reward_weights["time"]
        energy_context = slider_node.reward_weights["energy"]
        reset_env = energy_context > 1.5 or time_context > 1.5
        desired_speed = slider_node.reward_weights["speed"]
        env.unwrapped.heading_context[:] = energy_context
        env.unwrapped.speed_context[:] = time_context
        #print(time_context)
        # run everything in inference mode
        with torch.inference_mode():
            # get keyboard command
            delta_pose, list_bool = teleop_interface.advance()
            gripper_command, is_wind_speed = list_bool
             
            delta_pose = delta_pose.astype("float32")
            if gripper_command:
                env.reset()
                gripper_command = False

            # convert to torch
            delta_pose = torch.tensor(delta_pose, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
            # pre-process actions
            if is_wind_speed:
                wind_speed += 0.1*delta_pose[:, -1] if torch.abs(delta_pose[:, -1]) > 0.6 else 0
                env.unwrapped._aerodynamics.update_wind(wind_speed=wind_speed)
            else: 
                wind_direction += 0.1*delta_pose[:, -1] if torch.abs(delta_pose[:, -1]) > 0.6 else 0
                wind_modulo = (wind_direction + torch.pi)%(2*torch.pi) - torch.pi
                env.unwrapped._aerodynamics.update_wind(wind_direction=wind_modulo)
            
            #print(f"delta: {delta_pose} \twind: {wind_speed} \twind_direc: {wind_direction}\n")
            #wind_direction += 
            actions = pre_process_actions(delta_pose)
            #print(actions)
            # apply actions
            teleop_interface.add_callback("L", env.reset)
            # env stepping
            obs, rew, dones, _, _ = env.step(actions)
            obs = obs['policy']

            env.unwrapped.desired_speed_b[:] = desired_speed
            aero_force = env.unwrapped._aerodynamic_force_b.squeeze(0)
            thruster_force = env.unwrapped._thruster_forces.squeeze(0)
            lin_speed = env.unwrapped._robot.data.root_lin_vel_b
            aoa = (180/torch.pi)*env.unwrapped._aerodynamics.angle_of_attack
            app_angle = (180/torch.pi)*env.unwrapped._aerodynamics.apparent_wind_angle # in degree
            sail = (180/torch.pi)*env.unwrapped.joint_angle_mapped_pi
            head_w = (180/torch.pi)*env.unwrapped._robot.data.heading_w
            head_wrt_wind = torch.abs(head_w - (180/torch.pi)*env.unwrapped._aerodynamics.Beta_w)
            lift = env.unwrapped._aerodynamics.wind_lift_b
            drag = env.unwrapped._aerodynamics.wind_drag_b
            ld_ratio = torch.norm(lift, dim=-1)/torch.norm(drag, dim=-1)
            #ld_ratio = torch.abs(aero_force[:, 0]/(aero_force[:, 1]+1e-6))
            robot_pos = env.unwrapped._robot.data.root_link_pos_w[:, :2]
            goal_pos =  env.unwrapped._desired_pos_w[:, :2]
            energy = env.unwrapped.energy
            episode_energy = env.unwrapped.episode_energy     
            lift_coeff = env.unwrapped._aerodynamics.lift_coeff
            drag_coeff = env.unwrapped._aerodynamics.drag_coeff
            sum_angle = sail + app_angle + aoa
            desired_pos = torch.abs(env.unwrapped.desired_pos_b)
            rew_progress = env.unwrapped.reward_progress
            rew_bearing = env.unwrapped.reward_bearing
            rew_energy = env.unwrapped.reward_energy
            rew_backward = env.unwrapped.reward_backward
            loss = env.unwrapped.loss_discrim_heading
            loss_disc = torch.tensor([loss.item()], device=rew_backward.device) if loss is not None else torch.zeros_like(rew_energy)
            



            if reset_env:
                env.reset()

            ros_node.publish(obs, actions, rew, aero_force, thruster_force, lin_speed, aoa, app_angle, sail, 
                             head_w, head_wrt_wind, ld_ratio, robot_pos, goal_pos, energy, episode_energy, lift, 
                             drag, lift_coeff, drag_coeff, sum_angle, desired_pos, rew_progress, rew_bearing, rew_energy, rew_backward, loss_disc)
            
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
