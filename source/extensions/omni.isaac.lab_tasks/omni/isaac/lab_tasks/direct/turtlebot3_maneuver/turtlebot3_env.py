# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms, transform_points, quat_from_euler_xyz
import numpy as np

##
# Pre-defined configs
##
from omni.isaac.lab_assets import TURTLEBOT3_BURGER_CFG, CUBOID_MARKER_CFG


class TurtleBot3EnvWindow(BaseEnvWindow):
    """Window manager for the TurtleBot3 environment."""

    def __init__(self, env: TurtleBot3Env, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)  


# CSI-Code-WhenIsGood-9cspihx
@configclass
class TurtleBot3EnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 100.0 #30
    physics_dt = 1 / 60.0  # 60 Hz
    decimation = 3
    step_dt = physics_dt * decimation  # 20 Hz
    action_space = 2 #gym.spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32)
    observation_space = 16 
    """gym.spaces.Dict({
                        "observation": gym.spaces.Box(low=-1, high=1, shape=(16, ), dtype=np.float32), 
                        "achieved_goal": gym.spaces.Box(low=np.array([-np.inf, -np.inf, -np.pi]), high=np.array([np.inf, np.inf, np.pi]), shape=(3, ), dtype=np.float32), # [x, y, theta]
                        "desired_goal": gym.spaces.Box(low=np.array([-np.inf, -np.inf, -np.pi]), high=np.array([np.inf, np.inf, np.pi]), shape=(3, ), dtype=np.float32),  # [x, y, theta]
                    })"""
    state_space = 0
    debug_vis = True

    ui_window_class_type = TurtleBot3EnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
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
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        debug_vis=True,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=10.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = TURTLEBOT3_BURGER_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    cone_cfg = RigidObjectCfg(

        prim_path="/World/envs/env_.*/Cone",
        spawn=sim_utils.ConeCfg(
            radius=0.1,
            height=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),

    )

    # camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/base_link/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-5.0, 0.0, 15.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=80,
        height=80,
    )
    

    max_energy = 2.0  # Max of 1.0 per thruster
    max_available_energy = max_energy
    max_robot_speed = 1.5 # Max speed for energy optimization

    lin_velocity_command_scale = 0.4 # Scale for the velocity commands output by the policy
    ang_velocity_command_scale = 1.5
    # reward scales
    distance_reward_scale = 0.0
    distance_progress_reward_scale = 1.4 #2 #0.5 #30 #5 # 6 too much
    bearing_progress_reward_scale = 0.0

    goal_reached_threshold = 0.3
    bearing_reached_threshold = 0.3
    goal_reached_scale =  800.0 #

    energy_penalty_scale = -1 #0.5 #0.5 #0.08  #-0.001
    backwards_penalty_scale = -0.5
    time_penalty_scale = -1 #-0.008 #
    penalty_inefficient_sailing_scale = -0.1
    tack_penalty_scale = -10
    bearing_penalty_scale = 0.5
    beargin_penalty_coef = -4 #0.5 #-4
    lift_drag_ratio_scale = 0.1
    acord_reward_scale = 0.5
    speed_penalty_scale = -0.1

    # Environment
    min_target_distance = 5.0 
    max_target_distance = 10.0
    min_target_bearing =  45*torch.pi/180 #-torch.pi / 2
    max_target_bearing = 120*torch.pi/180 #torch.pi / 2
    max_cross_track = 8.0


class TwistToWheels:
    def __init__(self, wheel_radius=0.035, wheel_base=0.23):
        self.r = wheel_radius
        self.b = wheel_base

    def convert(self, vx:torch.Tensor, wz:torch.Tensor):
        # Compute wheel angular velocities (rad/s)
        w_l = (vx - 0.5 * self.b * wz) / self.r
        w_r = (vx + 0.5 * self.b * wz) / self.r
        return torch.stack([w_l, w_r], dim=1)


class TurtleBot3Env(DirectRLEnv):
    cfg: TurtleBot3EnvCfg

    def __init__(self, cfg: TurtleBot3EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Actions
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)

        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "1_distance_progress",
                "2_goal_reached",
                "3_energy",
                "4_backwards",
                "5_bearing_penalty",
                "6_time",
                "7_tack_penalty",
                "8_lift_drag_ratio",    
            ]
        }
        # Get the link ids
        self.base_link_id, _ = self._robot.find_bodies("a__namespace_base_link")
        
        self.left_wheel_joint_id, _ = self._robot.find_joints("a__namespace_wheel_left_joint")
        self.right_wheel_joint_id, _ = self._robot.find_joints("a__namespace_wheel_right_joint")

        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()


        # Buffers
        self.distance = torch.zeros(self.num_envs, device=self.device)
        self.previous_distance = torch.zeros(self.num_envs, device=self.device)
        self.distance_progress = torch.zeros(self.num_envs, device=self.device)
        self.initial_distance = torch.zeros(self.num_envs, device=self.device)
        self.cross_track_error = torch.zeros(self.num_envs, device=self.device)
        self.position_progress = torch.zeros(self.num_envs, device=self.device)

        

        self.bearing = torch.zeros(self.num_envs, device=self.device)
        self.previous_bearing = torch.zeros(self.num_envs, device=self.device)
        self.bearing_progress = torch.zeros(self.num_envs, device=self.device)
        self.initial_bearing = torch.zeros(self.num_envs, device=self.device)
        self.use_thruster = torch.zeros(self.num_envs, device=self.device)
        self.desired_orientation = torch.zeros(self.num_envs, device=self.device)

        self.energy = torch.zeros(self.num_envs, device=self.device)
        self.episode_energy = torch.zeros_like(self.energy)
        self.max_available_episode_energy = torch.zeros_like(self.energy)
        self.ratio_energy_usage = torch.zeros_like(self.energy)
        self.episode_avg_speed = torch.zeros_like(self.energy)
        self.episode_avg_lift_drag_ratio = torch.zeros_like(self.energy)
        self.desired_pos_b = torch.zeros(self.num_envs, 2, device=self.device)
        self.desired_trajectory_b = torch.zeros(self.num_envs, 10, 2) # 10 pts
        self.desired_speed_b = torch.zeros(self.num_envs, device=self.device)
        self.desired_bearing = torch.zeros(self.num_envs, device=self.device)


        self.joint_pos_target = torch.zeros(self.num_envs, device=self.device)
        self.sail_angle = torch.zeros(self.num_envs, device=self.device)
        self.joint_angle_mapped_pi = torch.zeros(self.num_envs, device=self.device)
        self.actual_angle_of_attack = torch.zeros(self.num_envs, device=self.device)
        
        self.previous_robot_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.initial_robot_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        self.previous_sail_action = torch.zeros_like(self.actions[:, -1])

        self.episode_number = torch.ones(self.num_envs, device=self.device)
        self.global_step = 0

        self.is_Training = True

        self.reward_progress = torch.zeros(self.num_envs, device=self.device)
        self.reward_success = torch.zeros(self.num_envs, device=self.device)
        self.reward_bearing = torch.zeros(self.num_envs, device=self.device)
        self.reward_energy = torch.zeros(self.num_envs, device=self.device)
        self.reward_backward = torch.zeros(self.num_envs, device=self.device)
        self.reward_acord = torch.zeros(self.num_envs, device=self.device)
        self.reward_aero = torch.zeros(self.num_envs, device=self.device)
        self.reward_time = torch.zeros(self.num_envs, device=self.device)

        self.wheels_joint_cmds = torch.zeros((self.num_envs, 2), device=self.device)
        
        self.normalize_heading = torch.zeros(self.num_envs, device=self.device)
        self.normalized_energy = torch.zeros(self.num_envs, device=self.device)

        self.max_aero_force = torch.zeros(self.num_envs, device=self.device)  # Max aerodynamic force for the current episode

        # ============================================================================================#
        # ======================== Markers for the wind visualization ================================#
        # ============================================================================================#

        self.env_pos = self._terrain.env_origins[0, :2].cpu().numpy()

        self.twist_to_wheels = TwistToWheels(wheel_radius=0.035, wheel_base=0.23)
        # ============================================================================================#
        # ============================================================================================#


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        #self.cone = RigidObject(self.cfg.cone_cfg)
        #self.scene.rigid_objects["cone"] = self.cone
        #self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        
        #self.scene.sensors["tiled_camera"] = self._tiled_camera

        self.visualize = False

        
    def _pre_physics_step(self, actions: torch.Tensor):
        
        self._actions = actions.clone().clamp(-1.0, 1.0)

        lin_velocity = self._actions[:, 0] * self.cfg.lin_velocity_command_scale
        ang_velocity = self._actions[:, 1] * self.cfg.ang_velocity_command_scale

        self.wheels_joint_cmds = self.twist_to_wheels.convert(lin_velocity, ang_velocity) 

        #=====================================================================================================#

    def get_info(self):
        
        info = {
            "lin_vel_b": self._robot.data.root_lin_vel_b,                  # (N, 3)
            "ang_vel_b": self._robot.data.root_ang_vel_b,                  # (N, 3)
            "robot_pos_w": self._robot.data.root_pos_w,                    # (N, 3)
            "heading_w": self._robot.data.heading_w,                       # (N,)
            "lin_target_wrench": torch.zeros((self.num_envs, 1), device=self.device),   # (N, 1)
            "ang_target_wrench": torch.zeros((self.num_envs, 1), device=self.device),   # (N, 1)
            "energy": self.energy,                                         # (N,)
            "episode_energy": self.episode_energy,                         # (N,)
            "max_available_energy": self.max_available_episode_energy,     # (N,)
            "ratio_energy_usage": self.ratio_energy_usage,                 # (N,)
            "reward_progress": self.reward_progress,                       # (N,)
            "reward_acord": self.reward_acord,                             # (N,) 
            "reward_energy": self.reward_energy,                           # (N,)
            "reward_goal": self.reward_success,                            # (N,)
            "reward_backward": self.reward_backward,                       # (N,)
            "reward_aero": self.reward_aero,                               # (N,)
            "bearing": self.bearing,                                       # (N,)
            "distance": self.distance,                                     # (N,)
            "norm_error_lin": torch.zeros((self.num_envs, 1), device=self.device),                         # (N,)
            "norm_error_ang": torch.zeros((self.num_envs, 1), device=self.device),                         # (N,)
            "goal_pos": self._desired_pos_w[:, :2],                        # (N, 2)
            "desired_wrench_b": torch.zeros((self.num_envs, 1), device=self.device), # (N, 3)

            # Combined error vector (N, 3)
            "norm_error_cat": torch.cat(
                [
                    torch.zeros((self.num_envs, 1), device=self.device),
                    torch.zeros((self.num_envs, 1), device=self.device),
                ],
                dim=-1,
            ),

        }

        return info   
    
    
    def _apply_action(self):
        # only apply thruster forces if they are not zero, otherwise it disables external previous forces.

        self._robot.set_joint_velocity_target(target=self.wheels_joint_cmds, joint_ids=[self.left_wheel_joint_id[0], self.right_wheel_joint_id[0]])


    def compute_reward(self, achieved_goal, desired_goal, info={}, is_her=True):
        # Case 1: HER → NumPy arrays
        alpha = 0.5
        beta = 1 - alpha

        if isinstance(achieved_goal, np.ndarray):
            
            pos_error = np.linalg.norm(achieved_goal[:, :2] - desired_goal[:, :2], axis=1)
            norm_pos_error = (pos_error-self.cfg.min_target_distance)/(self.cfg.max_target_distance-self.cfg.min_target_distance)
            heading_diff = achieved_goal[:, 2] - desired_goal[:, 2]
            heading_error = np.abs(np.arctan2(np.sin(heading_diff), np.cos(heading_diff)))
            success = np.logical_and(pos_error <= self.cfg.goal_reached_threshold, heading_error<=self.cfg.bearing_reached_threshold).astype(float)

            return -alpha*norm_pos_error + success - (1-alpha)
            
        # Case 2: IsaacLab internal reward → Torch tensors
        elif isinstance(achieved_goal, torch.Tensor):

            ag = achieved_goal
            dg = desired_goal
            pos_error = torch.norm(ag[:, :2] - dg[:, :2], dim=1)
            norm_pos_error = (pos_error-self.cfg.min_target_bearing)/(self.cfg.max_target_distance-self.cfg.min_target_distance)
            heading_diff = ag[:, 2] - dg[:, 2]
            heading_error = torch.atan2(torch.sin(heading_diff), torch.cos(heading_diff)).abs()
            success = torch.logical_and(pos_error <= self.cfg.goal_reached_threshold, heading_error<=self.cfg.bearing_reached_threshold).float()
            success = pos_error <= self.cfg.goal_reached_threshold
            return -alpha*norm_pos_error, success, - (1-alpha)
        else:
            raise TypeError(f"Unsupported type for compute_reward: {type(achieved_goal)}")

    
    def compute_reward_maneuver_net(self, achieved_goal, desired_goal, info={}, rew_type:str="default"):
        """
            type: hourglass, default, max
        """

        delta_goal = achieved_goal[:, :2] - desired_goal[:, :2]
        error_bias = (1.0, 2.0) 

        if rew_type=="hourglass":
            reward = self.get_reward_hourglass(delta_goal, error_bias)
        elif rew_type=="max":
            pass
        else:
            reward = self.get_reward_box(delta_goal, error_bias)

        return reward

    def get_reward_box(self, delta_goal, error_bias):

        """
            copied and adapted from: 
            https://github.com/MelodieDANIEL/4ws_actor_critic_maneuvering/blob/main/reward_shape.ipynb

            delta_goal: batch of (delta_x, delta_y), size: (batch_size, 2) the error in position along the x and y axes in the robot frame.
            error_bias: (bias_x, bias_y) the bias to apply to the error along the x and y axes. This can be used to shape 
            the reward to encourage certain behaviors, such as prioritizing progress along the x-axis (towards the goal) 
            over the y-axis (cross-track error).

            bias: (bias_x, bias_y) the bias to apply to the error along the x and y axes. This can be used to shape

        """

        if isinstance(delta_goal, torch.Tensor):

            coords = delta_goal
            bias = torch.ones_like(coords) * torch.tensor(error_bias, device=coords.device)
            reward = -torch.linalg.norm(coords * bias, dim=1)
        else:
            coords = delta_goal
            bias = np.ones_like(coords) * np.array(error_bias)
            reward = -np.linalg.norm(coords * bias, axis=1)

        return reward
    
    def get_reward_hourglass(self, delta_goal, error_bias):

        """
            copied and adapted from: 
            https://github.com/MelodieDANIEL/4ws_actor_critic_maneuvering/blob/main/reward_shape.ipynb

            delta_goal: batch of (delta_x, delta_y), size: (batch_size, 2) the error in position along the x and y axes in the robot frame.
            error_bias: (bias_x, bias_y) the bias to apply to the error along the x and y axes. This can be used to shape 
            the reward to encourage certain behaviors, such as prioritizing progress along the x-axis (towards the goal) 
            over the y-axis (cross-track error).

            bias: (bias_x, bias_y) the bias to apply to the error along the x and y axes. This can be used to shape

        """

        if isinstance(delta_goal, torch.Tensor):

            coords = delta_goal
            bias = torch.ones_like(coords) * torch.tensor(error_bias, device=coords.device)
            delta = torch.min((coords[:, 0].abs() - coords[:, 1].abs()), torch.zeros_like(coords[:, 0]))
            coords[:, 1] -= torch.sign(coords[:, 1]) * delta * 1.0
            reward = -torch.linalg.norm(coords * bias, dim=1)
        else:
            coords = delta_goal
            bias = np.ones_like(coords) * np.array(error_bias)
            delta = min((abs(coords[:, 0]) - abs(coords[:, 1])), 0)
            coords[:, 1] -= np.sign(coords[:, 1]) * delta * 1.0
            reward = -np.linalg.norm(coords * bias, axis=1)

        return reward


    def _get_observations(self) -> dict:

        # Observations
        self.global_step += 1
        # Desired position in the robot frame (2D)
        self.desired_pos_b_3d, _ = subtract_frame_transforms(
            self._robot.data.root_link_state_w[:, :3], self._robot.data.root_link_state_w[:, 3:7], self._desired_pos_w
        )
        self.desired_pos_b[:, :2] = self.desired_pos_b_3d[:, :2]
        self.distance = torch.linalg.norm(self.desired_pos_b, dim=1)
        self.bearing = torch.atan2(self.desired_pos_b[:, 1], self.desired_pos_b[:, 0])

        self.energy = torch.sum(torch.square(self._actions[:, :2]), dim=1)
        self.normalized_energy = self.energy / self.cfg.max_energy

        self.ratio_energy_usage = self.episode_energy / self.max_available_episode_energy
        
        self.energy = torch.sum(torch.square(self._actions[:, :2]), dim=1)
        
        base_obs = torch.cat(
            [
                self._actions,  # 2
                self._robot.data.root_lin_vel_b[:, :2]/self.cfg.max_robot_speed,  # 2
                self._robot.data.root_ang_vel_b[:, 2].unsqueeze(1),  # 1
                torch.cos(self.bearing).unsqueeze(1),  # 1
                torch.sin(self.bearing).unsqueeze(1),  # 1
                torch.cos(self._robot.data.heading_w).unsqueeze(1),  # 1
                torch.sin(self._robot.data.heading_w).unsqueeze(1),  # 1
                (self.distance/self.initial_distance).unsqueeze(1),  # 1

            ],
            dim=1,
        )
        
        current_config = torch.cat(
            [self._robot.data.root_pos_w[:, :2]], dim=1
        ) 

        target_config = torch.cat(
            [self._desired_pos_w[:, :2]], dim=1
        ) 
        
        if isinstance(self.cfg.observation_space, int):
            obs = torch.cat([base_obs, current_config, target_config], dim=1)
        else:
            obs = {
                "observation":base_obs, # 10
                "achieved_goal": current_config, # 3
                "desired_goal":target_config, # 3
            }
    
        observations = {"policy": obs}

        if not self.is_Training:
            self.extras.update({
                "info":self.get_info()
                })
            
        return observations

    def _get_rewards(self) -> torch.Tensor:
        
        current_config = torch.cat(
            [self._robot.data.root_pos_w[:, :2], self._robot.data.heading_w.reshape(-1, 1)], dim=1
        ) 

        target_config = torch.cat(
            [self._desired_pos_w[:, :2], self.desired_orientation.reshape(-1, 1)], dim=1
        )

        reward_progress, goal_reward, time_reward = self.compute_reward(current_config, target_config)
        rew_maneuver = self.compute_reward_maneuver_net(current_config, target_config)
        self.reward_progress = reward_progress.clone()
        self.reward_success = goal_reward.clone()
        self.reward_time[:] = time_reward

        rewards = {  
            "2_goal_reached": 0*self.reward_success,
            "1_distance_progress":rew_maneuver,
            "6_time": self.reward_time,
        }
        #print(f"rewards: {rewards} rew_lift_drag_ratio: {lift_drag_ratio}\n")
        # #"6_time": time_reward
        self.previous_distance = self.distance.clone()
        self.previous_robot_pos = self._robot.data.root_pos_w.clone()
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        
        return reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        current_config = torch.cat(
            [self._robot.data.root_pos_w[:, :2], self._robot.data.heading_w.reshape(-1, 1)], dim=1
        ) 

        target_config = torch.cat(
            [self._desired_pos_w[:, :2], self.desired_orientation.reshape(-1, 1)], dim=1
        )
        pos_error = torch.norm(target_config[:, :2] - current_config[:, :2], dim=1)
        heading_diff = target_config[:, 2] - current_config[:, 2]
        heading_error = torch.atan2(torch.sin(heading_diff), torch.cos(heading_diff)).abs()

        done = pos_error <= self.cfg.goal_reached_threshold #torch.logical_or(pos_error <= self.cfg.goal_reached_threshold, heading_error<=self.cfg.bearing_reached_threshold).float()

        #done = torch.logical_or(done, goal_passed)
        if (torch.any(done) or torch.any(time_out)) and self.num_envs==1:
            print(f"done: {done} \tdistance: {self.distance} \ttime: {time_out}     \
                \tepisod_length: {self.episode_length_buf}: > {self.max_episode_length - 1}")
            
        #self.total_reward[self.send_goals==1] = 0
        return done, time_out
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = self.distance[env_ids].mean()
        final_bearing_to_goal = self.bearing[env_ids].mean()
        final_energy =  self.energy[env_ids].mean()
        consumed_energy = (self.episode_energy[env_ids]/(self.episode_length_buf[env_ids])).mean() # /self.max_available_episode_energy[env_ids]
        extras = dict()
        total_reward = 0
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            total_reward += episodic_sum_avg.item()
            extras["Episode_Reward/" + key] = episodic_sum_avg
            self._episode_sums[key][env_ids] = 0.0
    


        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/done"] = torch.count_nonzero(self.reset_terminated[env_ids]).item() / len(env_ids)
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item() / len(
            env_ids
        )
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        extras["Metrics/final_bearing_to_goal"] = final_bearing_to_goal.item()
        extras["Metrics/final_energy"] = final_energy.item()
        extras["Metrics/consumed_energy"] = consumed_energy.item()
        extras["Metrics/average_speed"] = (self.episode_avg_speed[env_ids]/(self.episode_length_buf[env_ids])).mean().item()
        self.extras["log"].update(extras)

        extras = dict()

        self.extras["log"].update(extras)

        
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if self.num_envs > 1 and len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new goal position
        self.initial_bearing[env_ids] = torch.zeros_like(self._desired_pos_w[env_ids, 0]).uniform_(
            self.cfg.min_target_bearing, self.cfg.max_target_bearing
        )
        self.bearing[env_ids] = self.initial_bearing[env_ids]
        self.previous_bearing = self.initial_bearing[env_ids]

        self.desired_orientation[env_ids] = torch.zeros_like(self.initial_bearing[env_ids]).uniform_(
            -torch.pi, torch.pi
        )

        self.initial_distance[env_ids] = torch.zeros_like(self._desired_pos_w[env_ids, 0]).uniform_(
            self.cfg.min_target_distance, self.cfg.max_target_distance
        )
        self.distance[env_ids] = self.initial_distance[env_ids]
        self.previous_distance[env_ids] = self.initial_distance[env_ids]
        self._desired_pos_w[env_ids, 0] = torch.cos(self.initial_bearing[env_ids]) * self.initial_distance[env_ids]
        self._desired_pos_w[env_ids, 1] = torch.sin(self.initial_bearing[env_ids]) * self.initial_distance[env_ids]
        self._desired_pos_w[env_ids, 2] = 0.0  # only in 2D
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]

        # Initialize available energy
        max_dist_per_step = self.cfg.max_robot_speed*self.step_dt
        energy_percent = 1 #torch.rand_like(self.initial_distance[env_ids]) * 0.3 + 0.5  # Randomize energy percent between 0.5 and 1.0
        self.max_available_episode_energy[env_ids] = energy_percent*(self.initial_distance[env_ids]/max_dist_per_step) * self.cfg.max_energy   
        
        self.episode_number[env_ids]  = self.episode_number[env_ids] + 1 
        self.episode_energy[env_ids] = 0
        self.episode_avg_speed[env_ids] = 0
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_link_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.previous_robot_pos[env_ids] = self._robot.data.root_link_pos_w[env_ids]
        self.initial_robot_pos[env_ids] = self._robot.data.root_link_pos_w[env_ids]

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the_robot_mass first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.1, 0.1, 0.5)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)

        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
        