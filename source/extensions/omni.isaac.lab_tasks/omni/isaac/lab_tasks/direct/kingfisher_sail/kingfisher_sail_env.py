# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuator_force.foil_actuator_force import FoilActuator, FoilActuatorCfg
from omni.isaac.lab.actuator_force.actuator_force import PropellerActuator, PropellerActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.physics.hydrodynamics import Hydrodynamics, HydrodynamicsCfg
from omni.isaac.lab.physics.hydrostatics import Hydrostatics, HydrostaticsCfg
from omni.isaac.lab.physics.foil_dynamics import FoilDynamics, FoilDynamicsCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms, transform_points, quat_from_euler_xyz
import numpy as np
from .network import DiscriminatorNetwork
from omni.isaac.lab_tasks.utils.my_utils.boat_config import (sail_config, rudder_config, keel_config, hydrostatics_config, hydrodynamics_config, propeller_config,
                     sail_actuator_config, rudder_actuator_config, keel_actuator_config)
##
# Pre-defined configs
##
from omni.isaac.lab_assets import (KINGFISHER_SAIL_CFG, RED_ARROW_X_MARKER_CFG, MAROON_ARROW_X_MARKER_CFG,
 BLUE_ARROW_X_MARKER_CFG, CUBOID_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, YELLOW_ARROW_X_MARKER_CFG, MIMOSA_ARROW_X_MARKER_CFG,
  RED_BIG_ARROW_X_MARKER_CFG, BEIGE_ARROW_X_MARKER_CFG, ORANGE_ARROW_X_MARKER_CFG, MAGENTA_ARROW_X_MARKER_CFG)  # isort: skip
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip


class KingfisherSailEnvWindow(BaseEnvWindow):
    """Window manager for the Kingfisher environment."""

    def __init__(self, env: KingfisherSailEnv, window_name: str = "IsaacLab"):
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
class KingfisherSailEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 150.0 #30
    physics_dt = 1 / 60.0  # 60 Hz
    decimation = 3
    step_dt = physics_dt * decimation  # 20 Hz
    action_space = 3
    observation_space = 16 #12
    state_space = 0
    debug_vis = True

    ui_window_class_type = KingfisherSailEnvWindow

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
    robot: ArticulationCfg = KINGFISHER_SAIL_CFG.replace(prim_path="/World/envs/env_.*/Robot")

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
    
    sail_aerodyn_cfg: FoilDynamicsCfg = sail_config()
    rudder_hydrodyn_cfg: FoilDynamicsCfg = rudder_config()
    keel_hydrodyn_cfg: FoilDynamicsCfg = keel_config()

    sail_actuator_cfg: FoilActuatorCfg = sail_actuator_config(pos_from_com=(0.0, 0., 0.0))
    rudder_actuator_cfg: FoilActuatorCfg = rudder_actuator_config(pos_from_com=(-1.5, 0., -0.1))
    keel_actuator_cfg: FoilActuatorCfg = keel_actuator_config()
    
    # Hydrostatics
    hydrostatics_cfg: HydrostaticsCfg = hydrostatics_config()
    # Hydrdynamics
    hydrodynamics_cfg: HydrodynamicsCfg = hydrodynamics_config()
    # Thruster dynamics
    propeller_cfg: PropellerActuatorCfg = propeller_config()

    # markers configurations
    # Blue markers used to visualize apparent wind
    blue_marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
    blue_marker_cfg.prim_path = "/Visuals/Command/appWind"

    maroon_marker_cfg = MAROON_ARROW_X_MARKER_CFG.copy()
    maroon_marker_cfg.prim_path = "/Visuals/Command/inducedWind"

    # red markers used to visualize true wind
    red_marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
    red_marker_cfg.prim_path = "/Visuals/Command/trueWind"

    # red markers used to visualize true wind
    red_big_marker_cfg = RED_BIG_ARROW_X_MARKER_CFG.copy()
    red_big_marker_cfg.prim_path = "/Visuals/Command/truebigWind"

    # Green markers used to visualize thruster forces
    green_marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
    green_marker_cfg.prim_path = "/Visuals/Command/force"

    # Yellow markers used to visualize thruster forces
    yellow_marker_cfg = YELLOW_ARROW_X_MARKER_CFG.copy()
    yellow_marker_cfg.prim_path = "/Visuals/Command/force_yellow"

    # Green markers used to visualize thruster forces
    beige_marker_cfg = BEIGE_ARROW_X_MARKER_CFG.copy()
    beige_marker_cfg.prim_path = "/Visuals/Command/netforce_beige"

    # Magenta markers used to visualize thruster forces
    magenta_marker_cfg = MAGENTA_ARROW_X_MARKER_CFG.copy()
    magenta_marker_cfg.prim_path = "/Visuals/Command/sail_unit_magenta"

    # Mimosa markers used to visualize thruster forces
    mimosa_marker_cfg = MIMOSA_ARROW_X_MARKER_CFG.copy()
    mimosa_marker_cfg.prim_path = "/Visuals/Command/sail_unit_ortho_magenta"

    # Orange markers used to visualize thruster forces
    orange_marker_cfg = ORANGE_ARROW_X_MARKER_CFG.copy()
    orange_marker_cfg.prim_path = "/Visuals/Command/hydro_force_orange"

    max_energy = 2.0  # Max of 1.0 per thruster
    max_available_energy = max_energy
    max_robot_speed = 1.5 # Max speed for energy optimization

    # reward scales
    distance_reward_scale = 0.0
    distance_progress_reward_scale = 3 #2 #0.5 #30 #5 # 6 too much
    bearing_progress_reward_scale = 0.0

    goal_reached_threshold = 0.1
    goal_reached_scale =  800.0 #

    energy_penalty_scale = -3 #0.5 #0.5 #0.08  #-0.001
    backwards_penalty_scale = -10.
    time_penalty_scale = -0.5 #-0.008 #
    penalty_inefficient_sailing_scale = -0.1
    tack_penalty_scale = -10
    bearing_penalty_scale = 0.5
    beargin_penalty_coef = -4 #0.5 #-4
    lift_drag_ratio_scale = 0.1
    acord_reward_scale = 0.5
    speed_penalty_scale = -0.1

    # Environment
    min_target_distance = 30.0 
    max_target_distance = 35.0
    min_target_bearing =  0*torch.pi/180 #-torch.pi / 2
    max_target_bearing = 0*torch.pi/180 #torch.pi / 2
    max_cross_track = 10.0

    downwind_flag = True
    upwind_flag = True
    beam_flag = True
    close_hauled_flag = True
    broad_reach_flag = True
    fixed_wind_flag = False
    wind_direction_randomization_flag = True
    wind_speed_randomization_flag = False

    

class KingfisherSailEnv(DirectRLEnv):
    cfg: KingfisherSailEnvCfg

    def __init__(self, cfg: KingfisherSailEnvCfg, render_mode: str | None = None, **kwargs):
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
        # Get specific body indices
        self._base_link = self._robot.find_bodies("base_link")[0]
        self._left_thruster_id = self._robot.find_bodies("thruster_left")[0]
        self._right_thruster_id = self._robot.find_bodies("thruster_right")[0]
        self._sail_wing_id = self._robot.find_bodies("sailwing")[0]
        self._wing_joint_dof_id = self._robot.find_joints("wing_joint")[0]

        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Forces
        self._hydrodynamic_force = torch.zeros(self.num_envs, 1, 6, device=self.device)
        self._hydrostatic_force = torch.zeros(self.num_envs, 1, 6, device=self.device)
        self._sail_aerodynamic_force_b = torch.zeros(self.num_envs, 1, 6, device=self.device)
        self._rudder_hydrodynamics_force_b = torch.zeros(self.num_envs, 1, 6, device=self.device)
        self._keel_hydrodynamics_force_b = torch.zeros(self.num_envs, 1, 6, device=self.device)
        self._aerodynamic_force_wing = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._thruster_forces = torch.zeros(self.num_envs, 1, 6, device=self.device)
        
        self._no_torque = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self._hydrostatics = Hydrostatics(num_envs=self.num_envs, device=self.device, cfg=self.cfg.hydrostatics_cfg)

        self._hydrodynamics = Hydrodynamics(num_envs=self.num_envs, device=self.device, cfg=self.cfg.hydrodynamics_cfg)

        self._sail_aerodynamics = FoilDynamics(num_envs=self.num_envs, device=self.device, cfg=self.cfg.sail_aerodyn_cfg)
        self._sail_aerodynamics.init_flow_vector(
            self._robot.data.root_lin_vel_b[:, 0:2], self._robot.data.heading_w
        )
        self._rudder_hydrodynamics = FoilDynamics(num_envs=self.num_envs, device=self.device, cfg=self.cfg.rudder_hydrodyn_cfg)
        self._rudder_hydrodynamics.init_flow_vector(
            self._robot.data.root_lin_vel_b[:, 0:2], self._robot.data.heading_w
        )
        self._keel_hydrodynamics = FoilDynamics(num_envs=self.num_envs, device=self.device, cfg=self.cfg.keel_hydrodyn_cfg)
        self._keel_hydrodynamics.init_flow_vector(
            self._robot.data.root_lin_vel_b[:, 0:2], self._robot.data.heading_w
        )
        
        self._thruster_dynamics = PropellerActuator(
            num_envs=self.num_envs, device=self.device, dt=cfg.step_dt, cfg=self.cfg.propeller_cfg
        )
        self._sail_actuator = FoilActuator(
            num_envs=self.num_envs, dt=cfg.step_dt, dynamics=self._sail_aerodynamics, cfg=self.cfg.sail_actuator_cfg
        )
        self._rudder_actuator = FoilActuator(
            num_envs=self.num_envs, dynamics=self._rudder_hydrodynamics, dt=cfg.step_dt, cfg=self.cfg.rudder_actuator_cfg
        )
        self._keel_actuator = FoilActuator(
            num_envs=self.num_envs, dynamics=self._keel_hydrodynamics, dt=cfg.step_dt, cfg=self.cfg.keel_actuator_cfg
        )

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

        self.energy = torch.zeros(self.num_envs, device=self.device)
        self.episode_energy = torch.zeros_like(self.energy)
        self.max_available_episode_energy = torch.zeros_like(self.energy)
        self.ratio_energy_usage = torch.zeros_like(self.energy)
        self.episode_avg_speed = torch.zeros_like(self.energy)
        self.episode_avg_lift_drag_ratio = torch.zeros_like(self.energy)
        self.desired_pos_b = torch.zeros(self.num_envs, 2, device=self.device)
        self.desired_trajectory_b = torch.zeros(self.num_envs, 10, 2) # 10 pts
        self.desired_speed_b = torch.zeros(self.num_envs, device=self.device)
        self.prev_desired_pos_b = torch.zeros_like(self.desired_pos_b)
        self.desired_bearing = torch.zeros(self.num_envs, device=self.device)

        self.is_upwind = torch.zeros(self.num_envs, device=self.device)
        self.is_downwind = torch.zeros(self.num_envs, device=self.device)

        self.joint_pos_target = torch.zeros(self.num_envs, device=self.device)
        self.sail_angle = torch.zeros(self.num_envs, device=self.device)
        self.joint_angle_mapped_pi = torch.zeros(self.num_envs, device=self.device)
        self.actual_angle_of_attack = torch.zeros(self.num_envs, device=self.device)

        self.relative_wind_angle = torch.zeros(self.num_envs, device=self.device)
        
        self.previous_robot_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.initial_robot_pos = torch.zeros((self.num_envs, 3), device=self.device)

        self.thruster_left_randn = torch.zeros(self.num_envs, device=self.device)
        self.thruster_right_randn = torch.zeros(self.num_envs, device=self.device)
        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        self.previous_sail_action = torch.zeros_like(self.actions[:, -1])

        self.episode_number = torch.ones(self.num_envs, device=self.device)
        self.global_step = 0

        self.is_Training = True

        

        self.energy_context = torch.ones_like(self.episode_number)
        n = 10
        self.evaluation_context_set = 0.1*torch.arange(1, n+1, device=self.device).expand(self.num_envs, n)
        self.current_test_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        
        self.reward_progress = torch.zeros(self.num_envs, device=self.device)
        self.reward_bearing = torch.zeros(self.num_envs, device=self.device)
        self.reward_energy = torch.zeros(self.num_envs, device=self.device)
        self.reward_backward = torch.zeros(self.num_envs, device=self.device)
        self.reward_acord = torch.zeros(self.num_envs, device=self.device)
        self.reward_aero = torch.zeros(self.num_envs, device=self.device)
        
        self.normalize_heading = torch.zeros(self.num_envs, device=self.device)
        self.normalized_energy = torch.zeros(self.num_envs, device=self.device)

        self.corridor_width = torch.ones(self.num_envs, device=self.device)

        
        self.tack_side = torch.ones(self.num_envs, device=self.device)
        self.in_tack_mode = torch.ones(self.num_envs, device=self.device)
        self.sailing_mode = torch.zeros((self.num_envs, 3), device=self.device)

        self.tack_length = torch.ones(self.num_envs, device=self.device)  # Default tack leg length
        self.num_tack_waypoints = 10  # Default number of waypoints for tacking
        self.tack_waypoints = torch.zeros((self.num_envs, self.num_tack_waypoints, 3), device=self.device)  # 10 waypoints
        self.tack_valid_mask = torch.zeros((self.num_envs, self.num_tack_waypoints), dtype=torch.bool, device=self.device)  # Valid mask for waypoints

        self.next_tack_wpt_idx = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)  # Index of the next waypoint to reach
        self.max_aero_force = torch.zeros(self.num_envs, device=self.device)  # Max aerodynamic force for the current episode

        self.can_tack = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.new_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.new_velocities = torch.zeros((self.num_envs, 6), device=self.device)
        # ============================================================================================#
        # ======================== Markers for the wind visualization ================================#
        # ============================================================================================#
        n_markers = 500
        self.max_width = 5
        d = 5
        self.env_pos = self._terrain.env_origins[0, :2].cpu().numpy()

        # Generate random translations for markers within a specified range in the world frame
        self.marker_translations = np.random.uniform([-self.env_pos[0]-d, -self.env_pos[1]-d, 0], 
                        [self.env_pos[0]+d, self.env_pos[1]+d, 2], (n_markers, 3))  # Adjust bounds as needed
        
        self.red_marker_translations = np.random.uniform([-self.env_pos[0]-d, -self.env_pos[1]-d, 0], 
                        [self.env_pos[0]+d, self.env_pos[1]+d, 2], (n_markers//10, 3))
        
        self.green_marker_translations = np.zeros((3, 3)) # 3 arrows for 3 bodies, both hulls and sail
        
        # ============================================================================================#
        # ============================================================================================#


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cone = RigidObject(self.cfg.cone_cfg)
        self.scene.rigid_objects["cone"] = self.cone
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

        self.blue_marker = VisualizationMarkers(self.cfg.blue_marker_cfg)
        self.maroon_marker = VisualizationMarkers(self.cfg.maroon_marker_cfg)
        self.green_marker = VisualizationMarkers(self.cfg.green_marker_cfg)
        self.red_marker = VisualizationMarkers(self.cfg.red_marker_cfg)
        self.red_big_marker = VisualizationMarkers(self.cfg.red_big_marker_cfg)
        self.yellow_marker = VisualizationMarkers(self.cfg.yellow_marker_cfg)
        self.beige_marker = VisualizationMarkers(self.cfg.beige_marker_cfg)
        self.orange_marker = VisualizationMarkers(self.cfg.orange_marker_cfg)
        self.magenta_marker = VisualizationMarkers(self.cfg.magenta_marker_cfg)
        self.mimosa_marker = VisualizationMarkers(self.cfg.mimosa_marker_cfg)
        self.visualize = False

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        heading = torch.atan2(torch.sin(self._robot.data.heading_w), torch.cos(torch.sin(self._robot.data.heading_w)))
        # 1. Calculate relative wind
        heading_relative_wind = heading - self._sail_aerodynamics.Beta_w + torch.pi
        normalized_heading_relative_wind = ((heading_relative_wind + torch.pi) % (2 * torch.pi) - torch.pi)
        print(f"[WIND DEBUG] head_rel: {torch.rad2deg(heading_relative_wind)} --> {torch.rad2deg(normalized_heading_relative_wind)}")
        
        self.relative_wind_angle = normalized_heading_relative_wind.clone()

        self.new_velocities = self._robot.data.root_com_vel_w.clone()
        
        self.new_velocities[:, 5] = 4*self._actions[:, 0] * ((torch.norm(self.new_velocities[:, :3], dim=1))**2)  # angular velocity around z imitating rudder effect (proportional to the speed of the boat)

            
        """if not self.is_Training:
            print(f"[DEBUG] actions: {self._actions} new_velocities: {self.new_velocities} self._robot.data.root_com_vel_w: {self._robot.data.root_com_vel_w} ")"""
        # 3. Thruster & Hydrodynamic Calculations (for standard execution)
        if self._actions.shape[1] > 2:
            self._thruster_dynamics.set_target_cmd(self._actions[:, :2])
            self._thruster_forces[:, 0, :] = self._thruster_dynamics.update_forces()

        """self._thruster_dynamics.set_target_cmd(torch.cat([self._actions[:, 0:1], self._actions[:, 0:1]], dim=-1))
        self._thruster_forces[:, 0, :] = self._thruster_dynamics.update_forces()
"""
        # Compute forces
        robot_pos = self._robot.data.root_pos_w.clone()
        robot_quat = self._robot.data.root_quat_w.clone()
        robot_vel = self._robot.data.root_vel_w.clone()

        self._hydrostatic_force[:, 0, :] = self._hydrostatics.compute_archimedes_metacentric_local(
            robot_pos, robot_quat
        )
        self._hydrodynamic_force[:, 0, :] = self._hydrodynamics.ComputeHydrodynamicsEffects(robot_quat, robot_vel)

        current_joint_pos = self._sail_actuator.dynamics.foil_angle.reshape(self.num_envs, -1)
        self._sail_actuator.update_joint_cmd(current_joint_pos, self._actions[:, -1:])
        self._sail_actuator.update_forces(self._robot.data.heading_w, self._robot.data.root_lin_vel_b)
        self._sail_aerodynamic_force_b[:, 0, :] = self._sail_actuator.get_forces_and_torques()

        current_rudder_pos = self._rudder_actuator.dynamics.foil_angle.reshape(self.num_envs, -1)
        self._rudder_actuator.update_joint_cmd(current_rudder_pos, self._actions[:, 0:1])
        self._rudder_actuator.update_forces(self._robot.data.heading_w, self._robot.data.root_lin_vel_b)
        self._rudder_hydrodynamics_force_b[:, 0, :] = self._rudder_actuator.get_forces_and_torques()

        """if not self.is_Training:
            print(f"[DEBUG] rudder force: {self._rudder_hydrodynamics_force_b} \nrudder_angle: {self._rudder_actuator.joint_position} \nrudder_foil_angle:{self._rudder_actuator.dynamics.foil_angle}" )
            print(f"[DEBUG] rudder force: {self._sail_aerodynamic_force_b} \nsail_angle: {self._sail_actuator.joint_position} \nsail_foil_angle: {self._sail_actuator.dynamics.foil_angle}" )
"""
        if not self.is_Training:
            print(f"[JOINT] joint position: {torch.rad2deg(self.joint_pos_target)} relative_wind: {torch.rad2deg(self.relative_wind_angle)}")
            print(f"aoa: {self._sail_actuator.dynamics.angle_of_attack.rad2deg()}, speed: {self._robot.data.root_link_lin_vel_b[:, 0]}")

    def _apply_action(self):
        # only apply thruster forces if they are not zero, otherwise it disables external previous forces.
        lft_thruster_force = self._thruster_forces[..., :3]
        rgt_thruster_force = self._thruster_forces[..., 3:] 

        combined = self._hydrostatic_force + self._hydrodynamic_force

        combined[:, 0, :3] = combined[:, 0, :3] + self._sail_aerodynamic_force_b[:, 0, :3] #+ \
        #self._rudder_hydrodynamics_force_b[:, 0, :3] #+ self._keel_hydrodynamics_force_b[:, 0, :3]

        combined[:, 0, 3:] = combined[:, 0, 3:] + self._sail_aerodynamic_force_b[:, 0, 3:] #+ \
        #self._rudder_hydrodynamics_force_b[:, 0, 3:] #+ self._keel_hydrodynamics_force_b[:, 0, 3:]

        self._robot.set_external_force_and_torque(combined[..., :3], combined[..., 3:], body_ids=self._base_link)
        
        if lft_thruster_force.any():
            self._robot.set_external_force_and_torque(
                lft_thruster_force, self._no_torque, body_ids=self._left_thruster_id
            )
        if rgt_thruster_force.any():
            self._robot.set_external_force_and_torque(
                rgt_thruster_force, self._no_torque, body_ids=self._right_thruster_id
            )
        
        self.joint_pos_target = self._sail_actuator.get_joint_positions()
        
        #self.joint_pos_target[:] = torch.pi/2
        # Set psoition of the sail joint
        self._robot.set_joint_position_target(target=self.joint_pos_target.reshape(self.num_envs,-1), 
             joint_ids=self._wing_joint_dof_id
            )

        # This snippet is for the tacking maneuver, we artificially change the boat's heading and 
        # reset its velocity to zero to simulate a tack.
        # 5. Reset velocities to zero (energy loss penalty)
        
        # 6. Apply directly to simulation
        env_ids = self.can_tack.nonzero(as_tuple=False).squeeze(-1)
        #self._robot.write_root_pose_to_sim(self.new_pose, env_ids=env_ids)
        #self._robot.write_root_velocity_to_sim(self.new_velocities)
        

    def get_info(self):
        
        info = {
            "lin_vel_b": self._robot.data.root_lin_vel_b,                  # (N, 3)
            "ang_vel_b": self._robot.data.root_ang_vel_b,                  # (N, 3)
            "robot_pos_w": self._robot.data.root_pos_w,                    # (N, 3)
            "lift_force_b": self._sail_aerodynamics.flow_lift_b,           # (N, 3)
            "drag_force_b": self._sail_aerodynamics.flow_drag_b,           # (N, 3)
            "heading_w": self._robot.data.heading_w,                       # (N,)
            "lift_coeff": self._sail_aerodynamics.lift_coeff,              # (N,)
            "drag_coeff": self._sail_aerodynamics.drag_coeff,              # (N,)                    # (N,)
            "energy": self.energy,                                         # (N,)
            "episode_energy": self.episode_energy,                         # (N,)
            "max_available_energy": self.max_available_episode_energy,     # (N,)
            "ratio_energy_usage": self.ratio_energy_usage,                 # (N,)
            "reward_progress": self.reward_progress,                       # (N,)
            "reward_acord": self.reward_acord,                             # (N,) 
            "reward_energy": self.reward_energy,                           # (N,)
            "reward_backward": self.reward_backward,                       # (N,)
            "reward_aero": self.reward_aero,                               # (N,)
            "bearing": self.bearing,                                       # (N,)
            "distance": self.distance,                                     # (N,)
            "aoa": self._sail_aerodynamics.angle_of_attack,                # (N,)
            "app_wind_angle": self._sail_aerodynamics.apparent_flow_angle, # (N,)
            "true_wind_angle_b": self._sail_aerodynamics.true_flow_angle,    # (N,)
            "sail_angle": self._sail_aerodynamics.foil_angle,              # (N,)
            "aero_force": self._sail_aerodynamic_force_b,                  # (N, 6)
            "max_aero_force": self.max_aero_force,      # (N,)
            "thruster_force": self._thruster_forces,                       # (N, 6)
            "relative_wind_angle": self.relative_wind_angle.unsqueeze(1),   # (N, 1)
            "goal_pos": self._desired_pos_w[:, :2],                        # (N, 2)
            "reward_bearing": self.reward_bearing,                                 # (N,)
            "cross_track_error": self.cross_track_error,                   # (N,)   
            "true_wind_angle_w": self._sail_aerodynamics.Beta_w, # (N,)
            "energy_context": self.energy_context,                         # (N,)

            
        }

        return info
        
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
        
        # compute the cross-track error
        self.update_cross_track_error()
        obs = torch.cat(
            [
                self._actions,  # 3
                self._robot.data.root_lin_vel_b[:, :2],  # 2
                self._robot.data.root_ang_vel_b[:, 2].unsqueeze(1),  # 1
                torch.cos(self.bearing).unsqueeze(1),  # 1
                torch.sin(self.bearing).unsqueeze(1),  # 1
                self.distance.unsqueeze(1)/self.cfg.max_target_distance,  # 1
                self._sail_aerodynamics.foil_angle.reshape(self.num_envs, -1)/torch.pi, # 1
                self._sail_aerodynamics.apparent_flow_angle.reshape(self.num_envs, -1)/torch.pi, # 1
                torch.norm(self._sail_aerodynamics.apparent_flow_speed_b, dim=-1).reshape(self.num_envs, -1), #1 
                (self._sail_aerodynamics.Beta_w - self.bearing + torch.pi).reshape(self.num_envs, -1)/torch.pi, # 1
                (self.cross_track_error.unsqueeze(1)/self.cfg.max_cross_track).abs(), # 1
                self.energy.reshape(self.num_envs, -1), # 1
                self.energy_context.reshape(self.num_envs, -1) # 1
                
            ],
            dim=1,
        )   

        """self.desired_pos_b[:, 0].abs().unsqueeze(1)/self.cfg.max_target_distance,  # 1
        self.desired_pos_b[:, 1].unsqueeze(1)/self.cfg.max_target_distance,  # 1

        torch.cos(self._sail_aerodynamics.foil_angle.reshape(self.num_envs, -1)), # 1
        torch.sin(self._sail_aerodynamics.foil_angle.reshape(self.num_envs, -1)), # 1
        torch.cos(self._sail_aerodynamics.apparent_flow_angle).reshape(self.num_envs, -1), # 1
        torch.sin(self._sail_aerodynamics.apparent_flow_angle).reshape(self.num_envs, -1), # 1"""

        n = (self.cfg.episode_length_s/self.step_dt)/2
        # 1. Get the actual integer indices of the environments that need resetting
        env_indices = torch.nonzero(self.episode_length_buf % (n) == 0).squeeze(-1)
        
        num_resets = env_indices.numel()

        # 2. Only run the logic if at least one environment triggers the condition
        if num_resets > 0:
            if self.is_Training:
                # Securely sample uniform noise matching the exact shape of the target slice
                # Assuming energy_context is 1D or 2D. torch.rand handles (0,1) uniformly.
                target_shape = self.energy_context[env_indices].shape
                self.energy_context[env_indices] = torch.rand(target_shape, device=self.energy_context.device)
                
            else:
                # Sample context for evaluation uniformly from the discrete set
                # Generates an index pointer for just the environments being reset
                
                idx = torch.randint(0, self.evaluation_context_set.shape[1], size=(num_resets,), device=self.energy_context.device)
                
                # Pull from the evaluation set using advanced indexing
                # Note: This assumes evaluation_context_set aligns with your active env_indices
                #self.energy_context[env_indices] = self.evaluation_context_set[env_indices, idx]
                self.energy_context[env_indices] = self.evaluation_context_set[env_indices, self.current_test_idx%(self.evaluation_context_set.shape[1])]
                self.current_test_idx[env_indices] += 1
                print(f"\n[DEBUG] Sampled new evaluation contexts: {self.energy_context[env_indices]} episode length: {self.episode_length_buf[env_indices]}\n")

        if not self.is_Training:
            self.extras.update({
                "info":self.get_info()
                })
            #print(f"[DEBUG] cross_track_error: {self.cross_track_error}")
    
        observations = {"policy": obs}

        return observations

    def update_cross_track_error(self):
        # Calculate the cross-track error based on the desired trajectory and current position
        # For simplicity, let's assume the desired trajectory is a straight line from initial position to desired position
        start_pos = self.initial_robot_pos[:, :2]
        end_pos = self._desired_pos_w[:, :2]
        current_pos = self._robot.data.root_link_pos_w[:, :2]

        # Vector from start to end
        trajectory_vector = end_pos - start_pos
        trajectory_length = torch.norm(trajectory_vector, dim=1, keepdim=True)
        trajectory_unit_vector = trajectory_vector / (trajectory_length + 1e-8)

        # Vector from start to current position
        current_vector = current_pos - start_pos

        # Project current vector onto trajectory unit vector to find the closest point on the trajectory
        projection_length = torch.sum(current_vector * trajectory_unit_vector, dim=1, keepdim=True)
        closest_point_on_trajectory = start_pos + projection_length * trajectory_unit_vector

        # Cross-track error is the distance from the current position to the closest point on the trajectory
        cross_track_error_vector = current_pos - closest_point_on_trajectory
        self.cross_track_error = torch.norm(cross_track_error_vector, dim=1)

    
    def _get_rewards(self) -> torch.Tensor:

        current_robot_pos = self._robot.data.root_link_pos_w.clone()

        grad_direction = self._desired_pos_w - current_robot_pos #self.initial_robot_pos
        grad_direction = grad_direction / torch.norm(grad_direction, dim=-1, keepdim=True)
        root_vel_w = self._robot.data.root_lin_vel_w.clone()

        velocity_made_good = torch.sum(root_vel_w*grad_direction, dim=-1) 
        # 1. Scalar Distance Progress (Velocity Made Good)
        # Measures direct distance reduction to goal between steps.
        # Correctly rewards 45-degree tacking maneuvers without projection artifacts.
        progress_toward_goal = self.previous_distance - self.distance 
        diff_desired_pos_b = self.prev_desired_pos_b - self.desired_pos_b
        #progress_toward_goal = (diff_desired_pos_b[:, 0]) + 0.4*(diff_desired_pos_b[:, 1])

        alpha = 0.2
        #distance_progress_reward = (alpha*progress_toward_goal + (1 - alpha)*velocity_made_good) * self.cfg.distance_progress_reward_scale #/(self.energy.clamp(min=1))
        distance_progress_reward = (-(progress_toward_goal<0).float() + velocity_made_good*(torch.exp(-2*self.energy))) #* self.cfg.distance_progress_reward_scale #/(self.energy.clamp(min=1))
        
        # 2. Goal Reached
        goal_reward = torch.zeros(self.num_envs, device=self.device)
        goal_mask = self.distance < self.cfg.goal_reached_threshold
        goal_reward[goal_mask] = self.cfg.goal_reached_scale

        # 3. Penalize Moving Backwards (STRICTLY NEGATIVE)
        root_lin_vel_b_x = self._robot.data.root_lin_vel_b[:, 0]
        backwards_penalty = torch.zeros(self.num_envs, device=self.device)
        # Ensure this is a NEGATIVE value!
        backwards_penalty[root_lin_vel_b_x < -0.2] = -abs(self.cfg.backwards_penalty_scale) #*torch.abs(root_lin_vel_b_x[root_lin_vel_b_x < 0.0])*self.step_dt

        # 4. Small Step / Time Penalty (STRICTLY NEGATIVE)
        time_penalty = -abs(self.cfg.time_penalty_scale) * torch.ones(self.num_envs, device=self.device) *self.step_dt 

        tack_penalty = torch.zeros_like(time_penalty)
        tack_mask = torch.abs(self.cross_track_error/self.cfg.max_cross_track) > 1.
        tack_penalty[tack_mask] = -abs(self.cfg.tack_penalty_scale)*self.step_dt #*torch.abs((self.cross_track_error[tack_mask]-self.cfg.max_cross_track)/self.cfg.max_cross_track)*self.step_dt

        energy_norm = self.energy / self.cfg.max_energy
        energy_reward = self.cfg.energy_penalty_scale * energy_norm

        max__lift_force = self._sail_actuator.dynamics.get_max_aero_force(self._sail_actuator.dynamics.Beta_w)
        aero_force_w = transform_points(self._sail_aerodynamic_force_b[:, :, :3], 
                        quat=self._robot.data.body_state_w[:, self._base_link, 3:7].reshape(self.num_envs, -1), pos=None)
                    
        aero_force_w = aero_force_w.squeeze(dim=1)
        #print(f"aero_force_w: {aero_force_w.shape}")
        # Force magnitude
        aero_force_norm = torch.norm(
            aero_force_w,
            dim=-1,
            keepdim=True
        )

        # Normalized force direction, safely
        aero_force_direction_w = (
            aero_force_w
            / aero_force_norm.clamp(min=1e-6)
        )

        # Alignment between aerodynamic force and goal direction
        aero_alignment = torch.sum(
            aero_force_direction_w[:, :2] * grad_direction[:, :2],
            dim=-1
        )

        # Optional: don't reward a zero-force state.
        # This makes the reward proportional to useful aerodynamic force.
        aero_force_strength = (
            aero_force_norm.squeeze(-1)
            / max__lift_force
        ).clamp(0.0, 1.0)

        aero_reward = (
            aero_alignment * aero_force_strength*0.5
        )      
        
        #aero_reward =  (aero_force/self._sail_actuator.dynamics.max_aero_force)*torch.cos(self.bearing) 
        
        # Combine active rewards
        rewards = {  
            "1_distance_progress": 0.1*distance_progress_reward,
            "2_goal_reached": 0.5*goal_reward,
            "3_energy": aero_reward,
            "4_backwards": backwards_penalty,
            "6_time": 0*time_penalty,
            "7_tack_penalty": tack_penalty
        }

        # Sum total reward
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Update history
        self.previous_distance = self.distance.clone()
        self.previous_robot_pos = current_robot_pos.clone()
        self.prev_desired_pos_b = self.desired_pos_b.clone()

        # Episode logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        if not self.is_Training:
            print(f"\n[REWARD DEBUG] progress toward goal: {progress_toward_goal} \nvmg: {velocity_made_good}")
            print(f"reward_aero: {aero_reward} \nReward progress: {distance_progress_reward} \ncontext:{self.energy_context}")
            print(f"max_lift_force: {self.max_aero_force}")
            #print(f"[TOTAL REWARD] reward: {reward} desired_pos_b: {self.desired_pos_b}")

            #print(f"wind: {self._sail_actuator.dynamics.Uw} \napp_wind: {self._sail_actuator.dynamics.apparent_flow_speed_b.norm()} \naero_norm: {aero_force_norm}")
            print(f"[AERO FORCE DEBUG] align: {aero_alignment} \ngrad_pos: {grad_direction} \nlift_w: {aero_force_w}")
            print()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Desired position in the robot frame (2D)
        """self.desired_pos_b_3d, _ = subtract_frame_transforms(
            self._robot.data.root_link_state_w[:, :3], self._robot.data.root_link_state_w[:, 3:7], self._desired_pos_w
        )
        self.desired_pos_b[:, :2] = self.desired_pos_b_3d[:, :2]"""
        self.distance = torch.linalg.norm(self._desired_pos_w[:, :2] - self._robot.data.root_link_pos_w[:, :2], dim=1)
        
        # Finish episode if the goal is reached
        done = torch.zeros_like(time_out)
        done[self.distance < self.cfg.goal_reached_threshold] = True
        #done = torch.logical_or(done, goal_passed)
        if (torch.any(done) or torch.any(time_out)) and self.num_envs==1:
            print(f"done: {done} \tdistance: {self.distance} \ttime: {time_out}     \
                \tepisod_length: {self.episode_length_buf}: > {self.max_episode_length - 1}")
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

        if not self.is_Training:
            self.initial_bearing[env_ids] = torch.zeros_like(self._desired_pos_w[env_ids, 0]).uniform_(
                0.0, 0.0
            )
        self.bearing[env_ids] = self.initial_bearing[env_ids]
        self.previous_bearing = self.initial_bearing[env_ids]

        mask1 = torch.any(self.episode_number[env_ids] < 50)
        mask2 = torch.any(torch.logical_and(self.episode_number[env_ids] >= 50 , self.episode_number[env_ids] < 150))
        mask3 = torch.any(self.episode_number[env_ids] >= 150)
        self._sail_actuator.reset(env_ids=env_ids)
        self._rudder_actuator.reset(env_ids=env_ids)
        """if self.is_Training:
            if mask1:
                self.cfg.min_target_distance = 1
                self.cfg.max_target_distance = 5
            elif mask2:
                self.cfg.min_target_distance = 5
                self.cfg.max_target_distance = 7
            elif mask3:
                self.cfg.min_target_distance = 7
                self.cfg.max_target_distance = 10"""

        if not self.is_Training:
            self.cfg.min_target_distance = self.cfg.min_target_distance
            self.cfg.max_target_distance = self.cfg.min_target_distance


        self.initial_distance[env_ids] = torch.zeros_like(self._desired_pos_w[env_ids, 0]).uniform_(
            self.cfg.min_target_distance, self.cfg.max_target_distance
        )
        self.next_tack_wpt_idx[env_ids] = 0
        self.distance[env_ids] = self.initial_distance[env_ids]
        self.previous_distance[env_ids] = self.initial_distance[env_ids]
        self._desired_pos_w[env_ids, 0] = torch.cos(self.initial_bearing[env_ids]) * self.initial_distance[env_ids]
        self._desired_pos_w[env_ids, 1] = torch.sin(self.initial_bearing[env_ids]) * self.initial_distance[env_ids]
        self._desired_pos_w[env_ids, 2] = 0.0  # only in 2D
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]

        if self.is_Training:
            self.energy_context[env_ids] = torch.zeros_like(self.energy_context[env_ids]).uniform_(0, 1)
        else:
            # sample context for evalutation uniformly from a set of discrete contexts to reduce noise in evaluation metrics
            idx = torch.randint(0, self.evaluation_context_set.shape[1], size=(len(env_ids), ))
            
            self.energy_context[env_ids] = self.evaluation_context_set[torch.arange(len(env_ids)), self.current_test_idx%(self.evaluation_context_set.shape[1])]
            print(f"\n[DEBUG] Sampled new evaluation contexts: {self.energy_context[env_ids]} episode length: {self.episode_length_buf[env_ids]}\n")
            #self.current_test_idx[env_ids] += 1


        # Initialize markers pos
        d = 5
        n_markers = 500
        self.marker_translations = np.random.uniform([-self.env_pos[0]-d, -self.env_pos[1]-d, 1], 
                    [self.env_pos[0]+d, self.env_pos[1]+d, 1.6], (n_markers, 3))  # Adjust bounds as needed
        self.red_marker_translations = np.random.uniform([-self.env_pos[0]-d/3, -self.env_pos[1]-d/3, 1], 
                    [self.env_pos[0]+d/2, self.env_pos[1]+d/2, 1.5], (n_markers//5, 3))
        
        self.max_aero_force = self._sail_aerodynamics.get_max_aero_force(self._sail_aerodynamics.Uw)

        self.energy_context[env_ids] = torch.zeros_like(self.energy_context[env_ids]).uniform_(0, 1)

        desired_pos_b_3d, _ = subtract_frame_transforms(
                    self._robot.data.root_link_state_w[env_ids, :3], self._robot.data.root_link_state_w[env_ids, 3:7], self._desired_pos_w[env_ids]
                )
        self.prev_desired_pos_b[env_ids, :2] = desired_pos_b_3d[:, :2]
        
        if self.is_Training:

            self.desired_speed_b[env_ids] = torch.zeros_like(self.desired_speed_b[env_ids]).uniform_(0.2, 1.5)
            
            random = torch.rand_like(self.episode_number[env_ids])

            #upwind = torch.logical_and(random > 0.2, torch.logical_and(random <= 0.6, self.episode_number[env_ids] > 200))
            #upwind = torch.logical_and(random > 0.2, random <= 0.8)
            #downwind = torch.logical_and(random > 0.6,  self.episode_number[env_ids] > 200)

            downwind = torch.any(torch.logical_and(self.episode_number[env_ids]>50, self.episode_number[env_ids]<99)) #random > 0.8
            beam = torch.any(torch.logical_and(self.episode_number[env_ids]>99, self.episode_number[env_ids]<199))
            broad = torch.any(torch.logical_and(self.episode_number[env_ids]>199, self.episode_number[env_ids]<249))
            close = torch.any(torch.logical_and(self.episode_number[env_ids]>249, self.episode_number[env_ids]<349))
            """upwind = torch.any(torch.logical_and(self.episode_number[env_ids]>349, self.episode_number[env_ids]<400))
            random = torch.any(self.episode_number[env_ids]>400)"""
            
            self._sail_aerodynamics.reset_flow_condition(env_ids=env_ids, randomize_direction=self.cfg.wind_direction_randomization_flag, randomize_speed=self.cfg.wind_speed_randomization_flag, 
                                                                    upflow=self.cfg.upwind_flag, downflow=self.cfg.downwind_flag, beam=self.cfg.beam_flag, close=self.cfg.close_hauled_flag, broad=self.cfg.broad_reach_flag,
                                                                    fixed=self.cfg.fixed_wind_flag)

            """def reset_wind_condition(
                self,
                env_ids=None,
                upwind=False,
                downwind=False,
                beam=False,
                close=False,
                broad=False,
                randomize_direction=False,
                randomize_speed=False,
            )"""
        
        self.tack_side[env_ids] = torch.ones_like(self.tack_side[env_ids])
        self.in_tack_mode[env_ids] = torch.zeros_like(self.in_tack_mode[env_ids])
        
        self.episode_number[env_ids]  = self.episode_number[env_ids] + 1 
        self.episode_energy[env_ids] = 0
        self.episode_avg_speed[env_ids] = 0
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        if not self.is_Training:
            new_yaw = torch.pi/2*torch.ones_like(self._robot.data.heading_w[env_ids])  # Mirrored heading
            new_quat = quat_from_euler_xyz(
                torch.zeros_like(new_yaw),
                torch.zeros_like(new_yaw),
                new_yaw
            )
            default_root_state[:, 3:7] = new_quat
        else:
            random_heading = torch.rand_like(self._robot.data.heading_w[env_ids]) * 2 * torch.pi
            new_quat = quat_from_euler_xyz(
                torch.zeros_like(random_heading),
                torch.zeros_like(random_heading),
                random_heading
            )
            default_root_state[:, 3:7] = new_quat

        self._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_link_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.previous_robot_pos[env_ids] = self._robot.data.root_link_pos_w[env_ids]
        self.initial_robot_pos[env_ids] = self._robot.data.root_link_pos_w[env_ids]
        
        self.tack_length[env_ids] = self.initial_distance[env_ids] // 3

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
        
        if self.num_envs==1:
            # apparent wind visualization
            app_wind_world = transform_points(torch.cat((self._sail_aerodynamics.apparent_flow_speed_b, 
            torch.zeros((self.num_envs, 1), device=self.device)), dim=-1), 
            quat=self._robot.data.body_state_w[:, self._base_link, 3:7].reshape(self.num_envs, -1), 
            pos=None)[:, :2]
            #print(f"app: {self._sail_aerodynamics.apparent_flow_speed_b}")
            app_wind_speed_world = torch.norm(app_wind_world, dim=-1) # magnitude of app_wind
            app_wind_angle_world = torch.atan2(app_wind_world[:, 1], app_wind_world[:, 0])


            true_wind_world = transform_points(torch.cat((self._sail_aerodynamics.true_flow_speed2D_b, 
            torch.zeros((self.num_envs, 1), device=self.device)), dim=-1), 
            quat=self._robot.data.body_state_w[:, self._base_link, 3:7].reshape(self.num_envs, -1), 
            pos=None)[:, :2]
            true_wind_speed_world = torch.norm(true_wind_world, dim=-1) # magnitude of app_wind
            true_wind_angle_world = torch.atan2(true_wind_world[:, 1], true_wind_world[:, 0])

            induced_wind_world = transform_points(-self._robot.data.root_lin_vel_b, 
                quat=self._robot.data.body_state_w[:, self._base_link, 3:7].reshape(self.num_envs, -1), 
            pos=None)[:, :2]
            induced_wind_angle_world = torch.atan2(induced_wind_world[:, 1], induced_wind_world[:, 0])
            induced_wind_speed_world = torch.norm(induced_wind_world, dim=-1) 

            # sail unit vectors
            sail_unit_vector_world = transform_points(torch.cat((self._sail_aerodynamics.foil_unit_vector, 
            torch.zeros((self.num_envs, 1), device=self.device)), dim=-1), 
            quat=self._robot.data.body_state_w[:, self._base_link, 3:7].reshape(self.num_envs, -1), 
            pos=None)[:, :2]

            # sail ortho unit vector
            sail_ortho_unit_vector_world = transform_points(torch.cat((self._sail_aerodynamics.foil_ortho_unit_vector, 
            torch.zeros((self.num_envs, 1), device=self.device)), dim=-1), 
            quat=self._robot.data.body_state_w[:, self._base_link, 3:7].reshape(self.num_envs, -1), 
            pos=None)[:, :2]

            vec2d = torch.cat((sail_unit_vector_world, sail_ortho_unit_vector_world), dim=0)
            #print(app_wind_world.shape, true_wind_world.shape, vec2d.shape)

            self.vector_field1_visualization(vis_enabled=True, 
                                             vectors_2d=sail_unit_vector_world,
                                             body_ids=[3],
                                             offset=[0.0, 0.0, 0.35], marker_obj=self.magenta_marker, max_arrow_length=10)
            
            self.vector_field1_visualization(vis_enabled=True, 
                                             vectors_2d=sail_ortho_unit_vector_world,
                                             body_ids=[3],
                                             offset=[0.0, 0.0, 0.35], marker_obj=self.mimosa_marker)
            
            """print(f"Vector: app_transf: {app_wind_world} \ttrue: {true_wind_world} \tinduced:{induced_wind_world}")
            print(f"Norm: app_transf: {app_wind_speed_world} \ttrue: {true_wind_speed_world} \tinduced:{induced_wind_speed_world}")
            print(f"Angle: app_transf: {(180/torch.pi)*app_wind_angle_world} \ttrue: {(180/torch.pi)*true_wind_angle_world} \
                  \tinduced:{(180/torch.pi)*induced_wind_angle_world}\n")"""
            #wind_drag_world = transform_points(self._sail_aerodynamics.wind_drag[:, :3], quat=self._robot.data.root_link_state_w[:, 3:7], pos=None)
            self.vector_field_visualization(
                                vis_enabled=True,
                                directions=app_wind_angle_world,
                                magnitudes=app_wind_speed_world,
                                marker_obj=self.blue_marker,
                                marker_translations=self.marker_translations,
                                wrap_distance=2.5,
                                update_scale=0.1,
                                body_ids=[3],
                                offset=[0.0, 0.0, 0.2]
                            )
            
            self.vector_field_visualization(
                                vis_enabled=True,
                                directions=true_wind_world,
                                magnitudes=None,
                                marker_obj=self.red_big_marker,
                                marker_translations=self.marker_translations,
                                wrap_distance=2.5,
                                update_scale=0.1,
                                body_ids=[3],
                                offset=[0., 0.0, 0.2]
                            )
            
            # true wind visualization
            """self.vector_field_visualization(
                                vis_enabled=True,
                                directions=true_wind_world,
                                magnitudes=None,
                                marker_obj=self.red_marker,
                                marker_translations=self.red_marker_translations,
                                wrap_distance=2.0,
                                update_scale=0.05,
                            )"""
            
            self.vector_field_visualization(
                                vis_enabled=True,
                                directions=induced_wind_angle_world,
                                magnitudes=induced_wind_speed_world,
                                marker_obj=self.maroon_marker,
                                marker_translations=self.marker_translations,
                                wrap_distance=2.5,
                                update_scale=0.1,
                                body_ids=[3],
                                offset=[0.0, 0.0, 0.2]
                            )
            #print(f"\napp_speed: {app_wind_world} app_angle: {app_wind_angle_world}")
            #print(f"true_speed: {self._sail_aerodynamics.Uw*torch.cos(self._sail_aerodynamics.Beta_w), self._sail_aerodynamics.Uw*torch.sin(self._sail_aerodynamics.Beta_w)} :{true_wind_world} true_angle: {true_wind_angle_world}\n")

            
            thrust_left = self._thruster_forces[..., :3].clone().detach().reshape(self.num_envs, -1)
            thrust_right = self._thruster_forces[..., 3:].clone().detach().reshape(self.num_envs, -1)
            aerodynamic_force_b = self._sail_aerodynamic_force_b[:, :3].clone().detach().reshape(self.num_envs, -1)
            hydrodynamic_force = self._hydrodynamic_force.clone().detach().reshape(self.num_envs, -1)[:, :3]
            hydrodynamic_force_lateral = torch.zeros_like(hydrodynamic_force)
            hydrodynamic_force_forward = torch.zeros_like(hydrodynamic_force)

            hydrodynamic_force_lateral[:, 1] = hydrodynamic_force[:, 1]
            hydrodynamic_force_forward[:, 0] = hydrodynamic_force[:, 0]
            """thrust_left[:, :1] = self._actions[:, :1]
            thrust_right[:, :1] = self._actions[:, 1:2]"""
            #print(f"{thrust_left.shape} -> {thrust_right.shape} -> {self._sail_aerodynamic_force_b.squeeze(0)[:, :2].shape}")
            #array_forces_2D = torch.cat((thrust_left, thrust_right, self._sail_aerodynamics.wind_lift[:, :2], self._sail_aerodynamics.wind_drag[:, :2]), dim=0)
            #self.force_visualization(True, thrust_left.squeeze(0)[:, :2], body_ids=[1], marker_ob=self.green_marker)
            #self.force_visualization(True, thrust_right.squeeze(0)[:, :2], body_ids=[2], marker_ob=self.green_marker)
            wind_lift_world = transform_points(self._sail_aerodynamics.flow_lift_b[:, :3], 
                quat=self._robot.data.body_state_w[:, self._base_link, 3:7].reshape(self.num_envs, -1), pos=None)
            
            wind_drag_world = transform_points(self._sail_aerodynamics.flow_drag_b[:, :3], 
                quat=self._robot.data.body_state_w[:, self._base_link, 3:7].reshape(self.num_envs, -1), pos=None)

            aerodynamic_force_world = transform_points(aerodynamic_force_b[:, :3], 
                quat=self._robot.data.body_state_w[:, self._base_link, 3:7].reshape(self.num_envs, -1), pos=None)
            
            """wind_lift_world = transform_points(self._sail_aerodynamics.wind_lift_wing[:, :3], 
                quat=self._robot.data.body_state_w[:, self._sail_wing_id, 3:7].reshape(self.num_envs, -1), pos=None)
            
            wind_drag_world = transform_points(self._sail_aerodynamics.wind_drag_wing[:, :3], 
                quat=self._robot.data.body_state_w[:, self._sail_wing_id, 3:7].reshape(self.num_envs, -1), pos=None)
            
            aerodynamic_force_world = transform_points(aerodynamic_force_wing, 
                quat=self._robot.data.body_state_w[:, self._sail_wing_id, 3:7].reshape(self.num_envs, -1), pos=None)"""
            
            #print(self._robot.data.body_state_w[:, self._sail_wing_id, 3:7].shape)
            #print(f"aero_body: {aerodynamic_force_b.shape} \naero_world: {aerodynamic_force_world.shape}\n aero_wing: {self._aerodynamic_force_wing.shape}\n")
            hydrodynamic_force_world = transform_points(hydrodynamic_force, quat=self._robot.data.root_link_state_w[:, 3:7], pos=None)
            hydrodynamic_force_forward_world = transform_points(hydrodynamic_force_forward, 
                quat=self._robot.data.root_link_state_w[:, 3:7], pos=None)
            
            hydrodynamic_force_lateral_world = transform_points(hydrodynamic_force_lateral, 
                quat=self._robot.data.root_link_state_w[:, 3:7], pos=None)
            
            thrust_left_force_world = transform_points(thrust_left, 
                quat=self._robot.data.body_state_w[:, self._left_thruster_id, 3:7].reshape(self.num_envs, -1), pos=None)
            
            thrust_right_force_world = transform_points(thrust_right, 
                quat=self._robot.data.body_state_w[:, self._right_thruster_id, 3:7].reshape(self.num_envs, -1), pos=None)

            #hydrodynamic_force_world[:, 1] = 0
            heading_vec = torch.zeros_like(self._robot.data.root_link_lin_vel_b[:, :2])
            heading_vec[:, 0] = torch.cos(self._robot.data.heading_w)
            heading_vec[:, 1] = torch.sin(self._robot.data.heading_w)
            unit_heading_vec = torch.zeros_like(heading_vec)

            unit_heading_vec[:, 0] = heading_vec[:, 0]/torch.norm(heading_vec, dim=-1)
            unit_heading_vec[:, 1] = heading_vec[:, 1]/torch.norm(heading_vec, dim=-1)
            
            """self.force_visualization(True, wind_drag_world[:, :2], body_ids=[3], marker_ob=self.green_marker, offset=[0., 0., 0.15])
            self.force_visualization(True, wind_lift_world[:, :2], body_ids=[3], marker_ob=self.yellow_marker, offset=[0., 0., 0.15])
            self.force_visualization(True, aerodynamic_force_world[:, :2], body_ids=[3], marker_ob=self.beige_marker, offset=[0., 0., 0.15])"""

            self.vector_field1_visualization(vis_enabled=True, 
                                             vectors_2d=wind_drag_world[:, :2],
                                             body_ids=[3],
                                             offset=[0.0, 0.0, 0.15], marker_obj=self.green_marker, max_arrow_length=8)
            
            self.vector_field1_visualization(vis_enabled=True, 
                                             vectors_2d=wind_lift_world[:, :2],
                                             body_ids=[3],
                                             offset=[0.0, 0.0, 0.15], marker_obj=self.yellow_marker, max_arrow_length=8)
            
            self.vector_field1_visualization(vis_enabled=True, 
                                             vectors_2d=aerodynamic_force_world[:, :2],
                                             body_ids=[3],
                                             offset=[0.0, 0.0, 0.15], marker_obj=self.beige_marker, max_arrow_length=8)
            
            #self.force_visualization(True, hydrodynamic_force_world[:, :2], body_ids=[0], marker_ob=self.orange_marker, offset=[0., 0., 0.15])

            array_forces_2D = torch.cat((hydrodynamic_force_lateral_world[:, :2], hydrodynamic_force_forward_world[:, :2], 
                                         thrust_left_force_world[:, :2], thrust_right_force_world[:, :2]), dim=0)
            
            """self.force_visualization(True, thrust_left_force_world[:, :2], body_ids=[1], marker_ob=self.orange_marker, offset=[0., 0., 0.])
            self.force_visualization(True, thrust_right_force_world[:, :2], body_ids=[2], marker_ob=self.orange_marker, offset=[0., 0., 0.])"""
            self.force_visualization(True, array_forces_2D, body_ids=[0, 0, 1, 2], marker_ob=self.orange_marker, offset=[0., 0., 0.])
            """# force thruster visualization
            self.vector_field_visualization(
                                vis_enabled=True,
                                directions=self._actions,
                                magnitudes=self._actions,
                                marker_obj=self.red_marker,
                                marker_translations=self.red_marker_translations,
                                wrap_distance=2.0,
                                update_scale=0.05,
                                body_ids=[1, 2]
                            )
            
            # force wind visualization
            self.vector_field_visualization(
                                vis_enabled=True,
                                directions=self._sail_aerodynamic_force_b.squeeze(0),
                                magnitudes=self._sail_aerodynamic_force_b,
                                marker_obj=self.red_marker,
                                marker_translations=self.red_marker_translations,
                                wrap_distance=2.0,
                                update_scale=0.05,
                                body_ids=[3]
                            )"""

    def vector_field1_visualization(
        self,
        vis_enabled: bool,
        vectors_2d: torch.Tensor,
        body_ids: list[int],
        marker_obj,
        offset: list[float] | None = None,
        max_arrow_length: float = 5.0
    ):
        """
        Visualize multiple 2D vectors (forces, winds, etc.) at specified body positions.

        Args:
            vis_enabled: Whether to enable visualization.
            vectors_2d: (N, 2) tensor of 2D vectors to display.
            body_ids: List of body IDs corresponding to each vector.
            marker_obj: The marker object for visualization (e.g., self.blue_marker).
            offset: Optional [x, y, z] offset applied to all markers.
            max_arrow_length: Maximum visual arrow length.
        """
        if not vis_enabled or self.num_envs != 1:
            return

        n = len(body_ids)
        assert vectors_2d.shape[0] == n, f"vectors_2d and body_ids must have matching lengths: they have {vectors_2d.shape[0]} and {n}"

        # Compute rotation (yaw) angles for each vector
        alpha = torch.atan2(vectors_2d[:, 1], vectors_2d[:, 0]).cpu().numpy()
        
        # Build quaternion rotations around Z-axis (yaw)
        marker_orientations = np.zeros((n, 4))
        marker_orientations[:, 0] = np.cos(alpha / 2)  # w
        marker_orientations[:, 3] = np.sin(alpha / 2)  # z

        # Get body positions
        body_pos = self._robot.data.body_link_pos_w[:, body_ids, :].squeeze(0)
        marker_translations = body_pos.cpu().numpy()

        # Apply optional offset
        if offset is not None:
            marker_translations[:, 0] += offset[0]
            marker_translations[:, 1] += offset[1]
            marker_translations[:, 2] += offset[2]

        # Compute scaling based on vector magnitudes
        magnitudes = torch.norm(vectors_2d, dim=-1).cpu().numpy()
        marker_scales = np.ones_like(marker_translations)  # Default to [1, 1, 1] scaling

        #print(f"alpha: {alpha*(180/torch.pi)} \tmagn: {magnitudes}")

        max_magnitude = self._sail_aerodynamics.cfg.flow_speed + 2  # Or any configurable maximum
        scale_factors = (magnitudes / max_magnitude) * max_arrow_length
        marker_scales[:, 0] = scale_factors  # Scale along X only

        # Visualize
        marker_obj.visualize(translations=marker_translations, orientations=marker_orientations, scales=marker_scales)


    def vector_field_visualization(
        self,
        vis_enabled: bool,
        directions: torch.Tensor,
        magnitudes: torch.Tensor | None,
        marker_obj,
        marker_translations: np.ndarray,
        wrap_distance: float=4.,
        update_scale: float=0.1,
        z_clip: tuple[float, float] = (0.0, 4.0),
        body_ids: list[int] | None = None,
        offset:list[float] | None = None,
        max_arrow_length=5
    ):
        """
        Generic function for visualizing vector fields (wind, forces, etc.)

        Args:
            vis_enabled: Whether to enable the visualization.
            directions: A (N,) tensor of angles (radians) or 2D force vectors (N x 2).
            magnitudes: Optional (N,) tensor of speeds or force magnitudes.
            marker_obj: The marker object (e.g., self.blue_marker).
            marker_translations: N x 3 numpy array of marker positions (updated in-place).
            wrap_distance: Distance threshold to wrap markers around.
            update_scale: How much to move markers per timestep.
            z_clip: Min and max bounds for Z values.
            body_ids: Optional list of body IDs (used for static markers like forces).
            offset: Optional list of offset over the three axes [x, y, z]
        """
        if not vis_enabled or self.num_envs != 1:
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
            marker_translations[:, :2] += (speeds * self.step_dt * update_scale)

            current_pos = self._robot.data.root_pos_w[:, :2].clone().cpu().numpy()
            dist = np.abs(marker_translations[:, :2] - current_pos)

            for i in range(2):  # x and y
                marker_translations[:, i] = np.where(
                    dist[:, i] > wrap_distance,
                    current_pos[:, i] - wrap_distance / 2,
                    marker_translations[:, i]
                )

            marker_translations[:, 2] = np.clip(marker_translations[:, 2], *z_clip)
        else:
            body_pos = self._robot.data.body_link_pos_w[:, body_ids, :].squeeze(0)
            marker_translations[:] = body_pos.cpu().numpy()
            #print(f"{marker_translations.shape}")
            if offset is not None:
                marker_translations[:, 0] += offset[0]
                marker_translations[:, 1] += offset[1]
                marker_translations[:, 2] += offset[2]
                
            marker_scales = np.zeros_like(body_pos.cpu().numpy())
            scale = np.array([1, 1, 1])
            max_speeds = self._sail_aerodynamics.cfg.flow_speed + 2 # assuming 2m/s is boat max speed
            scale[0] = scale[0]*(1/max_speeds)*speeds*max_arrow_length
            # np.maximum(np.ones_like(speeds), (1/max_speeds)*speeds*max_arrow_length)
            marker_scales[:, :] = scale

        #print(f"{marker_scales.shape}:{body_ids}")
       
        # Visualize
        marker_obj.visualize(translations=marker_translations, orientations=marker_orientations, scales=marker_scales)

    def force_visualization(self, force_vis:bool, array_forces_2D: torch.Tensor, body_ids:list, marker_ob, 
                            offset:list[float] | None = None, max_arrow_length=8):
        """
            This function visualize the forces arrow applied on the bodies based on their direction
            
            Args: 
            force_vis: True means visualize the arrow, Force means not visualize them in the simulation
            array_forces_2D: concatenate of the 2D vector of the forces to be applied on the bodies
            body_ids: Ids of the bodies which will the forces be applied on.
            Please sort the body_ids in the order as the array_forces_2D is concatenated
            ex: [force_right_thruster, force_left_thruster, force-sail] ===> [right_hull_id, left_hull_id, sail_id]
            
        """

        if force_vis and self.num_envs==1:
        
            n = len(body_ids)
            alpha = torch.atan2(array_forces_2D[:, 1], array_forces_2D[:, 0]).cpu().numpy()

            # Compute quaternion for rotation along the wind direction (in the world frame)
            marker_orientations = np.zeros((n, 4))
            for i in range(len(alpha)):
                marker_orientations[i, :] = np.array([np.cos(alpha[i] / 2), 0, 0, np.sin(alpha[i] / 2)]) # # Rotate around Z-axis (yaw) 

            # Apply the same rotation to all markers
            body_pos = self._robot.data.body_link_pos_w[:, body_ids, :].squeeze(0)
            marker_translations = body_pos.cpu().numpy()
            if offset is not None:
                marker_translations[:, 0] += offset[0]
                marker_translations[:, 1] += offset[1]
                marker_translations[:, 2] += offset[2]
            #print(f"trans: {marker_translations} shape: {marker_translations.shape}")
            magnitude = torch.norm(array_forces_2D, dim=-1).cpu().numpy()
            #print(f"magnitude: {magnitude}")
            marker_scales = np.zeros_like(body_pos.cpu().numpy())
            scale = np.tile(np.array([1, 1, 1]), (len(body_ids), 1))
            max_force = 8
            for i in range(len(body_ids)):
                scale[i][0] = scale[i][0]*(1/max_force)*magnitude[i]*max_arrow_length
            marker_scales[:, :] = scale

            # Visualize markers in the world frame
            marker_ob.visualize(translations=marker_translations, orientations=marker_orientations, scales=marker_scales)


