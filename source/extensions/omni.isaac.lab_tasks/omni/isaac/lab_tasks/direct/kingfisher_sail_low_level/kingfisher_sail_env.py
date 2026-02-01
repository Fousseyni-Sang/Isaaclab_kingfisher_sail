# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuator_force.foil_actuator_force import FoilActuator, FoilActuatorCfg
from omni.isaac.lab.actuator_force.generic_foil_actuator_force import GenFoilActuator, GenFoilActuatorCfg
from omni.isaac.lab.actuator_force.actuator_force import PropellerActuator, PropellerActuatorCfg
from omni.isaac.lab.actuator_force.generic_actuator_force import GenPropellerActuatorCfg, GenPropellerActuator
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.physics.hydrodynamics import Hydrodynamics, HydrodynamicsCfg
from omni.isaac.lab.physics.hydrostatics import Hydrostatics, HydrostaticsCfg
from omni.isaac.lab.physics.foil_dynamics import FoilDynamics, FoilDynamicsCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms, transform_points, quat_from_euler_xyz
import numpy as np
from omni.isaac.lab.actuator_force.robot_actuator_system import RobotActuatorSystem
from ...utils.boat_config import (sail_config, rudder_config, keel_config, hydrostatics_config, hydrodynamics_config, 
                                propeller_config, sail_actuator_config, rudder_actuator_config, keel_actuator_config, 
                                kingfisher_propeller_config, jellyfish_propeller_config, vap2_propeller_config, 
                                vap4_propeller_config)

from .utils import sample_feasible_pairs_batch

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
    episode_length_s = 100 #9
    physics_dt = 1 / 60.0  # 60 Hz
    decimation = 3
    step_dt = physics_dt * decimation  # 20 Hz
    action_space = 2 #3
    base_observation_space = 4 #12
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

    sail_aerodyn_cfg: FoilDynamicsCfg = sail_config()
    sail_aerodyn_cfg.flow_speed = 8
    rudder_hydrodyn_cfg: FoilDynamicsCfg = rudder_config()
    keel_hydrodyn_cfg: FoilDynamicsCfg = keel_config()

    sail_actuator_cfg: FoilActuatorCfg = sail_actuator_config()
    rudder_actuator_cfg: FoilActuatorCfg = rudder_actuator_config()
    keel_actuator_cfg: FoilActuatorCfg = keel_actuator_config()

    generic_actuator_cfg: GenPropellerActuatorCfg = vap2_propeller_config()
    generic_foil_actuator_cfg: GenFoilActuatorCfg = GenFoilActuatorCfg(
        foils=[
            rudder_actuator_cfg, 
            keel_actuator_cfg,
        ]
    )
    
    action_space = sum([thr_cfg.num_dim for thr_cfg in generic_actuator_cfg.thrusters])
    observation_space = base_observation_space + action_space
    
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
    distance_progress_reward_scale = 1.3
    bearing_progress_reward_scale = 0.0

    goal_reached_threshold = 0.1
    goal_reached_scale =  800.0 #

    energy_penalty_scale = -1 
    backwards_penalty_scale = -1
    time_penalty_scale = -1 #-0.008 #

    bearing_penalty_scale = 0.5
    beargin_penalty_coef = -4 #0.5 #-4
    lift_drag_ratio_scale = 0.1
    acord_reward_scale = 0.5
    speed_penalty_scale = -0.1

    # Environment
    max_target_lin_wrench =  1.5
    min_target_lin_wrench =  0.*max_target_lin_wrench
    max_target_ang_wrench =  .5
    min_target_ang_wrench = - max_target_ang_wrench 
    
    


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
                "0_goal_reach",
                "1_distance_progress",
                "3_energy",
                "4_backwards",
                "5_lift_drag_ratio", 
                "6_time",   
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
        self.generic_thruster_forces = torch.zeros((self.num_envs, self.cfg.generic_actuator_cfg.num_thrusters, 3), 
                                                     device=self.device)
        self.generic_thruster_torques = torch.zeros_like(self.generic_thruster_forces)

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

        self._generic_thruster_dynamics = GenPropellerActuator(
            num_envs=self.num_envs, device=self.device, dt=cfg.step_dt, cfg=cfg.generic_actuator_cfg
            )

        # Buffers
        self.distance = torch.zeros(self.num_envs, device=self.device)
        self.previous_distance = torch.zeros(self.num_envs, device=self.device)
        self.distance_progress = torch.zeros(self.num_envs, device=self.device)
        self.initial_distance = torch.zeros(self.num_envs, device=self.device)
        
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

        self.desired_wrench_b = torch.zeros(self.num_envs, 6, device=self.device)
        self.current_wrench_b = torch.zeros(self.num_envs, 6, device=self.device)
        self.desired_bearing = torch.zeros(self.num_envs, device=self.device)

        self.is_upwind = torch.zeros(self.num_envs, device=self.device)
        self.is_downwind = torch.zeros(self.num_envs, device=self.device)
        self.root_accel_b = torch.zeros((self.num_envs, 3), device=self.device)
        self.joint_pos_target = torch.zeros(self.num_envs, device=self.device)
        self.sail_angle = torch.zeros(self.num_envs, device=self.device)
        self.joint_angle_mapped_pi = torch.zeros(self.num_envs, device=self.device)
        self.actual_angle_of_attack = torch.zeros(self.num_envs, device=self.device)
        
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
        self.interval_between_goals = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self.reward_progress = torch.zeros(self.num_envs, device=self.device)
        self.reward_bearing = torch.zeros(self.num_envs, device=self.device)
        self.reward_energy = torch.zeros(self.num_envs, device=self.device)
        self.reward_backward = torch.zeros(self.num_envs, device=self.device)
        self.reward_acord = torch.zeros(self.num_envs, device=self.device)
        self.reward_aero = torch.zeros(self.num_envs, device=self.device)
        
        self.reward_goal = torch.zeros(self.num_envs, device=self.device)

        self.normalize_heading = torch.zeros(self.num_envs, device=self.device)
        self.normalized_energy = torch.zeros(self.num_envs, device=self.device)

        self.next_tack_wpt_idx = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)  # Index of the next waypoint to reach
        self.max_aero_force = torch.zeros(self.num_envs, device=self.device)  # Max aerodynamic force for the current episode

        self.norm_error_lin = torch.zeros(self.num_envs, device=self.device)
        self.norm_error_ang = torch.zeros(self.num_envs, device=self.device)
        
        self.lin_target_wrench = torch.zeros((self.num_envs, 1), device=self.device)
        self.ang_target_wrench = torch.zeros((self.num_envs, 1), device=self.device)

        self.prev_lin_vel_b = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_ang_vel_b = torch.zeros((self.num_envs, 3), device=self.device)

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

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

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
        if not torch.isfinite(actions).all():
            print("🔥 NaN/Inf in LL obs", actions)
            raise RuntimeError("Invalid HL observation")
        self._actions = actions.clone().clamp(-1.0, 1.0)
        #self._actions[:, 2] = 0
        # Override the actions for debugging
        # self._actions[:,0] = 0.6
        # self._actions[:,1] = 0.6
        if not torch.isfinite(self._actions).all():
            print("🔥 NaN/Inf in LL obs", self._actions)
            raise RuntimeError("Invalid HL observation")
        # Compute the thruster forces based on the actions.
        # thrust_cmds = torch.tensor([0.0, 1.0], dtype=torch.float32, device=self.device)
        
        self._thruster_dynamics.set_target_cmd(self._actions[:, :2])
        self._thruster_forces[:, 0, :] = self._thruster_dynamics.update_forces()
        
        # === THRUSTERS === 
        thr_cmd_dim = self._generic_thruster_dynamics.total_cmd_dim 
        thruster_actions = self._actions[:, :thr_cmd_dim] 
        
        self._generic_thruster_dynamics.set_target_cmd(thruster_actions) 
        thruster_forces = self._generic_thruster_dynamics.update_forces() # (num_envs, num_thrusters, 3) 
        thruster_torques = self._generic_thruster_dynamics.thruster_torques # (num_envs, num_thrusters, 3) 
        
        # Store for _apply_action 
        self.generic_thruster_forces = thruster_forces 
        self.generic_thruster_torques = thruster_torques

        # Compute the hydrostatic and hydrodynamic forces
        robot_pos = self._robot.data.root_pos_w.clone()
        robot_quat = self._robot.data.root_quat_w.clone()
        robot_vel = self._robot.data.root_vel_w.clone()
        self._hydrostatic_force[:, 0, :] = self._hydrostatics.compute_archimedes_metacentric_local(
            robot_pos, robot_quat
        )
        self._hydrodynamic_force[:, 0, :] = self._hydrodynamics.ComputeHydrodynamicsEffects(robot_quat, robot_vel)

        current_joint_pos = self._sail_actuator.dynamics.foil_angle.reshape(self.num_envs, -1)
        self._sail_actuator.update_joint_cmd(current_joint_pos, self._actions[:, 1:2]) #2:3
        self._sail_actuator.update_forces(self._robot.data.heading_w, self._robot.data.root_lin_vel_b)
        self._sail_aerodynamic_force_b[:, 0, :] = self._sail_actuator.get_forces_and_torques()
        #print(f"SAIL FORCE: {self._sail_aerodynamic_force_b[:, 0, :]}")

        current_rudder_angle = self._rudder_actuator.dynamics.foil_angle.reshape(self.num_envs, -1)
        self._rudder_actuator.update_joint_cmd(current_rudder_angle, self._actions[:, 0:1])
        self._rudder_actuator.update_forces(self._robot.data.heading_w, self._robot.data.root_lin_vel_b)
        self._rudder_hydrodynamics_force_b[:, 0, :] = self._rudder_actuator.get_forces_and_torques()

        #current_keel_angle = self._keel_actuator.dynamics.foil_angle.reshape(self.num_envs, -1)
        #self._keel_actuator.update_joint_cmd(current_keel_angle, self._actions[:, 1:2])
        self._keel_actuator.update_forces(self._robot.data.heading_w, self._robot.data.root_lin_vel_b)
        self._keel_hydrodynamics_force_b[:, 0, :] = self._keel_actuator.get_forces_and_torques()

        #print(f"\nkeel: {self._keel_hydrodynamics_force_b} \nrudder: {self._rudder_hydrodynamics_force_b}")
        #=====================================================================================================#
        
    
    def _apply_action(self):
        # only apply thruster forces if they are not zero, otherwise it disables external previous forces.
        lft_thruster_force = self._thruster_forces[..., :3]
        rgt_thruster_force = self._thruster_forces[..., 3:] 

        #self._hydrodynamic_force[:, 0, :] = self._keel_hydrodynamics_force_b[:, 0, :]
        #print(self._hydrodynamic_force.shape, self._keel_hydrodynamics_force_b.shape)
        combined = self._hydrostatic_force + self._hydrodynamic_force

        combined[:, 0, :3] = combined[:, 0, :3] #+ self._sail_aerodynamic_force_b[:, 0, :3] + \
        #self._rudder_hydrodynamics_force_b[:, 0, :3] + self._keel_hydrodynamics_force_b[:, 0, :3]

        combined[:, 0, 3:] = combined[:, 0, 3:] #+ self._sail_aerodynamic_force_b[:, 0, 3:] + \
        #self._rudder_hydrodynamics_force_b[:, 0, 3:] + self._keel_hydrodynamics_force_b[:, 0, 3:]

        combined_thruster_forces = self.generic_thruster_forces.sum(dim=1)
        combined_thruster_torques = self.generic_thruster_torques.sum(dim=1)

        """print("idx: ", idx_gene==idx_orig)
        print("interp: ", inter_gene[0]==inter_orig)
        print("cmd: ", curr_cmd_orig==curr_cmd_gene)
        print("mag: ", mag_gene==mag_orig)
        print(f"gene_thru_force: {self.generic_thruster_forces}")
        print(f"gene_thru_force2: {self._generic_thruster_dynamics.thruster_forces}")
"""
        combined[:, 0, :3] += combined_thruster_forces
        combined[:, 0, 3:] += combined_thruster_torques

        #print(f"combined_force: {combined_thruster_forces}")
        self._robot.set_external_force_and_torque(combined[..., :3], combined[..., 3:], body_ids=self._base_link)
        
        """if self.generic_thruster_forces[:, 0, :].any():
            self._robot.set_external_force_and_torque(
                self.generic_thruster_forces[:, 0, :], self._no_torque, body_ids=self._left_thruster_id
            )
        if self.generic_thruster_forces[:, 1, :].any():
            self._robot.set_external_force_and_torque(
                self.generic_thruster_forces[:, 1, :], self._no_torque, body_ids=self._right_thruster_id
            )"""
        self.joint_pos_target = self._sail_actuator.get_joint_positions()
        # Set psoition of the sail joint
        """self._robot.set_joint_position_target(target=self.joint_pos_target.reshape(self.num_envs,-1), 
             joint_ids=self._wing_joint_dof_id
            )"""
    def get_added_obs_couples(self, hydrodynamics:FoilDynamics, obs:torch.Tensor):
        """ This function computes observation related to the foil which hydro-dynamics is given and 
        concatenate that to the observation given as argument """

        foil_obs = torch.cat([
            obs, 
            torch.cos(hydrodynamics.foil_angle.reshape(self.num_envs, -1)), # 1
            torch.sin(hydrodynamics.foil_angle.reshape(self.num_envs, -1)), # 1
            torch.cos(hydrodynamics.apparent_flow_angle).reshape(self.num_envs, -1), # 1
            torch.sin(hydrodynamics.apparent_flow_angle).reshape(self.num_envs, -1), # 1
            torch.norm(hydrodynamics.apparent_flow_speed_b, dim=-1).reshape(self.num_envs, -1), #1
        ], dim=-1)

        if not torch.isfinite(foil_obs).all():
            print("🔥 NaN/Inf in <<get_added_obs_couples>>: ", foil_obs)
            raise RuntimeError("<<get_added_obs_couples>>: Invalid HL observation")
        
        return foil_obs.clone()
    
    def _get_observations(self) -> dict:

        # Observations
        self.global_step += 1
        # Desired position in the robot frame (2D)
        self.desired_pos_b_3d, _ = subtract_frame_transforms(
            self._robot.data.root_link_state_w[:, :3], self._robot.data.root_link_state_w[:, 3:7], self._desired_pos_w
        )
        self.desired_pos_b[:, :2] = self.desired_pos_b_3d[:, :2]
        #self.distance = torch.linalg.norm(self.desired_pos_b, dim=1)
        self.bearing = torch.atan2(self.desired_pos_b[:, 1], self.desired_pos_b[:, 0])

        self.energy = torch.sum(torch.square(self._actions[:, :2]), dim=1)
        self.normalized_energy = self.energy / self.cfg.max_energy

        self.ratio_energy_usage = self.episode_energy / self.max_available_episode_energy
        
        self.energy = torch.sum(torch.square(self._actions[:, :2]), dim=1)
        error_lin = self.desired_wrench_b[:, 0]-self._robot.data.root_lin_vel_b[:, 0]
        self.norm_error_lin = torch.abs(error_lin)

        error_ang = self.desired_wrench_b[:, 5]-self._robot.data.root_ang_vel_b[:, 2]
        self.norm_error_ang = torch.abs(error_ang)

        delta_vel_lin_b = (self._robot.data.root_lin_vel_b-self.prev_lin_vel_b)
        delta_ang_vel_b = (self._robot.data.root_ang_vel_b-self.prev_ang_vel_b)

        obs = torch.cat(
            [
                self._actions,  # 3
                error_lin.reshape(self.num_envs, -1), # 1
                delta_vel_lin_b[:, 0].reshape(self.num_envs, -1), # 1
                error_ang.reshape(self.num_envs, -1), # 1
                delta_ang_vel_b[:, 2].reshape(self.num_envs, -1), # 1
                #((self.episode_length_buf%self.interval_between_goals)/self.interval_between_goals).reshape(self.num_envs, -1)
                
            ],
            dim=1,
        )
        
        if not torch.isfinite(delta_ang_vel_b[:, 2]).all():
            print("🔥 NaN/Inf in <<_get_observations>>: ", delta_ang_vel_b[:, 2])
            raise RuntimeError("<<_get_observations>>: Invalid HL observation")
        #print("before")
        #obs = self.get_added_obs_couples(self._rudder_hydrodynamics, obs)
        #print("after")
        #obs = self.get_added_obs_couples(self._keel_hydrodynamics, obs)
        
        observations = {"policy": obs}

        #print(f"\nprev: {self.prev_lin_vel_b} \ncurrent: {self._robot.data.root_lin_vel_b}  ")
        
        self.prev_lin_vel_b = self._robot.data.root_lin_vel_b.clone()
        self.prev_ang_vel_b = self._robot.data.root_ang_vel_b.clone()

        return observations

    def _get_rewards(self) -> torch.Tensor:
        
        self.current_wrench_b[:, 0] = self._robot.data.root_lin_vel_b[:, 0].clone()
        self.current_wrench_b[:, 5] = self._robot.data.root_ang_vel_b[:, 2].clone()
        distance_to_target = self.desired_wrench_b - self.current_wrench_b
        self.distance = distance_to_target[:, 0:1].clone()
        # normalize with FIXED scales (never by target!)
        distance_norm = torch.zeros_like(distance_to_target)
        #distance_norm[:,0] = distance_to_target[:,0] / self.cfg.max_target_lin_wrench
        #distance_norm[:,5] = distance_to_target[:,5] / self.cfg.max_target_ang_wrench
        goal_mask = torch.logical_and(distance_to_target[:, 0].abs()<=0.01, distance_to_target[:, 5].abs()<=0.01)
        #goal_mask = torch.logical_or(distance_to_target[:, 0].abs()<=0.1, distance_to_target[:, 5]>=1e6)
        goal_reach = 5*goal_mask
        s = 1.0
        self.reward_goal = goal_reach.clone()

        r1 = 0.5*(torch.exp(-5*torch.norm(distance_to_target[:, 0:1], dim=-1)/s)-1) + 0.5*(torch.exp(-5*torch.norm(distance_to_target[:, 5:], dim=-1)/s)-1)
        self.reward_progress = r1.clone()

        dot_wrench = torch.sum(self.desired_wrench_b*self.current_wrench_b, dim=1)
        r2 = 2*dot_wrench/ (torch.norm(self.desired_wrench_b, dim=1)*torch.norm(self.current_wrench_b, dim=1)+1e-6)
        r2[r2>0] = 0
        #r3 = torch.atan2(r2[:, 2])
        # Penalize going backwards
        backwards_penalty = torch.zeros(self.num_envs, device=self.device)
        root_lin_vel_b_x = self._robot.data.root_lin_vel_b[:, 0].clone()
        backwards_penalty[root_lin_vel_b_x < 0.0] = self.cfg.backwards_penalty_scale
        
        self.reward_backward = r2 #backwards_penalty.clone()
        self.episode_avg_speed += torch.norm(self._robot.data.root_lin_vel_b, dim=-1)
        
        aerodynamic_force_b = self._sail_aerodynamics.flow_lift_b.clone().reshape_as(self._sail_aerodynamic_force_b[:, :, :3])
        
        reward_forward = self._sail_aerodynamic_force_b[:, 0, 0]/self.max_aero_force

        energy_norm = self.energy / self.cfg.max_energy
        #energy_reward = torch.zeros_like(energy_norm)
        energy_coeff_aeroforce = torch.clip(reward_forward, min=0.2, max=1)**(-1)
        energy_reward = 0.3*energy_coeff_aeroforce*self.cfg.energy_penalty_scale*energy_norm#* self.energy_context
        self.reward_energy = 0.5*energy_reward.clone()

        lift_drag_ratio = 0.1*torch.max(torch.zeros_like(self._sail_aerodynamics.lift_coeff), self._sail_aerodynamics.lift_coeff/self._sail_aerodynamics.drag_coeff)
        lift_ratio = torch.abs(self._sail_aerodynamics.lift_coeff/self._sail_aerodynamics.max_cl)
        ratio_reward = self.cfg.lift_drag_ratio_scale*self.step_dt*(lift_ratio)
        
        #reward_speed = self.cfg.speed_penalty_scale*(torch.square(root_lin_vel_b_x - self.desired_speed_b))*self.step_dt
        #self.reward_energy = reward_speed.clone()

        
        lift_ratio = torch.abs(self._sail_aerodynamics.lift_coeff/self._sail_aerodynamics.max_cl)
        drag_coeff = self._sail_aerodynamics.drag_coeff
        
        reward_aero =   lift_ratio #0.5*lift_ratio
        
        self.reward_aero = reward_aero.clone()
        time_reward = self.cfg.time_penalty_scale*(~goal_mask)*self.step_dt
        rewards = {  
            "0_goal_reach": goal_reach,
            "1_distance_progress": self.reward_progress,
            "3_energy": 0*self.reward_energy,
            "4_backwards":  0*self.reward_backward,
            "5_lift_drag_ratio": 0*lift_ratio,
            "6_time": 0*time_reward,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        update_wrench_env_ids = (self.episode_length_buf%self.interval_between_goals == 0)
        self.reset_wrench_targets(env_ids=torch.nonzero(update_wrench_env_ids).squeeze(-1))
        
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        self.current_wrench_b[:, 0] = self._robot.data.root_lin_vel_b[:, 0].clone()
        self.current_wrench_b[:, 5] = self._robot.data.root_ang_vel_b[:, 2].clone()
        distance_to_target = torch.abs(self.desired_wrench_b - self.current_wrench_b)
        #goal_reach = torch.logical_and(distance_to_target[:, 0]<=0.1, distance_to_target[:, 5]<=0.1)
        #goal_reach = torch.logical_or(distance_to_target[:, 0]<=0.1, distance_to_target[:, 5]>=1e6)
        # Finish episode if the goal is reached
        done = time_out
        
        return done, time_out

    def reset_wrench_targets(self, env_ids: torch.Tensor | None = None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        params = {"V_min": self.cfg.min_target_lin_wrench, "V_max": self.cfg.max_target_lin_wrench, 
                  "omega_max": self.cfg.max_target_ang_wrench, "R_min":1,}
        
        target_wrenchs = sample_feasible_pairs_batch(env_ids=env_ids, device=self.device, params=params)

        if target_wrenchs is not None:
            self.desired_wrench_b[env_ids, 0] = target_wrenchs[:, 0]
            self.desired_wrench_b[env_ids, 5] = target_wrenchs[:, 1]

        """self.desired_wrench_b[env_ids, 0] = torch.zeros_like(self.desired_wrench_b[env_ids, 0]).uniform_(
            self.cfg.min_target_lin_wrench, self.cfg.max_target_lin_wrench
        )
        self.desired_wrench_b[env_ids, 5] = torch.zeros_like(self.desired_wrench_b[env_ids, 5]).uniform_(
            self.cfg.min_target_ang_wrench, self.cfg.max_target_ang_wrench
        )"""

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
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
        extras["Metrics/consumed_energy"] = consumed_energy.item()
        extras["Metrics/average_speed"] = (self.episode_avg_speed[env_ids]/(self.episode_length_buf[env_ids])).mean().item()
        self.extras["log"].update(extras)

        extras = dict()
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if self.num_envs > 1 and len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self.interval_between_goals[env_ids] = 300 #108 #torch.randint_like(input=self.interval_between_goals[env_ids], low=20, high=51)
        self._sail_actuator.reset(env_ids=env_ids)
        self._rudder_actuator.reset(env_ids=env_ids)
        self._keel_actuator.reset(env_ids=env_ids)
        
        self.reset_wrench_targets(env_ids=env_ids)
        
        # Initialize available energy
        max_dist_per_step = self.cfg.max_robot_speed*self.step_dt
        energy_percent = 1 #torch.rand_like(self.initial_distance[env_ids]) * 0.3 + 0.5  # Randomize energy percent between 0.5 and 1.0
        self.max_available_episode_energy[env_ids] = energy_percent*(self.initial_distance[env_ids]/max_dist_per_step) * self.cfg.max_energy
        
        # Initialize markers pos
        d = 5
        n_markers = 500
        self.marker_translations = np.random.uniform([-self.env_pos[0]-d, -self.env_pos[1]-d, 1], 
                    [self.env_pos[0]+d, self.env_pos[1]+d, 1.6], (n_markers, 3))  # Adjust bounds as needed
        self.red_marker_translations = np.random.uniform([-self.env_pos[0]-d/3, -self.env_pos[1]-d/3, 1], 
                    [self.env_pos[0]+d/2, self.env_pos[1]+d/2, 1.5], (n_markers//5, 3))
        
        self.max_aero_force = self._sail_aerodynamics.get_max_aero_force(self._sail_aerodynamics.Uw)

        if self.is_Training:
            
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
            
            self._sail_aerodynamics.reset_flow_condition(env_ids=env_ids, randomize_direction=False, randomize_speed=False, 
                                                    upflow=False, downflow=True, beam=True, close=False, broad=True,
                                                    fixed=True)

            self.thruster_left_randn[env_ids] = torch.zeros_like(self.thruster_left_randn[env_ids]).uniform_(0, 1)
            self.thruster_right_randn[env_ids] = torch.zeros_like(self.thruster_right_randn[env_ids]).uniform_(0, 1)

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
        
        if self.num_envs==1:
            # ATTENTION TO THE TRANSFORM_POINTS function if num_envs>1. SHAPE MIGHT BE INCONSISTENT
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


