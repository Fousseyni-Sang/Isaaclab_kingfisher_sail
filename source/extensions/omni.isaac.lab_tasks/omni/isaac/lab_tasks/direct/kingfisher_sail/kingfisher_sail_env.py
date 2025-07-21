# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuator_force.actuator_force import PropellerActuator, PropellerActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.physics.hydrodynamics import Hydrodynamics, HydrodynamicsCfg
from omni.isaac.lab.physics.hydrostatics import Hydrostatics, HydrostaticsCfg
from omni.isaac.lab.physics.aerodynamics import Aerodynamics, AerodynamicsCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms, transform_points, quat_from_euler_xyz
import numpy as np
from .network import DiscriminatorNetwork
from .buffer import ReplayBuffer

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

def cluster_context(context: torch.Tensor, device, n: int=5) -> torch.Tensor:
    # Create n intervals between 0 and 1 (exclusive)
    boundaries = torch.linspace(0, 1, n+1, device=device)
    
    # Use bucketize to find which interval each context value belongs to
    indices = torch.bucketize(context, boundaries) - 1  # Subtract 1 to match 0-based index
    
    # Map the indices to the boundary values
    context = boundaries[indices]
    
    return context

import torch

class TackManager:
    def __init__(self, min_upwind_angle: float, max_downwind_angle: float, max_cross_track: float):
        self.min_upwind_angle = min_upwind_angle
        self.max_downwind_angle = max_downwind_angle
        self.max_cross_track = max_cross_track

        self.tack_mode = None  # 0 = no tack, 1/2 = upwind left/right, 3/4 = downwind left/right
        self.desired_bearing = None

    def reset(self, num_envs: int, device=None):
        self.tack_mode = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.desired_bearing = torch.zeros(num_envs, device=device)

    def get_desired_bearing(self, bearing: torch.Tensor, wind_direction: torch.Tensor, cross_track: torch.Tensor) -> torch.Tensor:
        ks = bearing - wind_direction + torch.pi
        ks = torch.atan2(torch.sin(ks), torch.cos(ks))

        is_upwind = torch.abs(ks) < self.min_upwind_angle
        is_downwind = torch.abs(ks) > self.max_downwind_angle
        needs_switch = torch.abs(cross_track) > self.max_cross_track

        # Compute possible target angles
        desired_upwind_angle = torch.where(
            torch.logical_and(cross_track > 0, needs_switch),
            -self.min_upwind_angle + wind_direction + torch.pi,
             self.min_upwind_angle + wind_direction + torch.pi,
        )
        desired_downwind_angle = torch.where(
            torch.logical_and(cross_track > 0, needs_switch),
             -self.max_downwind_angle + wind_direction + torch.pi,
            self.max_downwind_angle + wind_direction + torch.pi,
        )

        # --- Start vectorized logic ---

        # Enter tack if in up/downwind and not in tack mode
        entering_upwind = is_upwind & (self.tack_mode == 0)
        entering_downwind = is_downwind & (self.tack_mode == 0)

        self.tack_mode = torch.where(
            entering_upwind,
            torch.where(cross_track > 0, torch.tensor(1, device=bearing.device), torch.tensor(2, device=bearing.device)),
            self.tack_mode
        )

        self.tack_mode = torch.where(
            entering_downwind,
            torch.where(cross_track > 0, torch.tensor(3, device=bearing.device), torch.tensor(4, device=bearing.device)),
            self.tack_mode
        )

        # Set desired bearing when entering tack
        self.desired_bearing = torch.where(
            entering_upwind, desired_upwind_angle, self.desired_bearing
        )
        self.desired_bearing = torch.where(
            entering_downwind, desired_downwind_angle, self.desired_bearing
        )

        
        # If already tacking and need to switch sides
        switching_upwind = (self.tack_mode == 1) & needs_switch
        switching_downwind = (self.tack_mode == 3) & needs_switch

        self.tack_mode = torch.where(switching_upwind, torch.tensor(2, device=bearing.device), self.tack_mode)
        self.tack_mode = torch.where(switching_downwind, torch.tensor(4, device=bearing.device), self.tack_mode)

        self.desired_bearing = torch.where(switching_upwind, desired_upwind_angle, self.desired_bearing)
        self.desired_bearing = torch.where(switching_downwind, desired_downwind_angle, self.desired_bearing)

        #print(f"tack_mode: {self.tack_mode}, desired_bearing: {self.desired_bearing*(180/torch.pi)}\n")
        # Exit tack mode if not upwind or downwind anymore
        exit_tack = ~(is_upwind | is_downwind)
        self.tack_mode = torch.where(exit_tack, torch.tensor(0, device=bearing.device), self.tack_mode)
        self.desired_bearing = torch.where(exit_tack, bearing, self.desired_bearing)

        return torch.atan2(torch.sin(self.desired_bearing), torch.cos(self.desired_bearing))  # wrap to [-π, π]


def generate_tacking_waypoints(start_pos:torch.Tensor, goal_pos:torch.Tensor, wind_direction:torch.Tensor,
                                     min_upwind_angle:float, tack_leg_length:torch.Tensor, max_num_waypoints=10):
    
    device = start_pos.device
    num_envs = start_pos.shape[0]

    waypoints = torch.zeros((num_envs, max_num_waypoints, 2), device=device)
    valid_mask = torch.zeros((num_envs, max_num_waypoints), dtype=torch.bool, device=device)
    waypoints[:, 0, :] = start_pos
    current_pos = start_pos.clone()
    tack_side = torch.ones(num_envs, device=device)
    
    finished = torch.zeros(num_envs, dtype=torch.bool, device=device)

    for step in range(1, max_num_waypoints-1):
        tack_angle = wind_direction + tack_side * min_upwind_angle + torch.pi
        tack_angle = torch.atan2(torch.sin(tack_angle), torch.cos(tack_angle))
        #print(f"tack_angle: {(180/torch.pi)*tack_angle}; wind_angle: {(180/torch.pi)*wind_direction}; min_angle: {(180/torch.pi)*min_upwind_angle}")
        
        tack_dir = torch.stack((torch.cos(tack_angle), torch.sin(tack_angle)), dim=1)
        
        next_pos = current_pos + tack_leg_length.reshape(num_envs, -1) * tack_dir

        waypoints[:, step, :] = next_pos
        valid_mask[:, step] = ~finished
        #print(f"step: {step} \tcurrent_pos: {current_pos} \tnext_pos: {next_pos} \tgoal_pos: {goal_pos}")
        to_goal_vec = goal_pos - current_pos
        to_next_vec = next_pos - current_pos

        goal_dir = to_goal_vec / (to_goal_vec.norm(dim=-1, keepdim=True) + 1e-6)
        tack_dir = to_next_vec / (to_next_vec.norm(dim=-1, keepdim=True) + 1e-6)

        goal_proj = torch.sum(goal_dir * tack_dir, dim=1)
        #print(f"goal_dir: {goal_dir} \ttack_dir: {tack_dir} \tgoal_proj: {goal_proj}")
        sail_away = goal_proj < 0.1
        #print(f"cos: {goal_proj} \tsailaway: {sail_away}")
        near_goal = torch.norm(goal_pos - next_pos, dim=1) < tack_leg_length
        #print(f"sail_away: {sail_away} \tnear_goal: {near_goal}")
        done_now = (~finished) & (sail_away | near_goal)
        next_pos[done_now] = goal_pos[done_now]
        finished |= done_now
        #print(f"done_now: {done_now} \tfinished: {finished}")
        current_pos = next_pos
        tack_side = -tack_side

        if finished.all():
            waypoints[:, step+1, :] = goal_pos
            break

    return waypoints, valid_mask

def get_desired_bearing_wpts(bearing: torch.Tensor, next_wpt_idx:torch.Tensor, tack_waypts:torch.Tensor ,robot:Articulation, sail_mode:torch.Tensor):
    """ Calculate the desired bearing based on the next waypoint and the robot's orientation."""

    next_wpt = tack_waypts[torch.arange(tack_waypts.shape[0]), next_wpt_idx]
    desired_pos_b_3d, _ = subtract_frame_transforms(
            robot.data.root_link_state_w[:, :3], robot.data.root_link_state_w[:, 3:7], next_wpt
        )
    desired_pos_b = torch.zeros_like(desired_pos_b_3d)
    desired_pos_b[:, :2] = desired_pos_b_3d[:, :2]
    
    upwind_mask = sail_mode[:, 0] == 1
    bearing[upwind_mask] = torch.atan2(desired_pos_b[:, 1], desired_pos_b[:, 0])[upwind_mask]

    distance = torch.linalg.norm(desired_pos_b, dim=1)
    bearing = torch.atan2(desired_pos_b[:, 1], desired_pos_b[:, 0])

    previous_wpt = tack_waypts[torch.arange(tack_waypts.shape[0]), torch.clamp(next_wpt_idx-1, min=0)]

    wpt_passed = torch.sum((next_wpt[:, :2] - previous_wpt[:, :2])*(robot.data.root_link_pos_w[:, :2] \
    - previous_wpt[:, :2]), dim=-1)>torch.square(torch.norm(next_wpt[:, :2] - next_wpt[:, :2], dim=-1))

    wpt_reached = distance < 0.5

    next_wpt_idx = torch.where(wpt_passed | wpt_reached, torch.clamp(next_wpt_idx+1, max=tack_waypts.shape[1]-1), next_wpt_idx)
    

    return bearing, next_wpt_idx



def get_desired_bearing(bearing: torch.Tensor, wind_direction: torch.Tensor, min_upwind_angle:float, 
                        max_downwind_angle:float, cross_track:torch.Tensor, max_cross_track:float, sail_mode:torch.Tensor, 
                        in_tack_mode:torch.Tensor, tack_side:torch.Tensor):
    # Calculate the desired heading based on the bearing and wind direction

    is_upwind = sail_mode[:, 0]==1
    is_downwind = sail_mode[:, 1]==1
    is_nominal = sail_mode[:, 2]==1

    # put tack_side to -1 if cross>0 and we're not already in tack_mode (to avoid flips every time) otherwise 
    # 1 (if not in tack_mode still, this avoids conflict with the condiion of cross>max_cross_track)
    tack_side = tack_side
    tack_side = torch.where(torch.logical_and(cross_track>0, in_tack_mode==0), -1.0, 
                            torch.where(torch.logical_and(cross_track<0, in_tack_mode==0), 1.0, tack_side))
    
    in_tack_mode = torch.where(is_nominal, 0, 1) # Check if in tack mode
    
    # switch tack side if in tack mode and cross_track > max_cross_track
    need_switch = torch.logical_and(torch.abs(cross_track) > max_cross_track, in_tack_mode==1)

    # Switch is you need to switch and only change side to the opposite. This is to avoid conflict with the first tack_side 
    # condition above cos you're changiing only in tack mode
    tack_side = torch.where(torch.logical_and(need_switch, tack_side==1), -tack_side, tack_side)
    tack_side = torch.where(torch.logical_and(need_switch, tack_side==-1), -tack_side, tack_side)

    desired_upwind_angle =   tack_side*min_upwind_angle + wind_direction
    """desired_downwind_angle = torch.where(torch.logical_and(cross_track>0, need_switch), 
                    max_downwind_angle + wind_direction + torch.pi, -max_downwind_angle + wind_direction + torch.pi)"""

    bearing[is_upwind] = torch.atan2(torch.sin(desired_upwind_angle[is_upwind]), torch.cos(desired_upwind_angle[is_upwind]))
    #bearing[sail_mode[:, 1]==1] = torch.atan2(torch.sin(desired_downwind_angle[is_downwind]), torch.cos(desired_downwind_angle[is_downwind]))
    
    #print(f"\nis_upwind: {is_upwind} \tis_downwind: {is_downwind}")
    return bearing, in_tack_mode, tack_side

def tack_corridor_reward(p_boat:torch.Tensor, p_k:torch.Tensor, p_k1:torch.Tensor, corridor_width, margin=0.0):
    
    """
    p_boat: [N, 2] tensor of boat positions
    p_k, p_k1: [2] tensors, path segment start and end
    corridor_width: float, total width between constraint lines
    margin: tolerance before penalty starts
    """
    # Path vector and normalized direction
    path_vec = p_k1 - p_k  # [2]
    path_dir = path_vec / torch.norm(path_vec, dim=-1).unsqueeze(-1)  # [2]

    # Vector from pk to boat
    boat_vec = p_boat - p_k  # [N, 2]

    # Cross-track error (signed)
    cross = boat_vec[:, 0] * path_dir[:, 1] - boat_vec[:, 1] * path_dir[:, 0]  # [N]

    # Penalize if outside half corridor + margin
    corridor_half = (corridor_width / 2.0) + margin
    penalty = torch.relu(torch.abs(cross) - corridor_half)

    # Optional: return negative penalty (reward)
    return -penalty, cross

import random
def step_motor(current_angle, desired_angle, resolution=torch.pi/100, max_speed=0.1, dt=0.02, precision=0.05):
    """
        Simple model of the Stepper motor actuator that respects the actuator model we have
    """
    # Quantize target to valid step angle
    desired_angle = torch.round(desired_angle / resolution) * resolution
    random_noise = random.gauss(0, precision)
    # Compute delta and limit rate
    max_delta = max_speed * dt
    delta = desired_angle.clone().reshape_as(current_angle) - current_angle
    clipped_delta = torch.clamp(delta, min=-max_delta, max=+max_delta)
    #print(f"current: {current_angle.shape}; clipped: {clipped_delta.shape} desired: {desired_angle.shape}")
    final_angle = current_angle.reshape_as(clipped_delta) + clipped_delta + random_noise*clipped_delta

    return torch.atan2(torch.sin(final_angle), torch.cos(final_angle))


# CSI-Code-WhenIsGood-9cspihx
@configclass
class KingfisherSailEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 150.0 #30
    physics_dt = 1 / 60.0  # 60 Hz
    decimation = 3
    step_dt = physics_dt * decimation  # 20 Hz
    action_space = 3
    observation_space = 19
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

    # Aerodynamics
    aerodynamics_cfg: AerodynamicsCfg = AerodynamicsCfg()
    aerodynamics_cfg.air_density = 1.225
    aerodynamics_cfg.wing_span = 1
    aerodynamics_cfg.wing_chord = 0.2
    aerodynamics_cfg.wind_direction = -90*torch.pi/180
    aerodynamics_cfg.wind_speed = 8
    aerodynamics_cfg.angle_of_attack = 20*torch.pi/180
    aerodynamics_cfg.min_upwind_angle = 45*torch.pi/180
    aerodynamics_cfg.max_downwind_angle = 140*torch.pi/180
    

    # Hydrostatics
    hydrostatics_cfg: HydrostaticsCfg = HydrostaticsCfg()
    hydrostatics_cfg.mass = 35.0  # Kg considering added sensors
    hydrostatics_cfg.width = 1.0  # Kingfisher/Heron width 1.0m in Spec Sheet
    hydrostatics_cfg.length = 1.3  # Kingfisher/Heron length 1.3m in Spec Sheet
    hydrostatics_cfg.waterplane_area = 0.33  # 0.15 width * 1.1 length * 2 hulls
    hydrostatics_cfg.draught_offset = 0.21986  # Distance from base_link to bottom of the hull
    hydrostatics_cfg.max_draught = 0.20  # Kingfisher/Heron draught 120mm in Spec Sheet
    hydrostatics_cfg.average_hydrostatics_force = 275.0

    # Hydrdynamics
    hydrodynamics_cfg: HydrodynamicsCfg = HydrodynamicsCfg()
    # linear Nominal [16.44998712, 15.79776044, 100, 13, 13, 6]
    # linear SID [0.0, 99.99, 99.99, 13.0, 13.0, 0.82985084]
    hydrodynamics_cfg.linear_damping = [0.0, 99.99, 99.99, 13.0, 13.0, 5.83] # not 5
    # quadratic Nominal [2.942, 2.7617212, 10, 5, 5, 5]
    # quadratic SID [17.257603, 99.99, 10.0, 5.0, 5.0, 17.33600724]
    hydrodynamics_cfg.quadratic_damping = [17.257603, 99.99, 10.0, 5.0, 5.0, 17.33600724]
    hydrodynamics_cfg.use_drag_randomization = False
    hydrodynamics_cfg.linear_damping_rand = [0.1, 0.1, 0.0, 0.0, 0.0, 0.1]
    hydrodynamics_cfg.quadratic_damping_rand = [0.1, 0.1, 0.0, 0.0, 0.0, 0.1]

    # Thruster dynamics
    propeller_cfg: PropellerActuatorCfg = PropellerActuatorCfg()
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
    max_available_energy = episode_length_s*max_energy
    max_robot_speed = 0.7 # Max speed for energy optimization

    # reward scales
    distance_reward_scale = 0.0
    distance_progress_reward_scale =  35 #5 # 6 too much
    bearing_progress_reward_scale = 0.0

    goal_reached_threshold = 0.1
    goal_reached_scale = 100.0 # 150

    energy_penalty_scale = -0.08  #-0.001
    backwards_penalty_scale = -0.05
    time_penalty_scale = -1 #-0.008 #
    penalty_inefficient_sailing_scale = -0.1
    tack_penalty_scale = -10
    bearing_penalty_scale = 0.1 #1.0
    beargin_penalty_coef = -0.5 #-4
    lift_drag_ratio_scale = 0.1
    acord_reward_scale = 0.5
    speed_penalty_scale = -0.1

    # Environment
    min_target_distance = 20.0 
    max_target_distance = 100.0
    min_target_bearing = 0 #-torch.pi / 2
    max_target_bearing = 5*torch.pi/180 #torch.pi / 2
    max_cross_track = 8.0



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
        self._aerodynamic_force_b = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._aerodynamic_force_wing = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._thruster_forces = torch.zeros(self.num_envs, 1, 6, device=self.device)
        
        self._no_torque = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self._hydrostatics = Hydrostatics(num_envs=self.num_envs, device=self.device, cfg=self.cfg.hydrostatics_cfg)

        self._hydrodynamics = Hydrodynamics(num_envs=self.num_envs, device=self.device, cfg=self.cfg.hydrodynamics_cfg)

        self._aerodynamics = Aerodynamics(num_envs=self.num_envs, device=self.device, cfg=self.cfg.aerodynamics_cfg)

        self._thruster_dynamics = PropellerActuator(
            num_envs=self.num_envs, device=self.device, dt=cfg.step_dt, cfg=self.cfg.propeller_cfg
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
        self.max_available_episode_energy = 0.6*self.max_episode_length*torch.ones_like(self.energy)
        self.episode_avg_speed = torch.zeros_like(self.energy)
        self.episode_avg_lift_drag_ratio = torch.zeros_like(self.energy)
        self.desired_pos_b = torch.zeros(self.num_envs, 2, device=self.device)
        self.desired_trajectory_b = torch.zeros(self.num_envs, 10, 2) # 10 pts
        self.desired_speed_b = torch.zeros(self.num_envs, device=self.device)
        self.desired_bearing = torch.zeros(self.num_envs, device=self.device)

        self.is_upwind = torch.zeros(self.num_envs, device=self.device)
        self.is_downwind = torch.zeros(self.num_envs, device=self.device)

        self.joint_pos_target = torch.zeros(self.num_envs, device=self.device)
        self.sail_angle = torch.zeros(self.num_envs, device=self.device)
        self.joint_angle_mapped_pi = torch.zeros(self.num_envs, device=self.device)
        
        self.previous_robot_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.initial_robot_pos = torch.zeros((self.num_envs, 3), device=self.device)

        self.thruster_left_randn = torch.zeros(self.num_envs, device=self.device)
        self.thruster_right_randn = torch.zeros(self.num_envs, device=self.device)
        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        self.previous_sail_action = torch.zeros_like(self.actions[:, -1])

        self.episode_number = torch.ones(self.num_envs, device=self.device)
        self.global_step = 0

        # Discriminator repplay buffer memory
        #self.disc_memory = ReplayBuffer(self.num_envs, 1000, 2, device=self.device)
        self.discriminator_energy = DiscriminatorNetwork(lr=0.0001, input_dims=1, fc1_dims=256, fc2_dims=256, prediction_dims=1, 
                                                   num_envs=self.num_envs, device=self.device)
        self.discriminator_time = DiscriminatorNetwork(lr=0.0001, input_dims=1, fc1_dims=256, fc2_dims=256, prediction_dims=1, 
                                                   num_envs=self.num_envs, device=self.device)

        self.loss_discrim_energy = None #torch.zeros(self.num_envs, device=self.device)
        self.loss_discrim_time = None #torch.zeros(self.num_envs, device=self.device)
        self.disc_prediction_mean = torch.zeros(self.num_envs, device=self.device)
        self.disc_prediction_std = torch.zeros(self.num_envs, device=self.device)
        self.disc_prediction_error = torch.zeros(self.num_envs, device=self.device)
        
        self.energy_context = torch.ones_like(self.episode_number)
        self.time_context = torch.ones_like(self.episode_number)
        self.progress_context = torch.ones_like(self.episode_number)
        self.is_Training = True

        self.reward_progress = torch.zeros(self.num_envs, device=self.device)
        self.reward_bearing = torch.zeros(self.num_envs, device=self.device)
        self.reward_energy = torch.zeros(self.num_envs, device=self.device)
        self.reward_backward = torch.zeros(self.num_envs, device=self.device)
        
        self.normalize_heading = torch.zeros(self.num_envs, device=self.device)
        self.normalized_energy = torch.zeros(self.num_envs, device=self.device)

        self.corridor_width = torch.ones(self.num_envs, device=self.device)

        self.tack_manager = TackManager(min_upwind_angle=self._aerodynamics.cfg.min_upwind_angle, 
                        max_downwind_angle=self._aerodynamics.cfg.max_downwind_angle, max_cross_track=3.0)
        self.tack_manager.reset(num_envs=self.num_envs, device=self.device)
        self.tack_side = torch.ones(self.num_envs, device=self.device)
        self.in_tack_mode = torch.ones(self.num_envs, device=self.device)
        self.sailing_mode = torch.zeros((self.num_envs, 3), device=self.device)

        self.tack_length = torch.ones(self.num_envs, device=self.device)  # Default tack leg length
        self.num_tack_waypoints = 10  # Default number of waypoints for tacking
        self.tack_waypoints = torch.zeros((self.num_envs, self.num_tack_waypoints, 3), device=self.device)  # 10 waypoints
        self.tack_valid_mask = torch.zeros((self.num_envs, self.num_tack_waypoints), dtype=torch.bool, device=self.device)  # Valid mask for waypoints

        self.next_tack_wpt_idx = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)  # Index of the next waypoint to reach
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
        """if not self.airfoil_set:
            self.xfoil.airfoil = Airfoil(self._aerodynamics.cfg.x_coords_naca, self._aerodynamics.cfg.y_coords_naca)  # set later
        
            self.xfoil.n_crit = 9
            self._aerodynamics.xfoil = self.xfoil
            self.airfoil_set=True"""

        self._actions = actions.clone().clamp(-1.0, 1.0)
        # Override the actions for debugging
        # self._actions[:,0] = 0.6
        # self._actions[:,1] = 0.6

        # Compute the thruster forces based on the actions.
        # thrust_cmds = torch.tensor([0.0, 1.0], dtype=torch.float32, device=self.device)
        self._thruster_dynamics.set_target_cmd(self._actions[:, :2])
        self._thruster_forces[:, 0, :] = self._thruster_dynamics.update_forces()

        # Compute the hydrostatic and hydrodynamic forces
        robot_pos = self._robot.data.root_pos_w.clone()
        robot_quat = self._robot.data.root_quat_w.clone()
        robot_vel = self._robot.data.root_vel_w.clone()
        self._hydrostatic_force[:, 0, :] = self._hydrostatics.compute_archimedes_metacentric_local(
            robot_pos, robot_quat
        )
        self._hydrodynamic_force[:, 0, :] = self._hydrodynamics.ComputeHydrodynamicsEffects(robot_quat, robot_vel)

        #=====================================================================================================#
        #=========================Sail angle control by the agent or the joytick==============================#
        #=====================================================================================================#
        # Compute the sail angle given the angle of attack (fixed to 20 degree for now)
        current_joint_pos = self._robot.data.joint_pos[:, self._wing_joint_dof_id].clone()

        # Check if teleoperation or if joint action is controled by the robot
        
        if actions.shape[1] == 3:
            # Agent directly controls the AoA (interpreted as a relative sail change)
            sail_wing_action_scale = 0.01 * torch.pi  # Small increment per timestep
            self.sail_angle = sail_wing_action_scale * actions[:, 2:3]  # Shape: (num_envs, 1)

            max_angle = torch.pi / 2  # or other joint limit
            self.joint_pos_target = actions[:, 2:3] * max_angle

            """# Compute the new joint target: current + increment, wrapped to [-2π, 2π]
            self.joint_pos_target = (current_joint_pos + self.sail_angle + 2 * torch.pi) % (4 * torch.pi) - 2 * torch.pi"""
            self.joint_pos_target = step_motor(current_joint_pos, self.joint_pos_target)
            # For AoA calculation, map the joint to [-π, π]
            joint_pos_mapped_pi = (self.joint_pos_target + torch.pi) % (2 * torch.pi) - torch.pi
            self.joint_angle_mapped_pi = joint_pos_mapped_pi.clone()
            # Compute AoA between wind direction and current sail orientation
            self._aerodynamics.angle_of_attack = self._aerodynamics.get_angle_of_attack(
                self._aerodynamics.apparent_wind_angle, self.joint_pos_target)
            
            self._aerodynamics.sail_angle = self.joint_pos_target #joint_pos_mapped_pi.clone()

        else:
            # Autopilot mode: sail angle is computed from desired AoA and apparent wind
            self.sail_angle = self._aerodynamics.get_sail_angle(
                self._aerodynamics.apparent_wind_angle,
                self._aerodynamics.angle_of_attack
            ).reshape(current_joint_pos.shape)

            self._aerodynamics.sail_angle = self.sail_angle.clone()

            # Compute joint error (sail_angle - current), wrapped to [-π, π]
            joint_error = (self.sail_angle - current_joint_pos + torch.pi) % (2 * torch.pi) - torch.pi

            # Compute new joint target (still wrapped to [-2π, 2π] for safety)
            self.joint_pos_target = (current_joint_pos + joint_error + 2 * torch.pi) % (4 * torch.pi) - 2 * torch.pi



        # Compute the aerodynamic (wind) effect on the sail wing in the boat frame
        robot_vel_b = self._robot.data.root_lin_vel_b.clone().detach()
        #print(f"\njoin-pos: {self.joint_pos_target} \nact: {actions[:, -1]} \ntot_act: {actions}\n")
        self._aerodynamic_force_b[:, 0, :] = self._aerodynamics.compute_wind_effect(
            self._aerodynamics.Uw, self._aerodynamics.Beta_w, self._robot.data.heading_w,
            robot_vel_b[:, :2]
        )
        yaw_angle = current_joint_pos.clone().detach().reshape(self.num_envs)
        wing_quat = quat_from_euler_xyz(torch.zeros_like(yaw_angle), torch.zeros_like(yaw_angle), yaw_angle)
        #print(f"wing_quat: {wing_quat.shape} \taero_force: {self._aerodynamic_force_b.reshape(self.num_envs, -1).shape}")
        aero_force_wing = transform_points(self._aerodynamic_force_b, quat=wing_quat, pos=None)
        #print(aero_force_wing.shape)
        self._aerodynamics.wind_lift_wing = transform_points(self._aerodynamics.wind_lift_b.reshape(self.num_envs, -1), quat=wing_quat, pos=None)
        self._aerodynamics.wind_drag_wing = transform_points(self._aerodynamics.wind_drag_b.reshape(self.num_envs, -1), quat=wing_quat, pos=None)
        #print(self._aerodynamic_force_b.shape, aero_force_wing.shape)
        self._aerodynamic_force_wing = aero_force_wing.reshape_as(self._aerodynamic_force_b)
        #print(self._aerodynamic_force_b[:, 0, :].shape)
        
        #print(f"{self._aerodynamics.Uw}: {self._aerodynamics.Beta_w}")

        #=====================================================================================================#
        

    def _apply_action(self):
        sail_wing_force_b = self._aerodynamic_force_b.clone()
        # only apply thruster forces if they are not zero, otherwise it disables external previous forces.
        lft_thruster_force = self._thruster_forces[..., :3]
        rgt_thruster_force = self._thruster_forces[..., 3:] #self._thruster_forces[..., 3:] #-
        torque = torch.zeros_like(self._no_torque)
        torque[:, 0, 2] = lft_thruster_force[:, 0, 0] 
        combined = self._hydrostatic_force + self._hydrodynamic_force
        combined[:, 0, :3] = combined[:, 0, :3] + sail_wing_force_b[:, 0, :]
        #combined[:, 0, 3:] = combined[:, 0, 3:] + torque[:, 0, :]
        self._robot.set_external_force_and_torque(combined[..., :3], combined[..., 3:], body_ids=self._base_link)

        apply_mask = self.episode_energy <= self.max_available_episode_energy
        env_ids = torch.nonzero(apply_mask, as_tuple=False).squeeze(-1)
        if self.is_Training:
            epsilon = 0.4  # 10% chance to deactivate thrusters
            random_mask_left = self.thruster_left_randn < epsilon
            random_mask_right = self.thruster_right_randn < epsilon

            #mask_energy = (self.episode_energy < self.max_available_episode_energy)
            #apply_mask =  random_mask & mask_energy #& 

            # Zero out forces where mask is False
            """lft_thruster_force[random_mask_left] = 0.0
            rgt_thruster_force[random_mask_right] = 0.0"""

        if lft_thruster_force.any():
            self._robot.set_external_force_and_torque(
                lft_thruster_force, self._no_torque, body_ids=self._left_thruster_id
            )
        if rgt_thruster_force.any():
            self._robot.set_external_force_and_torque(
                rgt_thruster_force, self._no_torque, body_ids=self._right_thruster_id
            )
        
        """if rgt_thruster_force.any():
            self._robot.set_external_force_and_torque(
                rgt_thruster_force, self._no_torque, body_ids=self._right_thruster_id
            )"""
    
        """
        if sail_wing_force_wing.any():
            
            torque = torch.zeros_like(self._no_torque)
            cog = self._robot.data.root_link_pos_w.clone().detach() # center of mass
            sail_pos = self._robot.data.body_link_pos_w[:, self._sail_wing_id, :]
            lever_arm = -(sail_pos.view(-1, 3) - cog)
        
            torque = torch.cross(lever_arm.view(-1, 3), sail_wing_force_wing.view(-1, 3), dim=-1).reshape_as(sail_wing_force_b)

            self._robot.set_external_force_and_torque(
                sail_wing_force_wing, self._no_torque, body_ids=self._sail_wing_id
            )"""
            #print(sail_wing_force)
            

        
        # Set psoition of the sail joint
        self._robot.set_joint_position_target(target=self.joint_pos_target.reshape(self.num_envs,-1), 
             joint_ids=self._wing_joint_dof_id
            )

        
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
        
        direction_wind_robot = torch.atan2(self._aerodynamics.true_wind_speed2D_b[:, 1], self._aerodynamics.true_wind_speed2D_b[:, 0])
        ks = self.bearing - direction_wind_robot + torch.pi
        ks = torch.atan2(torch.sin(ks), torch.cos(ks))
        
        #desired_bearing = # In your control loop:
        #self.desired_bearing = self.tack_manager.get_desired_bearing(self.bearing, self._aerodynamics.Beta_w, self.cross_track_error)
        
        
        upwind_mask = (torch.abs(ks) < self._aerodynamics.cfg.min_upwind_angle) 
        downwind_mask = (torch.abs(ks) > self._aerodynamics.cfg.max_downwind_angle) 
        self.sailing_mode[upwind_mask, 0] = 1.0
        self.sailing_mode[downwind_mask, 1] = 1.0
        self.sailing_mode[~upwind_mask & ~downwind_mask, 2] = 1.0
        RAD2DEG = 180.0 / torch.pi
        self.in_tack_mode = torch.where(self.sailing_mode[:, 2] == 0, 1.0, 0)
        """self.bearing, self.in_tack_mode, self.tack_side = get_desired_bearing(self.bearing, self._aerodynamics.Beta_w, self._aerodynamics.cfg.min_upwind_angle, 
                self._aerodynamics.cfg.max_downwind_angle, self.cross_track_error, self.cfg.max_cross_track, self.sailing_mode,
                self.in_tack_mode, self.tack_side)"""
        
        """print(f"\nbearing: {RAD2DEG*self.bearing} \tks: {RAD2DEG*ks} \tself.sailing_mode: {self.sailing_mode}")
        print(f"cross_track_error: {self.cross_track_error} \ttack_side: {self.tack_side} \tin_tack_mode: {self.in_tack_mode}\n")"""
        
        self.bearing, self.next_tack_wpt_idx = get_desired_bearing_wpts(bearing=self.bearing, next_wpt_idx=self.next_tack_wpt_idx,
                        tack_waypts=self.tack_waypoints, robot=self._robot, sail_mode=self.sailing_mode)
        
        sampling_rate = 400
        left_thruster_enabled = torch.ones_like(self.thruster_left_randn)
        right_thruster_enabled = torch.ones_like(self.thruster_left_randn)

        if self.is_Training:
            apply_mask = (self.episode_length_buf%sampling_rate)==0
            env_ids = torch.nonzero(apply_mask, as_tuple=False).squeeze(-1)
            self.energy_context[env_ids] = torch.zeros_like(self.time_context[env_ids]).uniform_(0, 1)
            
            #torch.zeros_like(self.time_context[env_ids]).uniform_(0, 1)
            """self.time_context[env_ids] = torch.zeros_like(self.time_context[env_ids]).uniform_(0, 1)
            #cluster_context(torch.zeros_like(self.time_context[env_ids]).uniform_(0, 1), device=self.device)
            self.progress_context[env_ids] = cluster_context(torch.abs(torch.normal(mean=torch.ones_like(self.progress_context[env_ids]), 
                            std=0.5*torch.ones_like(self.progress_context[env_ids]))), device=self.device)"""
            
            apply_mask = (self.episode_length_buf%500)==0
            env_ids = torch.nonzero(apply_mask, as_tuple=False).squeeze(-1)
            #self._aerodynamics.reset_wind_condition(env_ids=env_ids)    

            # Resample speed every 300 steps
            self.desired_speed_b[env_ids] = torch.zeros_like(self.desired_speed_b[env_ids]).uniform_(0.2, 1.5)

            epsilon = 0.4  # 10% chance to deactivate thrusters
            random_mask_left = self.thruster_left_randn < epsilon
            random_mask_right = self.thruster_right_randn < epsilon

            left_thruster_enabled[random_mask_left] = 0
            right_thruster_enabled[random_mask_right] = 0


        self.energy = torch.sum(torch.square(self._actions[:, :2]), dim=1)
        self.normalized_energy = self.energy / self.cfg.max_energy

        penalty, cross = tack_corridor_reward(p_boat=self._robot.data.root_link_pos_w[:, :2], 
                    p_k=self.initial_robot_pos[:, :2], p_k1=self._desired_pos_w[:, :2], corridor_width=8*self.corridor_width)
        self.cross_track_error = cross.clone()
        
        obs = torch.cat(
            [
                self._actions,  # 3
                self._robot.data.root_lin_vel_b[:, :2],  # 2
                self._robot.data.root_ang_vel_b[:, 2].unsqueeze(1),  # 1
                torch.cos(self.bearing).unsqueeze(1),  # 1
                torch.sin(self.bearing).unsqueeze(1),  # 1
                (self.distance).unsqueeze(1),  # 1
                (self.distance/self.initial_distance).unsqueeze(1),  # 1
                torch.cos(self._aerodynamics.angle_of_attack.reshape(self.num_envs, -1)), # 1
                torch.sin(self._aerodynamics.angle_of_attack.reshape(self.num_envs, -1)), # 1
                torch.cos(self._aerodynamics.apparent_wind_angle).reshape(self.num_envs, -1), # 1
                torch.sin(self._aerodynamics.apparent_wind_angle).reshape(self.num_envs, -1), # 1
                torch.norm(self._aerodynamics.apparent_wind_speed_b, dim=-1).reshape(self.num_envs, -1), # 1
                self.cross_track_error.reshape(self.num_envs, -1), #1
                self.sailing_mode.reshape(self.num_envs, -1), #3
                
            ],
            dim=1,
        )
        #self.energy_context.reshape(self.num_envs, -1), #1
        #desired_bearing.reshape(self.num_envs, -1), # 1
        #left_thruster_enabled.reshape(self.num_envs, -1), #1
        #right_thruster_enabled.reshape(self.num_envs, -1), #1
   
        #self.normalized_energy.reshape(self.num_envs, -1), # 1,
        #self.energy_context.reshape(self.num_envs, -1), # 1
        #self.corridor_width.reshape(self.num_envs, -1), #1
        #/((self.episode_length_buf+1)*self.cfg.max_energy))
        #self.desired_speed_b.reshape(self.num_envs, -1), # 1
        #self.energy_context.reshape(self.num_envs, -1), #1
        #/((self.episode_length_buf+1)*self.cfg.max_energy)
        #self.energy_context.reshape(self.num_envs, -1), # 1
        #self.desired_speed_b.reshape(self.num_envs, -1), # 1
        # self._actions,  # 2
        #self.time_context.reshape(self.num_envs, -1), # 1
        #
        #
        #print(f"obs: {obs.shape} \tenergy: {self.energy.shape} \tcontext: {self.energy_context.shape}")
        
        
        
        self.discriminator_energy.memory.store_transition(torch.cat((self.energy.reshape(self.num_envs, -1), 
                                            self.energy_context.reshape(self.num_envs, -1)), dim=-1))
        """
        self.discriminator_time.memory.store_transition(torch.cat((torch.norm(self._robot.data.root_lin_vel_b[:, :2], 
                        dim=-1).reshape(self.num_envs, -1), self.time_context.reshape(self.num_envs, -1)), dim=-1))"""
        
        if self.global_step % 20 == 0:
            #print(self.global_step)
            self.loss_discrim_energy, _ = self.discriminator_energy.learn() #, log_prob1
        #self.loss_discrim_time, log_prob2 = self.discriminator_time.learn()

        #print(f"loss1: {self.loss_discrim_energy} \tloss2: {self.loss_discrim_time} ")

        #print(f"\nangle: {self.sail_angle} \naoa: {self._aerodynamics.angle_of_attack}\n")
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        
        current_robot_pos = self._robot.data.root_link_pos_w.clone()
        self.position_progress = current_robot_pos - self.previous_robot_pos
        #grad_direction = self._desired_pos_w - self.previous_robot_pos #self.initial_robot_pos
        grad_direction = self._desired_pos_w - current_robot_pos
        grad_direction = grad_direction / torch.norm(grad_direction, dim=-1, keepdim=True)
        root_vel_w = self._robot.data.root_lin_vel_w.clone()
        
        # Distance progress
        #self.distance_progress = torch.sum(root_vel_w*grad_direction, dim=-1)/torch.norm(grad_direction, dim=-1)
        self.distance_progress = self.previous_distance - self.distance #torch.sum(self.position_progress*grad_direction, dim=-1)/torch.norm(grad_direction, dim=-1) 
        #self.previous_distance - self.distance #
        #torch.sum((current_robot_pos - self.previous_robot_pos)*(grad_direction), dim=-1)/torch.norm(grad_direction, dim=-1) #
        distance_progress_norm =  (self.distance_progress / self.initial_distance) #(1 - self.distance/self.initial_distance)*self.step_dt
        #print(f"new: {self.distance_progress} old: {distance_progress_norm}")
        #self.cfg.distance_progress_reward_scale = 1
        distance_progress_reward =  distance_progress_norm * self.cfg.distance_progress_reward_scale #*self.step_dt
        self.reward_progress = distance_progress_reward.clone()
        distance_progress_reward = distance_progress_reward.clone()

        # Downwind and upwind condition check
        direction_wind_robot = torch.atan2(self._aerodynamics.true_wind_speed2D_b[:, 1], self._aerodynamics.true_wind_speed2D_b[:, 0])
        ks = direction_wind_robot + torch.pi
        ks = torch.atan2(torch.sin(ks), torch.cos(ks))
        
        DEG2RAD = torch.pi/180
        #print(f"wind: {direction_wind_robot*(180/torch.pi)} \tbearing: {self.bearing*(180/torch.pi)} \tks: {ks*(180/torch.pi)}")
        #condition = (torch.cos(torch.pi/4*torch.ones_like(direction_wind_robot)) - torch.abs(torch.cos(direction_wind_robot)) < 0)
        self.is_upwind = torch.abs(ks)<self._aerodynamics.cfg.min_upwind_angle
        self.is_downwind = torch.abs(ks)>self._aerodynamics.cfg.max_downwind_angle
        condition = torch.logical_or(self.is_downwind, self.is_upwind)
        
        
        """distance_progress_reward[self.is_upwind] *= upwind_deviation[self.is_upwind]
        distance_progress_reward[self.is_downwind] *= downwind_deviation[self.is_downwind]"""
        #print(f"direction: {direction_wind_robot*(180/torch.pi)} \tcondition: {condition}")
        # 0.005*(1 - 2.5*torch.tanh(distance_progress_norm))
        #print(f"episod: {distance_progress_reward}: {torch.sum((current_robot_pos - self.previous_robot_pos)*(grad_direction), dim=-1)/torch.norm(grad_direction, dim=-1)}")
        # self.cfg.distance_progress_reward_scale*self.distance_progress #
        
        # Energy
        self.energy = torch.sum(torch.square(self._actions[:, :2]), dim=1)
        self.episode_energy += self.energy
        #-torch.sum(torch.square(self._actions[:, -1:]), dim=1)
        
        # Reached goal
        goal_reward = torch.zeros(self.num_envs, device=self.device)
        goal_reward[self.distance < self.cfg.goal_reached_threshold] = self.cfg.goal_reached_scale

        # Goal passed
        goal_passed = torch.sum((self._desired_pos_w[:, :2] - self.initial_robot_pos[:, :2])*(self._robot.data.root_link_pos_w[:, :2] \
        - self.initial_robot_pos[:, :2]), dim=-1)>torch.square(torch.norm(self._desired_pos_w[:, :2] - self.initial_robot_pos[:, :2], dim=-1))+3
        #goal_reward[goal_passed] = -0.1*self.cfg.goal_reached_scale

        # Penalize going backwards
        backwards_penalty = torch.zeros(self.num_envs, device=self.device)
        root_lin_vel_b_x = self._robot.data.root_lin_vel_b[:, 0]
        backwards_penalty[root_lin_vel_b_x < 0.0] = self.cfg.backwards_penalty_scale
        self.reward_backward = backwards_penalty.clone()
        self.episode_avg_speed += torch.norm(self._robot.data.root_lin_vel_b, dim=-1)
        #print(f"lin: {root_lin_vel_b_x}")

        # Reduce energy consumption
        # check max speed
        
        energy_norm = self.energy * self.step_dt / self.cfg.max_energy
        #energy_reward = torch.zeros_like(energy_norm)
        energy_reward = self.cfg.energy_penalty_scale * energy_norm #* self.energy_context
        self.reward_energy = energy_reward.clone()
        mask = root_lin_vel_b_x > self.cfg.max_robot_speed
        #energy_reward[mask] = 100*self.cfg.energy_penalty_scale * energy_norm[mask] #+ torch.abs(root_lin_vel_b_x-self.cfg.max_robot_speed)[mask])
        #print((torch.abs(self.previous_distance - self.distance)/torch.norm(self._robot.data.root_lin_vel_b, dim=-1)) * self.cfg.time_penalty_scale * self.step_dt * self.time_context)
        # Penalize bearing errors
        root_vel_w = self._robot.data.root_lin_vel_w.clone()
        bearing_penalty = self.cfg.bearing_penalty_scale*(torch.exp(self.cfg.beargin_penalty_coef * torch.abs(self.bearing)) - 1)
        #bearing_penalty = torch.sum(root_vel_w*(grad_direction), dim=-1)/(torch.norm(grad_direction, dim=-1)*torch.norm(root_vel_w, dim=-1))
        #bearing_penalty = 0.1*bearing_penalty*self.step_dt
        #bearing_penalty[condition] = 0.01*bearing_penalty[condition]
        #torch.zeros_like(self.bearing) #

        lift_drag_ratio = 0.1*torch.max(torch.zeros_like(self._aerodynamics.lift_coeff), self._aerodynamics.lift_coeff/self._aerodynamics.drag_coeff)
        lift_ratio = torch.max(torch.zeros_like(self._aerodynamics.lift_coeff), self._aerodynamics.lift_coeff/self._aerodynamics.max_cl)
        ratio_reward = self.cfg.lift_drag_ratio_scale*self.step_dt*(lift_ratio)
        #ratio_reward = self.cfg.lift_drag_ratio_scale*self.step_dt*(lift_drag_ratio)
        self.reward_bearing = bearing_penalty.clone()
        #self.episode_avg_lift_drag_ratio += 
        #print(f"max_ratio: {self._aerodynamics.max_cl_cd_ratio}")
        #print(f"\nlift: {self._aerodynamics.lift_coeff} \tratio: {lift_ratio} \tratio_rew: {ratio_reward}")
        
        # Don't penalize if too close to the goal as the bearing becomes unstable
        # bearing_penalty[self.distance < 0.2] = 0.0
        # Time
        time = torch.ones_like(self.energy) #torch.abs(self.previous_distance - self.distance)/torch.norm(self._robot.data.root_lin_vel_b, dim=-1)
        time_reward = 0.1*self.cfg.time_penalty_scale * time * self.step_dt #* (1- self.energy_context)
        #self.cfg.time_penalty_scale * self.step_dt * self.time_context * torch.ones(self.num_envs, device=self.device) 
        
        reward_speed = self.cfg.speed_penalty_scale*(torch.square(root_lin_vel_b_x - self.desired_speed_b))*self.step_dt
        self.reward_energy = reward_speed.clone()

        penalty_inefficient_sailing = torch.zeros_like(reward_speed)
        penalty_inefficient_sailing[condition] = 0.5*self.cfg.penalty_inefficient_sailing_scale*self.step_dt*torch.norm(self._aerodynamics.Uw, dim=-1)
        self.reward_backward = penalty_inefficient_sailing.clone()
        #print(f"time: {time_reward}")
        #print(f"distance_progress: {distance_progress_reward} \ttime: {time_reward} \tenergy: {energy_reward}")
        #print(f"penal: {bearing_penalty} : {torch.abs(self._robot.data.heading_w-self._aerodynamics.Beta_w)}")
        #(f"prog: {distance_progress_reward} \tdist: {bearing_penalty}")

        #print(f"bearing: {bearing_penalty.item()} \tspeed: {reward_speed.item()} \tprogr: {distance_progress_reward.item()}\n")

        predicted_energy_context, log_probs1, distrib1 = self.discriminator_energy.predict(
            self.episode_energy.reshape(self.num_envs, -1), requires_grad=False, reparameterize=False)
        self.disc_prediction_mean = distrib1.loc.clone()
        self.disc_prediction_std = distrib1.scale.clone()
        #predicted_energy_context = self.discriminator_energy.forward(self.episode_energy.reshape(self.num_envs, -1))
        predic_error_energy = torch.clamp(torch.abs(self.normalized_energy.reshape(-1)-self.energy_context.reshape(-1))**2, min=0.000001, max=0.99999)
        #print(f"\npredic_error_energy: {predic_error_energy} \tenergy_context: {self.energy_context} \tprediction: {self.normalized_energy}")
        self.disc_prediction_error = predic_error_energy.clone()
        #predic_error_energy = torch.abs(predicted_energy_context.reshape(-1) - self.energy_context.reshape(-1))
        #predicted_time_context, log_probs2, distrib2= self.discriminator_time.predict(self.episode_energy.reshape(self.num_envs, -1), requires_grad=False)
        #predic_error_time = torch.abs(predicted_time_context.reshape(-1) - self.time_context.reshape(-1))
        #print(f"\nenerg: {predicted_energy_context[:10], self.energy_context[:10]}")
        #print(f"speed: {predicted_time_context[:10], self.time_context[:10]}\n")
        #print(self.discriminator_energy.memory.state_memory)
        eps = 1e-6
        #safe_time_error = torch.clamp(predic_error_time, min=eps)
        
        distance_reward = 0.6*(- torch.tanh((self.distance/self.initial_distance)/1.6)) * self.step_dt

        #print(f"ks: {(180/torch.pi)*ks}")
        #safe_energy_error = torch.clamp(predic_error_energy, min=eps)
        #print(f"predic_error_energy: {predic_error_energy} \tenergy_context: {self.energy_context} \tprediction: {predicted_energy_context}")
        reward_acord = -self.cfg.acord_reward_scale * predic_error_energy # .pow(2) + predic_error_energy.pow(2))
        #reward_acord = self.cfg.acord_reward_scale * 0.5 * (- torch.log(safe_energy_error)) # - torch.log(safe_time_error)
        self.reward_backward = reward_acord.clone()
        
        tack_reward = torch.zeros_like(energy_reward)
        mask = torch.abs(self.cross_track_error)>5
        tack_reward[mask] = self.cfg.tack_penalty_scale*self.step_dt

        aerodynamic_force_b = self._aerodynamics.wind_lift_b.clone().reshape_as(self._aerodynamic_force_b)
        aerodynamic_force_world = transform_points(aerodynamic_force_b, 
                quat=self._robot.data.body_state_w[:, self._base_link, 3:7].reshape(self.num_envs, -1), pos=None)
        #print(f"inefficient_sailing: {penalty_inefficient_sailing} ")
        #print(f"{(self._desired_pos_w).shape}, {self.initial_robot_pos.shape} ,{aerodynamic_force_world[:, 0, :2].shape}")
        """force_dot_dist = 0.01*self.step_dt*torch.sum(aerodynamic_force_world[:, 0, :2]*(self._desired_pos_w-self.initial_robot_pos)[:, :2],
                                     dim=-1)/torch.norm(self._desired_pos_w-self._robot.data.root_link_pos_w, dim=-1)"""
        force_projection = 0.01*aerodynamic_force_b[:, 0, 0]*self.step_dt 
        #force_dot_dist = torch.where(force_dot_dist>0, force_dot_dist, (self.distance/self.initial_distance)*force_dot_dist)
        #print(f"force_dot_dist: {force_dot_dist} : \t{(self._desired_pos_w-self.initial_robot_pos)[:, :2]} \t{self._aerodynamic_force_b[:, 0, :2]}")
        #self.cross_track_error = cross.clone()
        #print(f"tack_reward: {tack_reward} \tcross: {self.cross_track_error}: {8*self.corridor_width} \tmask: {mask}")
        #print(f"reward_acord: {reward_acord}")
        #print(f"std1: {distrib1.scale} \tstd2: {distrib2.scale} ")
        rewards = {  
            "1_distance_progress": distance_progress_reward,
            "2_goal_reached": goal_reward,
            "3_energy": energy_reward,
            "4_backwards": backwards_penalty,
            "5_bearing_penalty": bearing_penalty,
            "6_time": time_reward,
            "7_tack_penalty": 0*distance_reward,
            "8_lift_drag_ratio": 0*force_projection,
        }
        #print(f"rewards: {rewards} rew_lift_drag_ratio: {lift_drag_ratio}\n")
        # #"6_time": time_reward
        self.previous_distance = self.distance.clone()
        self.previous_robot_pos = current_robot_pos.clone()
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Desired position in the robot frame (2D)
        self.desired_pos_b_3d, _ = subtract_frame_transforms(
            self._robot.data.root_link_state_w[:, :3], self._robot.data.root_link_state_w[:, 3:7], self._desired_pos_w
        )
        self.desired_pos_b[:, :2] = self.desired_pos_b_3d[:, :2]
        self.distance = torch.linalg.norm(self.desired_pos_b, dim=1)
        #self.bearing = torch.atan2(self.desired_pos_b[:, 1], self.desired_pos_b[:, 0])

        goal_passed = torch.sum((self._desired_pos_w[:, :2] - self.initial_robot_pos[:, :2])*(self._robot.data.root_link_pos_w[:, :2] \
        - self.initial_robot_pos[:, :2]), dim=-1)>torch.square(torch.norm(self._desired_pos_w[:, :2] - self.initial_robot_pos[:, :2], dim=-1))+3
        
        # Finish episode if the goal is reached
        done = torch.zeros_like(time_out)
        done[self.distance < self.cfg.goal_reached_threshold] = True
        done = torch.logical_or(done, goal_passed)
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
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
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
        extras["Contexts/time_weight_sampled"] = self.time_context[env_ids].mean().item()
        extras["Contexts/energy_weight_sampled"] = self.energy_context[env_ids].mean().item()
        extras["Contexts/disc_prediction_mean"] = self.disc_prediction_mean[env_ids].mean().item()
        extras["Contexts/disc_prediction_std"] = self.disc_prediction_std[env_ids].mean().item()
        extras["Contexts/disc_prediction_error"] = self.disc_prediction_error[env_ids].mean().item()
        extras["Contexts/progress_weight_sampled"] = self.progress_context[env_ids].mean().item()
        extras["Contexts/loss_discrim_energy"] = self.loss_discrim_energy.item() if self.loss_discrim_energy is not None else 0
        extras["Contexts/loss_discrim_time"] = self.loss_discrim_time.item() if self.loss_discrim_time is not None else 0
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

        mask1 = torch.any(self.episode_number[env_ids] < 50)
        mask2 = torch.any(torch.logical_and(self.episode_number[env_ids] >= 50 , self.episode_number[env_ids] < 150))
        mask3 = torch.any(self.episode_number[env_ids] >= 150)

        if self.is_Training:
            if mask1:
                self.cfg.max_target_distance = 15
                self.cfg.min_target_distance = 10
            elif mask2:
                self.cfg.max_target_distance = 25
                self.cfg.min_target_distance = 15
            elif mask3:
                self.cfg.max_target_distance = 60
                self.cfg.min_target_distance = 20

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

        # Initialize markers pos
        d = 5
        n_markers = 500
        self.marker_translations = np.random.uniform([-self.env_pos[0]-d, -self.env_pos[1]-d, 1], 
                    [self.env_pos[0]+d, self.env_pos[1]+d, 1.6], (n_markers, 3))  # Adjust bounds as needed
        self.red_marker_translations = np.random.uniform([-self.env_pos[0]-d/3, -self.env_pos[1]-d/3, 1], 
                    [self.env_pos[0]+d/2, self.env_pos[1]+d/2, 1.5], (n_markers//5, 3))
        
        if self.is_Training:
            self.energy_context[env_ids] = torch.zeros_like(self.time_context[env_ids]).uniform_(0, 1)
            #torch.zeros_like(self.time_context[env_ids]).uniform_(0, 1)
            #self.corridor_width[env_ids] = torch.zeros_like(self.corridor_width[env_ids]).uniform_(0.5, 1)
            self.time_context[env_ids] = torch.zeros_like(self.time_context[env_ids]).uniform_(0, 1)
            #cluster_context(torch.zeros_like(self.time_context[env_ids]).uniform_(0, 1), device=self.device)
            self.progress_context[env_ids] = cluster_context(torch.abs(torch.normal(mean=torch.ones_like(self.progress_context[env_ids]), 
                            std=0.5*torch.ones_like(self.progress_context[env_ids]))), device=self.device)
            
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
            
            self._aerodynamics.reset_wind_condition(env_ids=env_ids, randomize_direction=False, randomize_speed=True, 
                                                    upwind=True, downwind=False, beam=False, close=True, broad=False)

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
        
        self.tack_side[env_ids] = torch.ones_like(self.tack_side[env_ids])
        self.in_tack_mode = torch.zeros_like(self.in_tack_mode[env_ids])
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
        
        self.tack_length[env_ids] = self.initial_distance[env_ids] // 3
        #print(f"tack_length: {self.tack_length[env_ids]}")
        self.tack_waypoints[env_ids, :, :2], _ = generate_tacking_waypoints(
                    start_pos=self.initial_robot_pos[env_ids, :2], 
                    goal_pos=self._desired_pos_w[env_ids, :2], 
                    wind_direction=self._aerodynamics.Beta_w[env_ids],
                    min_upwind_angle=self._aerodynamics.cfg.min_upwind_angle, 
                    tack_leg_length=self.tack_length[env_ids],
                    max_num_waypoints=self.num_tack_waypoints
                )
        #print(f"tack_waypoints: {self.tack_waypoints[env_ids, :, :2]}")

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
            app_wind_world = transform_points(torch.cat((self._aerodynamics.apparent_wind_speed_b, 
            torch.zeros((self.num_envs, 1), device=self.device)), dim=-1), 
            quat=self._robot.data.body_state_w[:, self._base_link, 3:7].reshape(self.num_envs, -1), 
            pos=None)[:, :2]
            #print(f"app: {self._aerodynamics.apparent_wind_speed_b}")
            app_wind_speed_world = torch.norm(app_wind_world, dim=-1) # magnitude of app_wind
            app_wind_angle_world = torch.atan2(app_wind_world[:, 1], app_wind_world[:, 0])


            true_wind_world = transform_points(torch.cat((self._aerodynamics.true_wind_speed2D_b, 
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
            sail_unit_vector_world = transform_points(torch.cat((self._aerodynamics.sail_unit_vector, 
            torch.zeros((self.num_envs, 1), device=self.device)), dim=-1), 
            quat=self._robot.data.body_state_w[:, self._base_link, 3:7].reshape(self.num_envs, -1), 
            pos=None)[:, :2]

            # sail ortho unit vector
            sail_ortho_unit_vector_world = transform_points(torch.cat((self._aerodynamics.sail_ortho_unit_vector, 
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
            #wind_drag_world = transform_points(self._aerodynamics.wind_drag[:, :3], quat=self._robot.data.root_link_state_w[:, 3:7], pos=None)
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
            #print(f"true_speed: {self._aerodynamics.Uw*torch.cos(self._aerodynamics.Beta_w), self._aerodynamics.Uw*torch.sin(self._aerodynamics.Beta_w)} :{true_wind_world} true_angle: {true_wind_angle_world}\n")

            
            thrust_left = self._thruster_forces[..., :3].clone().detach().reshape(self.num_envs, -1)
            thrust_right = self._thruster_forces[..., 3:].clone().detach().reshape(self.num_envs, -1)
            aerodynamic_force_b = self._aerodynamic_force_b[:, :3].clone().detach().reshape(self.num_envs, -1)
            hydrodynamic_force = self._hydrodynamic_force.clone().detach().reshape(self.num_envs, -1)[:, :3]
            hydrodynamic_force_lateral = torch.zeros_like(hydrodynamic_force)
            hydrodynamic_force_forward = torch.zeros_like(hydrodynamic_force)

            hydrodynamic_force_lateral[:, 1] = hydrodynamic_force[:, 1]
            hydrodynamic_force_forward[:, 0] = hydrodynamic_force[:, 0]
            """thrust_left[:, :1] = self._actions[:, :1]
            thrust_right[:, :1] = self._actions[:, 1:2]"""
            #print(f"{thrust_left.shape} -> {thrust_right.shape} -> {self._aerodynamic_force_b.squeeze(0)[:, :2].shape}")
            #array_forces_2D = torch.cat((thrust_left, thrust_right, self._aerodynamics.wind_lift[:, :2], self._aerodynamics.wind_drag[:, :2]), dim=0)
            #self.force_visualization(True, thrust_left.squeeze(0)[:, :2], body_ids=[1], marker_ob=self.green_marker)
            #self.force_visualization(True, thrust_right.squeeze(0)[:, :2], body_ids=[2], marker_ob=self.green_marker)
            wind_lift_world = transform_points(self._aerodynamics.wind_lift_b[:, :3], 
                quat=self._robot.data.body_state_w[:, self._base_link, 3:7].reshape(self.num_envs, -1), pos=None)
            
            wind_drag_world = transform_points(self._aerodynamics.wind_drag_b[:, :3], 
                quat=self._robot.data.body_state_w[:, self._base_link, 3:7].reshape(self.num_envs, -1), pos=None)

            aerodynamic_force_world = transform_points(aerodynamic_force_b, 
                quat=self._robot.data.body_state_w[:, self._base_link, 3:7].reshape(self.num_envs, -1), pos=None)
            
            """wind_lift_world = transform_points(self._aerodynamics.wind_lift_wing[:, :3], 
                quat=self._robot.data.body_state_w[:, self._sail_wing_id, 3:7].reshape(self.num_envs, -1), pos=None)
            
            wind_drag_world = transform_points(self._aerodynamics.wind_drag_wing[:, :3], 
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
                                directions=self._aerodynamic_force_b.squeeze(0),
                                magnitudes=self._aerodynamic_force_b,
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

        max_magnitude = self._aerodynamics.cfg.wind_speed + 2  # Or any configurable maximum
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
            max_speeds = self._aerodynamics.cfg.wind_speed + 2 # assuming 2m/s is boat max speed
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
