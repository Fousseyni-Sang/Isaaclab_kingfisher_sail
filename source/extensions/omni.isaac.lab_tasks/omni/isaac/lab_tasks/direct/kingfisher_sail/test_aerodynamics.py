# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.lab.physics.aerodynamics import AerodynamicsCfg, Aerodynamics
import torch
import matplotlib.pyplot as plt
import numpy as np

# Aerodynamics
aerodynamics_cfg: AerodynamicsCfg = AerodynamicsCfg()
aerodynamics_cfg.air_density = 1.225
aerodynamics_cfg.wing_span = 1
aerodynamics_cfg.wing_chord = 0.2
aerodynamics_cfg.wind_direction = 170*torch.pi/180
aerodynamics_cfg.wind_speed = 5
aerodynamics_cfg.angle_of_attack = 20*torch.pi/180
aerodynamics_cfg.min_upwind_angle = 45*torch.pi/180
aerodynamics_cfg.max_downwind_angle = 140*torch.pi/180
device = 'cuda:0'

min_target_distance = 20.0 
max_target_distance = 100.0
min_target_bearing = 0 #-torch.pi / 2
max_target_bearing = 5*torch.pi/180 #torch.pi / 2
max_cross_track = 8.0
num_envs = 1  # Number of environments

in_tack_mode = torch.ones(num_envs, device=device)

tack_length = torch.ones(num_envs, device=device)  # Default tack leg length
num_tack_waypoints = 10  # Default number of waypoints for tacking
tack_waypoints = torch.zeros((num_envs, num_tack_waypoints, 3), device=device)  # 10 waypoints
tack_valid_mask = torch.zeros((num_envs, num_tack_waypoints), dtype=torch.bool, device=device)

_desired_pos_w = torch.zeros(num_envs, 3, device=device)
initial_distance = torch.zeros_like(_desired_pos_w[0, 0]).uniform_(
            min_target_distance, max_target_distance
        )

initial_robot_pos = torch.zeros((num_envs, 3), device=device)  # Initial robot position
initial_bearing = torch.zeros_like(_desired_pos_w[0, 0]).uniform_(
            min_target_bearing, max_target_bearing
        )
distance = initial_distance
previous_distance = initial_distance
_desired_pos_w[0, 0] = torch.cos(initial_bearing) * initial_distance
_desired_pos_w[0, 1] = torch.sin(initial_bearing) * initial_distance
_desired_pos_w[0, 2] = 0.0  # only in 2D


aerodynamics = Aerodynamics(num_envs=1, device=device, cfg=aerodynamics_cfg)

#=====================================================================================================================#
# Fonction definition for generating tacking waypoints
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
        print(f"tack_angle: {(180/torch.pi)*tack_angle}; wind_angle: {(180/torch.pi)*wind_direction}; min_angle: {(180/torch.pi)*min_upwind_angle}")
        
        tack_dir = torch.stack((torch.cos(tack_angle), torch.sin(tack_angle)), dim=1)
        
        next_pos = current_pos + tack_leg_length.reshape(num_envs, -1) * tack_dir

        waypoints[:, step, :] = next_pos
        valid_mask[:, step] = ~finished
        print(f"step: {step} \tcurrent_pos: {current_pos} \tnext_pos: {next_pos} \tgoal_pos: {goal_pos}")
        to_goal_vec = goal_pos - current_pos
        to_next_vec = next_pos - current_pos

        goal_dir = to_goal_vec / (to_goal_vec.norm(dim=-1, keepdim=True) + 1e-6)
        tack_dir = to_next_vec / (to_next_vec.norm(dim=-1, keepdim=True) + 1e-6)

        goal_proj = torch.sum(goal_dir * tack_dir, dim=1)
        print(f"goal_dir: {goal_dir} \ttack_dir: {tack_dir} \tgoal_proj: {goal_proj}")
        sail_away = goal_proj < 0.1
        #print(f"cos: {goal_proj} \tsailaway: {sail_away}")
        near_goal = torch.norm(goal_pos - next_pos, dim=1) < tack_leg_length
        print(f"sail_away: {sail_away} \tnear_goal: {near_goal}")
        done_now = (~finished) & (sail_away | near_goal)
        next_pos[done_now] = goal_pos[done_now]
        finished |= done_now
        print(f"done_now: {done_now} \tfinished: {finished}")
        current_pos = next_pos
        tack_side = -tack_side

        if finished.all():
            waypoints[:, step+1, :] = goal_pos
            break

    return waypoints, valid_mask

#def get_desired_heading_wpts


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

#=====================================================================================================================#

print(aerodynamics.Uw[:])
tack_waypoints[0, :, :2], _ = generate_tacking_waypoints(
                    start_pos=initial_robot_pos[:, :2], 
                    goal_pos=_desired_pos_w[:, :2], 
                    wind_direction=aerodynamics.Beta_w[:],
                    min_upwind_angle=aerodynamics.cfg.min_upwind_angle, 
                    tack_leg_length=5*tack_length,
                    max_num_waypoints=num_tack_waypoints)


waypoints = tack_waypoints[0, :, :2].cpu().numpy()
plt.plot(waypoints[:, 0], waypoints[:, 1], marker='o')
plt.arrow(_desired_pos_w[0, 0].item(), _desired_pos_w[0, 1].item()-10, 10*np.cos(aerodynamics.Beta_w[0].item()), 10*np.sin(aerodynamics.Beta_w[0].item()), color='skyblue', width=0.5)
plt.text(_desired_pos_w[0, 0].item(), _desired_pos_w[0, 1].item(), "Goal", fontsize=12)
plt.grid(True)
plt.axis('equal')
plt.title("Tacking Waypoint Planner")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("tack_waypoints.png")



