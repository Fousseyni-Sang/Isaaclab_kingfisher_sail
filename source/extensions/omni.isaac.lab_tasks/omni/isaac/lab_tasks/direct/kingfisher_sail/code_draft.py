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
    valid_mask[:, 0] = True
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
        #print(f"sail_away: {sail_away} \tnear_goal: {near_goal} goal_proj: {goal_proj}, tack_dir: {tack_dir}, goal_vec: \
        #{goal_dir} tack_angle: {(180/torch.pi)*tack_angle}")
        done_now = (~finished) & (sail_away | near_goal)
        #next_pos[done_now] = goal_pos[done_now]
        next_pos[near_goal] = goal_pos[near_goal]

        finished |= done_now
        #print(f"done_now: {done_now} \tfinished: {finished}")
        current_pos = next_pos
        tack_side = -tack_side
        #print(f"finished: {finished}, step: {step}, ")
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
    #bearing = torch.atan2(desired_pos_b[:, 1], desired_pos_b[:, 0])

    previous_wpt = tack_waypts[torch.arange(tack_waypts.shape[0]), torch.clamp(next_wpt_idx-1, min=0)]

    wpt_passed = torch.sum((next_wpt[:, :2] - previous_wpt[:, :2])*(robot.data.root_link_pos_w[:, :2] \
    - previous_wpt[:, :2]), dim=-1)>torch.square(torch.norm(next_wpt[:, :2] - previous_wpt[:, :2], dim=-1))

    wpt_reached = distance < 0.5
    """print(f"next_wpt: {next_wpt}, prev_wpt: {previous_wpt}, wpt_reached-wpt_passed: {wpt_reached, wpt_passed} distance: {distance}")
    print(f"tack_pts: {tack_waypts}, next_wpt_idx: {next_wpt_idx}")
    print(f"robot_pos: {robot.data.root_link_state_w[:, :3]}")"""
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
def step_motor(current_angle, desired_angle, resolution=torch.pi/100, max_speed=1.0, dt=0.05, precision=0.05):
    """ 
        Simple model of the Stepper motor actuator that respects the actuator model we have
    """
    # Quantize target to valid step angle
    desired_angle = torch.round(desired_angle / resolution) * resolution
    random_noise = torch.randn_like(current_angle)*precision
    # Compute delta and limit rate
    max_delta = max_speed * dt
    delta = desired_angle.clone().reshape_as(current_angle) - current_angle
    clipped_delta = torch.clamp(delta, min=-max_delta, max=+max_delta)
    #print(f"current: {current_angle.shape}; clipped: {clipped_delta.shape} desired: {desired_angle.shape}")
    final_angle = current_angle.reshape_as(clipped_delta) + clipped_delta + random_noise*clipped_delta

    return torch.atan2(torch.sin(final_angle), torch.cos(final_angle))