import torch
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import math

from omni.isaac.lab.physics.aerodynamics import AerodynamicsCfg, Aerodynamics
import torch
import matplotlib.pyplot as plt

# Aerodynamics
aerodynamics_cfg: AerodynamicsCfg = AerodynamicsCfg()
aerodynamics_cfg.air_density = 1.225
aerodynamics_cfg.wing_span = 1
aerodynamics_cfg.wing_chord = 0.2
aerodynamics_cfg.wind_direction = -90*torch.pi/180
aerodynamics_cfg.wind_speed = 5
aerodynamics_cfg.angle_of_attack = 20*torch.pi/180
device = 'cuda:0'

num_envs = 1
aerodynamics = Aerodynamics(num_envs=1, device=device, cfg=aerodynamics_cfg)
aerodynamic_force_b = torch.zeros(num_envs, 1, 3, device=device)

session = ort.InferenceSession("/home/GTL/fsangare/Isaaclab_kingfisher_sail/logs/rl_games/kingfisher_direct/2025-06-30_17-57-59/nn/last_kingfisher_direct_ep_50_rew_0.5999513.onnx")
#loaded_agent = torch.load("runs/exp1/last_kingfisher_direct_ep_200_rew_-13.527881.pth", weights_only=False)

# Input name and dummy observation
input_name = session.get_inputs()[0].name

# Simulation parameters
num_steps = 5000
change_every_k = 500

# Data containers
joint_positions = []
angle_of_attacks = []
wind_directions = []
rewards_aero = []
lift_coeffs = []
drag_coeffs = []

# Wind initialization
wind_dir_rad = np.radians(30)
wind_speed = 5.0  # example

import random
def step_motor(current_angle, desired_angle, resolution=torch.pi/100, max_speed=0.3, dt=0.02, precision=0.05):
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

current_joint_pos = 0
desired_pos_w = torch.tensor([10., 7.], device=device, dtype=torch.float32).reshape(num_envs, 2)
for step in range(num_steps):
    # Change wind direction and speed every k steps
    if step % change_every_k == 0:
        wind_dir_rad = np.radians(np.random.uniform(-180, 180))
    
    if step % 2000==0:
        wind_speed = np.random.uniform(2.0, 10.0)

    aerodynamics.update_wind(wind_direction=wind_dir_rad, wind_speed=wind_speed)
    # Fake observation (cos/sin of wind_dir, wind speed, previous joint pos)
    obs = np.array([
        math.cos(wind_dir_rad),
        math.sin(wind_dir_rad),
        wind_speed,
        joint_positions[-1] if joint_positions else 0.0
    ], dtype=np.float32).reshape(1, -1)

    # Inference
    output = session.run(None, {input_name: obs})
    action = output[0][0][0]  # scalar action
    sail_angle = action * math.pi  # scale output to [-π, π] (if tanh output)

    #sail_angle = step_motor(torch.tensor([current_joint_pos]), torch.tensor([sail_angle])).item()
    rad2degree = 180/np.pi
    # Store data
    joint_positions.append(sail_angle)
    angle_diff = ((wind_dir_rad - sail_angle + np.pi) % (2 * np.pi)) - np.pi  # Normalize
    wind_directions.append(wind_dir_rad*rad2degree)

    aerodynamics.angle_of_attack = aerodynamics.get_angle_of_attack(
            aerodynamics.apparent_wind_angle, torch.tensor([sail_angle], device=device))
    aerodynamics.sail_angle = torch.tensor([sail_angle], device=device)
    angle_of_attacks.append(aerodynamics.angle_of_attack.item()*rad2degree)
    # Compute rewards
    current_to_goal = desired_pos_w
    distance_to_goal = torch.norm(current_to_goal, dim=-1)
    goal_dir = current_to_goal[..., :2] / (distance_to_goal[..., None] + 1e-8)

    aerodynamic_force_b[:, 0, :] = aerodynamics.compute_wind_effect(
            aerodynamics.Uw, aerodynamics.Beta_w, torch.tensor([0], device=device),
            torch.zeros((1, 2), device=device)
        )
    
    lift_coeffs.append(aerodynamics.lift_coeff.item()) 
    drag_coeffs.append(aerodynamics.drag_coeff.item())

    #print(aerodynamic_force_b)
    # 4. Reward force alignment with goal
    force_projection = aerodynamic_force_b[:, 0, 0] #torch.sum(aerodynamic_force_b[:, 0, :2] * goal_dir, dim=-1)
    rewards_aero.append(force_projection.cpu().numpy())

    rand2deg = 180.0 / torch.pi
    current_joint_pos = sail_angle
    
#print(joint_positions)
#print(rewards_aero)
# Plot results

plt.figure(figsize=(18, 10))
plt.subplot(1, 4, 1)
plt.plot(np.array([joint_positions])*rad2degree, label='Sail Angle (rad)', color='green')
plt.plot(angle_of_attacks, label='Angle of Attack (rad)', color='orange')
plt.plot(wind_directions, label='Wind angle (rad)', color='red')
plt.legend()

plt.subplot(1, 4, 2)
plt.scatter(angle_of_attacks, rewards_aero, label='reward aero', color='orange')
plt.ylabel("reward_aero")
plt.xlabel("Step")
plt.legend()

plt.subplot(1, 4, 3)
plt.scatter(angle_of_attacks, lift_coeffs, label='lift coeff', color='orange')
plt.scatter(angle_of_attacks, drag_coeffs, label='drag coeff', color='blue')
plt.xlabel("Angle of Attack (rad)")
plt.ylabel("Coefficient")
plt.legend()

plt.subplot(1, 4, 4)
plt.scatter(angle_of_attacks, np.array([lift_coeffs])/np.array([drag_coeffs]), label='lift coeff', color='orange')
plt.xlabel("Angle of Attack (rad)")
plt.ylabel("Coefficient")
plt.legend()

plt.savefig("test_onnx.png")