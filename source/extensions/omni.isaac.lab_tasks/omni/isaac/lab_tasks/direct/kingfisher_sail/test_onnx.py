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

aerodynamics = Aerodynamics(num_envs=1, device=device, cfg=aerodynamics_cfg)

session = ort.InferenceSession("/home/fousseyni/asv-sawasp-fousseyni/Isaaclab_kingfisher_sail/logs/rl_games/kingfisher_direct/2025-06-25_15-26-10/nn/last_kingfisher_direct_ep_50_rew_-3.3877628.onnx")
#loaded_agent = torch.load("runs/exp1/last_kingfisher_direct_ep_200_rew_-13.527881.pth", weights_only=False)

# Input name and dummy observation
input_name = session.get_inputs()[0].name

# Simulation parameters
num_steps = 500
change_every_k = 100

# Data containers
joint_positions = []
angle_of_attacks = []
wind_directions = []

# Wind initialization
wind_dir_rad = np.radians(30)
wind_speed = 5.0  # example

for step in range(num_steps):
    # Change wind direction and speed every k steps
    if step % change_every_k == 0:
        wind_dir_rad = np.radians(np.random.uniform(0, 180))
    
    if step % 10==0:
        wind_speed = np.random.uniform(2.0, 10.0)

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
    rad2degree = 180/np.pi
    # Store data
    joint_positions.append(sail_angle*rad2degree)
    angle_diff = ((wind_dir_rad - sail_angle + np.pi) % (2 * np.pi)) - np.pi  # Normalize
    angle_of_attacks.append(angle_diff*rad2degree)
    wind_directions.append(wind_dir_rad*rad2degree)
    
print(joint_positions)
# Plot results
fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

ax[0].plot(joint_positions, label='Sail Angle (rad)', color='green')
ax[0].plot(angle_of_attacks, label='Angle of Attack (rad)', color='orange')
ax[0].plot(wind_directions, label='Wind angle (rad)', color='red')
ax[0].set_ylabel("Angle [rad]")
ax[0].legend()
ax[0].grid(True)

"""ax[1].plot(angle_of_attacks, label='Angle of Attack (rad)', color='orange')
ax[1].set_ylabel("AoA [rad]")
ax[1].set_xlabel("Step")
ax[1].legend()
ax[1].grid(True)"""

plt.tight_layout()
#plt.show()
fig.savefig("test_onnx.png")