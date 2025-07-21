import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# --- Find the latest log directory ---
eval_root = "eval_logs"
subdirs = sorted(
    [os.path.join(eval_root, d) for d in os.listdir(eval_root) if os.path.isdir(os.path.join(eval_root, d))],
    key=lambda x: x.split("/")[-1],  # assumes timestamp folder names
    reverse=True
)
latest_dir = subdirs[0]
print(f"Loading data from: {latest_dir}")
num_episodes = 10
# --- Load CSVs ---
df_traj = pd.read_csv(os.path.join(latest_dir, "trajectories.csv"))
df_aero = pd.read_csv(os.path.join(latest_dir, "aero_coeffs.csv"))
df_goals = pd.read_csv(os.path.join(latest_dir, "goals.csv"))
df_metrics = pd.read_csv(os.path.join(latest_dir, "metrics.csv"))
df_ep_length = pd.read_csv(os.path.join(latest_dir, "ep_length.csv"))
df_other_metrics = pd.read_csv(os.path.join(latest_dir, "other_metrics.csv"))

#print(df_traj.head)
# --- Plot Metrics ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(df_metrics["final_distance_to_goal"])
plt.title("Final Distance to Goal")

plt.subplot(1, 3, 2)
plt.plot(df_metrics["consumed_energy"])
plt.title("Consumed Energy")

plt.subplot(1, 3, 3)
plt.plot(df_metrics["disc_prediction_mean"])
plt.title("Discriminator Prediction Mean")
plt.savefig(os.path.join(latest_dir, "metrics.png"))

# --- Get episode_length ---

ep_length_list = df_ep_length["ep_length"].tolist()
"""for ep in df_ep_length["episode"].unique()[:3]:
    ep_length = df_ep_length[df_ep_length["episode"]==ep]
    
    ep_length_list.append(ep_length["ep_length"])
"""
print(f"episode_length: {ep_length_list}")
# --- Plot Trajectories ---
plt.figure(figsize=(12, 6))
for ep in df_traj["episode"].unique()[:num_episodes]:  # Limit to first 3 episodes
    ep_length = ep_length_list[ep]
    ep_traj = df_traj[df_traj["episode"] == ep]
    for env_id in ep_traj["env_id"].unique():
        env_traj = ep_traj[ep_traj["env_id"] == env_id]
        plt.plot(env_traj["x"][:ep_length], env_traj["y"][:ep_length], label=f"Ep{ep} Env{env_id}")
        goal = df_goals[(df_goals["episode"] == ep) & (df_goals["env_id"] == env_id)]
        plt.scatter(goal["goal_x"], goal["goal_y"], marker="x")
        #print(env_traj["x"].shape, env_traj["y"].shape)

wind_direc = (125 * np.pi / 180)  # Example wind direction in radians
wind_speed = 5  # Example wind speed
# Plot wind vector
wind_x = wind_speed * np.cos(wind_direc)
wind_y = wind_speed * np.sin(wind_direc)
plt.quiver(110, 0, wind_x, wind_y, angles='xy', scale_units='xy', scale=0.5, color='blue', label='Wind Vector')
plt.title("Trajectories with Goals")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(latest_dir, "trajectory_plot.png"))

# --- Plot Aero Coefficients ---
"""plt.figure(figsize=(12, 4))
for ep in df_aero["episode"].unique()[:num_episodes]:
    ep_length = ep_length_list[ep]
    
    ep_aero = df_aero[df_aero["episode"] == ep]
    for env_id in ep_aero["env_id"].unique():
        env_aero = ep_aero[ep_aero["env_id"] == env_id]
        plt.plot(np.arange(ep_length), env_aero["lift_coeff"][:ep_length], label=f"Lift Ep{ep} Env{env_id}")
        #print(env_aero["lift_coeff"][:ep_length].shape)
        plt.plot(np.arange(ep_length), env_aero["drag_coeff"][:ep_length], label=f"drag Ep{ep} Env{env_id}")
        print((env_aero["lift_coeff"][:ep_length]))

plt.title("Lift Coefficients")
plt.xlabel("Timestep")
plt.ylabel("Lift")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(latest_dir, "aero_coeffs.png"))"""
#plt.show()

# --- Settings ---
max_timestep = 3000  # truncate or pad to this length
lift_all = []
drag_all = []

# --- Aggregate Data ---
for ep in df_aero["episode"].unique()[:num_episodes]:
    ep_aero = df_aero[df_aero["episode"] == ep]
    for env_id in ep_aero["env_id"].unique():
        env_aero = ep_aero[ep_aero["env_id"] == env_id]
        
        lift = env_aero["lift_coeff"].values[:max_timestep]
        drag = env_aero["drag_coeff"].values[:max_timestep]

        # Pad with NaNs if shorter than max_timestep
        if len(lift) < max_timestep:
            pad = max_timestep - len(lift)
            lift = np.pad(lift, (0, pad), constant_values=np.nan)
            drag = np.pad(drag, (0, pad), constant_values=np.nan)

        lift_all.append(lift)
        drag_all.append(drag)

lift_all = np.stack(lift_all)  # shape: (num_trajs, max_timestep)
drag_all = np.stack(drag_all)

# --- Compute stats ---
lift_mean = np.nanmean(lift_all, axis=0)
lift_std = np.nanstd(lift_all, axis=0)
drag_mean = np.nanmean(drag_all, axis=0)
drag_std = np.nanstd(drag_all, axis=0)

# --- Plot ---
plt.figure(figsize=(12, 4))
timesteps = np.arange(max_timestep)

# Lift
plt.plot(timesteps, lift_mean, label="Lift Mean", color='blue')
plt.fill_between(timesteps, lift_mean - lift_std, lift_mean + lift_std, alpha=0.3, color='blue', label="Lift ±1 std")

# Drag
plt.plot(timesteps, drag_mean, label="Drag Mean", color='red')
plt.fill_between(timesteps, drag_mean - drag_std, drag_mean + drag_std, alpha=0.3, color='red', label="Drag ±1 std")

plt.title("Lift and Drag Coefficients (Mean ± Std)")
plt.xlabel("Timestep")
plt.ylabel("Coefficient")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(latest_dir, "aero_coeffs_mean_std.png"))


# --- Other Metrics ---
max_timestep = 3000  # truncate or pad to this length
energy_all = []
rew_progress_all = []
rew_backward_all = []
rew_energy_all = []
total_reward_all = []
distance_all = []
bearing_all = []

# --- Aggregate Data ---
for ep in df_other_metrics["episode"].unique()[:num_episodes]:
    ep_other_metrics = df_other_metrics[df_other_metrics["episode"] == ep]
    ep_length = ep_length_list[ep]
    for env_id in ep_other_metrics["env_id"].unique():
        env_other_metrics = ep_other_metrics[ep_other_metrics["env_id"] == env_id]
        
        energy = env_other_metrics["energy"].values[:max_timestep]
        reward_progress = env_other_metrics["reward_progress"].values[:max_timestep]
        reward_backward = env_other_metrics["reward_backward"].values[:max_timestep]
        reward_energy = env_other_metrics["reward_energy"].values[:max_timestep]
        total_reward = env_other_metrics["total_reward"].values[:max_timestep]
        bearing = env_other_metrics["bearing"].values[:max_timestep]
        distance = env_other_metrics["distance"].values[:max_timestep]

        """# Pad with NaNs if shorter than max_timestep
        if len(lift) < max_timestep:
            pad = max_timestep - len(lift)
            lift = np.pad(lift, (0, pad), constant_values=np.nan)
            drag = np.pad(drag, (0, pad), constant_values=np.nan)"""

        energy_all.append(energy)
        rew_progress_all.append(reward_progress)
        rew_backward_all.append(reward_backward)
        rew_energy_all.append(reward_energy)
        total_reward_all.append(total_reward)
        bearing_all.append(bearing)
        distance_all.append(distance)

        """lift_all.append(lift)
        drag_all.append(drag)"""


energy_all = np.stack(energy_all)
rew_progress_all = np.stack(rew_progress_all)
rew_backward_all = np.stack(rew_backward_all)
rew_energy_all = np.stack(rew_energy_all)
total_reward_all = np.stack(total_reward_all)
bearing_all = np.stack(bearing_all)
distance_all = np.stack(distance_all)

# --- Compute stats for other metrics ---
energy_mean = np.nanmean(energy_all, axis=0)
energy_std = np.nanstd(energy_all, axis=0)
rew_progress_mean = np.nanmean(rew_progress_all, axis=0)
rew_progress_std = np.nanstd(rew_progress_all, axis=0)      
rew_backward_mean = np.nanmean(rew_backward_all, axis=0)
rew_backward_std = np.nanstd(rew_backward_all, axis=0)
rew_energy_mean = np.nanmean(rew_energy_all, axis=0)
rew_energy_std = np.nanstd(rew_energy_all, axis=0)
total_reward_mean = np.nanmean(total_reward_all, axis=0)
total_reward_std = np.nanstd(total_reward_all, axis=0)
bearing_mean = np.nanmean(bearing_all, axis=0)
bearing_std = np.nanstd(bearing_all, axis=0)
distance_mean = np.nanmean(distance_all, axis=0)
distance_std = np.nanstd(distance_all, axis=0)

# --- Plot Other Metrics ---
plt.figure(figsize=(12, 4))
timesteps = np.arange(max_timestep-1) 

plt.subplot(2, 2, 1)
# Energy    
plt.plot(np.arange(len(energy_mean)) , energy_mean, label="Energy Mean", color='green')
plt.fill_between(np.arange(len(energy_mean)), energy_mean - energy_std, energy_mean + energy_std, alpha=0.3, color='green', label="Energy ±1 std")
plt.title("Energy consumption (Mean ± Std)")
plt.xlabel("Timestep")
plt.ylabel("Value")
plt.legend()

# Reward Progress
plt.subplot(2, 2, 2)
plt.plot(np.arange(len(rew_progress_mean)), rew_progress_mean, label="Reward Progress Mean", color='orange')
plt.fill_between(np.arange(len(rew_progress_mean)), rew_progress_mean - rew_progress_std, rew_progress_mean + rew_progress_std, alpha=0.3, color='orange', label="Reward Progress ±1 std")
plt.title("Reward progress (Mean ± Std)")
plt.xlabel("Timestep")
plt.ylabel("Value")
plt.legend()

# Reward Backward
plt.subplot(2, 2, 3)
plt.plot(np.arange(len(rew_backward_mean)), rew_backward_mean, label="Reward Backward Mean", color='purple')
plt.fill_between(np.arange(len(rew_backward_mean)), rew_backward_mean - rew_backward_std, rew_backward_mean + rew_backward_std, alpha=0.3, color='purple', label="Reward Backward ±1 std")
plt.title("Reward backward (Mean ± Std)")
plt.xlabel("Timestep")
plt.ylabel("Value")
plt.legend()

# Reward Energy
plt.subplot(2, 2, 4)
plt.plot(np.arange(len(rew_energy_mean)), rew_energy_mean, label="Reward Energy Mean", color='brown')
plt.fill_between(np.arange(len(rew_energy_mean)), rew_energy_mean - rew_energy_std, rew_energy_mean + rew_energy_std, alpha=0.3, color='brown', label="Reward Energy ±1 std")
plt.title("Reward Energy (Mean ± Std)")
plt.xlabel("Timestep")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(latest_dir, "other_metrics_mean_std.png"))

plt.figure()
# Total Reward  
plt.plot(np.arange(len(total_reward_mean)), total_reward_mean, label="Total Reward Mean", color='cyan')
plt.fill_between(np.arange(len(total_reward_mean)), total_reward_mean - total_reward_std, total_reward_mean + total_reward_std, alpha=0.3, color='cyan', label="Total Reward ±1 std")
plt.title("Total Reward (Mean ± Std)")
plt.xlabel("Timestep")
plt.ylabel("Value")
plt.legend()    
plt.savefig(os.path.join(latest_dir, "total_reward_std.png"))


plt.figure()
plt.subplot(2, 1, 1)
for ep_idx in range(len(bearing_all)):
    
    # Bearing
    plt.plot(np.arange(len(bearing_all[ep_idx])), np.rad2deg(bearing_all[ep_idx]), label=f"Bearing {ep_idx}")
    #plt.fill_between(np.arange(len(bearing_mean)), bearing_mean - bearing_std, bearing_mean + bearing_std, alpha=0.3, color='magenta', label="Bearing ±1 std")
    plt.title("Bearing")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.legend()    

plt.subplot(2, 1, 2)
for ep_idx in range(len(distance_all)):
    # Distance
    plt.plot(np.arange(len(distance_all[ep_idx])), distance_all[ep_idx], label=f"Distance {ep_idx}")
    plt.title("Distance")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()

plt.savefig(os.path.join(latest_dir, "bearing_distance.png")) 


"""plt.figure()
plt.subplot(2, 1, 1)
# Bearing
plt.plot(np.arange(len(bearing_mean)), bearing_mean, label="Bearing Mean", color='magenta')
plt.fill_between(np.arange(len(bearing_mean)), bearing_mean - bearing_std, bearing_mean + bearing_std, alpha=0.3, color='magenta', label="Bearing ±1 std")
plt.title("Bearing (Mean ± Std)")
plt.xlabel("Timestep")
plt.ylabel("Value")
plt.legend()    

plt.subplot(2, 1, 2)
# Distance
plt.plot(np.arange(len(distance_mean)), distance_mean, label="Distance Mean", color='teal')
plt.fill_between(np.arange(len(distance_mean)), distance_mean - distance_std, distance_mean + distance_std, alpha=0.3, color='teal', label="Distance ±1 std")
plt.title("Distance (Mean ± Std)")
plt.xlabel("Timestep")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(latest_dir, "bearing_distance_std.png"))  """ 


