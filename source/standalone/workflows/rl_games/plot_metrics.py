import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

# --- Argument Parsing ---
parser = argparse.ArgumentParser()
parser.add_argument("--sac", type=bool, default=False, help="plot SAC metrics")
parser.add_argument("--dir_path", type=str, default=None, help="directory to plot")
parser.add_argument("--wind_direction", type=float, default=None, help="direction of the true wind in degree.")

args = parser.parse_args()

# --- Find the latest log directory ---
if args.sac:
    eval_root = "eval_logs/sac" 
else:
    eval_root = "eval_logs/rl_games"

subdirs = sorted(
    [os.path.join(eval_root, d) for d in os.listdir(eval_root) if os.path.isdir(os.path.join(eval_root, d))],
    key=lambda x: x.split("/")[-1],  # assumes timestamp folder names
    reverse=True
)

latest_dir = subdirs[0] if args.dir_path is None else args.dir_path


print("=========================================================================================================")
print(f"root directory: {eval_root}")
print(f"Loading data from: {latest_dir}")

# --- Load CSVs ---
df_traj = pd.read_csv(os.path.join(latest_dir, "trajectories.csv"))
df_aero = pd.read_csv(os.path.join(latest_dir, "aero_data.csv"))
df_goals = pd.read_csv(os.path.join(latest_dir, "goals.csv"))
df_metrics = pd.read_csv(os.path.join(latest_dir, "metrics.csv"))
df_ep_length = pd.read_csv(os.path.join(latest_dir, "ep_length.csv"))
df_other_metrics = pd.read_csv(os.path.join(latest_dir, "other_metrics.csv"))
df_actions = pd.read_csv(os.path.join(latest_dir, "actions.csv"))
df_acord = pd.read_csv(os.path.join(latest_dir, "acord_data.csv"))

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
num_episodes = len(ep_length_list)
"""for ep in df_ep_length["episode"].unique()[:3]:
    ep_length = df_ep_length[df_ep_length["episode"]==ep]
    
    ep_length_list.append(ep_length["ep_length"])
"""
print(f"episode_length: {ep_length_list}")
# --- Plot Trajectories ---
plt.figure()
plt.subplot(2, 1, 1)
for ep in df_traj["episode"].unique()[:num_episodes]:  # Limit to first 3 episodes
    ep_length = ep_length_list[ep]
    ep_traj = df_traj[df_traj["episode"] == ep]
    for env_id in ep_traj["env_id"].unique():
        env_traj = ep_traj[ep_traj["env_id"] == env_id]
        plt.plot(env_traj["x"][:ep_length], env_traj["y"][:ep_length])
        goal = df_goals[(df_goals["episode"] == ep) & (df_goals["env_id"] == env_id)]
        plt.scatter(goal["goal_x"], goal["goal_y"], marker="x")
        #print(env_traj["x"].shape, env_traj["y"].shape)

if args.wind_direction is None:
    raise ValueError("Please provide wind_direction in seconds.")
wind_direc = (args.wind_direction * np.pi / 180)  # Example wind direction in radians
wind_speed = 5  # Example wind speed
# Plot wind vector
wind_x = wind_speed * np.cos(wind_direc)
wind_y = wind_speed * np.sin(wind_direc)
plt.quiver(110, 0, wind_x, wind_y, angles='xy', scale_units='xy', scale=0.5, color='blue', label='Wind Vector')
plt.title("Trajectories with Goals")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.legend()
plt.tight_layout()


plt.subplot(2, 1, 2)
for ep in df_acord["episode"].unique()[:num_episodes]:  # Limit to first 3 episodes
    ep_length = ep_length_list[ep]
    ep_context = df_acord[df_acord["episode"] == ep]
    for env_id in ep_context["env_id"].unique():
        env_context = ep_context[ep_context["env_id"] == env_id]
        context = env_context["actual_context"][:ep_length]
        plt.plot(np.arange(len(context)), context)
        

plt.title("energy context")
plt.legend()

plt.savefig(os.path.join(latest_dir, "trajectory_plot.png"))


# --- Plot acord predictions logs ---
plt.figure()
plt.subplot(2, 2, 1)
for ep in df_acord["episode"].unique()[:num_episodes]:  # Limit to first 3 episodes
    ep_length = ep_length_list[ep]
    ep_acord = df_acord[df_acord["episode"] == ep]
    ep_context = df_goals[df_goals["episode"] == ep]
    for env_id in ep_acord["env_id"].unique():
        context = df_goals[(df_goals["episode"] == ep) & (df_goals["env_id"] == env_id)]
        context_all = []
        env_traj = ep_acord[ep_acord["env_id"] == env_id]
        plt.plot(np.arange(len(ep_acord["predicted_context"][:ep_length])), ep_acord["predicted_context"][:ep_length], linestyle = 'dotted', label="pred")
        plt.plot(np.arange(len(ep_acord["actual_context"][:ep_length])), ep_acord["actual_context"][:ep_length], linestyle = 'dotted', label="act")

plt.title("predicted context")
plt.legend()
plt.tight_layout()

plt.subplot(2, 2, 2)
for ep in df_acord["episode"].unique()[:num_episodes]:  # Limit to first 3 episodes
    ep_length = ep_length_list[ep]
    ep_acord = df_acord[df_acord["episode"] == ep]
    ep_context = df_goals[df_goals["episode"] == ep]
    #context = df_goals[(df_goals["episode"] == ep) & (df_goals["env_id"] == env_id)]
    for env_id in ep_acord["env_id"].unique():
        env_traj = ep_acord[ep_acord["env_id"] == env_id]
        plt.plot(np.arange(len(ep_acord["mean"][:ep_length])), ep_acord["mean"][:ep_length])
        #plt.scatter(context["episode"], context["energy_context"], marker="x")

plt.title("mean predicted context")
plt.legend()
plt.tight_layout()

plt.subplot(2, 2, 3)
for ep in df_acord["episode"].unique()[:num_episodes]:  # Limit to first 3 episodes
    ep_length = ep_length_list[ep]
    ep_acord = df_acord[df_acord["episode"] == ep]
    for env_id in ep_acord["env_id"].unique():
        env_traj = ep_acord[ep_acord["env_id"] == env_id]
        plt.scatter(np.arange(len(ep_acord["std"][:ep_length])), ep_acord["std"][:ep_length])

plt.title("std predicted context")
plt.legend()
plt.tight_layout()


# --- Prediction context error ---
plt.subplot(2, 2, 4)
predicted_error_all = []
max_timestep = int(max(ep_length_list))
for ep in df_acord["episode"].unique()[:num_episodes]:  # Limit to first 3 episodes
    ep_length = ep_length_list[ep]
    ep_acord = df_acord[df_acord["episode"] == ep]
    for env_id in ep_acord["env_id"].unique():
        env_traj = ep_acord[ep_acord["env_id"] == env_id]
        error = ep_acord["predicted_context"][:ep_length]-ep_acord["actual_context"][:ep_length]
        
        # Pad with NaNs if shorter than max_timestep
        if len(error) < max_timestep:
            pad = max_timestep - len(error)
            error = np.pad(error, (0, pad), constant_values=np.nan)

        
        """plt.scatter(np.arange(len(ep_acord["predicted_context"][:ep_length])), 
                 ep_acord["predicted_context"][:ep_length]-context["energy_context"].item())
"""
        predicted_error_all.append(error)  

predicted_error_all = np.stack(predicted_error_all)
   # --- Compute stats ---
error_mean = np.nanmean(predicted_error_all, axis=0)
error_std = np.nanstd(predicted_error_all, axis=0)
timesteps = np.arange(max_timestep)

plt.plot(timesteps, error_mean, label="error Mean", color='blue')
plt.fill_between(timesteps, error_mean - error_std, error_mean + error_std, alpha=0.3, color='blue', label="error ±1 std")

plt.title("predicted context error")
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(latest_dir, "acord_data.png"))    

# --- Plot Action 0 ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
for ep in df_actions["episode"].unique()[:num_episodes]:
    ep_length = ep_length_list[ep]
    ep_actions = df_actions[df_actions["episode"] == ep]
    
    for env_id in ep_actions["env_id"].unique():
        env_actions = ep_actions[ep_actions["env_id"] == env_id]
        plt.plot(
            np.arange(ep_length),
            env_actions["action_0"][:ep_length],
            label=f"action_0 Ep{ep} Env{env_id}"
        )

plt.title("Action 0")
plt.xlabel("Timestep")
plt.ylabel("Action 0 value")
plt.legend()
plt.tight_layout()

# --- Plot Action 1 ---
plt.subplot(1, 3, 2)
for ep in df_actions["episode"].unique()[:num_episodes]:
    ep_length = ep_length_list[ep]
    ep_actions = df_actions[df_actions["episode"] == ep]
    
    for env_id in ep_actions["env_id"].unique():
        env_actions = ep_actions[ep_actions["env_id"] == env_id]
        plt.plot(
            np.arange(ep_length),
            env_actions["action_1"][:ep_length],
            label=f"action_1 Ep{ep} Env{env_id}"
        )

plt.title("Action 1")
plt.xlabel("Timestep")
plt.ylabel("Action 1 value")
plt.legend()
plt.tight_layout()

# --- Plot Action 2 ---
plt.subplot(1, 3, 3)
for ep in df_actions["episode"].unique()[:num_episodes]:
    ep_length = ep_length_list[ep]
    ep_actions = df_actions[df_actions["episode"] == ep]
    
    for env_id in ep_actions["env_id"].unique():
        env_actions = ep_actions[ep_actions["env_id"] == env_id]
        plt.plot(
            np.arange(ep_length),
            env_actions["action_2"][:ep_length],
            label=f"action_2 Ep{ep} Env{env_id}"
        )

plt.title("Action 2")
plt.xlabel("Timestep")
plt.ylabel("Action 2 value")
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(latest_dir, "actions_plot.png"))
plt.close()

# Energy plot
# --- Energy Data ---

plt.figure()
for ep in df_other_metrics["episode"].unique()[:num_episodes]:
    ep_other_metrics = df_other_metrics[df_other_metrics["episode"] == ep]
    ep_length = ep_length_list[ep]
    ep_context = df_acord[df_acord["episode"] == ep]

    for env_id in ep_other_metrics["env_id"].unique():
        env_other_metrics = ep_other_metrics[ep_other_metrics["env_id"] == env_id]
        env_context = ep_context[ep_context["env_id"] == env_id]
        
        energy = env_other_metrics["energy"].values[:ep_length]
        context = env_context["actual_context"].values[:ep_length]

        plt.scatter(energy, context, label="energy", alpha=0.1)
        #plt.plot(np.arange(len(context)), 2*context, label="context")

plt.xlabel("time")
plt.ylabel("energy")
plt.title("energy and context")
plt.savefig(os.path.join(latest_dir, "energy_acord.png"))
plt.close()


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
dt = 1/60  # Assuming 60 FPS
decimation = 3  # Number physics step for a policy step
max_timestep = int(max(ep_length_list))  # truncate or pad to this length
print(f"Max Timestep: {max_timestep}")
lift_all = []
drag_all = []
aoa_all = []
app_wind_angle_all = []  
sail_angle_all = []

# --- Aggregate Data ---
for ep in df_aero["episode"].unique()[:num_episodes]:
    ep_aero = df_aero[df_aero["episode"] == ep]
    
    for env_id in ep_aero["env_id"].unique():
        env_aero = ep_aero[ep_aero["env_id"] == env_id]
        
        lift = env_aero["lift_coeff"].values[:max_timestep]
        drag = env_aero["drag_coeff"].values[:max_timestep]
        aoa = env_aero["aoa"].values[:max_timestep]  
        sail = env_aero["sail_angle"].values[:max_timestep] 
        app_wind_angle = env_aero["app_wind_angle"].values[:max_timestep]

        # Pad with NaNs if shorter than max_timestep
        if len(lift) < max_timestep:
            pad = max_timestep - len(lift)
            lift = np.pad(lift, (0, pad), constant_values=np.nan)
            drag = np.pad(drag, (0, pad), constant_values=np.nan)
            aoa = np.pad(aoa, (0, pad), constant_values=np.nan)
            app_wind_angle = np.pad(app_wind_angle, (0, pad), constant_values=np.nan)
            sail = np.pad(sail, (0, pad), constant_values=np.nan)

        lift_all.append(lift)
        drag_all.append(drag)
        aoa_all.append(aoa)
        app_wind_angle_all.append(app_wind_angle)
        sail_angle_all.append(sail)

lift_all = np.stack(lift_all)  # shape: (num_trajs, max_timestep)
drag_all = np.stack(drag_all)
aoa_all = np.stack(aoa_all)
app_wind_angle_all = np.stack(app_wind_angle_all)
sail_angle_all = np.stack(sail_angle_all)

# --- Compute stats ---
lift_mean = np.nanmean(lift_all, axis=0)
lift_std = np.nanstd(lift_all, axis=0)
drag_mean = np.nanmean(drag_all, axis=0)
drag_std = np.nanstd(drag_all, axis=0)
aoa_mean = np.nanmean(aoa_all, axis=0)
aoa_std = np.nanstd(aoa_all, axis=0)
app_wind_angle_mean = np.nanmean(app_wind_angle_all, axis=0)
app_wind_angle_std = np.nanstd(app_wind_angle_all, axis=0)
sail_angle_mean = np.nanmean(sail_angle_all, axis=0)
sail_angle_std = np.nanstd(sail_angle_all, axis=0)

# --- Plot ---
plt.figure(figsize=(12, 4))
timesteps = np.arange(max_timestep)
#plt.subplot(2, 1, 1)
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

plt.figure(figsize=(12, 4))
# Angle of Attack (aoa)
plt.plot(timesteps, aoa_mean, label="aoa Mean", color='orange')
plt.fill_between(timesteps, aoa_mean - aoa_std, aoa_mean + lift_std, alpha=0.3, color='orange', label="aoa ±1 std")

# Apparent Wind Angle
plt.plot(timesteps, app_wind_angle_mean, label="App Wind Angle Mean", color='purple')
plt.fill_between(timesteps, app_wind_angle_mean - app_wind_angle_std, app_wind_angle_mean + app_wind_angle_std, alpha=0.3, 
                 color='purple', label="App Wind Angle ±1 std")

# Sail Angle
plt.plot(timesteps, sail_angle_mean, label="Sail Angle Mean", color='green')
plt.fill_between(timesteps, sail_angle_mean - sail_angle_std, sail_angle_mean + sail_angle_std, alpha=0.3, color='green', label="Sail Angle ±1 std")    

plt.title("aero_angles (Mean ± Std)")
plt.xlabel("Timestep")
plt.ylabel("angle (°)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(latest_dir, "aero_angles.png"))


# --- Other Metrics ---
#max_timestep = 3000  # truncate or pad to this length
energy_all = []
rew_progress_all = []
rew_backward_all = []
rew_energy_all = []
total_reward_all = []
distance_all = []
bearing_all = []
aero_force_x_all = []
aero_force_y_all = []
thruster_force_1_x_all = []
thruster_force_2_x_all = []
max_aero_force_all = [] 
episode_energy_all = []
ratio_energy_usage_all = []
max_available_energy_all = []
reward_aero_all = []
reward_acord_all = []

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
        reward_aero = env_other_metrics["reward_aero"].values[:max_timestep]
        reward_acord = env_other_metrics["reward_acord"].values[:max_timestep]
        total_reward = env_other_metrics["total_reward"].values[:max_timestep]
        #print(total_reward)

        bearing = env_other_metrics["bearing"].values[:max_timestep]
        distance = env_other_metrics["distance"].values[:max_timestep]
        aero_force_x = env_other_metrics["aero_force_x"].values[:max_timestep]
        aero_force_y = env_other_metrics["aero_force_y"].values[:max_timestep]
        thruster_force_1_x = env_other_metrics["thruster_force_1_x"].values[:max_timestep]
        thruster_force_2_x = env_other_metrics["thruster_force_2_x"].values[:max_timestep]
        max_aero_force = env_other_metrics["max_aero_force"].values[:max_timestep]
        episode_energy = env_other_metrics["episode_energy"].values[:max_timestep]
        max_available_energy = env_other_metrics["max_available_energy"].values[:max_timestep]
        ratio_energy_usage = env_other_metrics["ratio_energy_usage"].values[:max_timestep]

        """# Pad with NaNs if shorter than max_timestep
        if len(lift) < max_timestep:
            pad = max_timestep - len(lift)
            lift = np.pad(lift, (0, pad), constant_values=np.nan)
            drag = np.pad(drag, (0, pad), constant_values=np.nan)"""

        energy_all.append(energy)
        rew_progress_all.append(reward_progress)
        rew_backward_all.append(reward_backward)
        rew_energy_all.append(reward_energy)
        reward_acord_all.append(reward_acord)
        reward_aero_all.append(reward_aero)
        total_reward_all.append(total_reward)
        bearing_all.append(bearing)
        distance_all.append(distance)
        aero_force_x_all.append(aero_force_x)
        aero_force_y_all.append(aero_force_y)
        thruster_force_1_x_all.append(thruster_force_1_x)
        thruster_force_2_x_all.append(thruster_force_2_x)
        max_aero_force_all.append(max_aero_force)
        episode_energy_all.append(episode_energy)
        max_available_energy_all.append(max_available_energy)
        ratio_energy_usage_all.append(ratio_energy_usage)

    
        """lift_all.append(lift)
        drag_all.append(drag)"""



energy_all = np.stack(energy_all)
rew_progress_all = np.stack(rew_progress_all)
rew_backward_all = np.stack(rew_backward_all)
rew_energy_all = np.stack(rew_energy_all)
reward_aero_all = np.stack(reward_aero_all)
reward_acord_all = np.stack(reward_acord_all)
total_reward_all = np.stack(total_reward_all)
bearing_all = np.stack(bearing_all)
distance_all = np.stack(distance_all)
aero_force_x_all = np.stack(aero_force_x_all)
aero_force_y_all = np.stack(aero_force_y_all)
thruster_force_1_x_all = np.stack(thruster_force_1_x_all)
thruster_force_2_x_all = np.stack(thruster_force_2_x_all)
max_aero_force_all = np.stack(max_aero_force_all)
episode_energy_all = np.stack(episode_energy_all)
max_available_energy_all = np.stack(max_available_energy_all)
ratio_energy_usage_all = np.stack(ratio_energy_usage_all)


# --- Compute stats for other metrics ---
energy_mean = np.nanmean(energy_all, axis=0)
energy_std = np.nanstd(energy_all, axis=0)
rew_progress_mean = np.nanmean(rew_progress_all, axis=0)
rew_progress_std = np.nanstd(rew_progress_all, axis=0)      
rew_backward_mean = np.nanmean(rew_backward_all, axis=0)
rew_backward_std = np.nanstd(rew_backward_all, axis=0)
rew_energy_mean = np.nanmean(rew_energy_all, axis=0)
rew_energy_std = np.nanstd(rew_energy_all, axis=0)
rew_aero_mean = np.nanmean(reward_aero_all, axis=0)
rew_aero_std = np.nanstd(reward_aero_all, axis=0)
rew_acord_mean = np.nanmean(reward_acord_all, axis=0)
rew_acord_std = np.nanstd(reward_acord_all, axis=0)
total_reward_mean = np.nanmean(total_reward_all, axis=0)
total_reward_std = np.nanstd(total_reward_all, axis=0)
bearing_mean = np.nanmean(bearing_all, axis=0)
bearing_std = np.nanstd(bearing_all, axis=0)
distance_mean = np.nanmean(distance_all, axis=0)
distance_std = np.nanstd(distance_all, axis=0)
aero_force_x_mean = np.nanmean(aero_force_x_all, axis=0)
aero_force_x_std = np.nanstd(aero_force_x_all, axis=0)
aero_force_y_mean = np.nanmean(aero_force_y_all, axis=0)
aero_force_y_std = np.nanstd(aero_force_y_all, axis=0)
thruster_force_1_x_mean = np.nanmean(thruster_force_1_x_all, axis=0)
thruster_force_1_x_std = np.nanstd(thruster_force_1_x_all, axis=0)
thruster_force_2_x_mean = np.nanmean(thruster_force_2_x_all, axis=0)
thruster_force_2_x_std = np.nanstd(thruster_force_2_x_all, axis=0)
max_aero_force_mean = np.nanmean(max_aero_force_all, axis=0)
max_aero_force_std = np.nanstd(max_aero_force_all, axis=0)
episode_energy_mean = np.nanmean(episode_energy_all, axis=0)
episode_energy_std = np.nanstd(episode_energy_all, axis=0)
max_available_energy_mean = np.nanmean(max_available_energy_all, axis=0)
max_available_energy_std = np.nanstd(max_available_energy_all, axis=0)
ratio_energy_usage_mean = np.nanmean(ratio_energy_usage_all, axis=0)
ratio_energy_usage_std = np.nanstd(ratio_energy_usage_all, axis=0)


# --- Plot Other Metrics ---
plt.figure(figsize=[30, 10])
timesteps = np.arange(max_timestep-1) 

plt.subplot(2, 3, 1)
# Energy    
plt.plot(np.arange(len(energy_mean)) , energy_mean, label="Energy Mean", color='green')
plt.fill_between(np.arange(len(energy_mean)), energy_mean - energy_std, energy_mean + energy_std, alpha=0.3, color='green', label="Energy ±1 std")
plt.title("Energy consumption (Mean ± Std)")
plt.xlabel("Timestep")
plt.ylabel("Value")
plt.legend()

# Reward Progress
plt.subplot(2, 3, 2)
plt.plot(np.arange(len(rew_progress_mean)), rew_progress_mean, label="Reward Progress Mean", color='orange')
plt.fill_between(np.arange(len(rew_progress_mean)), rew_progress_mean - rew_progress_std, rew_progress_mean + rew_progress_std, alpha=0.3, color='orange', label="Reward Progress ±1 std")
plt.title("Reward progress (Mean ± Std)")
plt.xlabel("Timestep")
plt.ylabel("Value")
plt.legend()

# Reward Backward
plt.subplot(2, 3, 3)
plt.plot(np.arange(len(rew_backward_mean)), rew_backward_mean, label="Reward Backward Mean", color='purple')
plt.fill_between(np.arange(len(rew_backward_mean)), rew_backward_mean - rew_backward_std, rew_backward_mean + rew_backward_std, alpha=0.3, color='purple', label="Reward Backward ±1 std")
plt.title("Reward backward (Mean ± Std)")
plt.xlabel("Timestep")
plt.ylabel("Value")
plt.legend()

# Reward Energy
plt.subplot(2, 3, 4)
plt.plot(np.arange(len(rew_energy_mean)), rew_energy_mean, label="Reward Energy Mean", color='brown')
plt.fill_between(np.arange(len(rew_energy_mean)), rew_energy_mean - rew_energy_std, rew_energy_mean + rew_energy_std, alpha=0.3, color='brown', label="Reward Energy ±1 std")
plt.title("Reward Energy (Mean ± Std)")
plt.xlabel("Timestep")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()

plt.subplot(2, 3, 5)
# Reward acord
plt.plot(np.arange(len(rew_acord_mean)) , rew_acord_mean, label="rew_acord_mean", color='green')
plt.fill_between(np.arange(len(rew_acord_mean)), rew_acord_mean - rew_acord_std, rew_acord_mean + rew_acord_std, alpha=0.3, color='green', label="rew_acord ±1 std")
plt.title("reward acord (Mean ± Std)")
plt.xlabel("Timestep")
plt.ylabel("Value")
plt.legend()

# Reward aero
plt.subplot(2, 3, 6)
plt.plot(np.arange(len(rew_aero_mean)), rew_aero_mean, label="Reward aero Mean", color='orange')
plt.fill_between(np.arange(len(rew_aero_mean)), rew_aero_mean - rew_aero_std, rew_aero_mean + rew_aero_std, alpha=0.3, color='orange', label="Reward Progress ±1 std")
plt.title("Reward aero (Mean ± Std)")
plt.xlabel("Timestep")
plt.ylabel("Value")
plt.legend()

plt.savefig(os.path.join(latest_dir, "other_metrics_mean_std.png"))

plt.figure()
# Total Reward  
plt.plot(np.arange(len(total_reward_mean)), total_reward_mean, label="Total Reward Mean", color='cyan')
plt.fill_between(np.arange(len(total_reward_mean)), total_reward_mean - total_reward_std, total_reward_mean + total_reward_std, alpha=0.3, color='cyan', label="Total Reward ±1 std")
plt.title("Total Reward (Mean ± Std)")
plt.xlabel("Timestep")
plt.ylabel("Value")
plt.ylim([-1, 1])
plt.legend()    
plt.savefig(os.path.join(latest_dir, "total_reward_std.png"))

plt.figure()
# Max Aero Force
plt.plot(np.arange(len(max_aero_force_mean)), max_aero_force_mean, label="Max Aero Force Mean", color='magenta')
plt.fill_between(np.arange(len(max_aero_force_mean)), max_aero_force_mean - max_aero_force_std, max_aero_force_mean + max_aero_force_std, alpha=0.3, color='magenta', label="Max Aero Force ±1 std")

plt.plot(np.arange(len(aero_force_x_mean)), aero_force_x_mean, label="Aero Force X Mean", color='blue')
plt.fill_between(np.arange(len(aero_force_x_mean)), aero_force_x_mean - aero_force_x_std, aero_force_x_mean + aero_force_x_std, alpha=0.3, color='blue', label="Aero Force X ±1 std")

plt.plot(np.arange(len(aero_force_y_mean)), aero_force_y_mean, label="Aero Force Y Mean", color='red')
plt.fill_between(np.arange(len(aero_force_y_mean)), aero_force_y_mean - aero_force_y_std, aero_force_y_mean + aero_force_y_std, alpha=0.3, color='red', label="Aero Force Y ±1 std")

plt.plot(np.arange(len(thruster_force_1_x_mean)), thruster_force_1_x_mean, label="Thruster Force 1 X Mean", color='green')
plt.fill_between(np.arange(len(thruster_force_1_x_mean)), thruster_force_1_x_mean - thruster_force_1_x_std, thruster_force_1_x_mean + thruster_force_1_x_std, alpha=0.3, color='green', label="Thruster Force 1 X ±1 std")

plt.plot(np.arange(len(thruster_force_2_x_mean)), thruster_force_2_x_mean, label="Thruster Force 2 X Mean", color='orange')
plt.fill_between(np.arange(len(thruster_force_2_x_mean)), thruster_force_2_x_mean - thruster_force_2_x_std, thruster_force_2_x_mean + thruster_force_2_x_std, alpha=0.3, color='orange', label="Thruster Force 2 X ±1 std")

plt.title("Forces (Mean ± Std)")
plt.xlabel("Timestep")
plt.ylabel("Value")
plt.legend()    
plt.savefig(os.path.join(latest_dir, "forces_std.png"))

plt.figure()
plt.subplot(2, 1, 1)
# Episode Energy
plt.plot(np.arange(len(episode_energy_mean)), episode_energy_mean, label="Episode Energy Mean", color='purple')
plt.fill_between(np.arange(len(episode_energy_mean)), episode_energy_mean - episode_energy_std, episode_energy_mean + episode_energy_std, alpha=0.3, color='purple', label="Episode Energy ±1 std")

plt.plot(np.arange(len(max_available_energy_mean)), max_available_energy_mean, label="Max Available Energy Mean", color='brown')
plt.fill_between(np.arange(len(max_available_energy_mean)), max_available_energy_mean - max_available_energy_std, max_available_energy_mean + max_available_energy_std, alpha=0.3, color='brown', label="Max Available Energy ±1 std")  

plt.title("Episode Energy (Mean ± Std)")
plt.xlabel("Timestep")
plt.ylabel("Value")
plt.legend()

plt.subplot(2, 1, 2)
# Ratio Energy Usage
plt.plot(np.arange(len(ratio_energy_usage_mean)), ratio_energy_usage_mean, label="Ratio Energy Usage Mean", color='cyan')
plt.fill_between(np.arange(len(ratio_energy_usage_mean)), ratio_energy_usage_mean - ratio_energy_usage_std, ratio_energy_usage_mean + ratio_energy_usage_std, alpha=0.3, color='cyan', label="Ratio Energy Usage ±1 std")       

plt.title("Ratio Energy Usage (Mean ± Std)")
plt.xlabel("Timestep")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(latest_dir, "energy_figure.png"))


plt.figure()
for ep_idx in range(len(bearing_all)):
    
    # Bearing
    plt.plot(np.arange(len(bearing_all[ep_idx])), np.rad2deg(bearing_all[ep_idx]), label=f"Bearing {ep_idx}")
    #plt.fill_between(np.arange(len(bearing_mean)), bearing_mean - bearing_std, bearing_mean + bearing_std, alpha=0.3, color='magenta', label="Bearing ±1 std")
    plt.title("Bearing")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.legend()    

plt.savefig(os.path.join(latest_dir, "bearing.png")) 

plt.figure(figsize=(12, 4))
for ep_idx in range(len(distance_all)):
    # Distance
    plt.plot(np.arange(len(distance_all[ep_idx])), distance_all[ep_idx], label=f"Distance {ep_idx}")
    plt.title("Distance")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()

plt.savefig(os.path.join(latest_dir, "distance.png"))  


