import os
import pandas as pd
import matplotlib.pyplot as plt

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
        plt.scatter(goal["goal_x"], goal["goal_y"], color="red", marker="x")
        print(env_traj["x"].shape, env_traj["y"].shape)
plt.title("Trajectories with Goals")
plt.xlabel("x"); plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(latest_dir, "trajectory_plot.png"))

# --- Plot Aero Coefficients ---
plt.figure(figsize=(12, 4))
for ep in df_aero["episode"].unique()[:num_episodes]:
    ep_length = ep_length_list[ep]
    
    ep_aero = df_aero[df_aero["episode"] == ep]
    for env_id in ep_aero["env_id"].unique():
        env_aero = ep_aero[ep_aero["env_id"] == env_id]
        plt.plot(env_aero["lift_coeff"][:ep_length], label=f"Lift Ep{ep} Env{env_id}")
        plt.plot(env_aero["drag_coeff"][:ep_length], label=f"drag Ep{ep} Env{env_id}")
        print(env_aero["lift_coeff"][:ep_length])

plt.title("Lift Coefficients")
plt.xlabel("Timestep")
plt.ylabel("Lift")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(latest_dir, "aero_coeffs.png"))
#plt.show()
