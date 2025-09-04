import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

# --- Argument Parsing ---
parser = argparse.ArgumentParser()
parser.add_argument("--dir_path", type=str, default=None, help="directory to plot")

args = parser.parse_args()

# --- Find the latest log directory ---
eval_root = "runs/bip_rlgames"

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
df_ep_length = pd.read_csv(os.path.join(latest_dir, "ep_length.csv"))
df_vel_context = pd.read_csv(os.path.join(latest_dir, "acord.csv"))

# --- Get episode_length ---

ep_length_list = df_ep_length["ep_length"].tolist()
num_episodes = len(ep_length_list)

print(f"episode_length: {ep_length_list}")
# --- Plot Speed vs Context ---
plt.figure()
plt.subplot(2, 1, 1)
for ep in df_vel_context["episode"].unique()[:num_episodes]:  
    ep_length = ep_length_list[ep]
    ep_vel_context = df_vel_context[df_vel_context["episode"] == ep]
    for env_id in ep_vel_context["env_id"].unique():
        env_vel_context = ep_vel_context[ep_vel_context["env_id"] == env_id]
        plt.scatter(env_vel_context["lin_vel_x"], env_vel_context["vel_context"], alpha=0.1)

plt.title("Velocity vs Context Trajectories")
plt.xlabel("lin_vel_x")
plt.ylabel("context ")
plt.grid()  
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)
for ep in df_vel_context["episode"].unique()[:num_episodes]:  
    ep_length = ep_length_list[ep]
    ep_vel_context = df_vel_context[df_vel_context["episode"] == ep]
    for env_id in ep_vel_context["env_id"].unique():
        env_vel_context = ep_vel_context[ep_vel_context["env_id"] == env_id]
        plt.scatter(env_vel_context["angle"], env_vel_context["angle_context"], alpha=0.1)

plt.title("Velocity vs Context Trajectories")
plt.xlabel("angle")
plt.ylabel("context ")
plt.grid()  
plt.legend(loc='upper right')
plt.savefig(os.path.join(latest_dir, "states_vs_contexts.png"))

# --- Plot Speed ---
plt.figure()
plt.subplot(2, 1, 1)
for ep in df_vel_context["episode"].unique()[:num_episodes]:  
    ep_length = ep_length_list[ep]
    ep_vel_context = df_vel_context[df_vel_context["episode"] == ep]
    for env_id in ep_vel_context["env_id"].unique():
        env_vel_context = ep_vel_context[ep_vel_context["env_id"] == env_id]
        plt.plot(np.arange(len(env_vel_context["lin_vel_x"])), env_vel_context["lin_vel_x"], alpha=0.5)

plt.title("Velocity")
plt.xlabel("time step")
plt.ylabel("lin_vel_x") 
plt.grid()  
plt.legend(loc='upper right')

# --- Plot context ---
plt.subplot(2, 1, 2)
for ep in df_vel_context["episode"].unique()[:num_episodes]:  
    ep_length = ep_length_list[ep]
    ep_vel_context = df_vel_context[df_vel_context["episode"] == ep]
    for env_id in ep_vel_context["env_id"].unique():
        env_vel_context = ep_vel_context[ep_vel_context["env_id"] == env_id]
        plt.plot(np.arange(len(env_vel_context["vel_context"])), env_vel_context["vel_context"], alpha=0.5)

plt.title("Context")
plt.xlabel("time step")
plt.ylabel("context") 
plt.grid()  
plt.legend(loc='upper right')
plt.savefig(os.path.join(latest_dir, "vel_and_context.png"))

# --- Plot Angle ---
plt.figure()
plt.subplot(2, 1, 1)
for ep in df_vel_context["episode"].unique()[:num_episodes]:  
    ep_length = ep_length_list[ep]
    ep_vel_context = df_vel_context[df_vel_context["episode"] == ep]
    for env_id in ep_vel_context["env_id"].unique():
        env_vel_context = ep_vel_context[ep_vel_context["env_id"] == env_id]
        plt.plot(np.arange(len(env_vel_context["angle"])), env_vel_context["angle"], alpha=0.5)

plt.title("Angle")
plt.xlabel("time step")
plt.ylabel("angle") 
plt.grid()  
plt.legend(loc='upper right')

# --- Plot context ---
plt.subplot(2, 1, 2)
for ep in df_vel_context["episode"].unique()[:num_episodes]:  
    ep_length = ep_length_list[ep]
    ep_vel_context = df_vel_context[df_vel_context["episode"] == ep]
    for env_id in ep_vel_context["env_id"].unique():
        env_vel_context = ep_vel_context[ep_vel_context["env_id"] == env_id]
        plt.plot(np.arange(len(env_vel_context["angle_context"])), env_vel_context["angle_context"], alpha=0.5)

plt.title("Context")
plt.xlabel("time step")
plt.ylabel("context") 
plt.grid()  
plt.legend(loc='upper right')
plt.savefig(os.path.join(latest_dir, "angle_and_context.png"))
        

