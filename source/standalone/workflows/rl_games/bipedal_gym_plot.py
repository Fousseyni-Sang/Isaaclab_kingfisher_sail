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

def plot_state_vs(name:str, name_context:str="", vs_context=False):
    for ep in df_vel_context["episode"].unique()[:num_episodes]:  
        ep_length = ep_length_list[ep]
        ep_vel_context = df_vel_context[df_vel_context["episode"] == ep]
        for env_id in ep_vel_context["env_id"].unique():
            env_vel_context = ep_vel_context[ep_vel_context["env_id"] == env_id]
            if vs_context:
                plt.scatter(env_vel_context[f"{name}"], env_vel_context[f"{name_context}"], alpha=0.1)
            else:
                plt.plot(np.arange(len(env_vel_context[f"{name}"])), env_vel_context[f"{name}"])

    plt.title(f"{name} vs {name_context}") if vs_context else plt.title(f"{name}")
    plt.xlabel(f"{name if vs_context else 'time_step'}")
    plt.ylabel(f"{name_context}")
    plt.grid()  
    plt.legend(loc='upper right')

plt.figure()
plt.subplot(2, 2, 1)
plot_state_vs("lin_vel_x", "vel_context", True)

plt.subplot(2, 2, 2)
plot_state_vs("hull_angle", "hull_angle_context", True)

plt.subplot(2, 2, 3)
plot_state_vs("joint_pos", "joint_pos_context", True)
plt.savefig(os.path.join(latest_dir, "states_vs_contexts.png"))

# --- Plot Speed ---
plt.figure()
plt.subplot(2, 1, 1)
plot_state_vs("lin_vel_x")

# --- Plot context ---
plt.subplot(2, 1, 2)
plot_state_vs("vel_context")
plt.savefig(os.path.join(latest_dir, "vel_and_context.png"))

# --- Plot hull_Angle ---
plt.figure()
plt.subplot(2, 1, 1)
plot_state_vs("hull_angle")

# --- Plot context ---
plt.subplot(2, 1, 2)
plot_state_vs("hull_angle_context")
plt.savefig(os.path.join(latest_dir, "hull_angle_and_context.png"))
        
# --- Plot joint_pos ---
plt.figure()
plt.subplot(2, 1, 1)
plot_state_vs("joint_pos")

# --- Plot context ---
plt.subplot(2, 1, 2)
plot_state_vs("joint_pos_context")
plt.savefig(os.path.join(latest_dir, "joint_pos_and_context.png"))
