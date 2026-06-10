import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import seaborn as sns
import scipy
import statsmodels.api as sm

# --- Argument Parsing ---
parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default=None, help="directory to plot")

args = parser.parse_args()

# --- Find the latest log directory ---
eval_root = "./"

file_name = args.file_name
if file_name is None:
    raise Exception(f"this path: {file_name} is not a file name")

file_path = os.path.join(eval_root, file_name)
if not os.path.exists(file_path):
    raise Exception(f"this file: {file_path} is does not exist")

print("=========================================================================================================")
print(f"root directory: {eval_root}")

new_dir = os.path.join(eval_root, "workshop")
os.makedirs(new_dir, exist_ok=True)

print(f"Loading data from: {new_dir}")

# --- Load CSVs ---
df_turtlebot_data = pd.read_csv(file_path)

# --- Get episode_length ---

pd.options.display.max_rows = 100
df = pd.read_csv("csv_real_turtlebot.csv")
odom = df[["__time", "/odom/twist/twist/linear/x", "/odom/twist/twist/angular/z"]]
cmd_vel = df[["__time", "/commands/velocity/angular/z", "/commands/velocity/linear/x"]]
energy = df[["__time", "/energy_context/data"]] #.dropna(how="all")

#
odom.dropna(inplace=True)
cmd_vel.dropna(inplace=True)
energy.dropna(inplace=True)


font_size = 16

def gaussian_kernel_smoother(x, y, eval_x=None, bandwidth=0.01):
    """
        This function fit a gaussian smoothed version 
    """
    if eval_x is None:
        eval_x = np.linspace(x.min(), x.max(), 500)

    f = np.zeros_like(eval_x)
    g = np.zeros_like(eval_x)

    for i, x0 in enumerate(eval_x):
        w = np.exp(-0.5 * ((x - x0) / bandwidth)**2)
        w /= w.sum()

        f[i] = np.sum(w * y)
        g[i] = np.sum(w * (y**2))

    var = g - f**2
    std = np.sqrt(np.maximum(var, 0))

    return eval_x, f, std

# --- Plot Speed vs Context ---
def plot_state_vs(name:str, name_context:str="", vs_context=False):
    
    y = env_vel_context[f"{name}"].to_numpy()
    
    if vs_context:
        x = env_vel_context[f"{name_context}"].to_numpy()
        
        t = np.arange(len(x))
        rho = scipy.stats.spearmanr(x, y).correlation
        
        plt.text(0.2, 0.92, f"Spearman ρ = {rho:.2f}", transform=plt.gca().transAxes,
        fontsize=12, color="red", ha="left", va="top")
        
        # Compute Gaussian-kernel mean + std
        eval_x, mean_y, std_y = gaussian_kernel_smoother(x, y, bandwidth=0.08)

        # Scatter
        plt.scatter(x, y, s=3, alpha=0.1, color='blue')

        # Uncertainty band
        plt.fill_between(eval_x, mean_y - std_y, mean_y + std_y,
                        color='red', alpha=0.1, label='Gaussian ±1 std')

        # Mean trend
        plt.plot(eval_x, mean_y, color='red', linewidth=1)

    else:
        
        plt.plot(np.arange(len(env_vel_context[f"{name}"])), y)

        #plt.title(f"{name} vs {name_context}") if vs_context else plt.title(f"{name}")
        plt.xlabel(f"{name_context if vs_context else 'time_step'}")
        plt.ylabel(f"{name}")
        #plt.axis('equal')
        #plt.grid()  
        plt.legend(loc='upper right')



plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)

plot_state_vs("lin_vel_x", "vel_context", True)
plt.title(f"Monotonicity of linear velocity vs context", fontsize=font_size)
plt.xlabel(f"linear velocity context", fontsize=font_size)
plt.ylabel(f"linear velocity", fontsize=font_size)
#plt.legend(fontsize=16)

plt.subplot(1, 2, 2)
plot_state_vs("hull_angle", "hull_angle_context", True)
plt.title(f"Monotonicity of hull angle vs context", fontsize=font_size)
plt.xlabel(f"hull angle context", fontsize=font_size)
plt.ylabel(f"hull angle (rad)", fontsize=font_size)
#plt.legend(fontsize=16)
"""plt.subplot(2, 2, 3)
plot_state_vs("joint_pos", "joint_pos_context", True)"""
plt.savefig(os.path.join(new_dir, "states_vs_contexts.png"))

# --- Plot Speed ---
plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plot_state_vs("lin_vel_x")

# --- Plot context ---
plt.subplot(2, 1, 2)
plot_state_vs("vel_context")
plt.savefig(os.path.join(new_dir, "vel_and_context.png"))


# --- Plot hull_Angle ---
plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plot_state_vs("hull_angle")

# --- Plot context ---
plt.subplot(2, 1, 2)
plot_state_vs("hull_angle_context")
plt.savefig(os.path.join(new_dir, "hull_angle_and_context.png"))
        
"""# --- Plot joint_pos ---
plt.figure()
plt.subplot(2, 1, 1)
plot_state_vs("joint_pos")

# --- Plot context ---
plt.subplot(2, 1, 2)
plot_state_vs("joint_pos_context")
plt.savefig(os.path.join(new_dir, "joint_pos_and_context.png"))"""