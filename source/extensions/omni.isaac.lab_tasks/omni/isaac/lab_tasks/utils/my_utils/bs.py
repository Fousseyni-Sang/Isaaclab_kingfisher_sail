import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load feasibility map
df = pd.read_csv("eval_ll/rl_games/model_005/feasibility_map.csv")
feas = df["feasible"].to_numpy()

# Reconstruct the grid
N1, N2, N3 = 10, 10, 10  # <-- replace with your actual grid sizes

scale_vx_vy = 0.1
max_target_lin_wrench =  1.5
min_target_lin_wrench =  -0.*max_target_lin_wrench
max_target_ang_wrench =  0.5
min_target_ang_wrench = - max_target_ang_wrench 


# === Correct scaling for asymmetric ranges === 
Vx_min = min_target_lin_wrench 
Vx_max = max_target_lin_wrench 

Vy_min = -scale_vx_vy * max_target_lin_wrench 
Vy_max = scale_vx_vy * max_target_lin_wrench 

W_min = -max_target_ang_wrench 
W_max = max_target_ang_wrench 


vx_vals = np.linspace(Vx_min, Vx_max, N1)
vy_vals = np.linspace(Vy_min, Vy_max, N2)
w_vals  = np.linspace(W_min,  W_max,  N3)



VX, VY, W = np.meshgrid(vx_vals, vy_vals, w_vals, indexing="ij")

# Flatten to match feasibility vector
VXf = VX.flatten()
VYf = VY.flatten()
Wf  = W.flatten()

for k in range(N3):
    slice_mask = np.zeros_like(feas, dtype=bool)
    for i in range(N1):
        for j in range(N2):
            idx = i*N2*N3 + j*N3 + k
            slice_mask[idx] = True

    grid = feas[slice_mask].reshape(N1, N2)

    plt.imshow(grid, cmap="Greens", origin="lower")
    plt.title(f"Feasibility Map at w index {k}")
    plt.xlabel("vy index")
    plt.ylabel("vx index")
    plt.colorbar(label="Feasible")
    

    plt.savefig(f"eval_ll/rl_games/model_005/feasibility_shape_{i}.png")
