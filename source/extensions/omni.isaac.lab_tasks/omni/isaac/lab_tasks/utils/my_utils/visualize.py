import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import argparse

parser = argparse.ArgumentParser(description="Launch training of arbitrary number of LL RL agent from RL-Games.")

parser.add_argument("--model_id_list", type=str, default=None, help="comma-separated list of low level model ids to train "
"(e.g. '000,001,002'). Overrides --num_models if provided.")


args_cli = parser.parse_args()

model_id_list = args_cli.model_id_list.split(",") if args_cli.model_id_list else None
if model_id_list and '-' in model_id_list[0]:
    # Handle range format like "000-004"
    start_id, end_id = model_id_list[0].split("-")
    model_id_list = [f"{i:03d}" for i in range(int(start_id), int(end_id) + 1)]
# Load feasibility map
k=-1
plt.figure(figsize=(20, 15))
for j in range(16):
    
    if j%10==0:
        k+=1

    if model_id_list and f"0{k}{j%10}" not in model_id_list:
        print(f"Skipping LL model 0{k}{j%10} as it's not in the provided model_id_list.")
        continue
    
    df = pd.read_csv(f"eval_ll/rl_games/model_0{k}{j%10}/feasibility_map.csv")
    feas = df["feasible"].to_numpy()
    #print(len(feas))
    if len(feas)==121:
        feas = feas.reshape((11, 11))
        
    else:
        feas = feas.reshape((11, 11, 11))
        
        

    """for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(feas[:, i, :], cmap="gray", origin="lower")
        plt.title(f"Feasibility Map at vy index {i}")
        plt.xlabel("vy index")
        plt.ylabel("w index")
        #print(feas.nonzero())
        #plt.colorbar(label="Feasible")
        # plt.savefig(f"eval_ll/rl_games/model_0{k}{j%10}/feasibility_shape.png")
    """
    if feas.ndim == 3:
        plt.imshow(feas[:, 5, :], cmap="gray", origin="lower")
    else:
        plt.imshow(feas, cmap="gray", origin="lower")
    plt.title(f"Feasibility Map at vy index {5}")
    plt.xlabel("w index")
    plt.ylabel("vx index")
    plt.savefig(f"eval_ll/rl_games/model_0{k}{j%10}/feasibility_shape.png")
    
    