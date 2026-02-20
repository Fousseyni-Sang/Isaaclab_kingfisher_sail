import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load feasibility map
k=-1
plt.figure(figsize=(20, 15))
for j in range(16):
    
    if j%10==0:
        k+=1
    df = pd.read_csv(f"eval_ll/rl_games/model_0{k}{j%10}/feasibility_map.csv")
    feas = df["feasible"].to_numpy()
    
    feas = feas.reshape((10, 10, 10))

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
    
    plt.imshow(feas[:, 5, :], cmap="gray", origin="lower")
    plt.title(f"Feasibility Map at vy index {5}")
    plt.xlabel("w index")
    plt.ylabel("vx index")
    plt.savefig(f"eval_ll/rl_games/model_0{k}{j%10}/feasibility_shape.png")
    #print(feas.nonzero())
    #plt.colorbar(label="Feasible")

    #plt.savefig(f"eval_ll/rl_games/model_0{k}{j%10}/feasibility_shape.png")
    #plt.savefig(f"eval_ll/rl_games/model_0{k}{j%10}/feasibility_shape.png")

    