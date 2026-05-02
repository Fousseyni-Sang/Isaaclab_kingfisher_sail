#from omni.isaac.lab.app import AppLauncher

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn

import argparse


parser = argparse.ArgumentParser(description="Launch training of arbitrary number of LL RL agent from RL-Games.")

parser.add_argument("--model_id_list", type=str, default=None, help="comma-separated list of low level model ids to train "
"(e.g. '000,001,002'). Overrides --num_models if provided.")

class AutoEncoder(nn.Module):
  def __init__(self, indim, outdim, hdim, device="cuda:1") -> None:
    super(AutoEncoder, self).__init__()
    self.device = device

    self.encoder = nn.Sequential(
        nn.Linear(indim, hdim),
        nn.ReLU(),
        nn.Linear(hdim, hdim),
        nn.ReLU(),
        nn.Linear(hdim, outdim),
    )

    self.decoder = nn.Sequential(
        nn.Linear(outdim, hdim),
        nn.ReLU(),
        nn.Linear(hdim, hdim),
        nn.ReLU(),
        nn.Linear(hdim, indim),
        nn.Sigmoid()
    )

    self.loss_fn = nn.MSELoss()

    # IMPORTANT: move model to device
    self.to(self.device)

  def forward(self, x):
    x = self.decoder(self.encoder(x))

    return x


  def load(self, path:str):
    self.eval()
    checkpoint = torch.load(path, map_location=self.device, weights_only=True)
    self.encoder.load_state_dict(checkpoint["encoder"])
    self.decoder.load_state_dict(checkpoint["decoder"])


args_cli = parser.parse_args()
device = "cpu"
autoencoder = AutoEncoder(indim=121, outdim=4, hdim=128, device=device)

autoencoder.load("config/model.pt")

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

    feas_tensor = torch.tensor(feas, device=device, dtype=torch.float32)
    predicted_feas_tensor = autoencoder(feas_tensor.reshape(1, -1)) # shape (1, 4)

    #print(len(feas))
    if len(feas)==121:
        feas = feas.reshape((11, 11))
        predicted_feas_tensor = predicted_feas_tensor.cpu().detach().numpy().reshape((11, 11))
        
    else:
        feas = feas.reshape((11, 11, 11))
        predicted_feas_tensor = predicted_feas_tensor.cpu().detach().numpy().reshape((11, 11, 11))

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
        plt.figure(figsize=(15, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(feas, cmap="gray", origin="lower")
        plt.title(f"original Feasibility Map ")
        plt.xlabel("w index")
        plt.ylabel("vx index")

        plt.subplot(1, 2, 2)
        plt.imshow(predicted_feas_tensor, cmap="gray", origin="lower")
        plt.title(f"reconstructed Feasibility Map")
        plt.xlabel("w index")
        plt.ylabel("vx index")

    plt.savefig(f"eval_ll/rl_games/model_0{k}{j%10}/feasibility_shape.png")
    
    