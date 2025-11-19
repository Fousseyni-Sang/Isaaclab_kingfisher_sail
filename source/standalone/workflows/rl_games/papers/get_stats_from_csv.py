import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch

# --- Argument Parsing ---
parser = argparse.ArgumentParser()
parser.add_argument("--kingfisher", action="store_true", default=False, help="if true, the state to plot is energy and comes \
another .csv")

args = parser.parse_args()

from torchsort import soft_rank
def spearman_corr(x: torch.Tensor, y: torch.Tensor, regularization_strength: float = 1.0):
    """
    Compute differentiable Spearman correlation between x and y via soft ranks.
    
    Args:
        x, y: tensors of shape [N]
        regularization_strength: parameter for soft ranking smoothness

    Returns:
        Scalar tensor: approximate Spearman correlation in [-1, 1]
    """
    # Flatten to [1, N]
    x = torch.tensor(x.values, dtype=torch.float32).view(1, -1).cpu()
    y = torch.tensor(y.values, dtype=torch.float32).view(1, -1).cpu()

    # Compute soft ranks (shape [N])
    x_rank = soft_rank(x, regularization_strength=regularization_strength).squeeze(0)
    y_rank = soft_rank(y, regularization_strength=regularization_strength).squeeze(0)

    # Center ranks
    x_r = x_rank - x_rank.mean()
    y_r = y_rank - y_rank.mean()

    # Pearson on ranks
    cov = (x_r * y_r).sum()
    corr = cov / (torch.norm(x_r, 2) * torch.norm(y_r, 2) + 1e-8)
    return corr

latest_dir = "eval_logs/Papers exps/kingfisher1/2025_09_14_17_09_42_180_5_spear_seed10"
csv_name = "acord_data"

print("=========================================================================================================")
print(f"Loading data from: {latest_dir}")

# --- Load CSVs ---
df_ep_length = pd.read_csv(os.path.join(latest_dir, "ep_length.csv"))
df_vel_context = pd.read_csv(os.path.join(latest_dir, f"{csv_name}.csv"))
if args.kingfisher:
    df_other_metrics = pd.read_csv(os.path.join(latest_dir, "other_metrics.csv"))

# --- Get episode_length ---

ep_length_list = df_ep_length["ep_length"].tolist()

num_episodes = len(ep_length_list)

print(f"episode_length: {ep_length_list}")
def plot_state_vs(name:str, name_context:str=""):
    spearman_dict = {}
    for ep in df_vel_context["episode"].unique()[:num_episodes]:  
        ep_length = ep_length_list[ep]
        ep_vel_context = df_vel_context[df_vel_context["episode"] == ep]
        if args.kingfisher:
            ep_other_metrics = df_other_metrics[df_other_metrics["episode"] == ep]
            for env_id in ep_vel_context["env_id"].unique():
                env_vel_context = ep_vel_context[ep_vel_context["env_id"] == env_id]
                env_other_metrics = ep_other_metrics[ep_other_metrics["env_id"] == env_id]
                
                sp = spearman_corr(env_other_metrics["energy"], env_vel_context[f"{name_context}"])
                spearman_dict[f"energy/{name_context}"] = sp

        else:
            for env_id in ep_vel_context["env_id"].unique():
                env_vel_context = ep_vel_context[ep_vel_context["env_id"] == env_id]
                
                sp = spearman_corr(env_vel_context[f"{name}"], env_vel_context[f"{name_context}"])
                spearman_dict[f"{name}/{name_context}"] = sp

    
    print(spearman_dict)


plot_state_vs("energy", "actual_context")