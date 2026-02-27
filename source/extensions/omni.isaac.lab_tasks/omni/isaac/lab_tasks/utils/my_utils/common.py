import torch
import re
import os
import yaml
import glob
import pandas as pd
import shutil

def sample_pairs_batch(env_ids, params, device="cpu"):
    """
    env_ids: LongTensor of env indices that need new (V, omega) targets
    params dict keys:
        'V_min', 'V_max',
        'omega_max',
        'V_min_nonzero',
    """

    num = len(env_ids)
    if num == 0:
        return None

    # Oversample to reduce rejection rate
    N = num 

    # Sample candidates
    Vx = torch.empty((N, 1), device=device).uniform_(params['Vx_min'], params['Vx_max'])
    Vy = torch.empty((N, 1), device=device).uniform_(params['Vy_min'], params['Vy_max'])
    omega = torch.empty((N, 1), device=device).uniform_(-params['omega_max'], params['omega_max'])

    return torch.cat([Vx, Vy, omega], dim=-1)

def sample_test_pairs(env_ids, params, device="cpu"):

    """
    env_ids: LongTensor of env indices that need new (V, omega) targets
    params dict keys:
        'V_min', 'V_max',
        'omega_max',
        'V_min_nonzero',
    """

    num = len(env_ids)
    if num == 0:
        return None

    # Oversample to reduce rejection rate
    N = num 

    # Sample candidates
    V = torch.empty((N, 2), device=device).uniform_(params['V_min'], params['V_max'])
    omega = torch.empty((N, 1), device=device).uniform_(-params['omega_max'], params['omega_max'])

    return torch.cat([V, omega], dim=-1)


def sample_feasible_pairs_batch(env_ids, params, device="cpu", batch_factor=10):
    """
    env_ids: LongTensor of env indices that need new (V, omega) targets
    params dict keys:
        'V_min', 'V_max',
        'omega_max',
        'V_min_nonzero',
        'R_min'   <-- only constraint used besides bounds!
    """

    num = len(env_ids)
    if num == 0:
        return None

    # Oversample to reduce rejection rate
    N = num * batch_factor

    # Sample candidates
    V = torch.empty(N, device=device).uniform_(params['V_min'], params['V_max'])
    omega = torch.empty(N, device=device).uniform_(-params['omega_max'], params['omega_max'])

    # ---- FEASIBILITY CHECKS ----
    
    eps = 1e-6
    tiny_omega = omega.abs() < eps


    # 2) Minimum turn radius: |V/omega| >= R_min, except when omega ≈ 0
    mask_radius = torch.ones(N, dtype=torch.bool, device=device)
    nonzero_omega = ~tiny_omega
    mask_radius[nonzero_omega] = (
        (V[nonzero_omega].abs() / omega[nonzero_omega].abs()) >= params['R_min']
    )
    # if omega == 0 → straight line → infinite radius → allowed

    feasible = mask_radius

    # Extract feasible samples
    V_f = V[feasible]
    w_f = omega[feasible]

    # Not enough? Recursively fill the remainder
    if len(V_f) < num:
        extra_needed = num - len(V_f)
        extra_pairs = sample_feasible_pairs_batch(
            env_ids=torch.arange(extra_needed, device=device),
            params=params,
            device=device,
            batch_factor=batch_factor
        )
        return torch.cat(
            [torch.stack([V_f, w_f], dim=-1), extra_pairs],
            dim=0
        )[:num]

    # Enough feasible ones
    V_res = V_f[:num]
    w_res = w_f[:num]

    return torch.stack([V_res, w_res], dim=-1)

def make_checkpoint_path(ckpt_file_name:str, model_id:str):

    return

def load_ll_population(
    spec_root="outputs/ll",
    ckpt_root="logs/rl_games",
    feas_root="eval_ll/rl_games",
):
    population = []

    def extract_episode(filename): 
        m = re.search(r"_ep_(\d+)_", filename) 
        if m: 
            return int(m.group(1)) 
        return -1 

    # Loop over all spec directories: outputs/ll/ll_model_000/
    for name in sorted(os.listdir(spec_root)):
        if not name.startswith("ll_model_"):
            continue

        model_id = name.split("_")[-1]  # "000"
        spec_dir = os.path.join(spec_root, name)
        spec_file = os.path.join(spec_dir, "spec.yaml")

        if not os.path.exists(spec_file):
            continue

        # Load spec
        with open(spec_file, "r") as f:
            spec = yaml.safe_load(f)

        # ---------------------------------------------------------
        # Find checkpoint directory:
        # logs/rl_games/kingfisher_direct_low_level_ll_model_000/<timestamp>/nn/*.pth
        # ---------------------------------------------------------
        ckpt_parent = os.path.join(
            ckpt_root, f"kingfisher_direct_low_level_ll_model_{model_id}"
        )

        if not os.path.exists(ckpt_parent):
            continue

        # Find all timestamped subdirectories
        timestamps = sorted(os.listdir(ckpt_parent))
        if not timestamps:
            continue

        latest = timestamps[-1]  # pick the latest timestamp
        ckpt_dir = os.path.join(ckpt_parent, latest, "nn")

        # Find .pth file
        ckpt_files = [ f for f in os.listdir(ckpt_dir) if re.match(r".*\.pth$", f) ]
        
        if not ckpt_files:
            continue
        ckpt_files.sort(key=lambda m: extract_episode(m))
        checkpoint = os.path.join(ckpt_dir, ckpt_files[-1])

        # ---------------------------------------------------------
        # Feasibility map:
        # eval_ll/rl_games/model_000/feasibility_map.csv
        # ---------------------------------------------------------
        feas_dir = os.path.join(feas_root, f"model_{model_id}")
        feas_file = os.path.join(feas_dir, "feasibility_map.csv")

        feas_map = None
        if os.path.exists(feas_file):
            feas_map = pd.read_csv(feas_file)

        # ---------------------------------------------------------
        # Store entry
        # ---------------------------------------------------------
        population.append({
            "id": model_id,
            "spec": spec,
            "checkpoint": checkpoint,
            "feasibility": feas_map,
        })

    return population
import random
def deterministic_split(population, train_ratio=0.7):
    # Sort by model ID (string or int both work)
    population = sorted(population, key=lambda m: int(m["id"]))

    n = len(population)
    k = int(train_ratio * n)
    
    random.shuffle(population)
    
    train_set = population[:k]
    eval_set  = population[k:]

    return train_set, eval_set

def clear_hl_split(split_root="outputs/hl_split", ll_model_root="outputs/ll", 
                   ll_ckpt_root="logs/rl_games"):
    train_file = os.path.join(split_root, "train_ids.txt")
    eval_file = os.path.join(split_root, "eval_ids.txt")

    if os.path.exists(train_file):
        os.remove(train_file)
    if os.path.exists(eval_file):
        os.remove(eval_file)

    for name in os.listdir(ll_model_root): 
        full = os.path.join(ll_model_root, name) 
        if os.path.isdir(full) and name.startswith("model_"): 
            shutil.rmtree(full)

    for name in os.listdir(ll_ckpt_root): 
        full = os.path.join(ll_ckpt_root, name) 
        if os.path.isdir(full) and "ll_model_" in name: 
            shutil.rmtree(full)

def save_split(train_ids, eval_ids, root="outputs/hl_split"): 

    os.makedirs(root, exist_ok=True) 
    #clear_hl_split()

    with open(os.path.join(root, "train_ids.txt"), "w") as f: 
        for mid in train_ids: 
            f.write(mid + "\n") 

    with open(os.path.join(root, "eval_ids.txt"), "w") as f: 
        for mid in eval_ids: 
            f.write(mid + "\n") 

def load_split(root="outputs/hl_split"): 

    train_file = os.path.join(root, "train_ids.txt") 
    eval_file = os.path.join(root, "eval_ids.txt") 
    if not (os.path.exists(train_file) and os.path.exists(eval_file)): 
        return None, None 
    
    with open(train_file, "r") as f: 
        train_ids = [line.strip() for line in f.readlines()] 

    with open(eval_file, "r") as f: 
        eval_ids = [line.strip() for line in f.readlines()] 

    return train_ids, eval_ids

def load_ll_population_with_split(
    train_ratio=0.7,
    spec_root="outputs/ll",
    ckpt_root="logs/rl_games",
    feas_root="eval_ll/rl_games",
    split_root="outputs/hl_split",
):
    # Load full population
    population = load_ll_population(spec_root, ckpt_root, feas_root)

    # Try loading existing split
    train_ids, eval_ids = load_split(split_root)
    

    if train_ids is not None and eval_ids is not None:
        # Map IDs to population entries
        pop_by_id = {m["id"]: m for m in population}
        
        train_set = [pop_by_id[mid] for mid in train_ids if mid in pop_by_id]
        eval_set  = [pop_by_id[mid] for mid in eval_ids if mid in pop_by_id]

        return train_set, eval_set

    # Otherwise create deterministic split
    train_set, eval_set = deterministic_split(population, train_ratio)

    # Save the split for future runs
    save_split([m["id"] for m in train_set],
               [m["id"] for m in eval_set],
               split_root)

    return train_set, eval_set


def compute_act_dim(spec):
        # Thrusters
        num_thrusters = spec.get("num_thrusters", 0)
        for boolean in spec["holonomic"]:
            if boolean==True:
                num_thrusters += 1

        # Foils: count all foils except keel
        num_foils = 0
        if spec["has_rudder"]==True:
            num_foils += 1
        if spec["has_sail"]==True:
            num_foils += 1

        return num_thrusters + num_foils

def compute_obs_dim(spec):
        # Thrusters

        # Foils: count all foils except keel
        num_obs = 0
        # cos, sin and speed norm
        if spec["has_sail"]==True:
            num_obs += 3

        # cos, sin
        if spec["has_keel"]==True:
            num_obs += 2

        # cos, sin
        if spec["has_rudder"]==True:
            num_obs += 2

        # add +1 dim for the norm of the speed of the flow (water)
        if spec["has_rudder"]==True or spec["has_keel"]==True:  
            num_obs += 1
        

        return num_obs

class DiscretizerFeasMap:
    def __init__(self, device, num_envs, n_pts=11, vy=False):
    
        self.device = device
        self.num_envs = num_envs
        self.grid_index = 0
        
        # Normalized grids
        self.vx_norm_grid = torch.linspace(-1, 1, n_pts, device=self.device)
        self.vy_norm_grid = torch.linspace(-1, 1, n_pts, device=self.device)
        self.w_norm_grid = torch.linspace(-1, 1, n_pts, device=self.device)
        self.vy = vy
        if vy==False:
            # Create full 2D mesh
            print("Creating 2D mesh for discretization...")
            vx_mesh, w_mesh = torch.meshgrid(
                self.vx_norm_grid,
                self.w_norm_grid,
                indexing="ij"
            )
            self.feasibility_map = torch.zeros((n_pts, n_pts), device=self.device, dtype=torch.int)

            # Flatten into (121, 2)
            self.grid_pairs = torch.stack(
                [vx_mesh.flatten(), w_mesh.flatten()], dim=-1)  # shape (121, 2)

                # Flatten into (1331, 3)
            self.grid_pairs = torch.stack(
                [vx_mesh.flatten(), w_mesh.flatten()], dim=-1)  # shape (1331, 3)

            self.grid_ijk = torch.stack([ ((torch.arange(n_pts**2, device=self.device)) // n_pts), # i
                        (torch.arange(n_pts**2, device=self.device) % n_pts) # j 
                        ], dim=-1) # shape (n_pts**2, 2)

        else:
            # Create full 3D mesh
            print("Creating 3D mesh for discretization...")
            self.feasibility_map = torch.zeros((n_pts, n_pts, n_pts), device=self.device, dtype=torch.int)

            vx_mesh, vy_mesh, w_mesh = torch.meshgrid(
                self.vx_norm_grid,
                self.vy_norm_grid,
                self.w_norm_grid,
                indexing="ij"
            )

            # Flatten into (1331, 3)
            self.grid_pairs = torch.stack(
                [vx_mesh.flatten(), vy_mesh.flatten(), w_mesh.flatten()],
                dim=-1
            )  # shape (1331, 3)

            self.grid_ijk = torch.stack([ (torch.arange(n_pts**3, device=self.device) // n_pts**2), # i 
                            ((torch.arange(n_pts**3, device=self.device) % (n_pts**2)) // n_pts), # j 
                        (torch.arange(n_pts**3, device=self.device) % n_pts) # k 
                        ], dim=-1) # shape (n_pts**3, 3) 

        # For tracking which grid point each env is evaluating 
        self.env_grid_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

if __name__=="__main__":

    train_set, eval_set = load_ll_population_with_split()
    
    
