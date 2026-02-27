from omni.isaac.lab.app import AppLauncher 

app_launcher = AppLauncher() 
simulation_app = app_launcher.app 

from omni.isaac.lab_tasks.utils.my_utils.boat_config import (generate_boat_specs, 
                    generate_structured_boat_specs, load_existing_specs)
from omni.isaac.lab_tasks.utils.my_utils.common import clear_hl_split
import os
import yaml
import subprocess
import argparse

AGENTCFG = "source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/kingfisher_sail_low_level/agents/rl_games_ppo_cfg.yaml"
parser = argparse.ArgumentParser(description="Launch training of arbitrary number of LL RL agent from RL-Games.")
parser.add_argument("--num_models", type=int, default=None, help="number of low level agent to be trained.")
parser.add_argument("--num_envs", type=int, default=1024, help="number of environments to be used.")
parser.add_argument("--model_id_list", type=str, default=None, help="comma-separated list of low level model ids to train "
"(e.g. '000,001,002'). Overrides --num_models if provided.")
args_cli = parser.parse_args()


def save_spec(spec, path):
    # Delete existing spec before creating a new one

    if os.path.exists(path):
        os.remove(path)
        
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(spec, f)

def launch_ll_sweep(num_models=15):
    # Generate all boat specifications
    boat_specs, is_loaded = load_existing_specs(n=num_models) #load_existing_specs() or generate_structured_boat_specs() or generate_boat_specs(num_models)
    #is_loaded = False # IGNORE
    boat_specs = sorted(boat_specs, key=lambda s: int(s["model_id"]))
    if is_loaded:
        print(f"Loaded {len(boat_specs)} existing boat specs from outputs/ll/")
        print("Boat IDs:", [spec["id"] for spec in boat_specs])
    else:
        boat_specs = generate_structured_boat_specs() #generate_boat_specs(num_models)

    #clear_hl_split()
    
    model_id_list = args_cli.model_id_list.split(",") if args_cli.model_id_list else None
    if model_id_list and '-' in model_id_list[0]:
        # Handle range format like "000-004"
        start_id, end_id = model_id_list[0].split("-")
        model_id_list = [f"{i:03d}" for i in range(int(start_id), int(end_id) + 1)]

    print(f"============> model_id_list: {model_id_list}")
    for model_id, spec in enumerate(boat_specs):
        # ---------------------------------------------------------
        # 1. Prepare directories and environment variables
        # ---------------------------------------------------------
        
        if model_id_list and f"{model_id:03d}" not in model_id_list:
            print(f"Skipping LL model {model_id} as it's not in the provided model_id_list.")
            continue
        
        run_name = f"ll_model_{model_id:03d}"
        out_dir = f"outputs/ll/{run_name}"
        if not is_loaded:
            os.makedirs(out_dir, exist_ok=True)

            spec["model_id"] = f"{model_id:03d}"
            # Save the boat specification for this model
            save_spec(spec, f"{out_dir}/spec.yaml")

        # Environment variables passed to Isaac env
        env = os.environ.copy()
        env["LL_MODEL_ID"] = str(model_id)
        env["LL_OUTPUT_DIR"] = out_dir

        # ---------------------------------------------------------
        # 2. Load and patch the RL Games YAML config
        # ---------------------------------------------------------
        with open(AGENTCFG, "r") as f:
            cfg = yaml.safe_load(f)

        if spec["has_keel"] or spec["has_rudder"] or spec["has_sail"]:
            cfg["params"]["config"]["max_epochs"] = 1000
            
        else:
            cfg["params"]["config"]["max_epochs"] = 300
        # Unique agent name → unique checkpoint filename
        cfg["params"]["config"]["name"] = f"kingfisher_direct_low_level_{run_name}"

        # Save patched YAML for this model
        with open(AGENTCFG, "w") as f:
            yaml.dump(cfg, f)

        # ---------------------------------------------------------
        # 3. Launch training
        # ---------------------------------------------------------
        cmd = [
            "/mnt/gpu_storage/zrr/fsangare/isaaclab/bin/python",
            "source/standalone/workflows/rl_games/train.py",
            "--task", "Isaac-KingfisherSail-Direct-Low-v0",
            "--headless",
            "--num_envs", str(args_cli.num_envs),
            "--device", "cuda:1"
        ]

        print(f"Launching LL model {model_id} ({run_name})…")
        subprocess.run(cmd, env=env, check=True)

    #subprocess.run(["./kill_isaac.sh"], check=True)

    # Launch the eval process
    cmd = [
        "/mnt/gpu_storage/zrr/fsangare/isaaclab/bin/python",
        "source/standalone/workflows/rl_games/launch_play_ll.py",
        "--model_id_list", args_cli.model_id_list,
    ]
    subprocess.run(cmd, check=True)

    #subprocess.run(["./kill_isaac.sh", "Direct"], check=True)

    # Launch the HL training 
    cmd = [
            "/mnt/gpu_storage/zrr/fsangare/isaaclab/bin/python",
            "source/standalone/workflows/rl_games/train.py",
            "--task", "Isaac-KingfisherSail-Direct-High-v0",
            "--headless",
            "--num_envs", str(args_cli.num_envs),
            "--device", "cuda:1"
        ]
    subprocess.run(cmd, check=True)
    #subprocess.run(["./kill_isaac.sh"], check=True)

if __name__ == "__main__":
    launch_ll_sweep(args_cli.num_models)
    #subprocess.run(["./kill_isaac.sh", "launch_training_ll"], check=True)




