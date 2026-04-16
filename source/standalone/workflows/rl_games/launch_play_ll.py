from omni.isaac.lab.app import AppLauncher
app_launcher = AppLauncher()
simulation_app = app_launcher.app

import os
import yaml
import subprocess
import argparse

parser = argparse.ArgumentParser(description="Launch training of arbitrary number of LL RL agent from RL-Games.")

parser.add_argument("--model_id_list", type=str, default=None, help="comma-separated list of low level model ids to train "
"(e.g. '000,001,002'). Overrides --num_models if provided.")
parser.add_argument("--num_envs", type=int, default=20, help="Number of environments to simulate.")
parser.add_argument("--num_episode", type=int, default=50, help="Number of episodes for evaluation.") 


args_cli = parser.parse_args()

AGENTCFG = "source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/kingfisher_sail_low_level/agents/rl_games_ppo_cfg.yaml"
ISAAC_PATH = os.getenv("ISAAC_PATH", "/mnt/ssd-storage/fsangare/isaaclab")

def load_all_ll_models(root="outputs/ll"):
    """Return a sorted list of all LL model directories."""
    dirs = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isdir(path) and name.startswith("ll_model_"):
            dirs.append(path)
    return dirs


def launch_ll_play():
    # ---------------------------------------------------------
    # 1. Find all trained LL models
    # ---------------------------------------------------------
    model_dirs = load_all_ll_models()

    if not model_dirs:
        print("No LL models found in outputs/ll/")
        return
    
    model_id_list = args_cli.model_id_list.split(",") if args_cli.model_id_list else None
    if model_id_list and '-' in model_id_list[0]:
        # Handle range format like "000-004"
        start_id, end_id = model_id_list[0].split("-")
        model_id_list = [f"{i:03d}" for i in range(int(start_id), int(end_id) + 1)]

    print(f"============> model_id_list: {model_id_list}")

    for model_id, model_dir in enumerate(model_dirs):

        if model_id_list and f"{model_id:03d}" not in model_id_list:
            print(f"Skipping LL model {model_id} as it's not in the provided model_id_list.")
            continue

        run_name = os.path.basename(model_dir)
        
        # ---------------------------------------------------------
        # 1. Environment variables for the LL env
        # ---------------------------------------------------------
        env = os.environ.copy()
        env["LL_MODEL_ID"] = str(model_id)
        env["LL_OUTPUT_DIR"] = model_dir

        # ---------------------------------------------------------
        # 2. Patch the agent config IN PLACE
        # ---------------------------------------------------------
        with open(AGENTCFG, "r") as f:
            cfg = yaml.safe_load(f)

        cfg["params"]["config"]["name"] = f"kingfisher_direct_low_level_{run_name}"

        with open(AGENTCFG, "w") as f:
            yaml.safe_dump(cfg, f)

        # ---------------------------------------------------------
        # 3. Launch the play script
        # ---------------------------------------------------------
        cmd = [
            f"{ISAAC_PATH}/bin/python",
            "source/standalone/workflows/rl_games/play_low_level_optimized.py",
            "--task", "Isaac-KingfisherSail-Direct-Low-v0",
            "--headless",
            "--num_envs", f"{args_cli.num_envs}",
            "--device", "cuda:0",
            "--num_episode", f"{args_cli.num_episode}"
        ]

        print(f"\n=== Launching PLAY for {run_name} ===")
        subprocess.run(cmd, env=env, check=True)
        #subprocess.run(["./kill_isaac.sh", "Direct"], check=True)


if __name__ == "__main__":
    launch_ll_play()
    #subprocess.run(["./kill_isaac.sh", "launch_play_ll"], check=True)
