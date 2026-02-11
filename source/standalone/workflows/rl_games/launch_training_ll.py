from omni.isaac.lab.app import AppLauncher 

app_launcher = AppLauncher() 
simulation_app = app_launcher.app 

from omni.isaac.lab_tasks.utils.my_utils.boat_config import generate_boat_specs
import os
import yaml
import subprocess

AGENTCFG = "source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/kingfisher_sail_low_level/agents/rl_games_ppo_cfg.yaml"

def save_spec(spec, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(spec, f)

def launch_ll_sweep(num_models=50):
    # Generate all boat specifications
    boat_specs = generate_boat_specs(num_models)

    for model_id, spec in enumerate(boat_specs):

        # ---------------------------------------------------------
        # 1. Prepare directories and environment variables
        # ---------------------------------------------------------
        run_name = f"ll_model_{model_id:03d}"
        out_dir = f"outputs/ll/{run_name}"
        os.makedirs(out_dir, exist_ok=True)

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

        # Unique agent name → unique checkpoint filename
        cfg["params"]["config"]["name"] = f"kingfisher_direct_low_level_{run_name}"

        # Save patched YAML for this model
        with open(AGENTCFG, "w") as f:
            yaml.dump(cfg, f)

        # ---------------------------------------------------------
        # 3. Launch training
        # ---------------------------------------------------------
        cmd = [
            "./isaaclab.sh",
            "-p",
            "source/standalone/workflows/rl_games/train.py",
            "--task", "Isaac-KingfisherSail-Direct-Low-v0",
            "--headless",
            "--num_envs", "1024",
            "--device", "cuda:1"
        ]


        


        print(f"Launching LL model {model_id} ({run_name})…")
        subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    launch_ll_sweep(3)





