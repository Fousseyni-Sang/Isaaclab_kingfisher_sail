from omni.isaac.lab.app import AppLauncher
app_launcher = AppLauncher()
simulation_app = app_launcher.app

import os
import yaml
import subprocess

AGENTCFG = "source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/kingfisher_sail_low_level/agents/rl_games_ppo_cfg.yaml"


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

    for model_id, model_dir in enumerate(model_dirs):

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
            "/mnt/gpu_storage/zrr/fsangare/isaaclab/bin/python",
            "source/standalone/workflows/rl_games/play_low_level_optimized.py",
            "--task", "Isaac-KingfisherSail-Direct-Low-v0",
            "--headless",
            "--num_envs", "300",
            "--device", "cuda:1",
            "--num_episode", "400"
        ]

        print(f"\n=== Launching PLAY for {run_name} ===")
        subprocess.run(cmd, env=env, check=True)
        subprocess.run(["./kill_isaac.sh", 
                        "Direct"], check=True
                        )


if __name__ == "__main__":
    launch_ll_play()
    subprocess.run(["./kill_isaac.sh", 
                        "launch_play_ll"], check=True
                        )
