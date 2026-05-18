import argparse
from omni.isaac.lab.app import AppLauncher 

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--ros", action="store_true", default=False, help="Enable ROS2 publishing.")
parser.add_argument("--wind_direction", type=float, default=180.0, help="True wind direction in degrees.") 
parser.add_argument("--wind_speed", type=float, default=5.0, help="True wind speed.") 
parser.add_argument("--num_episode", type=int, default=10, help="Number of episodes for evaluation.") 
parser.add_argument("--ros_publish_interval", type=int, default=10, help="ROS publish interval in steps.") 
parser.add_argument("--model_id", type=str, default=None, help="low level model to run") 
parser.add_argument("--pattern", type=str, default="grid", help="pattern of points to give as goals (grid, zigzag, line, square)")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import Pyro5.api
import gymnasium as gym
import torch
from omni.isaac.lab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

@Pyro5.api.expose
class IsaacKyaServer:

    def __init__(self, task, checkpoint=None, num_envs=1, device="cuda:0"):

        # Load env config
        env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
        agent_cfg = load_cfg_from_registry(task, "rl_games_cfg_entry_point")

        # Create Isaac environment
        env = gym.make(task, cfg=env_cfg)
        env = RlGamesVecEnvWrapper(env, device, clip_obs=1000, clip_actions=1000)
        
        # register the environment to rl-games registry
        # note: in agents configuration: environment name must be "rlgpu"
        vecenv.register(
            "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

        
        self.env = env

        # Load PPO player
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = checkpoint
        runner = Runner()
        runner.load(agent_cfg)
        self.agent: BasePlayer = runner.create_player()
        self.agent.restore(checkpoint)
        self.agent.reset()

        # Reset env
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        self.last_obs = obs

    # -----------------------------------------------------------
    #  KYA METHODS
    # -----------------------------------------------------------

    def set_env_state(self, state_dict):
        """Inject arbitrary state into IsaacLab."""
        if hasattr(self.env.unwrapped, "set_env_state"):
            self.env.unwrapped.set_env_state(state_dict)
        else:
            raise RuntimeError("Env does not implement set_env_state().")

    def reset_env(self, state_dict=None):
        """Reset env, optionally to a custom state."""
        obs = self.env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        if state_dict is not None:
            self.set_env_state(state_dict)
        self.last_obs = obs
        return obs.tolist()

    def get_value(self, obs):
        """Query V(s)."""
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            v = self.agent.get_values(obs_t)
        return float(v.item())

    def get_policy(self, obs):
        """Query π(a|s)."""
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            res = self.agent.get_action_values(obs_t)
        return {k: v.cpu().numpy().tolist() for k, v in res.items()}

    def step_env(self, action):
        """Step IsaacLab from arbitrary state."""
        action_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        obs, rew, done, info = self.env.step(action_t)
        if isinstance(obs, dict):
            obs = obs["obs"]
        self.last_obs = obs
        return {
            "obs": obs.tolist(),
            "reward": float(rew[0]),
            "done": bool(done[0]),
            "info": {}
        }


def main():
    daemon = Pyro5.api.Daemon()
    ns = Pyro5.api.locate_ns()   
    isaac_server = IsaacKyaServer(task=args_cli.task, checkpoint=args_cli.checkpoint, num_envs=args_cli.num_envs)
    
    uri = daemon.register(isaac_server)
    
    ns.register("isaacserver.kya", uri)
    print(f"IsaacKyaServer ready at: {uri} --> Access with Pyro nameserver <PYRONAME:isaacserver.kya>")
    print("Waiting for KYA client to connect...")
    daemon.requestLoop()


if __name__ == "__main__":
    main()

    simulation_app.close()