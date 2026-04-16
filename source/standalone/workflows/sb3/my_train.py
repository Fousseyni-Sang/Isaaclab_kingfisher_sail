# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with Stable Baselines3.

Since Stable-Baselines3 does not support buffers living on GPU directly,
we recommend using smaller number of environments. Otherwise,
there will be significant overhead in GPU->CPU transfer.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-KingfisherSail-Direct-High-SP-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--algo", type=str, default="ppo", help="name of the algorith to use in SB3: (ppo, ddpg, sac)")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import random
from datetime import datetime
import torch

from stable_baselines3 import PPO, SAC, DDPG, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from omni.isaac.lab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg


@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with stable-baselines agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    if args_cli.algo=="sac":
        algo = SAC
        agent_cfg = agent_cfg["sac"]
    elif args_cli.algo=="ddpg":
        algo = DDPG
        agent_cfg = agent_cfg["ddpg"]
    else:
        algo = PPO
        agent_cfg = agent_cfg["ppo"]

    if agent_cfg["policy_kwargs"]["activation_fn"]=="ELU":
        agent_cfg["policy_kwargs"]["activation_fn"] = torch.nn.ELU
    elif agent_cfg["policy_kwargs"]["activation_fn"]=="ReLU":
        agent_cfg["policy_kwargs"]["activation_fn"] = torch.nn.ReLU
    elif agent_cfg["policy_kwargs"]["activation_fn"]=="Tanh":
        agent_cfg["policy_kwargs"]["activation_fn"] = torch.nn.Tanh
    else:
        raise f"activation: {agent_cfg.get('policy_kwargs', {}).get('activation_fn', None)} not supported. Choose (ELU, ReLU, Tanh)"
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    # max iterations for training
    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # directory for logging into
    log_dir = os.path.join("logs", "sb3", args_cli.task, args_cli.algo, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)
    # read configurations about the agent-training
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for stable baselines
    eval_env = Sb3VecEnvWrapper(env)
    env = Sb3VecEnvWrapper(env)

    
    max_episode_length = env.unwrapped.max_episode_length
    if hasattr(env.unwrapped.cfg, "num_low_level_steps"):
        learning_starts = max_episode_length + 100 #*env.unwrapped.num_envs//env.unwrapped.cfg.num_low_level_steps
    else:
        learning_starts = max_episode_length + 100 # max_episode_length*env.unwrapped.num_envs + 100

    if "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=True,
            norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
            norm_reward="normalize_value" in agent_cfg and agent_cfg.pop("normalize_value"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

        eval_env = VecNormalize(
            eval_env, 
            training=False, 
            norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
            norm_reward="normalize_value" in agent_cfg and agent_cfg.pop("normalize_value"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    # create agent from stable baselines
    if algo in [DDPG, SAC]: 
        agent = algo(policy_arch, env, **agent_cfg, replay_buffer_class=HerReplayBuffer, 
                    replay_buffer_kwargs=dict(n_sampled_goal=4, # Number of virtual goals to create per real transition
                    goal_selection_strategy="future",
                    #copy_info_dict=True,
                ),
                learning_starts=learning_starts)
    else:
        agent = algo(policy_arch, env, **agent_cfg)

    # configure the logger
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    agent.set_logger(new_logger)


    class IsaacEpisodeMetricsCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)

        def _on_step(self):
            infos = self.locals.get("infos", None)
            if infos is None:
                return True

            for info in infos:
                episode = info.get("episode", None)
                if episode is None:
                    continue

                # Log all Episode_Reward/* and Metrics/* keys
                for key, value in episode.items():
                    if key.startswith("Episode_Reward/") or key.startswith("Metrics/"):
                        # Convert torch tensors to float
                        if hasattr(value, "item"):
                            value = value.item()
                        self.logger.record(key, value)

            return True


    save_freq = n_timesteps//10
    eval_freq = 2_000_000
    # callbacks for agent
    checkpoint_callback = CheckpointCallback(save_freq=save_freq//env.unwrapped.num_envs, save_path=log_dir, name_prefix="kingfisher_model", verbose=2)
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir, eval_freq=eval_freq//eval_env.unwrapped.num_envs)
    isaac_cb = IsaacEpisodeMetricsCallback()

    callback = CallbackList(
        [checkpoint_callback, eval_callback, isaac_cb]
    )
    # train the agent
    agent.learn(total_timesteps=n_timesteps, callback=callback)
    # save the final model
    agent.save(os.path.join(log_dir, "model"))

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
