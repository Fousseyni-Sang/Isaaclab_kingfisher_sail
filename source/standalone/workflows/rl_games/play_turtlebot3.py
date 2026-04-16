# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse

import numpy

from omni.isaac.lab.app import AppLauncher 
import yaml

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

"""Rest everything follows."""


import gymnasium as gym
import math
import os
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.assets import retrieve_file_path
from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

# ROS2 (optional) 
import rclpy 
from omni.isaac.lab_tasks.utils.my_utils.ros2_Node import RewardWeightSubscriber, DynamicsRlAgentPublisher 
from datetime import datetime
import pandas as pd
    
def load_yaml_config(config_file):
    """Load the YAML configuration file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config['experiments'][0]['points']

def main():
    """Play with RL-Games agent."""
    
    # ROS init (optional) 
    if args_cli.ros: 
        rclpy.init() 
        dyn_ros_node = DynamicsRlAgentPublisher(args_cli.num_envs) 
        slider_names = ["time", "energy", "goal", "wind_direct", "desired_speed", "wind_speed"] 
        slider_node = RewardWeightSubscriber(slider_names) 
    else: 
        dyn_ros_node = None 
        slider_node = None 

    # parse env configuration
    spec_path = os.environ.get("LL_OUTPUT_DIR", None)
    temp = spec_path
    if spec_path==None and args_cli.model_id is not None:
        spec_path = f"outputs/ll/ll_model_{args_cli.model_id}" 
        os.environ["LL_OUTPUT_DIR"] = spec_path

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    if temp==None and args_cli.model_id is not None:
        agent_cfg["params"]["config"]["name"] = f"kingfisher_direct_low_level_ll_model_{args_cli.model_id}"


    config_file = f"config/{args_cli.pattern}.yaml"
    # Load YAML points
    points = load_yaml_config(config_file)
        
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    acord_dir = log_dir.split("/")
    acord_dir_name = ('/'.join(acord_dir[:-3]) + "/acord/" + '/'.join(acord_dir[-2:]))

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    # reset environment
    """env.unwrapped.is_Training = False
    env.unwrapped.discr_checkpoint = checkpoint_name_acord
    env.unwrapped.discriminator.load_checkpoint(checkpoint_name_acord)
    env.unwrapped.discriminator.eval()"""
    env.unwrapped.is_Training = False
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()

    num_envs = env.unwrapped.num_envs 
    env.unwrapped.cfg.max_episode_length = 500 # override default max episode length for evaluation
    num_episodes_target = args_cli.num_episode 
    episodes_finished = 0 
    # Per-env episode buffers (lists of dicts) 
    per_env_steps = [[] for _ in range(num_envs)] 
    per_env_goal = [None for _ in range(num_envs)] 
    all_episode_logs = []   # list of dicts: { "episode": i, "env_id": j, "steps": tensor }

    per_env_episode_lengths = [] 
    all_metrics = [] 
    step_idx = 0 
    next_goal_idx = 0
    env.unwrapped._desired_pos_w[:, :2] = torch.tensor(points[next_goal_idx], device=env.unwrapped._desired_pos_w.device)

    print(f"\n================ Vectorized Evaluation: {num_envs} envs =================\n") 
    
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.
    while simulation_app.is_running() and episodes_finished < num_episodes_target:
        # run everything in inference mode
        with torch.inference_mode():
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            # env stepping
            obs, rew, dones, extras = env.step(actions)

            robot_pos = extras["info"]["robot_pos_w"][..., :2] # (N, 2) 
            lift_coeff = extras.get("info", {}).get("lift_coeff", torch.zeros((num_envs,), device=rew.device)) # (N,) 
            drag_coeff = extras.get("info", {}).get("drag_coeff", torch.zeros((num_envs,), device=rew.device)) # (N,) 
            energy = extras.get("info", {}).get("energy", torch.zeros((num_envs,), device=rew.device)) # (N,) 
            episode_energy = extras.get("info", {}).get("episode_energy", torch.zeros((num_envs,), device=rew.device)) # (N,) 
            max_available_energy = extras.get("info", {}).get("max_available_energy", torch.zeros((num_envs,), device=rew.device)) # (N,) 
            ratio_energy_usage = extras.get("info", {}).get("ratio_energy_usage", torch.zeros((num_envs,), device=rew.device)) # (N,) 
            rew_progress = extras.get("info", {}).get("reward_progress", torch.zeros((num_envs,), device=rew.device)) # (N,) 
            rew_energy = extras.get("info", {}).get("reward_energy", torch.zeros((num_envs,), device=rew.device)) # (N,) 
            rew_backward = extras.get("info", {}).get("reward_backward", torch.zeros((num_envs,), device=rew.device)) # (N,) 
            rew_aero = extras.get("info", {}).get("reward_aero", torch.zeros((num_envs,), device=rew.device)) # (N,) 
            rew_acord = extras.get("info", {}).get("reward_acord", torch.zeros((num_envs,), device=rew.device)) # (N,) 
            bearing = extras.get("info", {}).get("bearing", torch.zeros((num_envs,), device=rew.device)) # (N,) 
            distance = extras["info"]["distance"] # (N,) 
            #print(f"Step: {step_idx}, Rewards: {rew}, Distance: {distance}")
            aoa = extras.get("info", {}).get("aoa", torch.zeros((num_envs, ), device=rew.device)) # (N,) 
            app_wind_angle = extras.get("info", {}).get("app_wind_angle", torch.zeros((num_envs, ), device=rew.device)) # (N,) 
            true_wind_angle = extras.get("info", {}).get("true_wind_angle", torch.zeros((num_envs, ), device=rew.device)) # (N,) 
            sail_angle = extras.get("info", {}).get("sail_angle", torch.zeros((num_envs,), device=rew.device)) # (N,) 
            aero_force = extras.get("info", {}).get("aero_force", torch.zeros((num_envs, 1, 6), device=rew.device)) # (N, 6) 
            max_aero_force = extras.get("info", {}).get("max_aero_force", torch.zeros((num_envs,), device=rew.device)) # (N,) 
            thruster_force = extras.get("info", {}).get("thruster_force", torch.zeros((num_envs, 1, 6), device=rew.device)) # (N, 6) 
            norm_error_lin = extras["info"]["norm_error_lin"] # (N,) 
            norm_error_ang = extras["info"]["norm_error_ang"] # (N,) 
            goal_pos = extras["info"]["goal_pos"] # (N, 2) 
            norm_error_cat=torch.cat( [norm_error_lin, norm_error_ang.reshape(-1, 1)], dim=-1 )
            # Save goal per env once 
            for env_id in range(num_envs): 
                if per_env_goal[env_id] is None: 
                    per_env_goal[env_id] = goal_pos[env_id].detach().cpu().clone() 

            # ROS publish (throttled) 
            if args_cli.ros: #  and (step_idx % args_cli.ros_publish_interval == 0): 
                dyn_ros_node.publish( 
                    obs=obs,
                    actions=actions, 
                    total_rew=rew, 
                    aero_force=aero_force, 
                    thruster_force=thruster_force,
                    lin_vel_b=extras.get("info", {}).get("lin_vel_b", torch.zeros((num_envs, 3), device=rew.device)), 
                    angle_of_attack=aoa, 
                    app_flow_angle=app_wind_angle, 
                    true_flow_angle=true_wind_angle, 
                    sail_angle=sail_angle, 
                    heading_w=extras.get("info", {}).get("heading_w", torch.zeros((num_envs,), device=rew.device)), 
                    lift_drag_ratio=extras.get("info", {}).get("lift_drag_ratio", torch.zeros((num_envs,), device=rew.device)), 
                    robot_pos_w=robot_pos, 
                    energy=energy, 
                    episode_energy=episode_energy, 
                    lift_force_b=extras.get("info", {}).get("lift_force_b", torch.zeros((num_envs,3), device=rew.device)), 
                    drag_force_b=extras.get("info", {}).get("drag_force_b", torch.zeros((num_envs,3), device=rew.device)), 
                    lift_coeff=lift_coeff, 
                    drag_coeff=drag_coeff, 
                    rew_progress=rew_progress, 
                    rew_energy=rew_energy, 
                    rew_backward=rew_backward, 
                    desired_wrench_b=extras.get("info", {}).get("desired_wrench_b", torch.zeros((num_envs, 6), device=rew.device)), 
                    ang_speed_b=extras.get("info", {}).get("ang_vel_b", torch.zeros((num_envs, 3), device=rew.device)), 
                    norm_error_cat=norm_error_cat, 
                    rew_goal=extras.get("info", {}).get("reward_goal", torch.zeros((num_envs,), device=rew.device)), 
                    rew_aero=rew_aero, 
                    distance=distance, 
                    ) 
                
            # Store per-step logs per env 
            for env_id in range(num_envs): 
                step_record = { 
                    "time_step": len(per_env_steps[env_id]), 
                    "x": robot_pos[env_id, 0].item(), 
                    "y": robot_pos[env_id, 1].item(), 
                    "lift_coeff": lift_coeff[env_id].item(), 
                    "drag_coeff": drag_coeff[env_id].item(), 
                    "energy": energy[env_id].item(), 
                    "reward_progress": rew_progress[env_id].item(), 
                    "reward_energy": rew_energy[env_id].item(), 
                    "reward_backward": rew_backward[env_id].item(), ""
                    "reward_aero": rew_aero[env_id].item(), 
                    "reward_acord": rew_acord[env_id].item(), 
                    "total_reward": rew[env_id].item(), 
                    "bearing": bearing[env_id].item(), 
                    "distance": distance[env_id].item(),
                    "aoa": aoa[env_id].item(),
                    "app_wind_angle": app_wind_angle[env_id].item(), 
                    "true_wind_angle": true_wind_angle[env_id].item(), 
                    "sail_angle": sail_angle[env_id].item(), 
                    "episode_energy": episode_energy[env_id].item(), 
                    "max_available_energy": max_available_energy[env_id].item(), 
                    "ratio_energy_usage": ratio_energy_usage[env_id].item(), 
                    "norm_error_lin_vx": norm_error_lin[env_id, 0].item(), 
                    "norm_error_lin_vy": norm_error_lin[env_id, 1].item() if norm_error_lin.shape[1] > 1 else 0.0, 
                    "norm_error_ang": norm_error_ang[env_id].item(), 
                    } 
                # actions, aero_force, thruster_force, energy_context, acord_prediction as vectors 
                step_record.update({
                    "action_0": actions[env_id, 0].item(), 
                    "action_1": actions[env_id, 1].item(), 
                    "action_2": actions[env_id, 2].item() if actions.shape[1] > 2 else 0.0, 
                    "aero_fx": aero_force[env_id, 0, 0].item(), 
                    "aero_fy": aero_force[env_id, 0, 1].item(), 
                    "aero_fz": aero_force[env_id, 0, 2].item(), 
                    "aero_mx": aero_force[env_id, 0, 3].item(), 
                    "aero_my": aero_force[env_id, 0, 4].item(), 
                    "aero_mz": aero_force[env_id, 0, 5].item(), 
                    "thr_fx": thruster_force[env_id, 0, 0].item(), 
                    "thr_fy": thruster_force[env_id, 0, 1].item(), 
                    "thr_fz": thruster_force[env_id, 0, 2].item(), 
                    "thr_mx": thruster_force[env_id, 0, 3].item(), 
                    "thr_my": thruster_force[env_id, 0, 4].item(), 
                    "thr_mz": thruster_force[env_id, 0, 5].item(), 
                    "max_aero_force": max_aero_force[env_id].item(), 
                                    }) 
                
                per_env_steps[env_id].append(step_record)
            
            # Handle episode termination per env 
            for env_id in range(num_envs): 

                if distance[env_id].item() < 0.7:
                    
                    next_goal_idx = (next_goal_idx + 1) % len(points)
                    env.unwrapped._desired_pos_w[env_id, :2] = torch.tensor(points[next_goal_idx], device=env.unwrapped._desired_pos_w.device)
                    print(f"[INFO] Env {env_id} reached goal, moving to next goal: {points[next_goal_idx]}")

                if dones[env_id]: 
                    ep_len = len(per_env_steps[env_id]) 
                    per_env_episode_lengths.append({ 
                        "episode": episodes_finished, 
                        "env_id": env_id, 
                        "ep_length": ep_len, }) 
                    # --------------------------------------------------------- 
                    # Convert per-step Python dicts → vectorized tensors 
                    # --------------------------------------------------------- 
                    episode_dict = {} 
                    keys = per_env_steps[env_id][0].keys() 
                    for key in keys: # Build a tensor of shape (T,) or (T, D) 
                        episode_dict[key] = torch.tensor( [step[key] for step in per_env_steps[env_id]] ) 

                    # --------------------------------------------------------- 
                    # Store the entire episode in a global list 
                    # --------------------------------------------------------- 
                    all_episode_logs.append({ 
                        "episode": episodes_finished, 
                        "env_id": env_id, 
                        "steps": episode_dict, }) 
                    # --------------------------------------------------------- 
                    # Metrics from env.extras["log"] 
                    # --------------------------------------------------------- 
                    if "log" in env.unwrapped.extras: 
                        metrics = env.unwrapped.extras["log"] 
                        if isinstance(metrics, dict): 
                            metrics_copy = metrics.copy() 
                            metrics_copy["episode"] = episodes_finished 
                            metrics_copy["env_id"] = env_id 
                            all_metrics.append(metrics_copy) 
                    print(f"[INFO] Episode {episodes_finished} finished (env {env_id}, length {ep_len})") 
                    episodes_finished += 1 

                    # --------------------------------------------------------- 
                    # Reset buffer for this env AFTER saving 
                    # --------------------------------------------------------- 
                    per_env_steps[env_id] = []

            

            step_idx += 1
            # perform operations for terminated episodes
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
    
    # Cleanup ROS 
    if args_cli.ros: 
        dyn_ros_node.destroy_node() 
        slider_node.destroy_node() 
        rclpy.shutdown()

    # ------------------------------------------------------------------------- 
    # Save logs 
    # ------------------------------------------------------------------------- 
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") 
    model_id = ("_").join(agent_cfg['params']['config']['name'].split("_")[-2:])
    output_dir = os.path.join("eval_ll", "rl_games", f"{model_id}") 
    os.makedirs(output_dir, exist_ok=True) 
    print(f"[INFO] Saving logs to: {output_dir}")

    # Trajectories + actions + aero + errors are all in per_env_steps 
    traj_records = [] 
    actions_records = [] 
    aero_records = [] 
    norm_error_records = [] 
    # Goals 
    goal_records = [] 
    for env_id, goal in enumerate(per_env_goal): 
        if goal is not None: 
            goal_records.append({ "env_id": env_id, "goal_x": goal[0].item(), "goal_y": goal[1].item(), }) 
            # Flatten per_env_steps into episode-indexed logs 
            # Here we treat each env's episodes sequentially; 
            # if you want strict episode IDs, # you can track them per env separately. 
            episode_counter = 0 
            for env_id in range(num_envs): 
                # per_env_steps[env_id] is empty now because we reset on done, 
                # but you can adapt this if you want to keep partial episodes. 
                pass 
            # Instead, we reconstruct from per_env_episode_lengths + all_metrics if needed. 
            # For simplicity, we only saved per-step logs in per_env_steps during episodes; 
            # if you want to persist them across episodes, you can accumulate them into 
            # a global list when an episode finishes. 

            # Minimal example: save episode lengths and goals 
            pd.DataFrame(per_env_episode_lengths).to_csv(os.path.join(output_dir, "ep_length.csv"), index=False) 
            pd.DataFrame(goal_records).to_csv(os.path.join(output_dir, "goals.csv"), index=False) 
            
            if all_metrics: 
                pd.DataFrame(all_metrics).to_csv(os.path.join(output_dir, "metrics.csv"), index=False) 

    rows = []
    
    for ep in all_episode_logs:
        ep_id = ep["episode"]
        env_id = ep["env_id"]
        steps = ep["steps"]
        T = steps["x"].shape[0]

        for t in range(T):
            rows.append({
                "episode": ep_id,
                "env_id": env_id,
                "time_step": t,
                "x": steps["x"][t].item(),
                "y": steps["y"][t].item(),
                "lift_coeff": steps["lift_coeff"][t].item(),
                "drag_coeff": steps["drag_coeff"][t].item(),
                "energy": steps["energy"][t].item(),
                "reward_progress": steps["reward_progress"][t].item(),
                "reward_energy": steps["reward_energy"][t].item(),
                "reward_backward": steps["reward_backward"][t].item(),
                "reward_aero": steps["reward_aero"][t].item(),
                "reward_acord": steps["reward_acord"][t].item(),
                "bearing": steps["bearing"][t].item(),
                "distance": steps["distance"][t].item(),
                "aoa": steps["aoa"][t].item(),
                "app_wind_angle": steps["app_wind_angle"][t].item(),
                "true_wind_angle": steps["true_wind_angle"][t].item(),
                "sail_angle": steps["sail_angle"][t].item(),
                # etc...
            })

    '''feas = extras["info"]["feasibility_map"]

        i, j, k = np.indices((10,10,10))
    df = pd.DataFrame({
        "i": i.reshape(-1),
        "j": j.reshape(-1),
        "k": k.reshape(-1),
        "feasible": feas.reshape(-1)
    })

    df.to_csv(os.path.join(output_dir, "feasibility_map.csv"), index=False)
    '''
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "all_steps.csv"), index=False)

    feasibility = extras["info"]["feasibility_map"]   # shape (10,10,10) or (10,10)
    feasibility_flattened = feasibility.reshape(-1)               # shape (1000,) or (100,)

    df = pd.DataFrame({"feasible": feasibility_flattened})
    df.to_csv(os.path.join(output_dir, "feasibility_map.csv"), index=False)


    print("[INFO] Evaluation complete.")


    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
