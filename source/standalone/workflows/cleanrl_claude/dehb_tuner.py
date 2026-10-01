"""Joint hyper-parameter + reward-shaping optimization, as in Dierkes et al., "Combining Automated Optimisation of
Hyperparameters and Reward Shape" (RLC 2024, https://arxiv.org/abs/2406.18293) and its reference code
https://github.com/ADA-research/combined_hpo_and_reward_shaping.

The paper's point: an RL algorithm's hyper-parameters and its reward weights are entangled, so tuning one with the
other left at some hand-picked default under-sells both. Instead of two separate sweeps (tuner.py for
hyper-parameters, launch_sweep.py's REWARD_SWEEP_AXES grid for the reward), this puts BOTH into one search space
and optimizes them together with DEHB (Differential Evolution HyperBand) -- the optimizer the paper itself uses.
DEHB is multi-fidelity: most trials run short (cheap, `min_fidelity` env steps), and only the promising ones are
re-run at a longer budget (up to `max_fidelity`), the way Hyperband/successive-halving does, but new configurations
come from differential evolution over the population of good configs instead of random sampling.

Like tuner.py, every trial launches the algorithm script as a fresh process (Isaac Sim cannot be restarted inside
one process): the sampled hyper-parameters become CLI flags, and the sampled reward weights become a per-trial
reward_cfg.yaml pointed at by KINGFISHER_REWARD_CFG -- the exact mechanism launch_sweep.py already uses, so a
result of this search is a normal reward_cfg.yaml you can drop into any other run.

    python dehb_tuner.py --script ppo.py --n_trials 60 --min_fidelity 100000 --max_fidelity 2000000 \
        --extra_args "--num_envs 2048"
    python dehb_tuner.py --script sac.py --n_trials 40 --mode hpo_only        # ablation: hyper-parameters only
    python dehb_tuner.py --script sac.py --n_trials 40 --mode reward_only     # ablation: reward weights only
    python dehb_tuner.py --script ppo.py --n_trials 60 \
        --space "reward.tack_penalty_scale:float:-30.0:0.0"                  # add/override one search dimension

Space entries use tuner.py's DSL: name:float:low:high[:log] | name:int:low:high | name:cat:a,b,c ("64/64" in a cat
value means "--name 64 64"). A reward-shaping entry is the same DSL with a "reward." prefix, e.g.
"reward.energy_penalty_scale:float:-20.0:-1.0" -- it edits reward_cfg.yaml's reward_scales.energy_penalty_scale
instead of becoming a CLI flag. --space overrides or adds to the defaults below; DEFAULT_SPACES (hyper-parameters,
imported from tuner.py so the two tools never drift apart) and DEFAULT_REWARD_SPACE (reward weights, the same axes
launch_sweep.py's REWARD_SWEEP_AXES sweeps by hand) are just a reasonable starting point.

Needs the `dehb` package (`pip install dehb`; pulls in ConfigSpace). A study resumes automatically -- re-running the
same command with the same --study_name continues from its checkpoint in --log_root instead of starting over.
"""
import copy
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import ConfigSpace as CS
import tyro
import yaml
from dehb import DEHB

from launch_sweep import DEFAULT_REWARD_CFG, deep_update
from tuner import DEFAULT_SPACES

HERE = Path(__file__).parent

# The reward_scales axes launch_sweep.py's REWARD_SWEEP_AXES sweeps by hand as a grid -- here they're search
# dimensions instead, jointly optimized alongside the hyper-parameters. Ranges are a starting point, not a claim
# about what's good; override with --space.
DEFAULT_REWARD_SPACE = [
    "reward.distance_progress_reward_scale:float:1.0:30.0",
    "reward.goal_reached_scale:float:200.0:3000.0",
    "reward.energy_penalty_scale:float:-20.0:-0.5",
    "reward.time_penalty_scale:float:-5.0:0.0",
    "reward.tack_penalty_scale:float:-20.0:0.0",
]


@dataclass
class Args:
    script: str  # e.g. ppo.py (looked up next to this file)
    n_trials: int = 40  # total function evaluations (DEHB: fevals) across every fidelity
    min_fidelity: int = 20_000  # cheapest trial, in env steps (--total_timesteps). Scale to your algorithm/hardware.
    max_fidelity: int = 500_000  # most expensive trial (only promising configs reach this), in env steps
    eta: int = 3  # Hyperband downsampling rate: 1/eta configs of each rung are promoted to the next fidelity
    mode: str = "joint"  # joint (paper's method) | hpo_only | reward_only  -- the paper's own ablation modes
    study_name: str = ""  # default: the script name; re-running the same name resumes its checkpoint
    space: tuple[str, ...] = ()  # extra / overriding search-space entries (hyper-parameter or "reward.<name>")
    extra_args: str = ""  # fixed flags forwarded to every trial, e.g. "--num_envs 2048"
    devices: tuple[str, ...] = ("cuda:0",)  # trials are assigned to devices round-robin
    seeds: int = 1  # training runs per trial (the score is their mean)
    metric: str = "mean_return"  # mean_return | mean_step_reward (used as a fallback if no episode finished)
    direction: str = "maximize"
    log_root: str = "logs/cleanrl_claude_tuning"
    seed: int = 1  # DEHB's own RNG seed (search reproducibility, not the training seed)


def to_hyperparameter(spec):
    """tuner.py's DSL -> a ConfigSpace hyperparameter. name:float:low:high[:log] | name:int:low:high | name:cat:a,b,c"""
    name, kind, *rest = spec.split(":")
    if kind == "float":
        return CS.Float(name, bounds=(float(rest[0]), float(rest[1])), log=len(rest) > 2 and rest[2] == "log")
    if kind == "int":
        return CS.Integer(name, bounds=(int(rest[0]), int(rest[1])), log=len(rest) > 2 and rest[2] == "log")
    if kind == "cat":
        return CS.Categorical(name, items=rest[0].split(","))
    raise ValueError(f"unknown space kind '{kind}' in '{spec}'")


def split_config(config):
    """A sampled Configuration -> (reward_scales overrides, CLI flags), splitting on the 'reward.' prefix."""
    reward, flags = {}, []
    for name, value in dict(config).items():
        if name.startswith("reward."):
            reward[name[len("reward."):]] = float(value)
        else:
            flags += [f"--{name}", *str(value).split("/")]
    return reward, flags


def main():
    args = tyro.cli(Args)
    script = HERE / args.script
    algo = script.stem
    study_name = args.study_name or algo

    hparam_space = DEFAULT_SPACES.get(algo, [])
    reward_space = DEFAULT_REWARD_SPACE
    if args.mode == "hpo_only":
        reward_space = []
    elif args.mode == "reward_only":
        hparam_space = []
    elif args.mode != "joint":
        raise ValueError(f"--mode must be joint | hpo_only | reward_only, got '{args.mode}'")
    specs = {s.split(":")[0]: s for s in [*hparam_space, *reward_space]}
    specs.update({s.split(":")[0]: s for s in args.space})
    if not specs:
        raise ValueError(f"empty search space for '{algo}' in --mode {args.mode}: pass --space entries")

    cs = CS.ConfigurationSpace(seed=args.seed)
    cs.add([to_hyperparameter(spec) for spec in specs.values()])
    print(f"[dehb_tuner] {algo} ({args.mode}): {len(specs)} search dimension(s): {list(specs)}")

    study_dir = Path(args.log_root) / f"dehb_{study_name}"
    study_dir.mkdir(parents=True, exist_ok=True)
    resuming = (study_dir / "dehb_state.json").exists()
    print(f"[dehb_tuner] {'resuming' if resuming else 'starting'} study at {study_dir}")

    base_reward_cfg = yaml.safe_load(open(DEFAULT_REWARD_CFG)) if reward_space or any(
        s.startswith("reward.") for s in specs) else {}
    sign = -1.0 if args.direction == "maximize" else 1.0  # DEHB minimizes 'fitness'
    trial_idx = 0

    def evaluate(config, fidelity):
        """Runs one config at one fidelity (averaged over --seeds training runs); returns (fitness, cost, info)."""
        nonlocal trial_idx
        reward_overrides, flags = split_config(config)
        device = args.devices[trial_idx % len(args.devices)]
        steps = int(round(fidelity))
        scores, t0 = [], time.time()

        for seed in range(args.seeds):
            tag = f"trial{trial_idx}_f{steps}_seed{seed}"
            trial_dir = study_dir / "runs" / tag
            trial_dir.mkdir(parents=True, exist_ok=True)
            result = trial_dir / "result.json"

            env = dict(os.environ)
            if reward_overrides:
                reward_cfg = deep_update(copy.deepcopy(base_reward_cfg), {"reward_scales": reward_overrides})
                reward_cfg_path = trial_dir / "reward_cfg.yaml"
                yaml.safe_dump(reward_cfg, open(reward_cfg_path, "w"))
                env["KINGFISHER_REWARD_CFG"] = str(reward_cfg_path)

            cmd = [sys.executable, str(script), "--headless", "--device", device, "--seed", str(seed + 1),
                   "--total_timesteps", str(steps), "--exp_name", tag, "--log_root", str(trial_dir / "train"),
                   "--result_file", str(result), *shlex.split(args.extra_args), *flags]
            print(f"[dehb_tuner] trial {trial_idx} fidelity {steps} seed {seed}: {' '.join(cmd[2:])}", flush=True)
            with open(trial_dir / "stdout.log", "w") as f:
                code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

            if code != 0 or not result.exists():
                print(f"[dehb_tuner] trial {trial_idx} seed {seed} FAILED (exit {code}), see {trial_dir/'stdout.log'}"
                      " -- scoring it as a very bad result instead of crashing the search.")
                scores.append(-1e9 if args.direction == "maximize" else 1e9)
                continue
            out = json.load(open(result))
            score = out[args.metric]
            score = out["mean_step_reward"] if score is None or math.isnan(score) else score
            scores.append(score)
            print(f"[dehb_tuner] trial {trial_idx} seed {seed}: {args.metric}={score:.4f}", flush=True)

        mean_score = sum(scores) / len(scores)
        trial_idx += 1
        return mean_score, time.time() - t0

    optimizer = DEHB(cs=cs, min_fidelity=args.min_fidelity, max_fidelity=args.max_fidelity, eta=args.eta,
                     n_workers=1, seed=args.seed, output_path=str(study_dir), resume=resuming)
    trial_idx = len(optimizer.traj)  # resume: continue trial numbering instead of overwriting trial0, trial1, ...

    while len(optimizer.traj) < args.n_trials:
        job = optimizer.ask()
        mean_score, cost = evaluate(job["config"], job["fidelity"])
        optimizer.tell(job, {"fitness": sign * mean_score, "cost": cost,
                             "info": {"raw_score": mean_score, "fidelity": job["fidelity"]}})
        print(f"[dehb_tuner] {len(optimizer.traj)}/{args.n_trials} evals done, "
              f"incumbent {args.metric}={sign * optimizer.get_incumbents()[1]:.4f}", flush=True)

    inc_config, inc_fitness = optimizer.get_incumbents()
    best_reward, best_flags = split_config(inc_config)
    best = dict(value=sign * inc_fitness, params=dict(dict(inc_config)), reward_scales=best_reward, flags=best_flags)
    print("[dehb_tuner] best value:", best["value"])
    print("[dehb_tuner] best config:", json.dumps(best["params"], indent=2))
    json.dump(best, open(study_dir / "best.json", "w"), indent=2)
    if best_reward:  # the winning reward config, ready to point KINGFISHER_REWARD_CFG at directly
        yaml.safe_dump(deep_update(copy.deepcopy(base_reward_cfg), {"reward_scales": best_reward}),
                       open(study_dir / "best_reward_cfg.yaml", "w"))
        print(f"[dehb_tuner] best reward config written to {study_dir / 'best_reward_cfg.yaml'}")


if __name__ == "__main__":
    main()
