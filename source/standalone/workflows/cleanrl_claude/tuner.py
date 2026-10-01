"""Optuna hyper-parameter search around any script of this directory (in the spirit of CleanRL's Tuner).

Every trial launches the script as a fresh process (Isaac Sim cannot be restarted inside one process), passes the
sampled hyper-parameters as CLI flags, and reads the score from the json the script writes (--result_file).

    python tuner.py --script ppo.py --n_trials 20 --extra_args "--num_envs 512 --total_timesteps 2000000"
    python tuner.py --script sac.py --devices cuda:0 cuda:1 --n_jobs 2 --seeds 2 --space "tau:float:0.001:0.05:log"

A space is  name:float:low:high[:log] | name:int:low:high | name:cat:a,b,c   ("64/64" in a cat = "--name 64 64").
--space overrides the default space of that script for the same name, or adds a new one. Pass every entry after
ONE "--space" (space-separated), not one "--space" per entry: a second "--space" replaces the first instead of
adding to it (a tyro tuple-flag quirk), e.g. --space "a:float:0:1" "b:int:1:5", not --space "a:..." --space "b:...".
Studies are stored in sqlite, so an interrupted search resumes when you re-run the same command.
"""
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import optuna
import tyro

HERE = Path(__file__).parent
PPO_SPACE = ["learning_rate:float:1e-5:1e-2:log", "gamma:float:0.95:0.999", "gae_lambda:float:0.9:0.99",
             "clip_coef:float:0.1:0.3", "ent_coef:float:1e-5:1e-1:log", "update_epochs:int:3:10",
             "num_steps:cat:24,48,96", "vf_coef:float:0.5:4.0"]
OFF_SPACE = ["gamma:float:0.95:0.999", "batch_size:cat:128,256,512", "updates_per_step:int:1:4", "hidden:cat:128,256,512"]
DEFAULT_SPACES = {
    "ppo": PPO_SPACE,
    "ppo_rnn": PPO_SPACE + ["lstm_hidden:cat:32,64,128"],
    "ppo_rnd": PPO_SPACE + ["int_coef:float:0.1:2.0", "ext_coef:float:1.0:4.0", "rnd_lr:float:1e-5:1e-3:log"],
    "ppo_rnd_rnn": PPO_SPACE + ["int_coef:float:0.1:2.0", "ext_coef:float:1.0:4.0", "rnd_lr:float:1e-5:1e-3:log",
                                "rnd_cell:cat:lstm,gru", "rnd_seq_len:cat:8,16,24,48"],
    # "bins" is one choice per trial, each choice a full per-dimension tuple ("/"-joined -- see the module
    # docstring); the 4-token defaults below assume the Kingfisher task's 4-D action space, adjust via --space
    # for another task/action count.
    "ppo_discrete": PPO_SPACE + ["bins:cat:2/2/3/3,3/3/5/5,2/2/5/5,3/3/9/9"],
    "sac": OFF_SPACE + ["q_lr:float:1e-4:3e-3:log", "policy_lr:float:1e-4:1e-3:log", "tau:float:0.001:0.05:log"],
    "ddpg": OFF_SPACE + ["q_lr:float:1e-4:3e-3:log", "actor_lr:float:1e-4:1e-3:log", "tau:float:0.001:0.05:log",
                         "exploration_noise:float:0.05:0.3"],
    "dqn": OFF_SPACE + ["learning_rate:float:1e-5:1e-3:log", "bins:cat:2/2/3/3,3/3/5/5,2/2/5/5",
                        "target_frequency:cat:250,500,1000,2000", "exploration_fraction:float:0.1:0.6"],
    "drqn": OFF_SPACE + ["learning_rate:float:1e-5:1e-3:log", "bins:cat:2/2/3/3,3/3/5/5,2/2/5/5", "seq_len:cat:8,16,32",
                         "target_frequency:cat:250,500,1000,2000"],
}


@dataclass
class Args:
    script: str  # e.g. ppo.py (looked up next to this file)
    n_trials: int = 20
    study_name: str = ""  # default: the script name
    storage: str = ""  # optuna storage url (default: sqlite:///logs/cleanrl_claude_tuning/<study>.db)
    space: tuple[str, ...] = ()  # extra / overriding search-space entries
    extra_args: str = ""  # fixed flags forwarded to every trial, e.g. "--num_envs 512 --total_timesteps 2000000"
    devices: tuple[str, ...] = ("cuda:0",)  # trials are assigned to devices round-robin
    n_jobs: int = 1  # trials running in parallel
    seeds: int = 1  # runs per trial (the score is their mean)
    metric: str = "mean_return"  # mean_return | mean_step_reward (used as a fallback if no episode finished)
    direction: str = "maximize"
    log_root: str = "logs/cleanrl_claude_tuning"


def suggest(trial, spec):
    name, kind, *rest = spec.split(":")
    if kind == "float":
        return name, trial.suggest_float(name, float(rest[0]), float(rest[1]), log=len(rest) > 2 and rest[2] == "log")
    if kind == "int":
        return name, trial.suggest_int(name, int(rest[0]), int(rest[1]))
    if kind == "cat":
        return name, trial.suggest_categorical(name, rest[0].split(","))
    raise ValueError(f"unknown space kind '{kind}' in '{spec}'")


def main():
    args = tyro.cli(Args)
    script = HERE / args.script
    algo = script.stem
    spaces = {s.split(":")[0]: s for s in DEFAULT_SPACES.get(algo, [])}
    spaces.update({s.split(":")[0]: s for s in args.space})
    study_name = args.study_name or algo
    os.makedirs(args.log_root, exist_ok=True)
    storage = args.storage or f"sqlite:///{args.log_root}/{study_name}.db"
    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True, direction=args.direction)

    def objective(trial):
        flags = []
        for spec in spaces.values():
            name, value = suggest(trial, spec)
            flags += [f"--{name}", *str(value).split("/")]
        device = args.devices[trial.number % len(args.devices)]
        scores = []
        for seed in range(args.seeds):
            tag = f"trial{trial.number}_seed{seed}"
            result = Path(args.log_root) / study_name / f"{tag}.json"
            result.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(script), "--headless", "--device", device, "--seed", str(seed + 1),
                   "--exp_name", tag, "--log_root", str(Path(args.log_root) / study_name / "runs"),
                   "--result_file", str(result), *shlex.split(args.extra_args), *flags]
            print(f"[tuner] trial {trial.number} seed {seed}: {' '.join(cmd[2:])}", flush=True)
            t0 = time.time()
            with open(result.with_suffix(".log"), "w") as f:
                code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
            if code != 0 or not result.exists():
                raise optuna.TrialPruned(f"trial process failed (exit {code}), see {result.with_suffix('.log')}")
            out = json.load(open(result))
            score = out[args.metric]
            if score is None or math.isnan(score):
                score = out["mean_step_reward"]
            scores.append(score)
            print(f"[tuner] trial {trial.number} seed {seed}: {args.metric}={score:.4f} ({time.time() - t0:.0f}s)", flush=True)
        return sum(scores) / len(scores)

    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs)
    print("[tuner] best value:", study.best_value)
    print("[tuner] best params:", json.dumps(study.best_params, indent=2))
    json.dump(dict(value=study.best_value, params=study.best_params), open(Path(args.log_root) / f"{study_name}_best.json", "w"), indent=2)


if __name__ == "__main__":
    main()
