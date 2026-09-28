"""Launch a sweep of cleanrl_claude trainings in tmux: algorithm x reward config x hyper-parameters x seeds.

Same idea and same output layout as rl_games/icra2027/ours/launch_reward_sweep.py, so plot_paper_eval_3D.py works
on the results without changes. For every (run, seed) one tmux pane runs, one after the other:

    train  ->  eval.py (paper sweep, writes the per-timestep csv)  ->  plot_paper_eval_3D.py

Layout (run_name = <algo>_<swept values>, e.g. ppo_distProg20_learning_rate0.0005):

    outputs/cleanrl_sweep/<run_name>/seed_<seed>/
        reward_cfg.yaml                       reward config of this run (KINGFISHER_REWARD_CFG points at it)
        train/<algo>/<task>/run/              tensorboard summaries, args.yaml, model_last.pt
        all_steps_swept_seed_<seed>.csv       eval csv (what plot_paper_eval_3D.py reads)
        eval_summary.json, *.png              success rate / time / energy summary and the figures

Aggregate the seeds of one configuration (no launcher needed):
    pysaac source/standalone/workflows/rl_games/icra2027/ours/plot_paper_eval_3D.py outputs/cleanrl_sweep/<run_name>/

One tmux window per device, --jobs_per_device panes each, jobs assigned round-robin. This script only dispatches
and returns; attach with `tmux attach -t cleanrl_sweep` (Ctrl-b n/p windows, Ctrl-b <arrow> panes, Ctrl-b d detach).

    pysaac source/standalone/workflows/cleanrl_claude/launch_sweep.py --algos ppo --seeds 1,2,3
    pysaac source/standalone/workflows/cleanrl_claude/launch_sweep.py --algos ppo,sac,dqn --devices cuda:0,cuda:1
    pysaac source/standalone/workflows/cleanrl_claude/launch_sweep.py --algos sac --extra_args "--her" --run_tag her
    pysaac source/standalone/workflows/cleanrl_claude/launch_sweep.py --dry_run
"""
import argparse
import copy
import itertools
import os
import shlex
import shutil
import subprocess

import yaml

TASK = "Isaac-KingfisherSail-Direct-v0"
KINGFISHER_SAIL_TASK_DIR = "source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/kingfisher_sail"
DEFAULT_REWARD_CFG = os.path.join(KINGFISHER_SAIL_TASK_DIR, "reward_cfg.yaml")
CLEANRL_DIR = "source/standalone/workflows/cleanrl_claude"
PLOT_SCRIPT = "source/standalone/workflows/rl_games/icra2027/ours/plot_paper_eval_3D.py"
ALGOS = ["ppo", "ppo_rnn", "ppo_rnd", "ppo_rnd_rnn", "ppo_discrete", "sac", "ddpg", "dqn", "drqn"]

# -----------------------------------------------------------------------------
# Sweep definition (cross product of every non-empty list; [] keeps the default)
# -----------------------------------------------------------------------------
# Reward scales of reward_cfg.yaml (exactly as in launch_reward_sweep.py). Only has an effect if the env reads
# KINGFISHER_REWARD_CFG (KingfisherSailEnvCfg.__post_init__ -> reward_cfg_loader.load_reward_cfg).
REWARD_SWEEP_AXES = {
    "distance_reward_scale": [],
    "distance_progress_reward_scale": [],
    "bearing_progress_reward_scale": [],
    "goal_reached_scale": [],
    "energy_penalty_scale": [],
    "backwards_penalty_scale": [],
    "time_penalty_scale": [],
    "tack_penalty_scale": [],
    "bearing_penalty_scale": [],
    "beargin_penalty_coef": [],
    "lift_drag_ratio_scale": [],
    "acord_reward_scale": [],
    "speed_penalty_scale": [],
}
# Algorithm flags to sweep, per algorithm (flag name as in the script's Args / configs/<algo>.yaml).
# Example: "ppo": {"learning_rate": [3e-4, 5e-4], "num_steps": [24, 48]}
HPARAM_AXES = {algo: {} for algo in ALGOS}

_SHORT_NAMES = {
    "distance_reward_scale": "distR", "distance_progress_reward_scale": "distProg",
    "bearing_progress_reward_scale": "brgProg", "goal_reached_scale": "goal", "energy_penalty_scale": "energy",
    "backwards_penalty_scale": "backwards", "time_penalty_scale": "time", "tack_penalty_scale": "tack",
    "bearing_penalty_scale": "brgPen", "beargin_penalty_coef": "brgCoef", "lift_drag_ratio_scale": "liftDrag",
    "acord_reward_scale": "acord", "speed_penalty_scale": "speed",
}


def build_runs(algo: str, run_tag: str) -> list:
    """One entry per combination: {"run_name", "algo", "reward_overrides", "hparams"}."""
    axes = [("reward", k, v) for k, v in REWARD_SWEEP_AXES.items() if v]
    axes += [("hparam", k, v) for k, v in HPARAM_AXES.get(algo, {}).items() if v]
    prefix = "_".join(p for p in (algo, run_tag) if p)
    if not axes:
        return [dict(run_name=prefix, algo=algo, reward_overrides={}, hparams={})]
    runs = []
    for combo in itertools.product(*(vals for _, _, vals in axes)):
        reward, hparams, parts = {}, {}, []
        for (kind, key, _), v in zip(axes, combo):
            (reward if kind == "reward" else hparams)[key] = v
            parts.append(f"{_SHORT_NAMES.get(key, key)}{v:g}")
        runs.append(dict(run_name="_".join([prefix, *parts]), algo=algo, hparams=hparams,
                         reward_overrides={"reward_scales": reward} if reward else {}))
    return runs


def deep_update(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch a cleanrl_claude sweep (train -> eval -> plot) in tmux.")
    p.add_argument("--algos", default="ppo", help=f"Comma-separated subset of {ALGOS}.")
    p.add_argument("--task", default=TASK)
    p.add_argument("--num_envs", type=int, default=None, help="Training envs (default: the algorithm's config).")
    p.add_argument("--total_timesteps", type=int, default=None, help="Training env steps (default: the algorithm's config).")
    p.add_argument("--extra_args", default="", help='Extra flags for every training run, e.g. "--her --hidden 512".')
    p.add_argument("--run_tag", default="", help="Added to run names (use it to tell apart runs with different --extra_args).")
    p.add_argument("--devices", default="cuda:0,cuda:1", help="Comma-separated GPUs; one tmux window each.")
    p.add_argument("--jobs_per_device", type=int, default=2, help="Concurrent jobs (tmux panes) per device.")
    p.add_argument("--seeds", default=None, help="Comma-separated seeds, e.g. '1,2,3' -> seed_<seed>/ subdirectories. "
                   "Omit for one run per config with the config's own seed and no seed subdirectory.")
    p.add_argument("--run_names", default=None, help="Comma-separated subset of generated run names.")
    p.add_argument("--sweep_dir", default="outputs/cleanrl_sweep")
    p.add_argument("--skip_eval", action="store_true", help="Train only.")
    p.add_argument("--eval_num_envs", type=int, default=10)
    p.add_argument("--eval_episodes_per_env", type=int, default=0, help="0 = the full paper sweep (angles x speeds).")
    p.add_argument("--success_radius", type=float, default=None, help="Default: goal.goal_reached_threshold of the reward config.")
    p.add_argument("--tmux_session", default="cleanrl_sweep")
    p.add_argument("--dry_run", action="store_true", help="Write the reward configs and print the commands, launch nothing.")
    return p.parse_args()


def prepare_run(sweep_dir: str, run: dict, seed, base_reward_cfg: dict) -> tuple:
    """Create this (run, seed)'s directory and reward_cfg.yaml; returns (effective_dir, reward_cfg_path)."""
    run_dir = os.path.abspath(os.path.join(sweep_dir, run["run_name"]))
    effective_dir = run_dir if seed is None else os.path.join(run_dir, f"seed_{seed}")
    os.makedirs(effective_dir, exist_ok=True)
    reward_cfg = deep_update(copy.deepcopy(base_reward_cfg), run["reward_overrides"])
    path = os.path.join(effective_dir, "reward_cfg.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(reward_cfg, f)
    return effective_dir, path


def build_run_block(run, seed, run_dir, reward_cfg_path, device, python_bin, args, success_radius) -> str:
    algo = run["algo"]
    env_assign = f"KINGFISHER_REWARD_CFG={shlex.quote(reward_cfg_path)}"
    train_root = os.path.join(run_dir, "train")
    checkpoint = os.path.join(train_root, algo, args.task, "run", "model_last.pt")
    csv_path = os.path.join(run_dir, f"all_steps_swept_{os.path.basename(run_dir)}.csv")  # what plot_paper_eval_3D.py expects

    train = [env_assign, shlex.quote(python_bin), os.path.join(CLEANRL_DIR, f"{algo}.py"), "--headless", "--device", device,
             "--task", args.task, "--exp_name", "run", "--log_root", shlex.quote(train_root)]
    if seed is not None:
        train += ["--seed", str(seed)]
    if args.num_envs:
        train += ["--num_envs", str(args.num_envs)]
    if args.total_timesteps:
        train += ["--total_timesteps", str(args.total_timesteps)]
    for k, v in run["hparams"].items():
        train += [f"--{k}", str(v)]
    train.append(args.extra_args)
    stages = [" ".join(train)]

    if not args.skip_eval:
        stages.append(" ".join([
            env_assign, shlex.quote(python_bin), os.path.join(CLEANRL_DIR, "eval.py"), "--headless", "--device", device,
            "--num_envs", str(args.eval_num_envs), "--episodes_per_env", str(args.eval_episodes_per_env),
            "--checkpoint", shlex.quote(checkpoint), "--csv_path", shlex.quote(csv_path)]))
        stages.append(" ".join([shlex.quote(python_bin), PLOT_SCRIPT, "--success-radius", str(success_radius),
                                "--outdir", shlex.quote(run_dir), shlex.quote(csv_path)]))

    label = run["run_name"] if seed is None else f"{run['run_name']} seed={seed}"
    return f"echo '=== [{device}][{label}] starting ({len(stages)} stage(s)) ==='\n" + " && \\\n  ".join(stages)


def write_slot_script(slot_label: str, blocks: list, path: str) -> None:
    lines = ["#!/usr/bin/env bash", ""]
    for block in blocks:
        lines += [block, ""]
    lines += [f"echo '=== slot {slot_label}: all {len(blocks)} assigned job(s) finished ==='", "exec bash"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(path, 0o755)


def tmux(*a: str) -> None:
    subprocess.run(["tmux", *a], check=True)


def launch_tmux(slot_scripts: dict, devices: list, jobs_per_device: int, session: str, cwd: str) -> None:
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        raise RuntimeError(f"tmux session '{session}' already exists: attach (`tmux attach -t {session}`), kill it, "
                           f"or pick another --tmux_session.")
    tmux("new-session", "-d", "-s", session, "-n", devices[0], "-c", cwd)
    for i, device in enumerate(devices):
        if i > 0:
            tmux("new-window", "-t", session, "-n", device, "-c", cwd)
        window = f"{session}:{i}"
        for _ in range(jobs_per_device - 1):
            tmux("split-window", "-t", window, "-c", cwd)
        if jobs_per_device > 1:
            tmux("select-layout", "-t", window, "tiled")
        panes = subprocess.run(["tmux", "list-panes", "-t", window, "-F", "#{pane_index}"], capture_output=True,
                               text=True, check=True).stdout.split()
        for idx, pane in enumerate(panes):
            if (device, idx) in slot_scripts:
                tmux("send-keys", "-t", f"{window}.{pane}", f"bash {shlex.quote(slot_scripts[(device, idx)])}", "Enter")


def main() -> None:
    args = parse_args()
    isaac_path = os.environ.get("ISAAC_PATH")
    if not isaac_path:
        raise RuntimeError("ISAAC_PATH is not set (used to locate the Isaac Lab python binary).")
    python_bin = os.path.join(isaac_path, "bin", "python")
    if not args.dry_run and shutil.which("tmux") is None:
        raise RuntimeError("tmux is not installed / not on PATH.")

    with open(DEFAULT_REWARD_CFG) as f:
        base_reward_cfg = yaml.safe_load(f)
    success_radius = args.success_radius or base_reward_cfg.get("goal", {}).get("goal_reached_threshold", 0.3)

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    bad = [a for a in algos if a not in ALGOS]
    if bad:
        raise ValueError(f"unknown algorithm(s) {bad}; choose from {ALGOS}")
    runs = [r for a in algos for r in build_runs(a, args.run_tag)]
    if args.run_names:
        keep = args.run_names.split(",")
        runs = [r for r in runs if r["run_name"] in keep]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [None]
    jobs = [(r, s) for r in runs for s in seeds]
    os.makedirs(args.sweep_dir, exist_ok=True)

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    slots = [(d, i) for d in devices for i in range(args.jobs_per_device)]
    per_slot = {s: [] for s in slots}
    for i, job in enumerate(jobs):
        per_slot[slots[i % len(slots)]].append(job)

    scripts_dir = os.path.join(args.sweep_dir, "_tmux_scripts")
    os.makedirs(scripts_dir, exist_ok=True)
    slot_scripts = {}
    for (device, idx), slot_jobs in per_slot.items():
        if not slot_jobs:
            continue
        blocks = []
        for run, seed in slot_jobs:
            run_dir, cfg_path = prepare_run(args.sweep_dir, run, seed, base_reward_cfg)
            blocks.append(build_run_block(run, seed, run_dir, cfg_path, device, python_bin, args, success_radius))
        label = f"{device}#{idx}"
        path = os.path.join(scripts_dir, f"{device.replace(':', '')}_{idx}.sh")
        write_slot_script(label, blocks, path)
        slot_scripts[(device, idx)] = path
        print(f"[{label}] {len(slot_jobs)} job(s): {[r['run_name'] + ('' if s is None else f'(seed={s})') for r, s in slot_jobs]}")
        if args.dry_run:
            print("\n".join(blocks))

    if args.dry_run:
        print(f"[dry_run] wrote {len(jobs)} job(s) under {args.sweep_dir}; nothing launched.")
        return
    print(f"=== Dispatching {len(jobs)} job(s) into tmux session '{args.tmux_session}' "
          f"({len(devices)} window(s) x {args.jobs_per_device} pane(s)) ===")
    launch_tmux(slot_scripts, devices, args.jobs_per_device, args.tmux_session, os.getcwd())
    print(f"Attach with: tmux attach -t {args.tmux_session}")


if __name__ == "__main__":
    main()
