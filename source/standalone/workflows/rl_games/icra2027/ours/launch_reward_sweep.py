"""Launch a sweep of Isaac-KingfisherSail-Direct-v0 trainings, one per reward configuration.

Runs are dispatched into a tmux session, laid out as one WINDOW per device (e.g.
"cuda:0", "cuda:1") split into --jobs_per_device PANES each, so every concurrent
job gets its own visible, scrollable pane instead of every process's stdout mixing
together in one terminal. Runs are assigned to panes round-robin; each pane runs
its assigned runs one after another (train -> eval -> plot, next run only starting
after the previous one's chain finishes or fails).

This script only *dispatches* into tmux and returns immediately -- it does not
block waiting for training to finish. Attach with:

    tmux attach -t kingfisher_sweep

and use the standard tmux bindings to move around: Ctrl-b n / Ctrl-b p to switch
windows (devices), Ctrl-b <arrow> to switch panes (jobs) within a window, Ctrl-b d
to detach (leaves everything running).

For every entry in REWARD_SWEEP below, this creates outputs/reward_sweep/<run_name>/
and, inside it:

    1. Starts from the checked-in default reward config
       (source/extensions/.../direct/kingfisher_sail/reward_cfg.yaml), applies the
       entry's `overrides` on top of it, and saves the result as reward_cfg.yaml.
    2. Points the environment at that file via the KINGFISHER_REWARD_CFG environment
       variable, which KingfisherSailEnvCfg.__post_init__ reads through
       reward_cfg_loader.load_reward_cfg(). This is set per-command (inline env
       assignment), so concurrent panes never share state through it.
    3. Trains, giving the run a unique RL-Games experiment name and pinning it to its
       pane's device via Hydra CLI overrides (NOT by editing the checked-in agent
       yaml -- with several runs launching at once, editing a shared file out from
       under other in-flight processes would race):
           <ISAAC_PATH>/bin/python source/standalone/workflows/rl_games/train.py \
               --task Isaac-KingfisherSail-Direct-v0 --num_envs <N> --headless \
               --device <pane_device> \
               agent.params.config.name=kingfisher_direct_sweep_<run_name> \
               agent.params.config.device=<pane_device> agent.params.config.device_name=<pane_device>
    4. Plays the checkpoint just trained -- pointed at the right logs via
       --experiment_name (same reasoning: an explicit CLI arg instead of a shared
       file) -- which writes its per-timestep CSV as all_steps_swept_<run_name>.csv
       inside that same run directory:
           <ISAAC_PATH>/bin/python .../play_kingfisher_paper_eval.py \
               --task Isaac-KingfisherSail-Direct-v0 --num_envs <N> --headless \
               --device <pane_device> --experiment_name kingfisher_direct_sweep_<run_name>
    5. Plots that CSV, saving the figures into the same run directory:
           <ISAAC_PATH>/bin/python .../plot_paper_eval.py \
               --success-radius <R> --outdir outputs/reward_sweep/<run_name> \
               outputs/reward_sweep/<run_name>/all_steps_swept_<run_name>.csv

Seeds: pass --seeds 12,23,13 to run EVERY reward configuration on each of those
seeds (so you can tell a genuinely better reward shape from a lucky seed). Each
seed becomes a `seed_<seed>/` subdirectory under that configuration's directory
-- outputs/reward_sweep/<run_name>/seed_<seed>/ -- holding that seed's own copy
of reward_cfg.yaml, CSV, and plots; it also gets its own RL-Games experiment
name (.../kingfisher_direct_sweep_<run_name>_seed<seed>/), so seeds never share
a log directory. Omitting --seeds preserves the previous behaviour exactly: one
run per reward config, using the rl_games agent yaml's own default seed, with
no seed subdirectory.

Run from the repository root, e.g.:

    pysaac source/standalone/workflows/rl_games/icra2027/ours/launch_reward_sweep.py
    pysaac source/standalone/workflows/rl_games/icra2027/ours/launch_reward_sweep.py --devices cuda:0,cuda:1 --jobs_per_device 3
    pysaac source/standalone/workflows/rl_games/icra2027/ours/launch_reward_sweep.py --run_names baseline,high_energy_penalty
    pysaac source/standalone/workflows/rl_games/icra2027/ours/launch_reward_sweep.py --seeds 12,23,13
    pysaac source/standalone/workflows/rl_games/icra2027/ours/launch_reward_sweep.py --dry_run
    pysaac source/standalone/workflows/rl_games/icra2027/ours/launch_reward_sweep.py --skip_eval
    pysaac source/standalone/workflows/rl_games/icra2027/ours/launch_reward_sweep.py --tmux_session my_sweep
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

KINGFISHER_SAIL_TASK_DIR = (
    "source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/kingfisher_sail"
)
DEFAULT_REWARD_CFG = os.path.join(KINGFISHER_SAIL_TASK_DIR, "reward_cfg.yaml")

ICRA2027_OURS_DIR = "source/standalone/workflows/rl_games/icra2027/ours"
TRAIN_SCRIPT = "source/standalone/workflows/rl_games/train.py"
PLAY_SCRIPT = os.path.join(ICRA2027_OURS_DIR, "play_kingfisher_paper_eval.py")
PLOT_SCRIPT = os.path.join(ICRA2027_OURS_DIR, "plot_paper_eval.py")

SWEEP_OUTPUT_DIR = "outputs/reward_sweep"
TMUX_SCRIPTS_DIR = os.path.join(SWEEP_OUTPUT_DIR, "_tmux_scripts")

# -----------------------------------------------------------------------------
# Reward sweep definition
#
# One list of values per reward scale, matching the `reward_scales` section of
# reward_cfg.yaml. Leave a list EMPTY ([]) to keep that reward at its default
# value (from reward_cfg.yaml) for every run: it is then not swept and does not
# appear in run names.
#
# REWARD_SWEEP is generated as the cross product of every non-empty list below:
# one run per combination, with every other reward left at its default. With
# no lists populated, a single "baseline" run (pure defaults) is produced.
# -----------------------------------------------------------------------------
REWARD_SWEEP_AXES = {
    "distance_reward_scale": [],
    "distance_progress_reward_scale": [2.0, 5.0, 10., 20.0],
    "bearing_progress_reward_scale": [],
    "goal_reached_scale": [0, 1000, 2000],
    "energy_penalty_scale": [-5, -10, -20],
    "backwards_penalty_scale": [],
    "time_penalty_scale": [0, -1.0, -2.0, -5.0],
    "tack_penalty_scale": [],
    "bearing_penalty_scale": [],
    "beargin_penalty_coef": [],
    "lift_drag_ratio_scale": [],
    "acord_reward_scale": [],
    "speed_penalty_scale": [],
}

# Short aliases used to keep generated run names / log dirs readable.
_SHORT_NAMES = {
    "distance_reward_scale": "distR",
    "distance_progress_reward_scale": "distProg",
    "bearing_progress_reward_scale": "brgProg",
    "goal_reached_scale": "goal",
    "energy_penalty_scale": "energy",
    "backwards_penalty_scale": "backwards",
    "time_penalty_scale": "time",
    "tack_penalty_scale": "tack",
    "bearing_penalty_scale": "brgPen",
    "beargin_penalty_coef": "brgCoef",
    "lift_drag_ratio_scale": "liftDrag",
    "acord_reward_scale": "acord",
    "speed_penalty_scale": "speed",
}


def build_reward_sweep(axes: dict) -> list:
    """Cross product of every non-empty axis in `axes` into REWARD_SWEEP entries.

    Axes left as an empty list are dropped entirely, so that reward stays at its
    reward_cfg.yaml default in every generated run. If every axis is empty, a
    single pure-default "baseline" run is returned.
    """
    active_axes = {key: values for key, values in axes.items() if values}
    if not active_axes:
        return [{"run_name": "baseline", "overrides": {}}]

    keys = list(active_axes.keys())
    sweep = []
    for combo in itertools.product(*(active_axes[key] for key in keys)):
        overrides = dict(zip(keys, combo))
        run_name = "_".join(f"{_SHORT_NAMES.get(k, k)}{v:g}" for k, v in overrides.items())
        sweep.append({"run_name": run_name, "overrides": {"reward_scales": overrides}})
    return sweep


REWARD_SWEEP = build_reward_sweep(REWARD_SWEEP_AXES)


def deep_update(base: dict, overrides: dict) -> dict:
    """Recursively merge `overrides` into `base`, in place, and return it."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a reward-config sweep for Isaac-KingfisherSail-Direct-v0.")
    parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel training environments per run.")
    parser.add_argument(
        "--devices",
        type=str,
        default="cuda:0,cuda:1",
        help="Comma-separated GPUs to spread runs across, e.g. 'cuda:0,cuda:1'. One tmux window per device.",
    )
    parser.add_argument(
        "--jobs_per_device",
        type=int,
        default=3,
        help="Number of runs (train or eval, one at a time each) to keep in flight concurrently per device, "
        "as that many tmux panes split within that device's window. Tune to your VRAM budget: e.g. 3-4 for "
        "--num_envs 4096 on a 24GB GPU.",
    )
    parser.add_argument(
        "--run_names",
        type=str,
        default=None,
        help="Comma-separated subset of REWARD_SWEEP run_names to launch. Defaults to all.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Write the per-run reward config files but do not launch tmux/training/eval/plotting.",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        default=False,
        help="Train only: skip the play (evaluation) and plotting stages for every run.",
    )
    parser.add_argument("--eval_num_envs", type=int, default=10, help="Number of parallel environments for play/eval.")
    parser.add_argument(
        "--success_radius",
        type=float,
        default=None,
        help="--success-radius passed to plot_paper_eval.py. Defaults to the reward "
        "config's goal.goal_reached_threshold.",
    )
    parser.add_argument(
        "--tmux_session",
        type=str,
        default="kingfisher_sweep",
        help="Name of the tmux session to create. Must not already exist.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds to run every reward configuration on, e.g. '12,23,13'. Each becomes a "
        "seed_<seed>/ subdirectory under that config's run directory, with its own reward_cfg.yaml copy, CSV, "
        "and plots, and its own RL-Games experiment name. Omit to run once per config using the rl_games agent "
        "yaml's own default seed (no seed subdirectory).",
    )
    return parser.parse_args()


def prepare_run(run: dict, seed: int | None, base_reward_cfg: dict) -> tuple[str, str]:
    """Write this (run, seed)'s reward_cfg.yaml and return (effective_dir, reward_cfg_path).

    With no seed, effective_dir is the run's own directory (previous behavior, unchanged).
    With a seed, effective_dir is a seed_<seed>/ subdirectory under it, so every seed of a
    given reward config gets its own reward_cfg.yaml copy, CSV, and plots, nested under
    that config's directory.
    """
    run_name = run["run_name"]
    run_dir = os.path.abspath(os.path.join(SWEEP_OUTPUT_DIR, run_name))
    effective_dir = run_dir if seed is None else os.path.join(run_dir, f"seed_{seed}")
    os.makedirs(effective_dir, exist_ok=True)

    reward_cfg = deep_update(copy.deepcopy(base_reward_cfg), run.get("overrides", {}))
    reward_cfg_path = os.path.join(effective_dir, "reward_cfg.yaml")
    with open(reward_cfg_path, "w") as f:
        yaml.safe_dump(reward_cfg, f)

    return effective_dir, reward_cfg_path


def build_run_block(run: dict, seed: int | None, run_dir: str, reward_cfg_path: str, device: str,
                     python_bin: str, args: argparse.Namespace, success_radius: float) -> str:
    """Shell commands (as one string) that train -> eval -> plot a single (run, seed) on `device`."""
    run_name = run["run_name"]
    seed_suffix = "" if seed is None else f"_seed{seed}"
    experiment_name = f"kingfisher_direct_sweep_{run_name}{seed_suffix}"
    env_assign = f"KINGFISHER_REWARD_CFG={shlex.quote(reward_cfg_path)}"

    train_cmd_parts = [
        env_assign, shlex.quote(python_bin), shlex.quote(TRAIN_SCRIPT),
        "--task", TASK,
        "--num_envs", str(args.num_envs),
        "--headless",
        "--device", device,
    ]
    if seed is not None:
        train_cmd_parts += ["--seed", str(seed)]
    train_cmd_parts += [
        f"agent.params.config.name={experiment_name}",
        f"agent.params.config.device={device}",
        f"agent.params.config.device_name={device}",
    ]
    stages = [" ".join(train_cmd_parts)]

    if not args.skip_eval:
        play_cmd = " ".join([
            env_assign, shlex.quote(python_bin), shlex.quote(PLAY_SCRIPT),
            "--task", TASK,
            "--num_envs", str(args.eval_num_envs),
            "--headless",
            "--device", device,
            "--experiment_name", experiment_name,
        ])
        # Matches play_kingfisher_paper_eval.py's own naming: it names/places the CSV after
        # the basename of KINGFISHER_REWARD_CFG's directory, which is run_dir here -- either
        # the run_name itself (no seed) or "seed_<seed>" (nested under the run's directory).
        csv_path = os.path.join(run_dir, f"all_steps_swept_{os.path.basename(run_dir)}.csv")
        plot_cmd = " ".join([
            shlex.quote(python_bin), shlex.quote(PLOT_SCRIPT),
            "--success-radius", str(success_radius),
            "--outdir", shlex.quote(run_dir),
            shlex.quote(csv_path),
        ])
        stages += [play_cmd, plot_cmd]

    label = run_name if seed is None else f"{run_name} seed={seed}"
    banner = f"echo '=== [{device}][{label}] starting ({len(stages)} stage(s)) ==='"
    chained = " && \\\n  ".join(stages)
    return f"{banner}\n{chained}"


def write_slot_script(slot_label: str, run_blocks: list, script_path: str) -> None:
    """Write one pane's script: its assigned jobs, back to back, each independent of the others."""
    lines = ["#!/usr/bin/env bash", ""]
    for block in run_blocks:
        lines.append(block)
        lines.append("")
    lines.append(f"echo '=== slot {slot_label}: all {len(run_blocks)} assigned job(s) finished ==='")
    lines.append("exec bash")  # keep the pane open at an interactive shell afterwards
    with open(script_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(script_path, 0o755)


def tmux(*tmux_args: str) -> None:
    subprocess.run(["tmux", *tmux_args], check=True)


def tmux_pane_indices(window_target: str) -> list:
    result = subprocess.run(
        ["tmux", "list-panes", "-t", window_target, "-F", "#{pane_index}"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.split()


def launch_tmux_sweep(slot_scripts: dict, devices: list, jobs_per_device: int, session_name: str,
                       cwd: str) -> None:
    if subprocess.run(["tmux", "has-session", "-t", session_name], capture_output=True).returncode == 0:
        raise RuntimeError(
            f"A tmux session named '{session_name}' already exists. Attach with `tmux attach -t {session_name}`, "
            f"kill it with `tmux kill-session -t {session_name}`, or pick a different --tmux_session."
        )

    tmux("new-session", "-d", "-s", session_name, "-n", devices[0], "-c", cwd)

    for device_idx, device in enumerate(devices):
        if device_idx == 0:
            window_target = f"{session_name}:0"
        else:
            tmux("new-window", "-t", session_name, "-n", device, "-c", cwd)
            window_target = f"{session_name}:{device_idx}"

        for _ in range(jobs_per_device - 1):
            tmux("split-window", "-t", window_target, "-c", cwd)
        if jobs_per_device > 1:
            tmux("select-layout", "-t", window_target, "tiled")

        for pane_idx, pane_num in enumerate(tmux_pane_indices(window_target)):
            script_path = slot_scripts.get((device, pane_idx))
            if script_path is None:
                continue  # more panes than runs assigned to this device (fewer runs than slots)
            pane_target = f"{window_target}.{pane_num}"
            tmux("send-keys", "-t", pane_target, f"bash {shlex.quote(script_path)}", "Enter")


def main() -> None:
    args = parse_args()

    if shutil.which("tmux") is None:
        raise RuntimeError("tmux is not installed / not on PATH. Install it (e.g. `apt install tmux`) and retry.")

    isaac_path = os.environ.get("ISAAC_PATH")
    if not isaac_path:
        raise RuntimeError("ISAAC_PATH environment variable is not set (used to locate the Isaac Lab python binary).")
    python_bin = os.path.join(isaac_path, "bin", "python")

    with open(DEFAULT_REWARD_CFG, "r") as f:
        base_reward_cfg = yaml.safe_load(f)

    default_success_radius = base_reward_cfg.get("goal", {}).get("goal_reached_threshold", 0.3)
    success_radius = args.success_radius if args.success_radius is not None else default_success_radius

    os.makedirs(SWEEP_OUTPUT_DIR, exist_ok=True)

    run_filter = args.run_names.split(",") if args.run_names else None
    runs_to_launch = [r for r in REWARD_SWEEP if not run_filter or r["run_name"] in run_filter]
    for r in REWARD_SWEEP:
        if run_filter and r["run_name"] not in run_filter:
            print(f"Skipping '{r['run_name']}' (not in --run_names).")

    seeds = [int(s.strip()) for s in args.seeds.split(",")] if args.seeds else [None]
    jobs_to_launch = [(run, seed) for run in runs_to_launch for seed in seeds]

    def job_label(run: dict, seed) -> str:
        return run["run_name"] if seed is None else f"{run['run_name']}(seed={seed})"

    if args.dry_run:
        for run, seed in jobs_to_launch:
            run_dir, reward_cfg_path = prepare_run(run, seed, base_reward_cfg)
            print(f"[dry_run] Wrote reward config for '{job_label(run, seed)}' to '{reward_cfg_path}'.")
        return

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    slots = [(device, pane_idx) for device in devices for pane_idx in range(args.jobs_per_device)]
    total_slots = len(slots)

    # Round-robin assignment of jobs to slots; each slot gets its own ordered list of jobs.
    jobs_per_slot = {slot: [] for slot in slots}
    for i, job in enumerate(jobs_to_launch):
        jobs_per_slot[slots[i % total_slots]].append(job)

    os.makedirs(TMUX_SCRIPTS_DIR, exist_ok=True)
    cwd = os.getcwd()

    slot_scripts = {}
    for (device, pane_idx), jobs in jobs_per_slot.items():
        if not jobs:
            continue
        blocks = []
        for run, seed in jobs:
            run_dir, reward_cfg_path = prepare_run(run, seed, base_reward_cfg)
            blocks.append(build_run_block(run, seed, run_dir, reward_cfg_path, device, python_bin, args, success_radius))
        slot_label = f"{device}#{pane_idx}"
        script_path = os.path.join(TMUX_SCRIPTS_DIR, f"{device.replace(':', '')}_{pane_idx}.sh")
        write_slot_script(slot_label, blocks, script_path)
        slot_scripts[(device, pane_idx)] = script_path
        print(f"[{slot_label}] {len(jobs)} job(s): {[job_label(r, s) for r, s in jobs]}")

    print(
        f"=== Dispatching {len(jobs_to_launch)} job(s) ({len(runs_to_launch)} reward config(s) x "
        f"{len(seeds)} seed(s)) into tmux session '{args.tmux_session}': "
        f"{len(devices)} window(s) (one per device) x {args.jobs_per_device} pane(s) each "
        f"= {total_slots} slot(s) ==="
    )
    launch_tmux_sweep(slot_scripts, devices, args.jobs_per_device, args.tmux_session, cwd)

    print(f"Attach with: tmux attach -t {args.tmux_session}")
    print("Ctrl-b n / Ctrl-b p to switch devices (windows), Ctrl-b <arrow> to switch jobs (panes), Ctrl-b d to detach.")


if __name__ == "__main__":
    main()
