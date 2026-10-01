# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of Isaac Lab (Isaac Sim 4.2, Python 3.10, old `omni.isaac.lab*` package names) used for RL research on an
autonomous sailing boat: the **Kingfisher with a sail** (`Isaac-KingfisherSail-Direct-v0`). Almost all active work is in:

- the task: `source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/kingfisher_sail/`
- the rl_games paper pipeline: `source/standalone/workflows/rl_games/icra2027/ours/`
- the CleanRL-style algorithm suite: `source/standalone/workflows/cleanrl_claude/` (has its own detailed `README.md`; read it before changing anything there)

The rest of Isaac Lab is upstream code and is rarely touched.

## Running things

All scripts run from the repo root with `pysaac` (the Isaac Lab python launcher, used in every script docstring).
Isaac Sim flags such as `--headless`, `--device cuda:1` and `--num_envs N` work on every training/eval script.

```bash
# rl_games training / play of the task
pysaac source/standalone/workflows/rl_games/train.py --task Isaac-KingfisherSail-Direct-v0 --headless --num_envs 4096
pysaac source/standalone/workflows/rl_games/icra2027/ours/play_kingfisher_paper_eval.py --task Isaac-KingfisherSail-Direct-v0 --headless --num_envs 10 --experiment_name <name>

# reward-config sweep in tmux (train -> paper eval -> plot); --dry_run prints commands only
pysaac source/standalone/workflows/rl_games/icra2027/ours/launch_reward_sweep.py --devices cuda:0,cuda:1 --jobs_per_device 3 --seeds 12,23,13
tmux attach -t kingfisher_sweep

# cleanrl_claude (algo in ppo, ppo_rnn, ppo_rnd, ppo_rnd_rnn, ppo_discrete, sac, ddpg, dqn, drqn)
pysaac source/standalone/workflows/cleanrl_claude/ppo.py --headless --num_envs 4096
pysaac source/standalone/workflows/cleanrl_claude/eval.py --headless --num_envs 10 --checkpoint <run>/model_last.pt --episodes_per_env 2
pysaac source/standalone/workflows/cleanrl_claude/launch_sweep.py --algos ppo,sac --seeds 1,2,3   # tmux session: cleanrl_sweep

# ONNX export needs no Isaac Sim: use the plain Isaac Lab python, not pysaac
/home/GTL/fsangare/isaaclab/bin/python source/standalone/workflows/cleanrl_claude/export_onnx.py --checkpoint <run>/model_last.pt

# TensorBoard
pysaac -m tensorboard.main --logdir logs/cleanrl_claude --bind_all
```

Formatting/linting is the upstream Isaac Lab pre-commit setup (`./isaaclab.sh -f`). There are no tests for the
Kingfisher code. The practical smoke test is a short run with tiny settings (e.g. `--num_envs 32 --total_timesteps 15000`
for cleanrl_claude, or `launch_sweep.py` with those settings) followed by `eval.py`.

Isaac Sim cannot be restarted inside one Python process. That is why the sweep launchers and `tuner.py` start every run as a separate subprocess.

## Architecture

### The environment (`direct/kingfisher_sail/`)
- `kingfisher_sail_env.py`: `KingfisherSailEnvCfg` + `KingfisherSailEnv` (a `DirectRLEnv`). Actions are 4-D in `[-1, 1]`:
  `[thruster_left, thruster_right, rudder, sail]`. The observation has 16 dims. It runs at 20 Hz (physics 60 Hz, decimation 3), and episodes last 250 s.
- The boat physics are **custom modules added to the core `omni.isaac.lab` extension**, not upstream Isaac Lab:
  `omni.isaac.lab.physics.{hydrostatics,hydrodynamics,foil_dynamics}` and `omni.isaac.lab.actuator_force.{foil_actuator_force,actuator_force}`.
  Their parameters come from `omni.isaac.lab_tasks.utils.my_utils.boat_config`. The asset is `KINGFISHER_SAIL_CFG` in `omni.isaac.lab_assets`.
  These forces are computed in torch and applied as external wrenches on the body.
- Wind is randomized per episode (points of sail flagged by `*_flag` attributes on the cfg). Evaluation uses a fixed
  wind-angle × wind-speed grid per env slot (`is_paper_eval`).
- Reward components are logged as `Episode_Reward/<n>_<name>` (`1_distance_progress` … `8_lift_drag_ratio`), along with
  `Episode_Termination/*`, `Metrics/*` and `Contexts/*`, through `extras` at reset. Both training stacks pick them up automatically.
- Agent configs live in `agents/` (`rl_games_ppo_cfg.yaml` is the reference PPO config).

### Reward configuration (important)
`KingfisherSailEnvCfg.__post_init__` calls `reward_cfg_loader.load_reward_cfg`, which **overwrites cfg attributes from YAML**.
It resolves the file in this order: explicit path → `$KINGFISHER_REWARD_CFG` → `reward_cfg.yaml` next to the env. So the values
in `reward_cfg.yaml` win over the defaults written in the Python class. Edit the YAML (or add the attribute to the cfg class
first: unknown keys raise `ValueError`). Sweeps give each run its own YAML via an inline `KINGFISHER_REWARD_CFG=...`.
They never edit shared files, because concurrent runs would race on them.

### Evaluation / paper pipeline (`rl_games/icra2027/ours/`)
`play_kingfisher_paper_eval.py` writes one CSV row per env step (`all_steps_swept_<run>.csv`). `plot_paper_eval.py` and
`plot_paper_eval_3D.py` read those CSVs. `plot_paper_eval_3D.py <run_dir>/` also aggregates `seed_*/` subdirectories.
`cleanrl_claude/eval.py` writes the **same CSV schema** and uses the same `outputs/<sweep>/<run_name>/seed_<seed>/` layout,
so both stacks share the plotting scripts. Keep that schema compatible when changing either side.

### cleanrl_claude
One self-contained script per algorithm, with shared code in `common.py` (CLI, env wrapper, `DiscreteEnv`, logging, GAE)
and `buffers.py` (replay, HER, sequence sampling). Hyper-parameter priority: `Args` dataclass default → `configs/<algo>.yaml` → CLI flag.
The `ppo*.yaml` files `include:` `configs/kingfisher_ppo.yaml` (a copy of the rl_games yaml), and `common.py` translates the
rl_games key names (`RL_GAMES_KEYS`). Checkpoints store their algorithm name and args, so `eval.py` and `export_onnx.py`
work on any of them. The older `workflows/cleanrl/` directory is unrelated; leave it alone.

## Outputs
`logs/`, `outputs/`, `eval_logs/`, `runs/` and `*.usd` are gitignored. Training logs go to `logs/rl_games/...` and
`logs/cleanrl_claude/<algo>/<task>/<run>/`. Sweeps go to `outputs/reward_sweep/` and `outputs/cleanrl_sweep/`. Tuning goes to `logs/cleanrl_claude_tuning/`.
