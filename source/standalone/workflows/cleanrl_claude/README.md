# cleanrl_claude

Small, single-file, CleanRL-style algorithms that run directly on your Isaac Lab tasks.
Every script is 100–200 lines, uses vectorized envs on the GPU, and logs to TensorBoard.

| Script | Algorithm | Action space | Status |
|---|---|---|---|
| `ppo.py` | PPO, hyper-parameters from the rl_games yaml | continuous | ppo:OK |
| `ppo_rnn.py` | PPO with an LSTM policy | continuous | ppo_rnn:OK |
| `ppo_rnd.py` | PPO + Random Network Distillation | continuous | ppo_rnd:OK |
| `ppo_rnd_rnn.py` | PPO + sequence-based RND (recurrent target/predictor) | continuous | ppo_rnd_rnn:ran end-to-end (train, eval, LSTM and GRU), not yet trained to convergence |
| `ppo_discrete.py` | PPO on a discretized action space (`--bins`, one entry per action dim) | discrete | ppo_discrete:OK |
| `sac.py` | SAC (+ optional HER) | continuous | sac:OK, critic export checked too (`--her` ran only in an earlier short test) |
| `ddpg.py` | DDPG (+ optional HER) | continuous | ddpg:OK, critic export checked too (`--her` ran only in an earlier short test) |
| `dqn.py` | Double DQN, discretized actions (`--bins`) | discrete | dqn:OK |
| `drqn.py` | Recurrent DQN (sequence replay, burn-in) | discrete | drqn:OK |
| `eval.py` | paper-sweep evaluation of any checkpoint | – | eval:OK (all 9 algorithms) |
| `launch_sweep.py` | tmux sweep: train → eval → plot | – | launch_sweep:OK |
| `tuner.py` | Optuna search around any script | – | tuner:OK (2-trial SAC run, done before the config refactor, not re-run) |
| `dehb_tuner.py` | joint hyper-parameter + reward-shaping search (DEHB) | – | dehb_tuner:OK (fake-trainee ask/tell/resume checks + a real 6-trial joint SAC run) |
| `export_onnx.py` | export a checkpoint's actor + critic to ONNX | – | export_onnx:OK (all 9 algorithms, checked against onnxruntime) |

**"OK" means:** the script ran end to end through `launch_sweep.py` (tiny settings: 32 envs, 15k steps), wrote
TensorBoard scalars including the env's reward components, saved a checkpoint, was evaluated by `eval.py`, and its CSV
was read by `plot_paper_eval_3D.py`. It does **not** mean the algorithm was shown to learn: nobody trained any of them
long enough to compare, except where noted -- the user has since trained `ppo`, `ppo_rnn` and `ppo_rnd` to convergence
(rewards converged); the rest are still being validated.

Support files: `common.py` (CLI, env wrapper, discretizer, logging, GAE), `buffers.py` (replay buffer, HER,
sequence sampling), `configs/` (one yaml per algorithm, see below).

Nothing outside this directory was modified.

## Launching

Run from the project root, like your other scripts. All Isaac Sim flags (`--headless`, `--device cuda:1`, ...) work.
The algorithm flags come from the `Args` dataclass at the top of each script; `--help` lists them.

```bash
pysaac source/standalone/workflows/cleanrl_claude/ppo.py --headless --device cuda:0 --num_envs 4096
pysaac source/standalone/workflows/cleanrl_claude/ppo_rnn.py --headless --num_envs 4096
pysaac source/standalone/workflows/cleanrl_claude/ppo_rnd.py --headless --num_envs 4096 --int_coef 1.0
pysaac source/standalone/workflows/cleanrl_claude/ppo_discrete.py --headless --num_envs 4096 --bins 2 2 5 3

pysaac source/standalone/workflows/cleanrl_claude/sac.py  --headless --num_envs 64 --total_timesteps 2000000
pysaac source/standalone/workflows/cleanrl_claude/ddpg.py --headless --num_envs 64
pysaac source/standalone/workflows/cleanrl_claude/sac.py  --headless --num_envs 64 --her     # goal-conditioned baseline

pysaac source/standalone/workflows/cleanrl_claude/dqn.py  --headless --num_envs 64 --bins 3 3 5 5
pysaac source/standalone/workflows/cleanrl_claude/drqn.py --headless --num_envs 64 --bins 3 3 5 5 --seq_len 16
```

Default task: `Isaac-KingfisherSail-Direct-v0` (change with `--task`). Logs and checkpoints go to
`logs/cleanrl_claude/<algo>/<task>/<timestamp>/` (`summaries/` for TensorBoard, `*_last.pt`, `args.yaml`):

```bash
pysaac -m tensorboard.main --logdir logs/cleanrl_claude --bind_all
```

Useful flags for every script: `--total_timesteps` (env steps summed over all envs), `--seed`, `--exp_name`,
`--save_interval`, `--result_file` (writes the final score as json), `--cfg`, `--log_root`.

## Configs

Every algorithm has a yaml in `configs/<algo>.yaml`, loaded automatically at start-up. Priority: dataclass default →
yaml → command line flag. Use `--cfg other.yaml` for another file or `--cfg ""` to ignore the yaml.

* `ppo*.yaml` start from `configs/kingfisher_ppo.yaml`, a copy of your `rl_games_ppo_cfg.yaml`, through `include:`, and add the
  variant's own keys (`lstm_hidden`, the RND weights, `bin_*`). The rl_games file is read as it is; the loader translates
  its names: `tau→gae_lambda`, `horizon_length→num_steps`, `mini_epochs→update_epochs`, `e_clip→clip_coef`,
  `entropy_coef→ent_coef`, `critic_coef→vf_coef`, `grad_norm→max_grad_norm`, `clip_value→clip_vloss`,
  `reward_shaper.scale_value→reward_scale`, MLP units and activation, `max_epochs`, `seed`, `clip_actions`.
  The unused "scaling" residual-network block of the rl_games yaml is ignored.
* `sac.yaml`, `ddpg.yaml`, `dqn.yaml`, `drqn.yaml` are flat: the keys are the field names of `Args` in the script.

The PPO network follows the yaml: one shared MLP torso (`separate: False`), a state-independent log-std starting at 0,
and `value_bootstrap` on time-outs. The adaptive-lr KL is the sampled estimate `(ratio-1)-log(ratio)`, not rl_games' closed-form Gaussian KL.

## Discretization

`common.DiscreteEnv(env, bins, mode)` discretizes the continuous `[-1, 1]` actions with its own number of levels per action
dimension. The env is not changed: it still receives continuous actions. `bins` is a plain list, one entry per action
dimension, **in the env's own action order** -- generic to any task and any number of actions, not tied to Kingfisher's
layout. For Kingfisher (`[thruster_left, thruster_right, rudder, sail]`):

```yaml
bins: [2, 2, 5, 3]   # thruster_left, thruster_right, rudder, sail
                     # 2 levels = {-1, +1}   3 = {-1, 0, +1}   5 = {-1, -0.5, 0, 0.5, 1}
```

or on the command line: `--bins 2 2 5 3` (space-separated, same order). The length of `bins` must equal the env's
action dimension. Levels are equally spaced in `[-1, 1]` (1 level = 0, fixing that dimension). An even number of
levels has no 0 action. For another task, just give a `bins` list as long as its action space, in its action order;
nothing in `ppo_discrete.py`/`dqn.py`/`drqn.py`/`common.py` needs to change.

* `mode="multi"` (`ppo_discrete.py`): the action is `(N, 4)` integers; one categorical per dimension, each with its own number of choices.
* `mode="joint"` (`dqn.py`, `drqn.py`): every combination is one action, `2 x 2 x 5 x 3 = 60` in the example above (defaults: `3 x 3 x 5 x 5 = 225`).

## HER (`sac.py`, `ddpg.py`, flag `--her`)

The replay buffer stores robot pose and goal position for each step. When sampling, with probability `--her_ratio`
the goal is replaced by the robot position `1..--her_k` steps later in the same episode ("future" strategy). The goal
features in the observation (`--goal_obs_idx 6 7 8` = cos bearing, sin bearing, distance / `max_target_distance`) are recomputed in the
robot frame, and the reward becomes sparse: `--her_goal_reward` when the robot is within `goal_reached_threshold` of the goal, else 0.
This replaces the dense shaped reward of the env (that reward cannot be recomputed for a different goal), so `--her`
is a goal-conditioned sparse-reward baseline, not a variant of your reward. The pose/goal are read in `IsaacEnv.her_state()`; adapt that
function and `--goal_obs_idx` for another task.

## TensorBoard logging

All scripts write to `logs/cleanrl_claude/<algo>/<task>/<exp_name or timestamp>/summaries` (no wandb):

```bash
pysaac -m tensorboard.main --logdir logs/cleanrl_claude --bind_all
```

* `charts/*`: steps per second, learning rate / epsilon, mean step reward, mean episodic return, losses, ...
* Everything the env reports at each reset, averaged over the envs that finished (weighted by their number), as the
  env names it: `Episode_Reward/<component>` (every reward component, summed per episode, e.g. `1_distance_progress`,
  `3_energy`, `6_time`), `Episode_Termination/*`, `Metrics/*`, `Contexts/*`. Nothing to wire in per algorithm.

## Evaluation (`eval.py`)

Same protocol as `play_kingfisher_paper_eval.py`: every env runs `n_wind_angles × n_wind_speeds` episodes at the fixed
context of its slot (`is_paper_eval`), one csv row per step in `all_steps_swept_<run>.csv`, with the same columns
(`env_id, time_step, energy, distance, energy_context, true_wind_*, rb_pos_*, sail_angle, aero_force, aoa, ...,
local_episode_idx, goal_reached, timed_out`), so `plot_paper_eval_3D.py` reads it unchanged. Success is read from the sim's own
`reset_terminated` flag. It also writes `g_pos_x/g_pos_y` (the real goal, used by the trajectory plots) and the per-step reward components.

```bash
pysaac source/standalone/workflows/cleanrl_claude/eval.py --headless --num_envs 10 \
    --checkpoint logs/cleanrl_claude/ppo/Isaac-KingfisherSail-Direct-v0/<run>/model_last.pt
# quick check: --episodes_per_env 2      sample instead of deterministic actions (PPO/SAC): --stochastic
```

It prints and saves (`eval_summary.json`, and TensorBoard in `<checkpoint dir>/eval`): success rate, time-out rate, mean
time to goal, mean energy of the successes, success by context / wind speed / wind angle, and the mean reward components per episode.
Any of the 8 algorithms works; the checkpoint stores its own algorithm name and arguments.
Csv location: `--csv_path`, else next to `KINGFISHER_REWARD_CFG` (as the rl_games script does), else next to the checkpoint.

Differences from `play_kingfisher_paper_eval.py`: `aero_force` is filled (that script looked it up one level too deep and
always wrote 0), and the loop reads each info tensor once per step instead of calling `.item()` per env.

## Sweeps (`launch_sweep.py`)

The counterpart of `launch_reward_sweep.py`: one tmux window per GPU, several panes per window, and in every pane
train → `eval.py` → `plot_paper_eval_3D.py`, for every combination of algorithm × reward config × hyper-parameters × seed.
Edit `REWARD_SWEEP_AXES` (same reward scales as in your launcher; `[]` keeps the default) and `HPARAM_AXES` (per algorithm) at the top of the file.

```bash
pysaac source/standalone/workflows/cleanrl_claude/launch_sweep.py --algos ppo --seeds 1,2,3
pysaac source/standalone/workflows/cleanrl_claude/launch_sweep.py --algos ppo,sac,dqn --devices cuda:0,cuda:1 --jobs_per_device 3
pysaac source/standalone/workflows/cleanrl_claude/launch_sweep.py --algos sac --extra_args "--her" --run_tag her
pysaac source/standalone/workflows/cleanrl_claude/launch_sweep.py --dry_run          # write configs, print commands
tmux attach -t cleanrl_sweep
```

Layout (identical to the reward sweep, so the plot script needs nothing special): 
`outputs/cleanrl_sweep/<run_name>/seed_<seed>/{reward_cfg.yaml, all_steps_swept_seed_<seed>.csv, eval_summary.json, *.png, train/...}`.
Aggregate the seeds of one configuration with
`pysaac source/standalone/workflows/rl_games/icra2027/ours/plot_paper_eval_3D.py outputs/cleanrl_sweep/<run_name>/`.
Notes: `--num_envs` / `--total_timesteps` are only passed when given (otherwise the algorithm's yaml decides);
the reward axes only change training if the env reads `KINGFISHER_REWARD_CFG` (the `__post_init__` hook); `--eval_episodes_per_env 0` is the full paper sweep, which is slow.

## ONNX export (`export_onnx.py`)

Exports any checkpoint's actor and critic to a plain ONNX graph: the deterministic action, the log-density of that
action (continuous / `ppo_discrete` policies), and a value (V(obs) for the PPO family, Q(obs, action) for
SAC/DDPG, max Q for DQN/DRQN). No Isaac Sim: it reads every dimension from the checkpoint's own weight shapes, so
run it with the Isaac Lab python directly, not `pysaac`:

```bash
/home/GTL/fsangare/isaaclab/bin/python source/standalone/workflows/cleanrl_claude/export_onnx.py \
    --checkpoint logs/cleanrl_claude/ppo/Isaac-KingfisherSail-Direct-v0/<run>/model_last.pt
```

It cross-checks the exported graph against the torch model on a random batch (onnxruntime) and prints the max
difference (all near float32 precision, `1e-6` or better, in my tests across all 9 algorithms). `--batch_size N`
bakes in a fixed batch instead of a dynamic one; `--skip_verify` skips the onnxruntime check.

Recurrent policies (`ppo_rnn`, `drqn`) take the LSTM state as extra inputs (`h_in`, `c_in`, and `reset` for
`ppo_rnn`) and return the next state (`h_out`, `c_out`): the caller keeps the state and feeds it back in on the next call.

SAC and DDPG need a saved critic to export a value: only checkpoints trained after this feature was added carry
one (a `critic` key next to the actor). Older checkpoints export the actor only, with a printed warning.

DQN and DRQN have no actor distribution (value-based): they export `(action, q_values, value)` (`value` = max Q),
no `log_prob`. DDPG's policy is deterministic: `(action, value)`, no `log_prob` either.

One caveat: for `ppo` / `ppo_rnn` / `ppo_rnd` / `ppo_rnd_rnn` the action distribution's std is a single learned
number, not a function of the observation. So the log-prob of the exported action (the mean itself) is a
constant, the same for every observation -- it does not carry information about the state. `ppo_discrete` and
`sac` don't have this issue (their log-prob genuinely varies with the observation).

## Hyper-parameter search

`tuner.py` runs an Optuna study. Each trial starts the script as a separate process with the sampled flags
(Isaac Sim cannot restart inside one process) and reads the score from `--result_file`.

```bash
python tuner.py --script ppo.py --n_trials 20 --extra_args "--num_envs 512 --total_timesteps 2000000"
python tuner.py --script sac.py --n_trials 30 --devices cuda:0 cuda:1 --n_jobs 2 --seeds 2 \
                --extra_args "--num_envs 64 --total_timesteps 500000"
python tuner.py --script dqn.py --space "learning_rate:float:1e-5:1e-3:log" "bins:cat:2/2/3/3,3/3/5/5"
```

(Run `tuner.py` with `pysaac` if you want the same Python; it only needs `optuna` and `tyro`.)

* Default search spaces per script are in `DEFAULT_SPACES`; `--space` adds or overrides an entry.
  Format: `name:float:low:high[:log]`, `name:int:low:high`, `name:cat:a,b,c` (`64/64` inside a cat means `--name 64 64`).
  Give every entry after ONE `--space` (space-separated, as above): a second `--space` on the command line replaces
  the first instead of adding to it (a `tyro` tuple-flag quirk, not specific to this tool).
* Score = `--metric` (`mean_return` of the last 100 finished episodes; falls back to `mean_step_reward` when no
  episode finished, which happens with the very long kingfisher episodes and short trials). Use `--metric mean_step_reward` for short runs.
* Results are in `logs/cleanrl_claude_tuning/` (sqlite study, one log and json per trial, `<study>_best.json`).
  Re-running the same command resumes the study. Failed trials are marked pruned; their log is next to the json.
* Add a new algorithm: copy any script, make it write the result with `write_result(...)`, and add an entry to `DEFAULT_SPACES` (or pass `--space` flags).

## Joint hyper-parameter + reward-shaping search (`dehb_tuner.py`)

Implements Dierkes et al., ["Combining Automated Optimisation of Hyperparameters and Reward Shape"](https://arxiv.org/abs/2406.18293)
(RLC 2024) and its [reference code](https://github.com/ADA-research/combined_hpo_and_reward_shaping): instead of
tuning hyper-parameters and reward weights separately, it puts both in ONE search space and optimizes them
together with **DEHB** (Differential Evolution HyperBand, the optimizer the paper itself uses) -- the same
multi-fidelity idea as Hyperband (most trials run short and cheap; only the promising ones are re-run at a longer
budget), but new configurations come from differential evolution over the population instead of random sampling.

```bash
pip install dehb   # ConfigSpace, dask, pandas etc. come with it; not installed by default

python dehb_tuner.py --script ppo.py --n_trials 60 --min_fidelity 100000 --max_fidelity 2000000 \
    --extra_args "--num_envs 2048"
python dehb_tuner.py --script sac.py --n_trials 40 --mode hpo_only       # ablation: hyper-parameters only
python dehb_tuner.py --script sac.py --n_trials 40 --mode reward_only    # ablation: reward weights only
python dehb_tuner.py --script ppo.py --n_trials 60 \
    --space "reward.tack_penalty_scale:float:-30.0:0.0" "learning_rate:float:1e-5:1e-2:log"
```

* `--mode joint` (default) searches hyper-parameters (`DEFAULT_SPACES`, imported from `tuner.py`) and reward weights
  (`DEFAULT_REWARD_SPACE`, the same `reward_scales` keys `launch_sweep.py`'s `REWARD_SWEEP_AXES` sweeps by hand) at
  once -- `hpo_only`/`reward_only` are the paper's own ablations, one half of the space at a time.
  A reward entry is `reward.<key>:...` in `--space` and edits `reward_cfg.yaml`'s `reward_scales.<key>`, using the
  same `KINGFISHER_REWARD_CFG` mechanism `launch_sweep.py` uses -- it has no effect unless the env reads it.
* `--min_fidelity`/`--max_fidelity` are the training length (`--total_timesteps`) of the cheapest and most expensive
  trial; `--eta` (default 3) is Hyperband's downsampling rate. The defaults are small (smoke-test scale); scale them
  to your algorithm and hardware for a real search, the same way the other scripts' small defaults are meant to be raised.
* Results are in `logs/cleanrl_claude_tuning/dehb_<study>/`: DEHB's own checkpoint (`dehb_state.json`,
  `history.parquet.gzip`, `incumbent.json` -- re-running the same `--study_name` resumes from there), `runs/trial<i>_f<fidelity>_seed<s>/`
  per trial (its `reward_cfg.yaml` if any, training logs, `result.json`), and at the end `best.json` plus, if the
  winning config touched any reward key, `best_reward_cfg.yaml` -- a normal reward config, ready to point
  `KINGFISHER_REWARD_CFG` at directly, or drop into `launch_sweep.py`'s reward directory.
* A failed trial (non-zero exit, e.g. an invalid sampled flag combination) is scored as a very bad result instead of
  crashing the search -- DEHB has no "pruned" trial concept like Optuna's.

## Notes and limits

* Isaac Lab resets finished envs inside `step()`. The returned obs of a finished env is therefore already the
  first obs of the next episode. Off-policy targets use `terminated` only for the bootstrap mask, so
  time-outs bootstrap from the reset observation (small bias). PPO uses rl_games' `value_bootstrap` approximation instead.
* Off-policy scripts have `--updates_per_step` (gradient steps per vectorized env step). With many envs, raise it to keep
  the update-to-data ratio reasonable.
* The old, unrelated `../cleanrl/` directory was not touched.
