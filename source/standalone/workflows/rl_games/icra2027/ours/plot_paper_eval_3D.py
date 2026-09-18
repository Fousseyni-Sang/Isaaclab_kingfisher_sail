"""
3D variant of plot_paper_eval.py's Pareto-front analysis, plus per-condition trajectories.

Same input format as plot_paper_eval.py (a per-timestep all_steps_swept*.csv -- see
that file's module docstring for the exact column meanings). Reuses
episodes_from_timeseries(), plot_pareto_front(), plot_success_rate(), and
plot_energy_heatmap() from plot_paper_eval.py unchanged, so pareto_front.png,
success_rate.png, and energy_heatmap.png are identical to what plot_paper_eval.py
produces. The only thing replaced is pareto_by_windspeed.png (small multiples faceted
by wind-speed bin): here it becomes two true 3D plots -- context x wind speed x mean
time-to-goal, and context x wind speed x mean energy -- since context and wind speed
are both small discrete sets in this data (see module docstring of plot_paper_eval.py),
not something that needs binning.

Also adds, per (true_wind_speed, true_wind_angle_w) condition, one trajectory plot: the
(rb_pos_x, rb_pos_y) path of every episode run under that condition, colored by that
episode's context. A full paper-eval sweep has one episode per env per condition, so
each plot shows one trajectory per context value tested (10 in the current sweep design).
Each trajectory is marked with its start ('o') and its actual goal position ('x') --
NOT its last logged position, which for a failed/timed-out episode is nowhere near the
goal. The goal is read from g_pos_x/g_pos_y if the CSV has them (a future per-timestep
format); otherwise it's reconstructed from the fixed reset geometry in
KingfisherSailEnvCfg (--goal-min/max-distance, --goal-min/max-bearing), relative to that
episode's own start position -- see plot_trajectories()'s docstring for the exact math.

Usage:
    # one CSV, figures saved next to it
    python plot_paper_eval_3D.py outputs/reward_sweep/<run>/seed_12/all_steps_swept_seed_12.csv

    # several CSVs at once (each saved into its own directory)
    python plot_paper_eval_3D.py path/to/a.csv path/to/b.csv path/to/c.csv

    # every all_steps_swept*.csv found anywhere under outputs/reward_sweep/
    python plot_paper_eval_3D.py --all

    # skip the (slower, numerous) trajectory plots
    python plot_paper_eval_3D.py --all --skip-trajectories
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  -- registers the '3d' projection

from plot_paper_eval import (
    episodes_from_timeseries,
    plot_pareto_front,
    plot_success_rate,
    plot_energy_heatmap,
)


# ---------------------------------------------------------------------------
# 3D replacement for plot_pareto_by_windspeed: context x wind speed x <metric>,
# one point (and a light interpolated surface) per (context, wind speed) pair,
# averaged over wind angle -- same averaging plot_pareto_by_windspeed did within
# each wind-speed bin, just without needing bins since wind speed here is already
# a small discrete set.
# ---------------------------------------------------------------------------
def plot_context_wind_3d(
    df_success: pd.DataFrame,
    value_col: str,
    zlabel: str,
    title: str,
    outpath: str,
    context_col: str = "context",
    wind_speed_col: str = "wind_speed",
) -> pd.DataFrame:
    """3D scatter (+ light trisurf) of mean `value_col` over (context, wind_speed).

    df_success must already be filtered to successful episodes (same reasoning as
    plot_pareto_by_windspeed: time/energy aren't meaningful for failed episodes).
    """
    df = df_success.copy()
    # Episode-level context/wind-speed carry tiny float jitter (e.g. context stored
    # as 0.1000000014901161) -- round before grouping so episodes at the "same"
    # nominal condition actually land in the same group.
    df["_context_key"] = df[context_col].round(4)
    df["_wind_key"] = df[wind_speed_col].round(4)

    summary = (
        df.groupby(["_context_key", "_wind_key"])
        .agg(z_mean=(value_col, "mean"), n=(value_col, "count"))
        .reset_index()
    )

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    x, y, z = summary["_context_key"].to_numpy(), summary["_wind_key"].to_numpy(), summary["z_mean"].to_numpy()

    sc = ax.scatter(x, y, z, c=z, cmap="viridis", s=45, edgecolor="k", depthshade=True)
    if len(np.unique(x)) >= 3 and len(np.unique(y)) >= 3:
        # Delaunay-triangulated surface -- works on scattered (context, wind_speed)
        # points without needing a perfectly regular grid.
        try:
            ax.plot_trisurf(x, y, z, cmap="viridis", alpha=0.35, linewidth=0.1, antialiased=True)
        except Exception as exc:
            print(f"[WARN] Could not fit a surface for '{title}' ({exc}); scatter points only.")

    ax.set_xlabel("Context (energy budget)")
    ax.set_ylabel("Wind speed")
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label=zlabel)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return summary


# ---------------------------------------------------------------------------
# Per-(wind_speed, wind_angle) trajectory plots, colored by context.
# ---------------------------------------------------------------------------
def plot_trajectories(
    df_ts: pd.DataFrame,
    outdir: str,
    episode_group_cols=("env_id", "local_episode_idx"),
    context_col: str = "energy_context",
    wind_speed_col: str = "true_wind_speed",
    wind_angle_col: str = "true_wind_angle_w",
    x_col: str = "rb_pos_x",
    y_col: str = "rb_pos_y",
    timestep_col: str = "time_step",
    goal_x_col: str = "g_pos_x",
    goal_y_col: str = "g_pos_y",
    goal_min_distance: float = 30.0,
    goal_max_distance: float = 35.0,
    goal_min_bearing: float = 0.0,
    goal_max_bearing: float = 0.0,
) -> int:
    """One PNG per (wind_speed, wind_angle) condition: every episode's path under that
    condition, colored by its context. Returns the number of plots written.

    Expects the raw per-timestep dataframe (not the per-episode summary), since it
    needs the full (x, y) path over time.

    The goal marker ('x') is NOT the episode's last logged position -- a failed or
    timed-out episode never gets near the goal, so its last position says nothing
    about where the goal actually was. If the CSV has goal_x_col/goal_y_col (a future
    per-timestep format), those per-episode values are used directly. Otherwise the
    goal is reconstructed from the fixed reset geometry used by paper-eval runs
    (KingfisherSailEnvCfg: min/max_target_distance, min/max_target_bearing): a point at
    the midpoint distance/bearing of those ranges, offset from the episode's OWN start
    position (its first recorded (x, y)) -- matching how the goal is actually sampled
    relative to each robot's reset pose, not a single shared world coordinate.
    """
    os.makedirs(outdir, exist_ok=True)
    group_cols = list(episode_group_cols)

    has_goal_cols = goal_x_col in df_ts.columns and goal_y_col in df_ts.columns
    fallback_goal_distance = (goal_min_distance + goal_max_distance) / 2
    fallback_goal_bearing = (goal_min_bearing + goal_max_bearing) / 2

    # One (sorted) trajectory dataframe per episode, grouped once up front rather
    # than re-scanning the full (potentially million-row) dataframe per episode.
    episode_trajectories = {
        key: g.sort_values(timestep_col) for key, g in df_ts.groupby(group_cols)
    }

    # Each episode's condition + context, from its first row (constant within an episode).
    episode_conditions = (
        df_ts.groupby(group_cols)
        .first()[[wind_speed_col, wind_angle_col, context_col]]
        .reset_index()
    )
    episode_conditions["_speed_key"] = episode_conditions[wind_speed_col].round(3)
    episode_conditions["_angle_key"] = episode_conditions[wind_angle_col].round(1)

    context_values = episode_conditions[context_col]
    norm = plt.Normalize(vmin=context_values.min(), vmax=context_values.max())
    cmap = plt.get_cmap("viridis")

    conditions = episode_conditions[["_speed_key", "_angle_key"]].drop_duplicates()
    n_written = 0

    for _, cond in conditions.iterrows():
        speed_key, angle_key = cond["_speed_key"], cond["_angle_key"]
        episodes_here = episode_conditions[
            (episode_conditions["_speed_key"] == speed_key) & (episode_conditions["_angle_key"] == angle_key)
        ].sort_values(context_col)

        fig, ax = plt.subplots(figsize=(6, 6))
        for _, ep in episodes_here.iterrows():
            key = tuple(ep[c] for c in group_cols)
            traj = episode_trajectories[key]
            color = cmap(norm(ep[context_col]))

            start_x, start_y = traj[x_col].iloc[0], traj[y_col].iloc[0]
            if has_goal_cols:
                goal_x, goal_y = traj[goal_x_col].iloc[0], traj[goal_y_col].iloc[0]
            else:
                goal_x = start_x + fallback_goal_distance * np.cos(fallback_goal_bearing)
                goal_y = start_y + fallback_goal_distance * np.sin(fallback_goal_bearing)

            ax.plot(traj[x_col], traj[y_col], color=color, linewidth=1.5, alpha=0.85,
                     label=f"context={ep[context_col]:.2f}")
            ax.scatter(start_x, start_y, color=color, marker="o", s=25, zorder=3)
            ax.scatter(goal_x, goal_y, color=color, marker="x", s=70, linewidths=2.2, zorder=4)

        ax.set_xlabel("x position")
        ax.set_ylabel("y position")
        goal_source = "g_pos_x/g_pos_y" if has_goal_cols else "reconstructed from reset geometry"
        ax.set_title(
            f"Trajectories -- wind speed {speed_key:g}, wind angle {angle_key:g}deg\n"
            f"({len(episodes_here)} episode(s), colored by context; o=start, x=goal [{goal_source}])"
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Context (energy budget)")
        fig.tight_layout()

        fname = f"trajectory_speed{speed_key:g}_angle{angle_key:g}.png"
        fig.savefig(os.path.join(outdir, fname), dpi=150)
        plt.close(fig)
        n_written += 1

    return n_written


# ---------------------------------------------------------------------------
def process_csv(csv_path: str, args: argparse.Namespace) -> None:
    outdir = args.outdir or os.path.dirname(os.path.abspath(csv_path)) or "."
    os.makedirs(outdir, exist_ok=True)

    print(f"\n=== {csv_path} -> {outdir} ===")
    df_ts = pd.read_csv(csv_path)

    df = episodes_from_timeseries(df_ts, success_radius=args.success_radius) if args.timeseries else df_ts

    required = {"context", "wind_speed", "wind_angle", "total_energy", "time_to_goal", "success"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns after processing: {missing}")

    df_success = df[df["success"] == 1].copy()
    if df_success.empty:
        print(f"[WARN] No successful episodes in {csv_path} -- skipping pareto/energy/3D plots "
              f"(still attempting trajectories, and success_rate.png).")
    else:
        summary = plot_pareto_front(df, os.path.join(outdir, "pareto_front.png"))
        plot_context_wind_3d(
            df_success, "time_to_goal", "Mean time to goal",
            "Time-to-goal vs. context and wind speed\n(averaged over wind angle)",
            os.path.join(outdir, "context_wind_time_3d.png"),
        )
        plot_context_wind_3d(
            df_success, "total_energy", "Mean electrical energy used",
            "Energy use vs. context and wind speed\n(averaged over wind angle)",
            os.path.join(outdir, "context_wind_energy_3d.png"),
        )
        plot_energy_heatmap(df_success, os.path.join(outdir, "energy_heatmap.png"))
        print(summary.to_string(index=False))

    plot_success_rate(df, os.path.join(outdir, "success_rate.png"))

    if not args.skip_trajectories:
        if args.timeseries:
            traj_outdir = os.path.join(outdir, "trajectories")
            n = plot_trajectories(
                df_ts, traj_outdir,
                goal_min_distance=args.goal_min_distance,
                goal_max_distance=args.goal_max_distance,
                goal_min_bearing=args.goal_min_bearing,
                goal_max_bearing=args.goal_max_bearing,
            )
            print(f"Saved {n} trajectory plot(s) to {traj_outdir}/")
        else:
            print("[WARN] --timeseries is off (per-episode CSV given directly): "
                  "trajectory plots need the raw per-timestep positions, skipping.")


def main() -> None:
    parser = argparse.ArgumentParser(description="3D pareto plots + per-condition trajectories for eval CSVs.")
    parser.add_argument(
        "csv_path", nargs="*", default=[],
        help="One or more per-timestep CSV files to plot (space-separated for multiple).",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Also process every all_steps_swept*.csv found recursively under --sweep-dir.",
    )
    parser.add_argument(
        "--sweep-dir", default="outputs/reward_sweep",
        help="Root directory searched when --all is given (default: outputs/reward_sweep). Only ever read, never written to.",
    )
    parser.add_argument(
        "--outdir", default=None,
        help="Directory to save figures into. Defaults to each CSV's own directory -- "
        "recommended when passing multiple CSVs / --all, since a single shared --outdir "
        "would make later CSVs overwrite earlier ones' figures.",
    )
    parser.add_argument(
        "--timeseries", action="store_true", default=True,
        help="Set if the CSV is per-timestep (default: True, matching all_steps_swept*.csv). "
        "Also required for the trajectory plots.",
    )
    parser.add_argument("--success-radius", type=float, default=1.0,
                         help="Distance-to-goal threshold counted as success.")
    parser.add_argument("--skip-trajectories", action="store_true", default=False,
                         help="Skip the per (wind_speed, wind_angle) trajectory plots.")
    parser.add_argument(
        "--goal-min-distance", type=float, default=30.0,
        help="KingfisherSailEnvCfg.min_target_distance -- used to reconstruct the goal position "
        "for trajectory plots when the CSV has no g_pos_x/g_pos_y columns (default: 30.0).",
    )
    parser.add_argument(
        "--goal-max-distance", type=float, default=35.0,
        help="KingfisherSailEnvCfg.max_target_distance (default: 35.0).",
    )
    parser.add_argument(
        "--goal-min-bearing", type=float, default=0.0,
        help="KingfisherSailEnvCfg.min_target_bearing, in radians (default: 0.0).",
    )
    parser.add_argument(
        "--goal-max-bearing", type=float, default=0.0,
        help="KingfisherSailEnvCfg.max_target_bearing, in radians (default: 0.0).",
    )
    args = parser.parse_args()

    csv_paths = list(dict.fromkeys(args.csv_path))  # de-dupe, keep order

    if args.all:
        pattern = os.path.join(args.sweep_dir, "**", "all_steps_swept*.csv")
        found = sorted(glob.glob(pattern, recursive=True))
        print(f"[--all] Found {len(found)} CSV(s) under {args.sweep_dir}")
        for p in found:
            if p not in csv_paths:
                csv_paths.append(p)

    if not csv_paths:
        parser.error("No CSV files given -- pass one or more csv_path arguments, or use --all with --sweep-dir.")

    failures = []
    for csv_path in csv_paths:
        try:
            process_csv(csv_path, args)
        except Exception as exc:
            print(f"[ERROR] Failed to process {csv_path}: {exc}")
            failures.append(csv_path)

    print(f"\n=== Done: {len(csv_paths) - len(failures)}/{len(csv_paths)} CSV(s) processed successfully ===")
    if failures:
        print("Failed:")
        for f in failures:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
