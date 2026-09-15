"""
Pareto-front analysis for context-conditioned energy/time trade-off.

Expected input: a per-TIMESTEP CSV from the CORRECTED sweep loop
(sweep_eval_loop.py's all_steps_swept.csv), with columns:
    env_id             : parallel-env index. Context is fixed per env_id for
                          the whole run, so this also identifies context.
    local_episode_idx  : this env's own episode counter (0-44 for a full
                          5-angle x 9-speed sweep). Episode identity is the
                          PAIR (env_id, local_episode_idx) -- there is no
                          single global `episode` column in this schema
                          (the older all_steps.csv format had one; pass
                          episode_group_cols=("episode",) to
                          episodes_from_timeseries() if reading that format).
    time_step          : step index within the episode
    energy             : INSTANTANEOUS per-step energy/power term (summed
                          here to get total energy used, since it is not
                          cumulative)
    distance           : distance to goal at this step (used to detect
                          success)
    energy_context     : the context/energy-cap value for this episode
                          (constant within an episode)
    true_wind_speed    : wind speed for this episode (constant within an
                          episode)
    true_wind_angle_w  : true wind angle in world frame (constant within an
                          episode)

Collapse to one row per episode with `episodes_from_timeseries()`, then feed
the result into the plotting functions below.

Usage:
    python pareto_energy_time.py path/to/all_steps_swept.csv --outdir ./figs
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import viridis


# ---------------------------------------------------------------------------
# Collapse a per-timestep log into one row per episode.
# ---------------------------------------------------------------------------
def episodes_from_timeseries(
    df_ts: pd.DataFrame,
    episode_group_cols=("env_id", "local_episode_idx"),
    energy_col: str = "energy",
    context_col: str = "energy_context",
    wind_speed_col: str = "true_wind_speed",
    wind_angle_col: str = "true_wind_angle_w",
    distance_col: str = "distance",
    timestep_col: str = "time_step",
    success_radius: float = 1.0,
) -> pd.DataFrame:
    """Collapse per-timestep rows into one row per episode.

    Episode identity is the composite key `episode_group_cols` -- by default
    (env_id, local_episode_idx), matching the corrected sweep loop's output.
    Pass episode_group_cols=("episode",) instead if reading the older
    all_steps.csv format that has a single global `episode` column.

    Success detection: if the CSV has a `goal_reached` column (written by the
    corrected sweep_eval_loop.py, sourced directly from the sim's own
    reset_terminated flag), that is used and is authoritative. Otherwise
    falls back to inferring success from `distance` dropping at or below
    `success_radius` -- NOTE this fallback will systematically miss episodes
    whose true crossing row was the contaminated reset-boundary row (see
    sweep_eval_loop.py docstring): every episode's last kept row sits just
    ABOVE the env's actual goal_reached_threshold, never below it, so
    success_radius needs to be set a little looser than the true threshold
    (e.g. 0.35 if the env's threshold is 0.3) or every episode will register
    as a failure. Prefer having `goal_reached` in the CSV whenever possible.

    total_energy and time_to_goal are computed up to the last kept row for
    goal_reached episodes (so failed/lingering steps don't pollute the
    frontier); for non-goal-reached (timed out) episodes, energy is summed
    over the whole episode and time_to_goal is the episode length (censored).
    """
    group_cols = list(episode_group_cols)
    has_explicit_flag = "goal_reached" in df_ts.columns
    records = []
    for key, g in df_ts.sort_values(timestep_col).groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)

        if has_explicit_flag:
            success = bool(g["goal_reached"].iloc[0])
            t_goal = g[timestep_col].max()
            total_energy = g[energy_col].sum()
            time_to_goal = t_goal
        else:
            reached = g[g[distance_col] <= success_radius]
            success = len(reached) > 0
            if success:
                t_goal = reached[timestep_col].iloc[0]
                g_up_to_goal = g[g[timestep_col] <= t_goal]
                total_energy = g_up_to_goal[energy_col].sum()
                time_to_goal = t_goal
            else:
                total_energy = g[energy_col].sum()
                time_to_goal = g[timestep_col].max()

        record = dict(zip(group_cols, key))
        record.update(dict(
            context=g[context_col].iloc[0],
            wind_speed=g[wind_speed_col].iloc[0],
            wind_angle=g[wind_angle_col].iloc[0],
            total_energy=total_energy,
            time_to_goal=time_to_goal,
            success=int(success),
        ))
        records.append(record)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 1. Main Pareto front: energy vs. time, one point per context,
#    averaged over all wind conditions, with error bars.
# ---------------------------------------------------------------------------
def plot_pareto_front(df_all: pd.DataFrame, outpath: str):
    """
    df_all must be the FULL (unfiltered) per-episode dataframe, including
    failures -- NOT pre-filtered to successes. This function does its own
    success-rate accounting against the true denominator (all attempted
    episodes per context), and separately computes energy/time stats from
    only the successful subset (failures don't have a meaningful
    "time_to_goal"/"total_energy actually spent reaching the goal").

    Passing an already-success-filtered dataframe here silently makes
    success_rate always 1.0 and n mean "n_success" instead of "n_total" --
    that bug is exactly what happened before this fix.
    """
    df_success = df_all[df_all["success"] == 1]

    counts = (
        df_all.groupby("context")
        .agg(n_total=("success", "count"), n_success=("success", "sum"))
        .reset_index()
    )
    counts["success_rate"] = counts["n_success"] / counts["n_total"]

    stats = (
        df_success.groupby("context")
        .agg(
            energy_mean=("total_energy", "mean"),
            energy_std=("total_energy", "std"),
            time_mean=("time_to_goal", "mean"),
            time_std=("time_to_goal", "std"),
        )
        .reset_index()
    )

    summary = counts.merge(stats, on="context", how="left").sort_values("context")

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = viridis(np.linspace(0, 1, len(summary)))

    ax.plot(summary["time_mean"], summary["energy_mean"], "-", color="gray", zorder=1, alpha=0.6)
    ax.errorbar(
        summary["time_mean"], summary["energy_mean"],
        xerr=summary["time_std"], yerr=summary["energy_std"],
        fmt="none", ecolor="gray", alpha=0.5, zorder=1,
    )
    sc = ax.scatter(
        summary["time_mean"], summary["energy_mean"],
        c=summary["context"], cmap="viridis", s=90, zorder=2, edgecolor="k",
    )

    for _, row in summary.iterrows():
        ax.annotate(f"{row['context']:.1f}", (row["time_mean"], row["energy_mean"]),
                     textcoords="offset points", xytext=(6, 6), fontsize=8)

    ax.set_xlabel("Mean time to goal")
    ax.set_ylabel("Mean electrical energy used")
    ax.set_title("Energy-time vs. context\n(averaged over wind speed/angle sweep)")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Context")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return summary


# ---------------------------------------------------------------------------
# 2. Small multiples: same curve faceted by wind speed bucket.
# ---------------------------------------------------------------------------
def plot_pareto_by_windspeed(df: pd.DataFrame, outpath: str, n_speed_bins: int = 4):
    df = df.copy()
    df["wind_speed_bin"] = pd.qcut(df["wind_speed"], q=n_speed_bins, duplicates="drop")

    bins = sorted(df["wind_speed_bin"].unique())
    fig, axes = plt.subplots(1, len(bins), figsize=(5 * len(bins), 5), sharey=True, sharex=True)
    if len(bins) == 1:
        axes = [axes]

    for ax, b in zip(axes, bins):
        sub = df[df["wind_speed_bin"] == b]
        summary = (
            sub.groupby("context")
            .agg(energy_mean=("total_energy", "mean"), time_mean=("time_to_goal", "mean"))
            .reset_index()
            .sort_values("context")
        )
        ax.plot(summary["time_mean"], summary["energy_mean"], "-", color="gray", alpha=0.6)
        sc = ax.scatter(summary["time_mean"], summary["energy_mean"],
                         c=summary["context"], cmap="viridis", s=60, edgecolor="k")
        ax.set_title(f"wind speed {b}")
        ax.set_xlabel("Mean time to goal")

    axes[0].set_ylabel("Mean electrical energy used")
    fig.colorbar(sc, ax=axes, label="Context", fraction=0.02, pad=0.02)
    fig.suptitle("Energy-time Trade-off across wind speed conditions")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Success rate vs. context (robustness / failure-mode check).
# ---------------------------------------------------------------------------
def plot_success_rate(df: pd.DataFrame, outpath: str):
    summary = df.groupby("context")["success"].mean().reset_index().sort_values("context")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(summary["context"], summary["success"], "o-", color="black")
    ax.set_xlabel("Context (energy budget)")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Task success rate vs. energy budget")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return summary


# ---------------------------------------------------------------------------
# 4. Heatmap: energy as a function of (context, wind_angle), averaged over wind_speed.
# ---------------------------------------------------------------------------
def plot_energy_heatmap(df: pd.DataFrame, outpath: str, n_angle_bins: int = 12):
    df = df.copy()
    df["wind_angle_bin"] = pd.cut(df["wind_angle"], bins=n_angle_bins)
    pivot = df.pivot_table(
        index="context", columns="wind_angle_bin", values="total_energy", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="magma", origin="lower")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{c:.1f}" for c in pivot.index])
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{iv.mid:.0f}" for iv in pivot.columns], rotation=90)
    ax.set_xlabel("Wind angle (deg, bin center)")
    ax.set_ylabel("Context")
    ax.set_title("Mean energy use by context and wind angle\n(averaged over wind speed)")
    fig.colorbar(im, ax=ax, label="Mean energy")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to per-episode CSV")
    parser.add_argument("--outdir", default=".", help="Directory to save figures")
    parser.add_argument("--timeseries", action="store_true", default=True,
                         help="Set if the CSV is per-timestep (default: True, "
                              "matching the all_steps.csv logger format)")
    parser.add_argument("--success-radius", type=float, default=1.0,
                         help="Distance-to-goal threshold counted as success")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    print(sorted(df.energy_context.unique()))
    if args.timeseries:
        df = episodes_from_timeseries(df, success_radius=args.success_radius)

    required = {"context", "wind_speed", "wind_angle", "total_energy", "time_to_goal", "success"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Only compute time/energy stats on successful episodes to avoid
    # timeout-capped time_to_goal values distorting the frontier.
    df_success = df[df["success"] == 1].copy()

    if df_success.empty:
        raise ValueError(
            "No episodes registered as successful -- df_success is empty, so "
            "the Pareto plots can't be built. If your CSV has a 'goal_reached' "
            "column this shouldn't happen; if it doesn't, your --success-radius "
            "is probably tighter than the env's real goal_reached_threshold. "
            "See the episodes_from_timeseries() docstring: the reset-boundary "
            "fix means every episode's last kept row sits just ABOVE the true "
            "threshold, never below it, so --success-radius needs a small "
            "margin above the env's actual threshold (e.g. 0.35 for a 0.3 "
            "threshold), not the exact value."
        )

    summary = plot_pareto_front(df, f"{args.outdir}/pareto_front.png")
    plot_pareto_by_windspeed(df_success, f"{args.outdir}/pareto_by_windspeed.png")
    plot_success_rate(df, f"{args.outdir}/success_rate.png")
    plot_energy_heatmap(df_success, f"{args.outdir}/energy_heatmap.png")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()