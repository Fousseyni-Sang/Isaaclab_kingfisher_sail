"""
Pareto-front analysis for the sequential single-env, one-context-per-run
sweep (all_policies_swept.csv style: env_id always 0, local_episode_idx
0-44 repeating for every separately-run context).

Usage:
    python plot_paper_eval.py all_policies_swept.csv --outdir ./figs --success-radius 0.3
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import viridis


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

    Always disambiguates by context_col in addition to episode_group_cols --
    required here because you run one context per invocation, reusing the
    same env_id/local_episode_idx range every time. Without this, episodes
    from DIFFERENT contexts that happen to share (env_id, local_episode_idx)
    silently merge into one row.
    """
    group_cols = list(episode_group_cols)
    if context_col not in group_cols:
        group_cols.append(context_col)

    has_explicit_flag = "goal_reached" in df_ts.columns
    records = []
    for key, g in df_ts.sort_values(timestep_col).groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)

        if has_explicit_flag:
            success = bool(g["goal_reached"].iloc[0])
            total_energy = g[energy_col].sum()
            time_to_goal = g[timestep_col].max()
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


def plot_pareto_front(df_all: pd.DataFrame, outpath: str):
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


def plot_pareto_by_windspeed(df: pd.DataFrame, outpath: str, n_speed_bins: int = 4):
    df = df.copy()
    df["wind_speed_bin"] = pd.qcut(df["wind_speed"], q=min(n_speed_bins, df["wind_speed"].nunique()), duplicates="drop")

    bins = sorted(df["wind_speed_bin"].unique())
    fig, axes = plt.subplots(1, len(bins), figsize=(5 * len(bins), 5), sharey=True, sharex=True)
    if len(bins) == 1:
        axes = [axes]

    sc = None
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
                         c=summary["context"], cmap="viridis", s=60, edgecolor="k",
                         vmin=df["context"].min(), vmax=df["context"].max())
        ax.set_title(f"wind speed {b}")
        ax.set_xlabel("Mean time to goal")

    axes[0].set_ylabel("Mean electrical energy used")
    if sc is not None:
        fig.colorbar(sc, ax=axes, label="Context", fraction=0.02, pad=0.02)
    fig.suptitle("Energy-time trade-off across wind speed conditions")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


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


def plot_energy_heatmap(df: pd.DataFrame, outpath: str, n_angle_bins: int = 12):
    df = df.copy()
    df["wind_angle_bin"] = pd.cut(df["wind_angle"], bins=n_angle_bins)
    pivot = df.pivot_table(
        index="context", columns="wind_angle_bin", values="total_energy", aggfunc="mean", observed=False,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to per-timestep CSV")
    parser.add_argument("--outdir", default=".", help="Directory to save figures")
    parser.add_argument("--success-radius", type=float, default=0.35,
                         help="Distance-to-goal threshold counted as success "
                              "(only used as a fallback if the CSV has no goal_reached column)")
    args = parser.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    df_raw = pd.read_csv(args.csv_path, low_memory=False)

    # Clean up mixed-type goal_reached/timed_out columns and any stray
    # incomplete rows (e.g. from an interrupted run) before aggregating.
    if "goal_reached" in df_raw.columns:
        df_raw = df_raw.dropna(subset=["goal_reached"])
        df_raw["goal_reached"] = df_raw["goal_reached"].astype(bool)
    if "timed_out" in df_raw.columns:
        df_raw["timed_out"] = df_raw["timed_out"].astype(bool)

    print("Contexts present in raw file:", sorted(df_raw["energy_context"].unique()))

    df = episodes_from_timeseries(df_raw, success_radius=args.success_radius)

    print("Episodes per context after grouping (should be close to the number of "
          "episodes actually run per context, e.g. up to 45):")
    print(df.groupby("context").size())

    required = {"context", "wind_speed", "wind_angle", "total_energy", "time_to_goal", "success"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns after aggregation: {missing}")

    df_success = df[df["success"] == 1].copy()
    if df_success.empty:
        raise ValueError(
            "No episodes registered as successful. If your CSV has a 'goal_reached' "
            "column this shouldn't happen; otherwise --success-radius is probably "
            "too tight -- try loosening it (e.g. 0.35-0.5)."
        )

    summary = plot_pareto_front(df, f"{args.outdir}/pareto_front.png")
    plot_pareto_by_windspeed(df_success, f"{args.outdir}/pareto_by_windspeed.png")
    plot_success_rate(df, f"{args.outdir}/success_rate.png")
    plot_energy_heatmap(df_success, f"{args.outdir}/energy_heatmap.png")

    print()
    print(summary.to_string(index=False))
    print()
    print(f"[INFO] Figures saved to {args.outdir}/")


if __name__ == "__main__":
    main()