"""
analysis.py
-----------
Step 6 of the project: figures and the post-optimality analysis for the report.

  A. strategy comparison                                  -> fig1_cumulative_points.png
  B. post-optimality: look-ahead H = 1..10 (full season)  -> sens_horizon.csv, fig2_optimal_horizon.png

Run:  python analysis.py          (both parts, ~20 min)
      python analysis.py A        (only one part)
Needs results/season_*.csv from baselines.py for part A.
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import model as M
import simulate as S

RES = Path("results")
FIG = RES / "figures"

# ---- chart style (validated categorical palette, fixed order; recessive grid) ----
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a63", "#e4e3de"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # slot 1..4, never cycled
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 2, "legend.frameon": False,
})


def load():
    return pd.read_csv("data/processed/player_gw.csv"), pd.read_csv("data/processed/xp.csv")


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)
    print(f"  saved {FIG / name}")


# --------------------------------------------------------------------------- A
def part_a():
    """Cumulative points of the realistic strategies (perfect information is reported in the table only)."""
    labels = {"milp_H3": "MILP, 3-week look-ahead", "crowd": "Follow the crowd",
              "no_transfers": "No transfers"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ends = []
    for i, (key, label) in enumerate(labels.items()):
        log = pd.read_csv(RES / f"season_{key}.csv")
        ax.plot(log.GW, log.total, color=SERIES[i], label=label)
        ends.append([log.total.iloc[-1], f"{label}  {log.total.iloc[-1]:,}"])
    # end labels: push apart so they never overlap (min gap = 4.5% of the top value)
    ends.sort()
    gap = 0.045 * ends[-1][0]
    label_y = []
    for y, _ in ends:
        label_y.append(y if not label_y else max(y, label_y[-1] + gap))
    for (y, text), ly in zip(ends, label_y):
        ax.annotate(text, (38, y), xytext=(38.6, ly), textcoords="data", va="center",
                    fontsize=8.5, color=INK)
    perfect = pd.read_csv(RES / "season_perfect_H3.csv").total.iloc[-1]
    ax.set_title("Cumulative points over the 2025-26 season")
    ax.set_xlabel("Gameweek")
    ax.set_ylabel("Cumulative points")
    ax.set_xlim(1, 50)
    ax.set_xticks([1, 10, 20, 30, 38])
    ax.legend(loc="upper left", fontsize=8.5)
    ax.text(0.99, 0.02, f"Perfect-information reference (not shown): {perfect:,} pts",
            transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
    save(fig, "fig1_cumulative_points.png")


# --------------------------------------------------------------------------- B
def part_b(df, xp, horizons=range(1, 11), n_keep=None):
    """Full-season total for each look-ahead H (hit cost -4)."""
    rows = []
    for H in horizons:
        log = S.run_season(df, xp, f"sens_H{H}", horizon=H, n_keep=n_keep, verbose=False)
        rows.append({"horizon": H, "total_points": int(log.points.sum()),
                     "transfers": int(log.transfers.sum()), "hits": int(log.hits.sum()),
                     "solve_time_s": round(log.solve_time.sum(), 1)})
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "sens_horizon.csv", index=False)
    print(tab.to_string(index=False))
    plot_horizon(tab)
    return tab


def plot_horizon(tab, main_h=3):
    """Season points vs. look-ahead H, with the main model and the best tested H highlighted."""
    s = tab.sort_values("horizon")
    best = s.loc[s.total_points.idxmax()]
    main = s.loc[s.horizon == main_h].iloc[0]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(s.horizon, s.total_points, color=SERIES[0], marker="o", markersize=7)
    ax.scatter([best.horizon], [best.total_points], s=160, color=SERIES[0],
               edgecolor=SURFACE, linewidth=2, zorder=3)
    at_edge = best.horizon == s.horizon.max()
    label = "Best of tested H" + (" (largest H tested)" if at_edge else "")
    ax.annotate(f"{label}\nH = {int(best.horizon)}  ({int(best.total_points):,} pts)",
                (best.horizon, best.total_points), xytext=(6 if at_edge else 0, 14), textcoords="offset points",
                fontsize=9.5, fontweight="bold", color=INK, va="bottom", ha="right" if at_edge else "center")
    if main.horizon != best.horizon:
        ax.scatter([main.horizon], [main.total_points], s=160, color=SERIES[1],
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"Main model\nH = {int(main.horizon)}  ({int(main.total_points):,} pts)",
                    (main.horizon, main.total_points), xytext=(0, 14), textcoords="offset points",
                    fontsize=9.5, fontweight="bold", color=SERIES[1], va="bottom", ha="center")
    for row in s.itertuples():
        if row.horizon not in (best.horizon, main.horizon):
            ax.annotate(f"{int(row.total_points):,}", (row.horizon, row.total_points), xytext=(0, -16),
                        textcoords="offset points", ha="center", fontsize=8.5, color=MUTED)
    ax.set_title(f"Season points vs. look-ahead horizon (hit cost -{M.HIT_COST})")
    ax.set_xlabel("Look-ahead H (gameweeks planned at each deadline)")
    ax.set_ylabel("Season points")
    ax.set_xticks(s.horizon)
    span = s.total_points.max() - s.total_points.min()
    bottom = s.total_points.min() - 0.2 * span
    ax.set_ylim(bottom, s.total_points.max() + 0.55 * span)
    ax.vlines(best.horizon, bottom, best.total_points, color=MUTED, linestyle="--", linewidth=1, zorder=1)
    save(fig, "fig2_optimal_horizon.png")


if __name__ == "__main__":
    parts = [a.upper() for a in sys.argv[1:]] or list("AB")
    df, xp = load()
    RES.mkdir(exist_ok=True)
    for p in parts:
        print(f"\n=== part {p} ===")
        if p == "A": part_a()
        if p == "B": part_b(df, xp)
