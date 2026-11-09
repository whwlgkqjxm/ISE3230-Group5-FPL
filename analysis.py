"""
analysis.py
-----------
Step 6 of the project: post-optimality analysis, figures and tables for the report.

  A. strategy comparison            -> fig1_cumulative_points.png
  B. look-ahead H x planned hit cost -> sens_horizon_hit.csv, fig2_sensitivity_heatmap.png
  C. budget                          -> sens_budget_season.csv, sens_budget_gw1.csv, fig3_budget_gw1.png
  D. LP-relaxation shadow prices + ban/force-in test per player (GW1 model)
                                     -> lp_duals_gw1.csv, player_sensitivity_gw1.csv
  E. model size vs solve time        -> model_size_time.csv, fig4_solve_time.png
  F. forecast vs actual (main model) -> fig5_expected_vs_actual.png

Run:  python analysis.py          (everything, ~10 min)
      python analysis.py A D      (only some parts)
Needs results/season_*.csv from baselines.py for part A.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gurobipy as gp
import model as M
import simulate as S

RES = Path("results")
FIG = RES / "figures"

# ---- chart style (validated categorical palette, fixed order; recessive grid) ----
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a63", "#e4e3de"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # slot 1..4, never cycled
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
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
              "myopic_H1": "MILP, 1-week look-ahead", "no_transfers": "No transfers"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (key, label) in enumerate(labels.items()):
        log = pd.read_csv(RES / f"season_{key}.csv")
        ax.plot(log.GW, log.total, color=SERIES[i], label=label)
        ax.annotate(f"{label}  {log.total.iloc[-1]:,}", (log.GW.iloc[-1], log.total.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK)
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
def part_b(df, xp, horizons=(1, 2, 3, 4, 5), hit_costs=(4, 6, 8), n_keep=None):
    """Season total for each look-ahead H and each hit cost used when PLANNING (real penalty stays 4)."""
    rows = []
    for H in horizons:
        for hc in hit_costs:
            log = S.run_season(df, xp, f"sens_H{H}_hit{hc}", horizon=H, plan_hit_cost=hc,
                               n_keep=n_keep, verbose=False)
            rows.append({"horizon": H, "plan_hit_cost": hc, "total_points": int(log.points.sum()),
                         "transfers": int(log.transfers.sum()), "hits": int(log.hits.sum()),
                         "solve_time_s": round(log.solve_time.sum(), 1)})
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "sens_horizon_hit.csv", index=False)
    grid = tab.pivot(index="horizon", columns="plan_hit_cost", values="total_points")
    print(grid.to_string())

    fig, ax = plt.subplots(figsize=(6, 4.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", BLUES)
    im = ax.imshow(grid.values, cmap=cmap, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(grid.columns)), [f"-{c}" for c in grid.columns])
    ax.set_yticks(range(len(grid.index)), grid.index)
    ax.set_xlabel("Hit cost assumed when planning (points)")
    ax.set_ylabel("Look-ahead H (gameweeks)")
    ax.set_title("Season points by look-ahead and hit cost")
    mid = (grid.values.max() + grid.values.min()) / 2
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.values[i, j]
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=9,
                    color="white" if v > mid else INK)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Season points")
    save(fig, "fig2_sensitivity_heatmap.png")
    return tab


# --------------------------------------------------------------------------- C
def part_c(df, xp, season_budgets=(95, 100, 105), gw1_budgets=np.arange(90, 110.5, 1), n_keep=None):
    rows = []
    for bud in season_budgets:
        log = S.run_season(df, xp, f"sens_budget{bud}", horizon=3, budget=float(bud),
                           n_keep=n_keep, verbose=False)
        rows.append({"budget": bud, "total_points": int(log.points.sum())})
    pd.DataFrame(rows).to_csv(RES / "sens_budget_season.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))

    # parametric analysis of the GW1 model: optimal expected points vs. budget
    P, G, xpd, price, pos, club, _ = M.build_inputs(df, xp, 1, 3, n_keep=n_keep)
    gw1 = []
    for bud in gw1_budgets:
        r = M.solve(1, P, G, xpd, price, price, pos, club, [], float(bud), 0)
        gw1.append({"budget": float(bud), "objective": round(r["objective"], 2)})
    gw1 = pd.DataFrame(gw1)
    gw1["marginal_per_1m"] = gw1.objective.diff().round(2)
    gw1.to_csv(RES / "sens_budget_gw1.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gw1.budget, gw1.objective, color=SERIES[0], marker="o", markersize=4)
    base = gw1.loc[gw1.budget == 100, "objective"].iloc[0]
    ax.annotate(f"£100m: {base:.1f}", (100, base), xytext=(8, -14), textcoords="offset points",
                fontsize=8.5, color=INK)
    ax.set_title("GW1 model: optimal expected points vs. budget")
    ax.set_xlabel("Budget (£m)")
    ax.set_ylabel("Objective (discounted expected points, 3 GWs)")
    save(fig, "fig3_budget_gw1.png")


# --------------------------------------------------------------------------- D
def part_d(df, xp, n_keep=None):
    """Shadow prices from the LP relaxation + ban/force-in sensitivity per player (GW1 model)."""
    P, G, xpd, price, pos, club, name = M.build_inputs(df, xp, 1, 3, n_keep=n_keep)
    r = M.solve(1, P, G, xpd, price, price, pos, club, [], 100.0, 0)
    milp = r["model"]
    lp = milp.relax()
    lp.Params.OutputFlag = 0
    lp.optimize()
    print(f"  MILP objective {milp.ObjVal:.2f}  |  LP relaxation {lp.ObjVal:.2f}  "
          f"(gap {100 * (lp.ObjVal - milp.ObjVal) / milp.ObjVal:.2f}%)")

    keep = ("budget_", "squad15_", "quota_", "club_", "xi_", "minform_", "maxform_", "one_captain_")
    duals = pd.DataFrame([{"constraint": c.ConstrName, "shadow_price": round(c.Pi, 4),
                           "slack": round(c.Slack, 4)}
                          for c in lp.getConstrs() if c.ConstrName.startswith(keep)])
    duals = duals[duals.constraint.str.endswith("_1")]          # GW1 constraints
    duals = duals.reindex(duals.shadow_price.abs().sort_values(ascending=False).index)
    duals.to_csv(RES / "lp_duals_gw1.csv", index=False)
    print(duals.head(12).to_string(index=False))

    # LP reduced costs are degenerate here (x and b are tied by the flow equation), so we measure
    # player sensitivity directly on the MILP:
    #   chosen player     -> ban him (x = 0)  : objective loss = his value over the best replacement
    #   non-chosen player -> force him (x = 1): objective loss = how much better he would need to be
    chosen = set(r["squad"])
    others = sorted([p for p in P if p not in chosen], key=lambda p: -sum(xpd[p, g] for g in G))[:10]
    rows = []
    for p in list(chosen) + others:
        var = milp.getVarByName(f"x[{p},1]")
        old = (var.LB, var.UB)
        var.LB, var.UB = (0, 0) if p in chosen else (1, 1)
        milp.optimize()
        loss = milp.ObjVal - r["objective"] if milp.SolCount else float("nan")
        rows.append({"player": name[p], "position": pos[p], "price": price[p],
                     "in_optimal_squad": p in chosen, "xp_3gw": round(sum(xpd[p, g] for g in G), 2),
                     "test": "ban" if p in chosen else "force in", "objective_change": round(loss, 2)})
        var.LB, var.UB = old
    sens = pd.DataFrame(rows).sort_values(["in_optimal_squad", "objective_change"], ascending=[False, True])
    sens.to_csv(RES / "player_sensitivity_gw1.csv", index=False)
    print(sens.to_string(index=False))
    milp.dispose()


# --------------------------------------------------------------------------- E
def part_e(df, xp, gws=(1, 20), horizons=(1, 3, 5, 8), shares=(0.1, 0.25, 0.5, 1.0)):
    """How the solve time grows with the number of candidates and the look-ahead."""
    rows = []
    for t in gws:
        counts = df[(df.GW == t) & (df.available == 1)].position.value_counts()
        for H in horizons:
            for sh in shares:
                n_keep = {k: max(M.QUOTA[k] + 2, int(round(v * sh))) for k, v in counts.items()}
                P, G, xpd, price, pos, club, _ = M.build_inputs(df, xp, t, H, n_keep=n_keep)
                r = M.solve(t, P, G, xpd, price, price, pos, club, [], 100.0, 0)
                rows.append({"GW": t, "horizon": H, "candidates": len(P), "variables": r["n_vars"],
                             "constraints": r["n_constrs"], "solve_time_s": round(r["runtime"], 3)})
                r["model"].dispose()
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "model_size_time.csv", index=False)
    print(tab.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    sub = tab[tab.GW == gws[0]]
    for i, H in enumerate(horizons):
        s = sub[sub.horizon == H].sort_values("variables")
        ax.plot(s.variables, s.solve_time_s, color=SERIES[i], marker="o", markersize=5, label=f"H = {H}")
        ax.annotate(f"H = {H}", (s.variables.iloc[-1], s.solve_time_s.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5)
    ax.set_title(f"Solve time vs. model size (GW{gws[0]} squad selection)")
    ax.set_xlabel("Number of variables")
    ax.set_ylabel("Gurobi solve time (s)")
    ax.legend(fontsize=8.5)
    save(fig, "fig4_solve_time.png")


# --------------------------------------------------------------------------- F
def part_f():
    log = pd.read_csv(RES / "season_milp_H3.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(log.GW, log.expected, color=SERIES[0], label="Expected (model)")
    ax.plot(log.GW, log.points + 4 * log.hits, color=SERIES[1], label="Actual (before hits)")
    ax.set_title("Main model: expected vs. actual points per gameweek")
    ax.set_xlabel("Gameweek")
    ax.set_ylabel("Points")
    ax.set_xticks([1, 10, 20, 30, 38])
    ax.legend(fontsize=8.5)
    corr = np.corrcoef(log.expected, log.points + 4 * log.hits)[0, 1]
    ax.text(0.99, 0.02, f"correlation = {corr:.2f}", transform=ax.transAxes, ha="right",
            fontsize=8, color=MUTED)
    save(fig, "fig5_expected_vs_actual.png")


if __name__ == "__main__":
    parts = [a.upper() for a in sys.argv[1:]] or list("ABCDEF")
    df, xp = load()
    RES.mkdir(exist_ok=True)
    for p in parts:
        print(f"\n=== part {p} ===")
        {"A": part_a, "F": part_f}.get(p, lambda: None)()
        if p == "B": part_b(df, xp)
        if p == "C": part_c(df, xp)
        if p == "D": part_d(df, xp)
        if p == "E": part_e(df, xp)
