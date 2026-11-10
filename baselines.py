"""
baselines.py
------------
Step 5 of the project: compare our MILP strategy with simpler strategies.
All strategies are scored the same way (simulate.py, real points, real FPL rules).

  1. no_transfers   pick the best squad at GW1 (8-week look-ahead) and never transfer;
                    only the weekly XI and captain are chosen
  2. crowd          "follow the crowd": each week move toward the most-owned valid squad
                    using free transfers only (ownership at the deadline replaces xP)
  3. myopic_H1      our MILP, but it only looks 1 week ahead
  4. milp_H3        our MILP with a 3-week look-ahead (main model)
  5. perfect_H3     same MILP fed with the REAL future points (not achievable in reality;
                    shows how much is lost because the future is unknown)

Output: results/season_<strategy>.csv for each strategy + results/summary.csv
"""
import pandas as pd
import simulate as S

N_GW = 38


def xp_from_values(df, value_col, horizon):
    """Build a table shaped like xp.csv, but filled with another per-(player, GW) value."""
    rows = []
    val = df.pivot(index="element", columns="GW", values=value_col)
    avail = df.pivot(index="element", columns="GW", values="available")
    for t in range(1, N_GW + 1):
        players = avail.index[avail[t] == 1]
        for g in range(t, min(t + horizon, N_GW + 1)):
            rows.append(pd.DataFrame({"decision_gw": t, "element": players, "GW": g,
                                      "xp": val.loc[players, g].values}))
    return pd.concat(rows, ignore_index=True)


def ownership(df):
    """Ownership (number of managers) at each GW deadline, scaled to 'points-like' numbers."""
    raw = pd.read_csv("data/raw/merged_gw.csv").drop_duplicates(["element", "GW"])
    own = raw[["element", "GW", "selected"]]
    out = df.merge(own, on=["element", "GW"], how="left")
    out["selected"] = (out.groupby("element")["selected"].transform(lambda s: s.ffill().bfill())
                       .fillna(0) / 1e5)
    return out


def run_all(n_keep=None, hold_horizon=8):
    df = pd.read_csv("data/processed/player_gw.csv")
    xp = pd.read_csv("data/processed/xp.csv")
    logs = {}

    logs["no_transfers"] = S.run_season(df, xp, "no_transfers", horizon=hold_horizon, max_transfers=0,
                                        n_keep=n_keep, verbose=False)
    crowd_df = ownership(df)
    logs["crowd"] = S.run_season(df, xp_from_values(crowd_df, "selected", 1), "crowd", horizon=1,
                                 plan_hit_cost=1000, n_keep=n_keep, verbose=False)  # free transfers only
    logs["myopic_H1"] = S.run_season(df, xp, "myopic_H1", horizon=1, n_keep=n_keep, verbose=False)
    logs["milp_H3"] = S.run_season(df, xp, "milp_H3", horizon=3, n_keep=n_keep, verbose=False)
    logs["perfect_H3"] = S.run_season(df, xp_from_values(df, "points", 3), "perfect_H3", horizon=3,
                                      n_keep=n_keep, verbose=False)

    summary = pd.DataFrame([{
        "strategy": k,
        "total_points": int(v.points.sum()),
        "avg_per_GW": round(v.points.mean(), 1),
        "transfers": int(v.transfers.sum()),
        "hits": int(v.hits.sum()),
        "captain_pts": int(v.captain_pts.sum()),
        "solve_time_s": round(v.solve_time.sum(), 1),
    } for k, v in logs.items()])
    summary.to_csv("results/summary.csv", index=False)
    print("\n" + summary.to_string(index=False))
    return logs, summary


if __name__ == "__main__":
    run_all()
