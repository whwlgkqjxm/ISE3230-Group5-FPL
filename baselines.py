"""
baselines.py
------------
Step 5 of 6: compare the main model with other strategies.
Every strategy uses the same MILP and is scored the same way (simulate.py, real points,
FPL rules). Only what the MILP maximizes, or which transfers are allowed, is different.

  1. no_transfers   picks the GW1 squad by planning 10 GWs ahead and never transfers;
                    only the starting XI and the captain change each GW
  2. crowd          "follow the crowd": ownership (how many FPL managers own a player at the
                    deadline) replaces the expected points; free transfers only
  3. milp_H3        the main model: expected points, planning 3 GWs ahead
  4. perfect_H3     the same MILP with the real points of the next 3 GWs instead of the expected
                    points (not possible in reality; an upper reference point)

Output: results/season_<strategy>.csv for each strategy + results/summary.csv
"""
import pandas as pd
import simulate as S

N_GW = 38


def xp_from_values(df, value_col, horizon):
    """A table shaped like xp.csv, filled with another value per player and GW."""
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
    """Ownership at each GW deadline, divided by 100,000 so that it is on a scale like points."""
    raw = pd.read_csv("data/raw/merged_gw.csv").drop_duplicates(["element", "GW"])
    own = raw[["element", "GW", "selected"]]
    out = df.merge(own, on=["element", "GW"], how="left")
    out["selected"] = (out.groupby("element")["selected"].transform(lambda s: s.ffill().bfill())
                       .fillna(0) / 1e5)
    return out


def run_all(hold_horizon=10):
    df = pd.read_csv("data/processed/player_gw.csv")
    xp = pd.read_csv("data/processed/xp.csv")
    logs = {}

    logs["no_transfers"] = S.run_season(df, xp, "no_transfers", horizon=hold_horizon, max_transfers=0,
                                        verbose=False)
    crowd_df = ownership(df)
    logs["crowd"] = S.run_season(df, xp_from_values(crowd_df, "selected", 1), "crowd", horizon=1,
                                 plan_hit_cost=1000, verbose=False)  # a hit cost of 1000: free transfers only
    logs["milp_H3"] = S.run_season(df, xp, "milp_H3", horizon=3, verbose=False)
    logs["perfect_H3"] = S.run_season(df, xp_from_values(df, "points", 3), "perfect_H3", horizon=3,
                                      verbose=False)

    summary = pd.DataFrame([{
        "strategy": k,
        "total_points": int(v.points.sum()),
        "transfers": int(v.transfers.sum()),
        "hits": int(v.hits.sum()),
        "captain_pts": int(v.captain_pts.sum()),
    } for k, v in logs.items()])
    summary.to_csv("results/summary.csv", index=False)
    print("\n" + summary.to_string(index=False))
    return logs, summary


if __name__ == "__main__":
    run_all()
