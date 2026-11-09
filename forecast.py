"""
forecast.py
-----------
Step 2 of the project: expected points (xP) for every player.

At the deadline of gameweek t we may ONLY use information from GW 1..t-1.
For each future GW g = t, t+1, ..., t+H-1:

    xP[p, g] = rate[p, t] * n_fixtures[club of p, g]

    rate[p, t] = ( points in last W GWs  +  K * prior[p] )
                 / ( matches in last W GWs +  K )

  - rate      = expected points per match (already includes the chance of not playing,
                because a benched player scores 0 in that match)
  - n_fixtures= 0 in a blank GW, 2 in a double GW (fixture list is published in advance)
  - prior     = points per match LAST season (2024-25, same GitHub repo);
                players with no PL history last season get a value predicted from their
                GW1 price (linear fit per position on last season's players)
  - K         = how strongly we trust the prior (shrinkage); W = window length

Output: data/processed/xp.csv  with columns  decision_gw, element, GW, xp
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
N_GW = 38

WINDOW = 6     # W: number of past GWs used
K_PRIOR = 4    # K: weight of the prior, in "matches"
HORIZON = 8    # longest look-ahead we will ever need


def load():
    df = pd.read_csv(OUT_DIR / "player_gw.csv")
    files = {"players_raw_2526.csv": "2025-26/players_raw.csv",
             "players_raw_2425.csv": "2024-25/players_raw.csv"}
    for local, remote in files.items():
        path = RAW_DIR / local
        if not path.exists():
            print(f"downloading {remote} ...")
            pd.read_csv(f"{BASE_URL}/{remote}").to_csv(path, index=False)
    # only static ids (id <-> code) are taken from the 2025-26 file -> no look-ahead
    ids_now = pd.read_csv(RAW_DIR / "players_raw_2526.csv")[["id", "code"]]
    last = pd.read_csv(RAW_DIR / "players_raw_2425.csv")[["code", "total_points"]]
    return df, ids_now, last


def build_prior(df, ids_now, last):
    """Prior points-per-match for every player, using only last season + GW-entry price."""
    first = (df[df.available == 1].sort_values("GW")
             .groupby("element").first()[["position", "price"]])
    first = first.join(ids_now.set_index("id"), how="left")
    first = first.merge(last, on="code", how="left").set_index(first.index)
    first["last_rate"] = first["total_points"] / N_GW

    prior = first["last_rate"].copy()
    for pos, grp in first.groupby("position"):
        known = grp.dropna(subset=["last_rate"])
        slope, intercept = np.polyfit(known["price"], known["last_rate"], 1)
        missing = grp["last_rate"].isna()
        prior.loc[grp.index[missing]] = intercept + slope * grp.loc[missing, "price"]
    print(f"prior: {first.last_rate.notna().sum()} players from last season, "
          f"{first.last_rate.isna().sum()} from price fit")
    return prior.clip(lower=0).rename("prior")


def forecast_all(df, prior, window=WINDOW, k=K_PRIOR, horizon=HORIZON):
    pts = df.pivot(index="element", columns="GW", values="points")
    fix = df.pivot(index="element", columns="GW", values="n_fixtures") * \
          df.pivot(index="element", columns="GW", values="available")
    team = df.pivot(index="element", columns="GW", values="team")
    avail = df.pivot(index="element", columns="GW", values="available")
    team_fix = df.drop_duplicates(["team", "GW"]).set_index(["team", "GW"])["n_fixtures"]
    prior = prior.reindex(pts.index).fillna(0)

    out = []
    for t in range(1, N_GW + 1):
        past = [g for g in range(max(1, t - window), t)]
        p_sum = pts[past].sum(axis=1) if past else 0
        f_sum = fix[past].sum(axis=1) if past else 0
        rate = (p_sum + k * prior) / (f_sum + k)

        players = avail.index[avail[t] == 1]          # must exist in the game at t
        club_now = team.loc[players, t]               # club known at the deadline
        for g in range(t, min(t + horizon, N_GW + 1)):
            nf = team_fix.reindex(pd.MultiIndex.from_arrays([club_now.values, [g] * len(players)])).values
            out.append(pd.DataFrame({"decision_gw": t, "element": players, "GW": g,
                                     "xp": rate.loc[players].values * nf}))
    return pd.concat(out, ignore_index=True)


def evaluate(xp, df):
    """How good is the 1-week-ahead forecast? (only for reporting)"""
    one = xp[xp.decision_gw == xp.GW].merge(df[["element", "GW", "points"]], on=["element", "GW"])
    corr = one[["xp", "points"]].corr().iloc[0, 1]
    mae = (one.xp - one.points).abs().mean()
    print(f"1-week-ahead forecast: correlation = {corr:.3f}, MAE = {mae:.2f} points")
    return corr, mae


if __name__ == "__main__":
    df, ids_now, last = load()
    prior = build_prior(df, ids_now, last)
    xp = forecast_all(df, prior)
    evaluate(xp, df)
    xp.to_csv(OUT_DIR / "xp.csv", index=False)
    print(f"saved {OUT_DIR / 'xp.csv'}  ({len(xp)} rows)")
