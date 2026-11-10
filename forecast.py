"""
forecast.py
-----------
Step 2 of the project: expected points (xP) for every player.

At the deadline of gameweek t we may ONLY use information from GW 1..t-1.
For each future GW g = t, t+1, ..., t+H-1:

    xP[p, g] = P(play)[p, t]  *  pts_per_app[p, t]  *  n_fixtures[club of p, g]

  P(play)      chance the player plays a match. Weighted share of his club's recent
               matches in which he played (last match counts most), shrunk toward a prior.
               (In the data, a player who missed the last 2 matches plays the next one
               only ~5% of the time, so recency matters a lot.)
  pts_per_app  points per match when he plays, over the last W gameweeks, shrunk toward
               a prior with weight K ("K pseudo-matches").
  n_fixtures   0 in a blank GW, 2 in a double GW (fixture list is published in advance).

  Priors come from LAST season (2024-25, same GitHub repo): minutes share and points per
  appearance. Players with no PL history last season get values predicted from their price
  (linear fit per position, using only players in the game at GW1 who have last-season data).

Output: data/processed/xp.csv  with columns  decision_gw, element, GW, xp
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
N_GW = 38

WINDOW = 6     # W: past GWs used for points per appearance
K_PRIOR = 4    # K: weight of the prior, in "matches"
PLAY_DECAY = 0.5   # weight of a match 1 week older (for P(play))
PLAY_WINDOW = 4    # past GWs used for P(play)
K_PLAY = 0.5       # weight of the prior for P(play)
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
    last = pd.read_csv(RAW_DIR / "players_raw_2425.csv")[["code", "total_points", "minutes", "starts"]]
    return df, ids_now, last


def build_prior(df, ids_now, last):
    """Priors (P(play), points per appearance) using only last season + GW-entry price."""
    gw1_ids = set(df[(df.GW == 1) & (df.available == 1)].element)
    first = (df[df.available == 1].sort_values("GW")
             .groupby("element").first()[["position", "price"]])
    first = first.join(ids_now.set_index("id"), how="left")
    first = first.merge(last, on="code", how="left").set_index(first.index)
    apps = np.maximum(first["starts"], first["minutes"] / 90)
    first["play"] = (first["minutes"] / (N_GW * 90)).clip(0, 1)
    first["ppa"] = first["total_points"] / (apps + 2)            # +2: shrink small samples

    for col in ["play", "ppa"]:
        for pos, grp in first.groupby("position"):
            # fit only on players in the game at GW1 -> uses no information from later GWs
            known = grp[grp.index.isin(gw1_ids)].dropna(subset=[col])
            slope, intercept = np.polyfit(known["price"], known[col], 1)
            missing = grp[col].isna()
            first.loc[grp.index[missing], col] = intercept + slope * grp.loc[missing, "price"]
    n_hist = first["minutes"].notna().sum()
    print(f"prior: {n_hist} players from last season, {len(first) - n_hist} from price fit")
    first["play"] = first["play"].clip(0.05, 1)
    first["ppa"] = first["ppa"].clip(lower=0)
    return first[["play", "ppa"]]


def forecast_all(df, prior, window=WINDOW, k=K_PRIOR, horizon=HORIZON,
                 play_window=PLAY_WINDOW, decay=PLAY_DECAY, k_play=K_PLAY):
    piv = lambda col: df.pivot(index="element", columns="GW", values=col)
    pts, nfix, mins, team, avail = piv("points"), piv("n_fixtures"), piv("minutes"), piv("team"), piv("available")
    fix = nfix * avail                                   # matches the player could have played
    apps = piv("apps") * avail                           # matches he actually played (per match)
    team_fix = df.drop_duplicates(["team", "GW"]).set_index(["team", "GW"])["n_fixtures"]
    prior = prior.reindex(pts.index)

    out = []
    for t in range(1, N_GW + 1):
        # points per appearance over the last `window` GWs
        past = list(range(max(1, t - window), t))
        p_sum = pts[past].sum(axis=1) if past else 0
        a_sum = apps[past].sum(axis=1) if past else 0
        ppa = (p_sum + k * prior["ppa"]) / (a_sum + k)

        # P(play): recency-weighted share of matches played
        recent = list(range(max(1, t - play_window), t))
        w = pd.Series({g: decay ** (t - 1 - g) for g in recent}, dtype=float)
        num = (apps[recent] * w).sum(axis=1) if recent else 0
        den = (fix[recent] * w).sum(axis=1) if recent else 0
        play = (num + k_play * prior["play"]) / (den + k_play)

        rate = (play * ppa).fillna(0)
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
