"""
forecast.py
-----------
Step 2 of 6: expected points (xP) of every player.

At the deadline of GW t, only data from GW 1..t-1 is used.
For each planned GW g = t, t+1, ..., t+H-1:

    xP[p, g] = P(play)[p, t]  *  PPA[p, t]  *  n_fixtures[club of p, g]

  P(play)      chance that the player plays a match: the share of his club's matches in the
               last 4 GWs that he played in, with recent GWs counting more, mixed with a prior.
               (In the data, a player who missed his club's last 2 matches plays the next one
               only about 6% of the time, so recent GWs count more.)
  PPA          points per match played over the last 6 GWs, mixed with a prior that counts as
               4 extra matches (K_PRIOR)
  n_fixtures   matches of his club in GW g as scheduled at the deadline of GW t
               (0 in a blank GW, 2 in a double GW; data/processed/fixtures_known.csv)

  The priors come from last season (2024-25, same data source): share of minutes played and
  points per match played. Players who were not in the Premier League last season get values
  estimated from their price (a straight-line fit per position, using only players in the game
  at GW1 who have last-season data).

Output: data/processed/xp.csv  with columns  decision_gw, element, GW, xp
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
N_GW = 38

WINDOW = 6         # past GWs used for PPA
K_PRIOR = 4        # weight of the PPA prior, in matches
PLAY_DECAY = 0.5   # for P(play), each GW counts half as much as the GW after it
PLAY_WINDOW = 4    # past GWs used for P(play)
K_PLAY = 0.5       # weight of the P(play) prior, in matches
HORIZON = 10       # longest planning horizon used (H = 10)


def load():
    df = pd.read_csv(OUT_DIR / "player_gw.csv")
    files = {"players_raw_2526.csv": "2025-26/players_raw.csv",
             "players_raw_2425.csv": "2024-25/players_raw.csv"}
    for local, remote in files.items():
        path = RAW_DIR / local
        if not path.exists():
            print(f"downloading {remote} ...")
            pd.read_csv(f"{BASE_URL}/{remote}").to_csv(path, index=False)
    # from the 2025-26 file only the player ids are used, so no future data enters the forecast
    ids_now = pd.read_csv(RAW_DIR / "players_raw_2526.csv")[["id", "code"]]
    last = pd.read_csv(RAW_DIR / "players_raw_2425.csv")[["code", "total_points", "minutes", "starts"]]
    return df, ids_now, last


def build_prior(df, ids_now, last):
    """Priors of P(play) and PPA: last season's numbers, or an estimate from the price for new players."""
    gw1_ids = set(df[(df.GW == 1) & (df.available == 1)].element)
    first = (df[df.available == 1].sort_values("GW")
             .groupby("element").first()[["position", "price"]])
    first = first.join(ids_now.set_index("id"), how="left")
    first = first.merge(last, on="code", how="left").set_index(first.index)
    apps = np.maximum(first["starts"], first["minutes"] / 90)
    first["play"] = (first["minutes"] / (N_GW * 90)).clip(0, 1)
    first["ppa"] = first["total_points"] / (apps + 2)            # +2: a player with few matches does not look too good

    for col in ["play", "ppa"]:
        for pos, grp in first.groupby("position"):
            # fit only on players in the game at GW1, so no later data is used
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
    pts, nfix, team, avail = piv("points"), piv("n_fixtures"), piv("team"), piv("available")
    fix = nfix * avail                                   # matches the player could have played
    apps = piv("apps") * avail                           # matches he actually played
    known = pd.read_csv(OUT_DIR / "fixtures_known.csv").set_index(["decision_gw", "team", "GW"])["n_fixtures"]
    prior = prior.reindex(pts.index)

    out = []
    for t in range(1, N_GW + 1):
        # PPA: points per match played over the last `window` GWs
        past = list(range(max(1, t - window), t))
        p_sum = pts[past].sum(axis=1) if past else 0
        a_sum = apps[past].sum(axis=1) if past else 0
        ppa = (p_sum + k * prior["ppa"]) / (a_sum + k)

        # P(play): share of matches played, recent GWs counting more
        recent = list(range(max(1, t - play_window), t))
        w = pd.Series({g: decay ** (t - 1 - g) for g in recent}, dtype=float)
        num = (apps[recent] * w).sum(axis=1) if recent else 0
        den = (fix[recent] * w).sum(axis=1) if recent else 0
        play = (num + k_play * prior["play"]) / (den + k_play)

        rate = (play * ppa).fillna(0)
        players = avail.index[avail[t] == 1]          # players in the game at GW t
        club_now = team.loc[players, t]               # club at the deadline of GW t
        for g in range(t, min(t + horizon, N_GW + 1)):
            nf = known.reindex(pd.MultiIndex.from_arrays([[t] * len(players), club_now.values,
                                                          [g] * len(players)])).values
            out.append(pd.DataFrame({"decision_gw": t, "element": players, "GW": g,
                                     "xp": rate.loc[players].values * nf}))
    return pd.concat(out, ignore_index=True)


def evaluate(xp, df):
    """Correlation between the expected points for the current GW and the real points."""
    one = xp[xp.decision_gw == xp.GW].merge(df[["element", "GW", "points"]], on=["element", "GW"])
    corr = one[["xp", "points"]].corr().iloc[0, 1]
    print(f"1-week-ahead forecast: correlation with the real points = {corr:.3f}")
    return corr


if __name__ == "__main__":
    df, ids_now, last = load()
    prior = build_prior(df, ids_now, last)
    xp = forecast_all(df, prior)
    evaluate(xp, df)
    xp.to_csv(OUT_DIR / "xp.csv", index=False)
    print(f"saved {OUT_DIR / 'xp.csv'}  ({len(xp)} rows)")
