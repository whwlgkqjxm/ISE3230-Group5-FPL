"""
simulate.py
-----------
Step 4 of 6: play the whole 2025-26 season with the MILP (rolling horizon).

For t = 1 .. 38:
  1. take the expected points forecast at the deadline of GW t (xp.csv, decision_gw == t)
  2. solve the MILP for GWs t .. t+H-1 (model.py)
  3. carry out only the decisions for GW t (squad, starting XI, captain, transfers)
  4. score them with the real points of GW t
  5. update the squad, bank, purchase prices and free transfers, then go to GW t+1

FPL scoring rules applied:
  - the captain scores double; if he does not play, the vice-captain scores double
  - automatic substitutions: a starter with 0 minutes is replaced by the first bench player
    (in bench order) who played and keeps a valid formation (a GK only by the bench GK)
  - each hit costs 4 points
  - selling price = purchase price + half of any price rise, rounded down to £0.1m

Output: results/season_<name>.csv  (one row per GW)
"""
from pathlib import Path
import math
import time
import pandas as pd
import model as M

OUT_DIR = Path("results")
N_GW = 38


def selling_price(bought, now):
    """FPL selling price: half of any price rise is kept (rounded down to £0.1m); a fall is taken in full."""
    if now <= bought:
        return now
    profit_tenths = math.floor(round((now - bought) * 10) / 2)
    return round(bought + profit_tenths / 10, 1)


def score_gw(lineup, bench, captain, vice, pos, minutes, points):
    """Real points of one GW, with the captain and automatic substitutions."""
    xi = list(lineup)
    for out in [p for p in lineup if minutes[p] == 0]:
        for sub in bench:
            if sub in xi or minutes[sub] == 0:
                continue
            if (pos[out] == "GK") != (pos[sub] == "GK"):
                continue
            new_xi = [sub if p == out else p for p in xi]
            n = {k: sum(pos[p] == k for p in new_xi) for k in M.MIN_START}
            if all(n[k] >= M.MIN_START[k] for k in M.MIN_START):
                xi = new_xi
                break
    total = sum(points[p] for p in xi)
    if minutes[captain] > 0:
        total += points[captain]
    elif vice in xi and minutes[vice] > 0:
        total += points[vice]
    return total


def run_season(df, xp, name="milp_H3", horizon=3, plan_hit_cost=None, max_transfers=None, verbose=True):
    """
    Play the 38 GWs and save one row per GW to results/season_<name>.csv.
    horizon        H, the number of GWs planned at each deadline
    plan_hit_cost  hit cost used inside the MILP; only the "crowd" strategy changes it
    max_transfers  most transfers per GW; only the "no transfers" strategy sets it (0)
    """
    squad, bought_at, bank, ft = [], {}, 100.0, 0     # start of the season: no players, £100m
    rows, total = [], 0
    by_gw = {t: g.set_index("element") for t, g in df.groupby("GW")}

    for t in range(1, N_GW + 1):
        info = by_gw[t]
        P, G, xpd, price, pos, club, pname = M.build_inputs(df, xp, t, horizon, owned=squad)
        sell = {p: selling_price(bought_at[p], price[p]) if p in bought_at else price[p] for p in P}

        tic = time.time()
        r = M.solve(t, P, G, xpd, price, sell, pos, club, squad0=squad, bank0=bank, ft0=ft,
                    hit_cost=M.HIT_COST if plan_hit_cost is None else plan_hit_cost,
                    max_transfers=max_transfers)
        solve_time = time.time() - tic

        # carry out the GW-t transfers (the bank is recomputed with the FPL prices)
        for p in r["sells"]:
            bank += sell[p]
            bought_at.pop(p)
        for p in r["buys"]:
            bank -= price[p]
            bought_at[p] = price[p]
        bank = round(bank, 1)
        squad = r["squad"]

        # vice-captain: the starter with the highest expected points in GW t after the captain
        vice = sorted([p for p in r["lineup"] if p != r["captain"]], key=lambda p: -xpd[p, t])[0]
        pts = score_gw(r["lineup"], r["bench"], r["captain"], vice, info["position"],
                       info["minutes"], info["points"])
        pts -= M.HIT_COST * r["hits"]
        total += pts

        rows.append({"GW": t, "points": pts, "total": total, "hits": r["hits"],
                     "transfers": 0 if t == 1 else len(r["buys"]), "free_transfers": ft,
                     "captain": pname[r["captain"]], "captain_pts": info.loc[r["captain"], "points"],
                     "bank": bank,
                     "expected": round(sum(xpd[p, t] for p in r["lineup"]) + xpd[r["captain"], t], 1),
                     "buys": ", ".join(pname[p] for p in r["buys"]) if t > 1 else "",
                     "sells": ", ".join(pname[p] for p in r["sells"]),
                     "n_vars": r["n_vars"], "n_constrs": r["n_constrs"], "solve_time": round(solve_time, 3),
                     "optimal": r["optimal"],
                     "squad": "; ".join(pname[p] for p in squad),
                     "lineup": "; ".join(pname[p] for p in r["lineup"]),
                     "vice": pname[vice], "bench": "; ".join(pname[p] for p in r["bench"])})
        if verbose:
            print(f"GW{t:2d}  {pts:4d} pts  total {total:5d}  transfers {rows[-1]['transfers']}"
                  f"  hits {r['hits']}  C: {pname[r['captain']]}")

        # free transfers for the next GW: unused ones carry over, +1, at most 5; GW16 has 5
        if t == 1:
            ft = 1
        else:
            ft = min(M.MAX_FT, ft - r["free_used"] + 1)
        if t + 1 == M.AFCON_GW:
            ft = M.MAX_FT

    log = pd.DataFrame(rows)
    OUT_DIR.mkdir(exist_ok=True)
    log.to_csv(OUT_DIR / f"season_{name}.csv", index=False)
    print(f"\n[{name}] season total: {total} points  (hits taken: {log.hits.sum()}, "
          f"transfers: {log.transfers.sum()}, solve time: {log.solve_time.sum():.1f}s, "
          f"GWs not solved to optimality: {list(log.GW[~log.optimal]) or 'none'})")
    return log


if __name__ == "__main__":
    df = pd.read_csv("data/processed/player_gw.csv")
    xp = pd.read_csv("data/processed/xp.csv")
    run_season(df, xp, name="milp_H3", horizon=3)
