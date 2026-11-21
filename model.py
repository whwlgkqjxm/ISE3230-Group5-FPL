"""
model.py
--------
Step 3 of 6: the MILP that chooses the team at one gameweek (GW) deadline, solved with Gurobi.

At the deadline of GW t the model plans the GWs  G = {t, ..., t+H-1}.  Only the decisions for
GW t are carried out. The rest of the plan is made again at the next deadline (rolling horizon,
see simulate.py).

Data
  P             candidate players (all players in the game at GW t)
  G             planned GWs
  xp[p,g]       expected points of player p in GW g (forecast.py)
  cost[p]       price of p at GW t (£m)
  sell[p]       selling price of p at GW t (£m)
  pos[p]        position: GK, DEF, MID or FWD
  club[p]       club
  squad0        current squad of 15 players (empty before GW1)
  bank0         money in the bank (£m)
  ft0           free transfers available at GW t

Decision variables (for every g in G)
  x[p,g]   1 if p is in the 15-player squad        (binary)
  y[p,g]   1 if p is in the starting XI            (binary)
  c[p,g]   1 if p is the captain                   (binary)
  b[p,g]   1 if p is bought before GW g            (binary)
  s[p,g]   1 if p is sold before GW g              (binary)
  ft[g]    free transfers available in GW g        (integer, 0..5)
  u[g]     free transfers used in GW g             (integer)
  h[g]     hits: transfers beyond the free ones    (integer)
  bank[g]  money left after the GW-g transfers     (continuous, >= 0)
  z[g]     1 if any transfer is made in GW g       (binary; created only when a club has more
           than 3 squad players after a real-life transfer, see constraint 12)

Objective
  max  sum_g  delta^(g-t) * [ sum_p xp[p,g] * (y[p,g] + c[p,g])            starting XI, captain twice
                              + beta * sum_p xp[p,g] * (x[p,g] - y[p,g])   bench, small weight
                              - kappa * h[g] ]                             hit penalty
  with delta = 0.9, beta = 0.1 and kappa = 4.
"""
import gurobipy as gp
from gurobipy import GRB

QUOTA = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}       # players per position in the squad
MIN_START = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}   # fewest starters per position
MAX_START = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}   # most starters per position
CLUB_LIMIT = 3         # most squad players from one club
HIT_COST = 4           # points lost per hit (kappa)
MAX_FT = 5             # most free transfers that can be saved
AFCON_GW = 16          # in 2025-26 every manager got 5 free transfers in GW16


def solve(t, P, G, xp, cost, sell, pos, club, squad0, bank0, ft0,
          hit_cost=HIT_COST, bench_w=0.1, discount=0.9, max_transfers=None,
          time_limit=120, verbose=False):
    """Build and solve the MILP at the deadline of GW t. Returns the decisions for GW t."""
    first_gw = len(squad0) == 0          # GW1: the first squad is picked from scratch
    m = gp.Model(f"fpl_gw{t}")
    m.Params.OutputFlag = int(verbose)
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = 1e-4

    # ---------- decision variables ----------
    x = m.addVars(P, G, vtype=GRB.BINARY, name="x")
    y = m.addVars(P, G, vtype=GRB.BINARY, name="y")
    c = m.addVars(P, G, vtype=GRB.BINARY, name="c")
    b = m.addVars(P, G, vtype=GRB.BINARY, name="b")
    s = m.addVars(P, G, vtype=GRB.BINARY, name="s")
    ft = m.addVars(G, vtype=GRB.INTEGER, lb=0, ub=MAX_FT, name="ft")
    u = m.addVars(G, vtype=GRB.INTEGER, lb=0, name="u")
    h = m.addVars(G, vtype=GRB.INTEGER, lb=0, name="h")
    bank = m.addVars(G, lb=0.0, name="bank")

    owned = set(squad0)
    # squad players above the club limit, per club (> 0 only after a real-life transfer, see (12))
    excess = {cl: sum(1 for p in owned if club.get(p) == cl) - CLUB_LIMIT for cl in set(club.values())}
    z = {}                              # z[g], created only when constraint (12) needs it
    # money received for a sale: the FPL selling price for a player already owned,
    # the current price for a player bought within the plan
    sell_value = {p: sell[p] if p in owned else cost[p] for p in P}

    # ---------- constraints ----------
    for i, g in enumerate(G):
        # squad and bank of the previous GW (for the first planned GW: the current ones)
        prev = (lambda p: 1 if p in owned else 0) if i == 0 else (lambda p, gp_=G[i - 1]: x[p, gp_])
        bank_prev = bank0 if i == 0 else bank[G[i - 1]]

        # (1) squad = previous squad + players bought - players sold
        m.addConstrs((x[p, g] == prev(p) + b[p, g] - s[p, g] for p in P), name=f"flow_{g}")
        # (2) only a squad player can be sold, only a player not in the squad can be bought
        m.addConstrs((s[p, g] <= prev(p) for p in P), name=f"sell_owned_{g}")
        m.addConstrs((b[p, g] <= 1 - prev(p) for p in P), name=f"buy_new_{g}")
        # (3) the bank changes only through sales and purchases; it cannot go below 0
        #     (the lower bound 0 is given when bank is created above)
        m.addConstr(bank[g] == bank_prev
                    + gp.quicksum(sell_value[p] * s[p, g] for p in P)
                    - gp.quicksum(cost[p] * b[p, g] for p in P), name=f"budget_{g}")

        # (4) 15 players: 2 GK, 5 DEF, 5 MID, 3 FWD
        m.addConstr(x.sum("*", g) == 15, name=f"squad15_{g}")
        for k, q in QUOTA.items():
            m.addConstr(gp.quicksum(x[p, g] for p in P if pos[p] == k) == q, name=f"quota_{k}_{g}")
        # (5) at most 3 players from one club
        # (12) excess[cl] > 0 when a real-life transfer left more than 3 squad players at a club.
        #      They may be kept, but once any transfer is made (z[g] = 1) the club must go back to 3.
        for cl in excess:
            members = gp.quicksum(x[p, g] for p in P if club[p] == cl)
            if excess[cl] > 0:
                if g not in z:
                    z[g] = m.addVar(vtype=GRB.BINARY, name=f"any_transfer_{g}")
                    m.addConstr(b.sum("*", g) <= 15 * z[g], name=f"any_transfer_link_{g}")
                m.addConstr(members <= CLUB_LIMIT + excess[cl] * (1 - z[g]), name=f"club_{cl}_{g}")
            else:
                m.addConstr(members <= CLUB_LIMIT, name=f"club_{cl}_{g}")

        # (6) 11 starters, all from the squad
        m.addConstrs((y[p, g] <= x[p, g] for p in P), name=f"start_in_squad_{g}")
        m.addConstr(y.sum("*", g) == 11, name=f"xi_{g}")
        # (7) valid formation: 1 GK and at least 3 DEF, 2 MID and 1 FWD
        for k in QUOTA:
            starters = gp.quicksum(y[p, g] for p in P if pos[p] == k)
            m.addConstr(starters >= MIN_START[k], name=f"minform_{k}_{g}")
            m.addConstr(starters <= MAX_START[k], name=f"maxform_{k}_{g}")
        # (8) one captain, who must be a starter
        m.addConstrs((c[p, g] <= y[p, g] for p in P), name=f"capt_starter_{g}")
        m.addConstr(c.sum("*", g) == 1, name=f"one_captain_{g}")

        if first_gw and i == 0:
            # GW1: the first 15 players are not transfers, so (9)-(11) are not used
            m.addConstr(h[g] == 0)
            m.addConstr(u[g] == 0)
            m.addConstr(ft[g] == 0)
        else:
            # (9) every player bought is either a free transfer or a hit
            m.addConstr(b.sum("*", g) == u[g] + h[g], name=f"transfers_{g}")
            if max_transfers is not None:      # only for the "no transfers" strategy
                m.addConstr(b.sum("*", g) <= max_transfers, name=f"max_transfers_{g}")
            # (10) no more free transfers are used than are available
            m.addConstr(u[g] <= ft[g], name=f"use_free_{g}")
            # (11) the plan starts with ft0; GW16 has 5; otherwise unused free transfers
            #      carry over and one new one is added (the limit of 5 is given when ft is created above)
            if i == 0:
                m.addConstr(ft[g] == ft0, name="ft_start")
            elif g == AFCON_GW:
                m.addConstr(ft[g] == MAX_FT, name="ft_afcon")
            else:
                g0 = G[i - 1]
                m.addConstr(ft[g] <= ft[g0] - u[g0] + 1, name=f"ft_roll_{g}")

    # ---------- objective ----------
    obj = gp.LinExpr()
    for i, g in enumerate(G):
        w = discount ** i
        obj += w * gp.quicksum(xp[p, g] * (y[p, g] + c[p, g]) for p in P)
        obj += w * bench_w * gp.quicksum(xp[p, g] * (x[p, g] - y[p, g]) for p in P)
        obj -= w * hit_cost * h[g]
    m.setObjective(obj, GRB.MAXIMIZE)

    m.optimize()
    if m.SolCount == 0:
        raise RuntimeError(f"GW{t}: no feasible solution (status {m.Status})")

    # decisions for GW t, the first planned GW
    g = G[0]
    on = lambda var: [p for p in P if var[p, g].X > 0.5]
    squad, lineup = on(x), on(y)
    bench = [p for p in squad if p not in lineup]
    bench_order = sorted(bench, key=lambda p: (pos[p] != "GK", -xp[p, g]))  # GK first, then by xP
    return {
        "squad": squad, "lineup": lineup, "captain": on(c)[0], "bench": bench_order,
        "buys": on(b), "sells": on(s),
        "hits": int(round(h[g].X)), "free_used": int(round(u[g].X)),
        "bank": bank[g].X, "runtime": m.Runtime,
        "n_vars": m.NumVars, "n_constrs": m.NumConstrs, "optimal": m.Status == GRB.OPTIMAL,
    }


# ---------------------------------------------------------------------------
# inputs of solve() from the processed data files
# ---------------------------------------------------------------------------
def build_inputs(df, xp, t, horizon, owned=()):
    """
    Inputs of solve() at the deadline of GW t.
    df = data/processed/player_gw.csv, xp = data/processed/xp.csv
    """
    G = list(range(t, min(t + horizon, 39)))       # the season ends at GW38
    X = xp[(xp.decision_gw == t) & xp.GW.isin(G)]
    info = df[df.GW == t].set_index("element")
    P = sorted(set(X.element) | set(owned))
    xpd = {(p, g): 0.0 for p in P for g in G}
    for p, g, v in X[["element", "GW", "xp"]].itertuples(index=False):
        xpd[p, g] = v
    return (P, G, xpd, info.price.to_dict(), info.position.to_dict(),
            info.team.to_dict(), info.name.to_dict())


if __name__ == "__main__":
    # demo: the optimal GW1 squad, planning 3 GWs ahead
    import pandas as pd
    df = pd.read_csv("data/processed/player_gw.csv")
    xp = pd.read_csv("data/processed/xp.csv")
    P, G, xpd, price, pos, club, name = build_inputs(df, xp, t=1, horizon=3)
    r = solve(1, P, G, xpd, price, price, pos, club, squad0=[], bank0=100.0, ft0=0)
    print(f"{len(P)} candidates, {r['n_vars']} variables, {r['n_constrs']} constraints, "
          f"solved in {r['runtime']:.2f}s\n")
    order = ["GK", "DEF", "MID", "FWD"]
    for p in sorted(r["squad"], key=lambda p: (order.index(pos[p]), p not in r["lineup"])):
        role = "(C)" if p == r["captain"] else ("XI" if p in r["lineup"] else "bench")
        print(f"{pos[p]:4} {name[p]:28} {club[p]:15} £{price[p]:.1f}m  {role}")
    print(f"\nsquad cost £{sum(price[p] for p in r['squad']):.1f}m, bank £{r['bank']:.1f}m")
