"""
model.py
--------
Step 3 of the project: the multi-gameweek MILP (solved with gurobipy).

At the deadline of GW t we plan GWs  G = {t, ..., t+H-1}  and later only carry out
the decisions for GW t (rolling horizon, see simulate.py).

Sets / data
  P            candidate players (available at t)
  G            planning gameweeks
  xp[p,g]      expected points (forecast.py)
  cost[p]      buy price at t (£m)          sell[p]  selling price at t (£m)
  pos[p], club[p]
  squad0       current 15 players (empty before GW1)
  bank0        money in the bank,   ft0  free transfers available at t

Decision variables (g in G)
  x[p,g]  1 if p is in the 15-man squad          (binary)
  y[p,g]  1 if p is in the starting 11            (binary)
  c[p,g]  1 if p is captain                       (binary)
  b[p,g]  1 if p is bought before GW g            (binary)
  s[p,g]  1 if p is sold before GW g              (binary)
  ft[g]   free transfers available in GW g        (integer 0..5)
  u[g]    free transfers used in GW g             (integer)
  h[g]    paid transfers ("hits", -4 points each) (integer)
  bank[g] money left after GW g transfers         (continuous >= 0)
  z[g]    1 if any transfer is made in GW g       (binary; only created when a club is over
          the 3-player limit after a real-life transfer)

Objective
  max  sum_g  d^(g-t) * [ sum_p xp[p,g]*(y[p,g] + c[p,g])            starters + captain bonus
                          + BENCH_W * sum_p xp[p,g]*(x[p,g]-y[p,g])     small value for bench cover
                          - 4*h[g] ]                                    transfer penalty
"""
import gurobipy as gp
from gurobipy import GRB

QUOTA = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}       # squad composition
MIN_START = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}   # valid formation (GK exactly 1)
MAX_START = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
CLUB_LIMIT = 3
HIT_COST = 4
MAX_FT = 5
AFCON_GW = 16          # free transfers topped up to 5 in GW16 (2025/26 rule)


def solve(t, P, G, xp, cost, sell, pos, club, squad0, bank0, ft0,
          hit_cost=HIT_COST, bench_w=0.1, discount=0.9, max_transfers=None,
          time_limit=60, verbose=False):
    """Build and solve the MILP for the deadline of GW t. Returns the GW-t decisions."""
    first_gw = len(squad0) == 0          # GW1: squad picked from scratch, no transfer rules
    m = gp.Model(f"fpl_gw{t}")
    m.Params.OutputFlag = int(verbose)
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = 1e-4

    # ---------- variables ----------
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
    z = {}                              # only created when a club is over the limit (rare)
    sell_value = {p: sell[p] if p in owned else cost[p] for p in P}

    # ---------- constraints ----------
    for i, g in enumerate(G):
        prev = (lambda p: 1 if p in owned else 0) if i == 0 else (lambda p, gp_=G[i - 1]: x[p, gp_])
        bank_prev = bank0 if i == 0 else bank[G[i - 1]]

        # squad flow: this week's squad = last week's + bought - sold
        m.addConstrs((x[p, g] == prev(p) + b[p, g] - s[p, g] for p in P), name=f"flow_{g}")
        # can only sell a player you own, can only buy a player you don't own
        m.addConstrs((s[p, g] <= prev(p) for p in P), name=f"sell_owned_{g}")
        m.addConstrs((b[p, g] <= 1 - prev(p) for p in P), name=f"buy_new_{g}")
        # budget: money only moves through transfers, never negative
        m.addConstr(bank[g] == bank_prev
                    + gp.quicksum(sell_value[p] * s[p, g] for p in P)
                    - gp.quicksum(cost[p] * b[p, g] for p in P), name=f"budget_{g}")

        # squad composition and club limit
        m.addConstr(x.sum("*", g) == 15, name=f"squad15_{g}")
        for k, q in QUOTA.items():
            m.addConstr(gp.quicksum(x[p, g] for p in P if pos[p] == k) == q, name=f"quota_{k}_{g}")
        # FPL rule: if a real-life transfer leaves us with more than 3 players from a club, we may
        # keep them, but any transfer we make must bring that club back to 3.
        #   z[g] = 1 if any transfer is made in GW g;  limit = 3 + excess * (1 - z[g])
        excess = {cl: sum(1 for p in owned if club.get(p) == cl) - CLUB_LIMIT for cl in set(club.values())}
        for cl in excess:
            members = gp.quicksum(x[p, g] for p in P if club[p] == cl)
            if excess[cl] > 0:
                if g not in z:
                    z[g] = m.addVar(vtype=GRB.BINARY, name=f"any_transfer_{g}")
                    m.addConstr(b.sum("*", g) <= 15 * z[g], name=f"any_transfer_link_{g}")
                m.addConstr(members <= CLUB_LIMIT + excess[cl] * (1 - z[g]), name=f"club_{cl}_{g}")
            else:
                m.addConstr(members <= CLUB_LIMIT, name=f"club_{cl}_{g}")

        # starting XI with a valid formation, captain among starters
        m.addConstrs((y[p, g] <= x[p, g] for p in P), name=f"start_in_squad_{g}")
        m.addConstr(y.sum("*", g) == 11, name=f"xi_{g}")
        for k in QUOTA:
            starters = gp.quicksum(y[p, g] for p in P if pos[p] == k)
            m.addConstr(starters >= MIN_START[k], name=f"minform_{k}_{g}")
            m.addConstr(starters <= MAX_START[k], name=f"maxform_{k}_{g}")
        m.addConstrs((c[p, g] <= y[p, g] for p in P), name=f"capt_starter_{g}")
        m.addConstr(c.sum("*", g) == 1, name=f"one_captain_{g}")

        # transfers: free ones first, the rest cost hit_cost points each
        if first_gw and i == 0:
            m.addConstr(h[g] == 0)
            m.addConstr(u[g] == 0)
            m.addConstr(ft[g] == 0)
            m.addConstr(s.sum("*", g) == 0)
        else:
            m.addConstr(b.sum("*", g) == u[g] + h[g], name=f"transfers_{g}")
            if max_transfers is not None:      # used by the "no transfers" baseline
                m.addConstr(b.sum("*", g) <= max_transfers, name=f"max_transfers_{g}")
            m.addConstr(u[g] <= ft[g], name=f"use_free_{g}")
            if i == 0:
                m.addConstr(ft[g] == ft0, name="ft_start")
            elif g == AFCON_GW:
                m.addConstr(ft[g] == MAX_FT, name="ft_afcon")
            else:
                # unused free transfers roll over (+1 per week, at most 5)
                g0 = G[i - 1]
                m.addConstr(ft[g] <= ft[g0] - u[g0] + 1, name=f"ft_roll_{g}")
                if first_gw and i == 1:
                    m.addConstr(ft[g] <= 1)

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

    g = G[0]
    on = lambda var: [p for p in P if var[p, g].X > 0.5]
    bench = [p for p in on(x) if p not in on(y)]
    bench_order = sorted(bench, key=lambda p: (pos[p] != "GK", -xp[p, g]))  # GK first, then by xP
    return {
        "squad": on(x), "lineup": on(y), "captain": on(c)[0], "bench": bench_order,
        "buys": on(b), "sells": on(s),
        "hits": int(round(h[g].X)), "free_used": int(round(u[g].X)),
        "bank": bank[g].X, "objective": m.ObjVal, "runtime": m.Runtime,
        "n_vars": m.NumVars, "n_constrs": m.NumConstrs, "model": m,
    }


# ---------------------------------------------------------------------------
# helper: turn the CSV files into the inputs of solve()
# ---------------------------------------------------------------------------
def build_inputs(df, xp, t, horizon, owned=(), n_keep=None):
    """
    df = data/processed/player_gw.csv, xp = data/processed/xp.csv
    n_keep = optional dict like {"GK": 10, "DEF": 30, ...} to keep only the best
             candidates per position (faster); owned players are always kept.
    """
    G = list(range(t, min(t + horizon, 39)))
    X = xp[(xp.decision_gw == t) & xp.GW.isin(G)]
    total = X.groupby("element").xp.sum()
    info = df[df.GW == t].set_index("element")
    P = list(total.index)
    if n_keep:
        P = []
        for k, n in n_keep.items():
            ids = info[info.position == k].index.intersection(total.index)
            P += list(total.loc[ids].sort_values(ascending=False).head(n).index)
    P = sorted(set(P) | set(owned))
    xpd = {(p, g): 0.0 for p in P for g in G}
    for p, g, v in X[X.element.isin(P)][["element", "GW", "xp"]].itertuples(index=False):
        xpd[p, g] = v
    return (P, G, xpd, info.price.to_dict(), info.position.to_dict(),
            info.team.to_dict(), info.name.to_dict())


if __name__ == "__main__":
    # demo: pick the GW1 squad with a 3-week look-ahead, using ALL players
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
