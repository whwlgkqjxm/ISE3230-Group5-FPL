"""
data_prep.py
------------
Step 1 of the project: download the 2025-26 FPL data and turn it into ONE clean
table with one row per (player, gameweek).

Output: data/processed/player_gw.csv
        data/processed/fixtures_known.csv  (matches per club and GW as known at each GW deadline)

Columns
  element      player id
  name         player name
  position     GK / DEF / MID / FWD
  GW           gameweek (1..38)
  team         club in that GW (27 players changed club mid-season)
  n_fixtures   number of matches the club plays that GW (0 = blank, 2 = double)
  available    1 if the player exists in the game that GW (new signings appear later)
  price        price in £m that GW
  minutes      minutes played (summed over the GW)
  apps         matches actually played that GW (0, 1 or 2; counted per match)
  points       actual FPL points (summed over the GW)

Data problems handled
  1. duplicate rows (same player, same fixture)       -> dropped
  2. double gameweeks (2 rows per player in one GW)    -> summed
  3. blank gameweeks (no row at all)                   -> filled with 0 points
  4. players joining mid-season                        -> available = 0 before first appearance
  5. club changes                                      -> team taken per GW
"""
from pathlib import Path
import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2025-26"
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
N_GW = 38


def download_raw():
    """Download the raw files once and keep a copy in data/raw (needed for the GitHub repo)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = {"merged_gw.csv": "gws/merged_gw.csv",
             "fixtures.csv": "fixtures.csv",
             "teams.csv": "teams.csv"}
    for local, remote in files.items():
        path = RAW_DIR / local
        if not path.exists():
            print(f"downloading {remote} ...")
            pd.read_csv(f"{BASE_URL}/{remote}").to_csv(path, index=False)
    return (pd.read_csv(RAW_DIR / "merged_gw.csv"),
            pd.read_csv(RAW_DIR / "fixtures.csv"),
            pd.read_csv(RAW_DIR / "teams.csv"))


def team_fixture_counts(fixtures, teams):
    """Number of matches each club plays in each GW (from the fixture list, known in advance)."""
    id2name = teams.set_index("id")["name"]
    home = fixtures[["event", "team_h"]].rename(columns={"team_h": "team_id"})
    away = fixtures[["event", "team_a"]].rename(columns={"team_a": "team_id"})
    both = pd.concat([home, away])
    counts = both.groupby(["team_id", "event"]).size().rename("n_fixtures").reset_index()
    counts["team"] = counts["team_id"].map(id2name)
    counts = counts.rename(columns={"event": "GW"})[["team", "GW", "n_fixtures"]]
    # every (team, GW) pair, 0 where the club has no match
    full = pd.MultiIndex.from_product([id2name.values, range(1, N_GW + 1)], names=["team", "GW"])
    return counts.set_index(["team", "GW"]).reindex(full, fill_value=0).reset_index()


def known_fixture_counts(fixtures, teams):
    """
    Matches per club in each future GW, as they were KNOWN at the deadline of each decision GW.
    fixtures.csv holds the final schedule; 5 matches were moved after the season started
    (data/raw/fixture_changes.csv, with the official announcement dates). Before a match was
    postponed it counts in its original GW; once postponed without a new date it counts in no GW
    (as in FPL); once the new date was announced it counts in its final GW.
    A change is known at decision GW t if it was announced before the day of t's first kick-off.
    """
    changes = pd.read_csv(RAW_DIR / "fixture_changes.csv").set_index("fixture_id")
    id2name = teams.set_index("id")["name"]
    fx = fixtures.copy()
    fx["ko"] = pd.to_datetime(fx["kickoff_time"]).dt.tz_localize(None)
    deadline_day = fx.groupby("event")["ko"].min().dt.normalize()     # first kick-off day of each GW
    rows = []
    for t in range(1, N_GW + 1):
        day = deadline_day[t]
        for f in fx.itertuples():
            gw = f.event
            if f.id in changes.index:
                c = changes.loc[f.id]
                if not pd.Timestamp(c.postponed_known) < day:
                    gw = c.original_gw                     # change not yet known
                elif not pd.Timestamp(c.new_date_known) < day:
                    gw = None                              # postponed, new date not yet known
            if gw is not None and gw >= t:
                rows += [(t, id2name[f.team_h], gw), (t, id2name[f.team_a], gw)]
    known = pd.DataFrame(rows, columns=["decision_gw", "team", "GW"])
    known = known.groupby(["decision_gw", "team", "GW"]).size().rename("n_fixtures")
    full = pd.MultiIndex.from_tuples([(t, tm, g) for t in range(1, N_GW + 1) for tm in id2name.values
                                      for g in range(t, N_GW + 1)], names=["decision_gw", "team", "GW"])
    return known.reindex(full, fill_value=0).reset_index()


def build_player_gw(merged, fixtures, teams):
    # 1. drop duplicate rows
    before = len(merged)
    merged = merged.drop_duplicates(subset=["element", "fixture"])
    print(f"duplicate rows removed: {before - len(merged)}")

    # 2. one row per (player, GW) — sum points/minutes over a double GW
    gw = (merged.groupby(["element", "GW"])
          .agg(name=("name", "first"),
               position=("position", "first"),
               team=("team", "first"),
               price=("value", "first"),
               minutes=("minutes", "sum"),
               apps=("minutes", lambda s: int((s > 0).sum())),   # per match, not per GW
               points=("total_points", "sum"))
          .reset_index())
    gw["price"] = gw["price"] / 10  # 55 -> 5.5 (£m)

    # 3. full grid: every player x every GW
    players = gw.groupby("element").agg(name=("name", "first"), position=("position", "first"),
                                        first_gw=("GW", "min"))
    grid = pd.MultiIndex.from_product([players.index, range(1, N_GW + 1)], names=["element", "GW"])
    df = gw.set_index(["element", "GW"]).drop(columns=["name", "position"]).reindex(grid).reset_index()
    df = df.merge(players, left_on="element", right_index=True)

    # 4. available only from the first GW the player shows up
    df["available"] = (df["GW"] >= df["first_gw"]).astype(int)

    # 5. team/price in a blank GW = last known value (or next known, for the very first GWs)
    df = df.sort_values(["element", "GW"])
    for col in ["team", "price"]:
        df[col] = df.groupby("element")[col].transform(lambda s: s.ffill().bfill())
    df[["minutes", "points", "apps"]] = df[["minutes", "points", "apps"]].fillna(0).astype(int)

    # number of matches that GW for the player's club
    df = df.merge(team_fixture_counts(fixtures, teams), on=["team", "GW"], how="left")

    cols = ["element", "name", "position", "GW", "team", "n_fixtures",
            "available", "price", "minutes", "apps", "points"]
    return df[cols].reset_index(drop=True)


def sanity_checks(df):
    assert not df.duplicated(["element", "GW"]).any(), "duplicate (player, GW)"
    assert df["position"].isin(["GK", "DEF", "MID", "FWD"]).all()
    assert df["n_fixtures"].notna().all()
    totals = df.groupby("name")["points"].sum().sort_values(ascending=False)
    print("\nTop 5 season totals (check vs. real FPL):")
    print(totals.head().to_string())
    print("\nFixtures per GW (blank / double GWs):")
    per_gw = df.drop_duplicates(["team", "GW"]).groupby("GW")["n_fixtures"].sum() // 2
    print(per_gw[per_gw != 10].to_string())


if __name__ == "__main__":
    merged, fixtures, teams = download_raw()
    df = build_player_gw(merged, fixtures, teams)
    sanity_checks(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "player_gw.csv", index=False)
    known = known_fixture_counts(fixtures, teams)
    known.to_csv(OUT_DIR / "fixtures_known.csv", index=False)
    n_diff = (known.merge(team_fixture_counts(fixtures, teams), on=["team", "GW"], suffixes=("", "_final"))
              .query("n_fixtures != n_fixtures_final")[["decision_gw", "team", "GW"]])
    print(f"saved {OUT_DIR / 'fixtures_known.csv'}  ({len(n_diff)} (decision GW, club, GW) entries differ "
          f"from the final fixture list)")
    print(f"\nsaved {OUT_DIR / 'player_gw.csv'}  ({len(df)} rows, {df.element.nunique()} players)")
