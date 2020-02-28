from itertools import product
import numpy as np
import os
import pandas as pd
import sys

try:
    from fpl_predictor.functions import now_as_string, open_json, save_json
    from fpl_predictor.points_test import PointsTest
    from fpl_predictor.squad_builder import SquadBuilder
    from nav import DIR_DATA
except ModuleNotFoundError:
    DIR, FILENAME = os.path.split(__file__)
    sys.path.append(os.path.dirname(DIR))
    from fpl_predictor.functions import now_as_string, open_json, save_json
    from fpl_predictor.points_test import PointsTest
    from fpl_predictor.squad_builder import SquadBuilder
    from nav import DIR_DATA


def load_master():
    """Load the master file of test results."""
    master_fp = os.path.join(DIR_DATA, "test_results", "results_master.csv")
    if not os.path.exists(master_fp):
        df = pd.DataFrame(columns=["metric", "n", "year", "week", "cross_seasons", "agg_func",
                                   "percent_chance", "min_minute_percent",
                                   "pv_first_team_score", "pv_benchboost_score",
                                   "raw_first_team_score", "raw_benchboost_score"])
        df.to_csv(master_fp, encoding="utf-8", index=False)
    df = pd.read_csv(master_fp, encoding="utf-8")
    df["agg_func"] = np.where(df["agg_func"].str.contains("mean"), "mean", df["agg_func"])
    df["agg_func"] = np.where(df["agg_func"].str.contains("sum"), "sum", df["agg_func"])
    assert set(df["agg_func"]).issubset(["mean", "sum"]), "Unknown values found in `agg_func` col."
    return df


def perform_test(metrics: list, ns: list, year: int, weeks: list,
                 cross_seasons: list = None, agg_funcs: list = None,
                 percent_chance: list = None, min_minute_percent: list = None):
    """Perform a test for every permutation of parameters provided.
    """
    collect_all_test_results()

    # Set default values:
    cross_seasons = [False] if not cross_seasons else cross_seasons
    agg_funcs = ["mean"] if not agg_funcs else agg_funcs
    assert set(agg_funcs).issubset(["mean", "sum"]), "Unknown `agg_funcs` values."
    percent_chance = [100] if not percent_chance else percent_chance
    min_minute_percent = [0.5] if not min_minute_percent else min_minute_percent

    # Create the directory to store JSON files of test results:
    now = now_as_string()
    new_dir_fp = os.path.join(DIR_DATA, "test_results", now)
    os.mkdir(new_dir_fp)

    # Identify all new test combinations in parameters:
    combinations = list(product(agg_funcs, cross_seasons, percent_chance,
                                min_minute_percent, metrics, ns, weeks))
    df = load_master()
    df_combos = tuple(zip(*[df["agg_func"], df["cross_seasons"],
                            df["percent_chance"], df["min_minute_percent"],
                            df["metric"], df["n"], df["week"]]))
    combinations = [c for c in combinations if c not in df_combos]
    print(f"Performing {len(combinations):,} tests")
    if not len(combinations):
        return

    # Iterate through new test combinations saving scores:
    first, count = combinations[0], 1
    af, cs, pc, mmp = first[0], first[1], first[2], first[3]
    sqb = SquadBuilder(agg_func=af, cross_seasons=cs, percent_chance=pc, min_minute_percent=mmp)
    for combo in combinations:
        af, cs, pc, mmp = combo[0], combo[1], combo[2], combo[3]
        metric, n, week = combo[4], combo[5], combo[6]
        if not all([sqb.agg_func == af,
                    sqb.cross_seasons == cs,
                    sqb.percent_chance == pc,
                    sqb.min_minute_percent == mmp]):
            sqb = SquadBuilder(agg_func=af, cross_seasons=cs, percent_chance=pc, min_minute_percent=mmp)
        print(f"Current test: {count:,}", end="\r", file=sys.stdout)
        scores = PointsTest(sqb, year=year, week=week, metric=metric, n=n)
        results = dict(metric=metric, n=n, year=year, week=week, cross_seasons=cs,
                       agg_func=str(af), percent_chance=pc, min_minute_percent=mmp)
        results = {**results, **scores.total_scores}
        json_fp = os.path.join(new_dir_fp, f"test_{count}.json")
        save_json(results, json_fp)
        count += 1

    print(f"Performed {count:,} tests, JSON results in: {new_dir_fp}")


def collect_all_test_results():
    """Aggregate all the JSON files of test results into 1 master CSV file."""
    df = load_master()
    test_dir = os.path.join(DIR_DATA, "test_results")
    dirs = [os.path.join(test_dir, d) for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    for d in dirs:
        json_files = [os.path.join(test_dir, d, f) for f in os.listdir(d) if f.endswith(".json")]
        for f in json_files:
            d = open_json(f)
            df = df.append(d, ignore_index=True)

    # Standardize function column so drop-duplicates works:
    df["agg_func"] = np.where(df["agg_func"].str.contains("mean"), "mean", df["agg_func"])
    df["agg_func"] = np.where(df["agg_func"].str.contains("sum"), "sum", df["agg_func"])
    assert set(df["agg_func"]).issubset(["mean", "sum"]), "Unknown values found in `agg_func`."
    df = df.drop_duplicates().reset_index(drop=True)

    # Save to CSV:
    master_fp = os.path.join(test_dir, "results_master.csv")
    df.to_csv(master_fp, encoding="utf-8", index=False)
    print(f"{len(df):,} test results saved in CSV at: {master_fp}")
    return df


if __name__ == "__main__":
    metrics = ["selected", "transfers_in", "ict_index", "total_points", "bonus", "bps"]
    ns = list(range(1, 38, 1))
    cross_seasons = [False, True]
    year = 2019
    weeks = list(range(1, 28, 1))
    perform_test(metrics=metrics, ns=ns, year=year, weeks=weeks, cross_seasons=cross_seasons)
