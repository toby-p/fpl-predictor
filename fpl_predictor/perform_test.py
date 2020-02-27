
import numpy as np
import os

from fpl_predictor.functions import now_as_string, save_json
from fpl_predictor.points_test import PointsTest
from fpl_predictor.squad_builder import SquadBuilder
from nav import DIR_DATA


def perform_test(metrics: list, ns: list, year: int, weeks: list,
                 cross_seasons: list = None, agg_funcs: list = None,
                 percent_chance: list = None, min_minute_percent: list = None):
    """Perform a test for every permutation of parameters provided.
    """
    # Set default values:
    cross_seasons = [False] if not cross_seasons else cross_seasons
    agg_funcs = [np.mean] if not agg_funcs else agg_funcs
    percent_chance = [100] if not percent_chance else percent_chance
    min_minute_percent = [0.5] if not min_minute_percent else min_minute_percent

    now = now_as_string()
    new_dir_fp = os.path.join(DIR_DATA, "test_results", now)
    os.mkdir(new_dir_fp)

    # Iterate through tests saving scores:
    count = 1
    for af in agg_funcs:
        for cs in cross_seasons:
            for pc in percent_chance:
                for mmp in min_minute_percent:
                    sqb = SquadBuilder(cross_seasons=cs, percent_chance=pc, min_minute_percent=mmp)

                    for metric in metrics:
                        for n in ns:
                            for week in weeks:
                                scores = PointsTest(sqb, year=year, week=week, metric=metric, n=n)
                                results = dict(
                                    metric=metric, n=n, year=year, week=week, cross_seasons=cs,
                                    agg_func=str(af), percent_chance=pc, min_minute_percent=mmp)
                                results = {**results, **scores.total_scores}
                                json_fp = os.path.join(new_dir_fp, f"test_{count}.json")
                                save_json(results, json_fp)
                                count += 1
    print(f"Performed {count:,} tests, JSON results in: {new_dir_fp}")
    return new_dir_fp
