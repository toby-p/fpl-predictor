
import os
import numpy as np
import pandas as pd

from fpl_predictor.functions import index_players, aggregate_index
from nav import DIR_STRUCTURED_DATA


class PlayerScorer:
    """Object to score players based on the chosen metric for the desired
    number of previous weeks."""

    def __init__(self, metric="total_points"):
        """Score all players based on the target `metric`.

        Args:
            metric (str): column of gameweek data to score players on.
        """
        fp = os.path.join(DIR_STRUCTURED_DATA, "master.csv")
        self._master = pd.read_csv(fp, encoding="utf-8")
        assert metric in self._master.columns, f"Invalid scoring metric: {metric}"
        self._metric = metric

        # Index the players based on score on the `metric`:
        self._ix_score = index_players(self._master, column=metric)

        # Index players by how many minutes they played:
        self._ix_player_minutes = index_players(self._master, column="minutes")

    def __player_minutes_percent(self, year, week, n, cross_seasons=False):
        """Calculate the percent of total minutes that each player played in the
        `n` weeks prior up to and including the gameweek.

        Args:
            year (int): gameweek year to count back from (inclusive).
            week (int): gameweek week to count back from (inclusive).
            n (int): number of gameweeks to aggregate.
            cross_seasons (bool): if not True, only gameweeks in the same year
                as `year` are returned (so `n` may be reduced).
        """
        df = self._ix_player_minutes.copy()
        df = aggregate_index(df, year, week, n, cross_seasons=cross_seasons, agg_func=np.sum)
        if not cross_seasons:
            n = min([week, n])
        return df / (n * 90)

    def score(self, year, week, n, agg_func=np.mean, cross_seasons=True):
        """Score all players based on the `metric` passed in the initialization
        of this class instance."""
        df = aggregate_index(self._ix_score, year, week, n,
                             agg_func=agg_func, cross_seasons=cross_seasons)
        df = pd.DataFrame(data=df, columns=[self._metric])
        df.reset_index(inplace=True)
        mins = self.__player_minutes_percent(year, week, n, cross_seasons=cross_seasons)
        df["minutes_percent"] = df["code"].map(mins)
        return df
