
import os
from numbers import Number
import numpy as np
import pandas as pd

from nav import DIR_STRUCTURED_DATA


class PlayerScorer:
    """Object to score players based on the chosen metric for the desired
    number of previous weeks."""

    def __init__(self, metric: str = "total_points"):
        """Score all players based on the target `metric`.

        Args:
            metric (str): column of gameweek data to score players on.
        """
        fp = os.path.join(DIR_STRUCTURED_DATA, "master.csv")
        self.master = pd.read_csv(fp, encoding="utf-8", low_memory=False)
        assert metric in self.master.columns, f"Invalid scoring metric: {metric}"
        self.metric = metric

        # Index the players based on score on the `metric`:
        self.scores = self.index_master(column=metric)

        # Index players by how many minutes they played:
        self.minutes = self.index_master(column="minutes")

    def index_master(self, column: str = "ict_index"):
        """Create an indexed DataFrame of player values for the given column of data
         in master. The returned DataFrame has player codes as columns, and
         year and gameweek as a nested index.

        Args:
            column (str): column in `master` to get data from.
        """
        df = pd.pivot_table(self.master, index=["year", "GW"], columns=["code"], values=column)
        return df

    def aggregate_master(self, year: int, week: int, n: int,
                         column: str = "ict_index", agg_func=np.mean,
                         cross_seasons: bool = True):
        """Aggregate the results from the `index_players` function across the `n`
        weeks up to and including the gameweek given by `year` and `week`.

        Args:
            year (int): gameweek year to count back from (inclusive).
            week (int): gameweek week to count back from (inclusive).
            n (int): number of gameweeks to aggregate.
            column (str): column in `master` to aggregate.
            agg_func (function): function to use to aggregate data.
            cross_seasons (bool): if not True, only gameweeks in the same year
                as `year` are returned (so `n` may be reduced).
        """
        df = self.index_master(column=column)
        ix_list = df.index.to_list()
        max_ix = ix_list.index((year, week)) + 1
        min_ix = max_ix - n
        indices = ix_list[min_ix: max_ix]
        if not cross_seasons:
            indices = [t for t in indices if t[0] == year]
        df = df.loc[indices]
        return agg_func(df).sort_values(ascending=False)

    def get_player_minutes_percent(self, year: int, week: int, n: int,
                                   cross_seasons: bool = False):
        """Calculate the percent of total minutes that each player played in the
        `n` weeks prior up to and including the gameweek.

        Args:
            year (int): gameweek year to count back from (inclusive).
            week (int): gameweek week to count back from (inclusive).
            n (int): number of gameweeks to aggregate.
            cross_seasons (bool): if not True, only gameweeks in the same year
                as `year` are returned (so `n` may be reduced).
        """
        s = self.aggregate_master(year, week, n, column="minutes",
                                  cross_seasons=cross_seasons, agg_func=np.sum)
        if not cross_seasons:
            n = min([week, n])
        return s / (n * 90)

    def get_all_scores(self, year: int, week: int, n: int, agg_func=np.mean,
                       cross_seasons: bool = True):
        """Score all players based on the `metric` passed in the initialization
        of this class instance."""
        df = self.aggregate_master(year, week, n, column=self.metric,
                                   agg_func=agg_func, cross_seasons=cross_seasons)
        df = pd.DataFrame(data=df, columns=[self.metric])
        df.reset_index(inplace=True)
        mins = self.get_player_minutes_percent(year, week, n, cross_seasons=cross_seasons)
        df["minutes_percent"] = df["code"].map(mins)
        return df.set_index("code")

    def get_player_gw_score(self, code: int, year: int, week: int):
        """Get an individual player's score for a given year-week."""
        score = self.scores.loc[(year, week), code]
        if isinstance(score, Number):
            return score
        else:
            raise TypeError(f"Can't find player score: code {code}, year: {year}, week: {week}")
