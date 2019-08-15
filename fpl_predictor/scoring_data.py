
import numpy as np
import pandas as pd

from fpl_predictor.functions import year_week_minus_n, next_week


class ScoringData:
    """Object to hold all of the relevant player scoring data to build a team
    and implement a strategy."""

    def __init__(self, dl, agg_col="total_points"):
        """
        Args:
            dl (DataLoader): object to get data from.
            agg_col (str): the column of gameweek data to use to score players
                to create their ranking. This effectively picks your team.
        """
        self.__master = dl.master
        self.__players_master = dl.players_master
        self._player_value_ix = self.index_players(column="value")
        self._player_minutes_ix = self.index_players(column="minutes")
        self._player_available_ix = self._index_player_availability
        self.scoring_ix = self.index_players(column=agg_col)
        self.scoring_ix_roi = self.index_players(column=agg_col, roi=True)
        self.positions = self.player_positions
        self.__player_names = dl.player_names
        self.__year_player_team = {k: {k1: dl.team_codes[v1] for k1, v1 in v.items()}
                                   for k, v in dl.year_player_team.items()}

    def index_players(self, column="ict_index", roi=False):
        """Create indexed DataFrame of player values for the given column per gameweek.

        Args:
            column (str): column in `master` to get data from.
            roi (bool): if True, divide scores by player values.
        """
        df = pd.pivot_table(self.__master, index=["year", "GW"], columns=["code"], values=column)
        if roi:
            values = pd.pivot_table(self.__master, index=["year", "GW"], columns=["code"], values="value")
            df = df / values
        return df

    @staticmethod
    def aggregate_index(index, year: int, week: int, n: int,
                        agg_func=np.mean, cross_seasons=True):
        """Aggregate the results from the `index_players` method across the `n`
        weeks up to and including the gameweek given by `year` and `week`.

        Args:
            index (pd.DataFrame): result from the `index_players` method.
            year (int): gameweek year to count back from (inclusive).
            week (int): gameweek week to count back from (inclusive).
            n (int): number of gameweeks to aggregate.
            agg_func (function): function to use to aggregate data.
            cross_seasons (bool): if not True, only gameweeks in the same year
                as `year` are returned (so `n` may be reduced).
        """
        df = index.copy()
        min_year, min_week = year_week_minus_n(year, week, n)
        if not cross_seasons:
            min_year = year
            min_week = max([1, week - n + 1])
        df = df.iloc[(((df.index.get_level_values("year") >= min_year) &
                       (df.index.get_level_values("GW") >= min_week)) |
                      (df.index.get_level_values("year") > min_year)) &
                     (((df.index.get_level_values("year") == year) &
                       (df.index.get_level_values("GW") <= week)) |
                      (df.index.get_level_values("year") < year))
                     ]
        return agg_func(df).sort_values(ascending=False)

    def get_player_values(self, year, week):
        """Get all player values in the given year and week."""
        ix = self._player_value_ix.copy()
        values = ix.loc[(year, week), :]
        return values.to_dict()

    def get_players_available(self, year, week):
        """Get all players who were available in the given year and week."""
        s = self._player_minutes_ix.loc[(year, week), :]
        return (s > 0).to_dict()

    @property
    def _index_player_availability(self):
        """Create indexed DataFrame of whether a player was available for a
        gameweek, using the proxy that they played >0 minutes the previous week.
        """
        df = self._player_minutes_ix
        df = df.shift(1)
        df[df > 0] = True
        df = df.replace(0, False)
        return df.fillna(True)

    def player_minutes_percent(self, year, week, n, cross_seasons=False):
        """Calculate the percent of total minutes that each player played in the
        `n` weeks prior up to and including the gameweek.

        Args:
            year (int): gameweek year to count back from (inclusive).
            week (int): gameweek week to count back from (inclusive).
            n (int): number of gameweeks to aggregate.
            cross_seasons (bool): if not True, only gameweeks in the same year
                as `year` are returned (so `n` may be reduced).
        """
        ix = self._player_minutes_ix.copy()
        df = self.aggregate_index(ix, year, week, n, cross_seasons=cross_seasons, agg_func=np.sum)
        if not cross_seasons:
            n = min([week, n])
        return df / (n * 90)

    @property
    def player_positions(self):
        """Get a nested dict of {year: {player code: position}} """
        df = self.__players_master[["year", "code", "position"]].copy()
        positions = dict()
        for year in df["year"].unique():
            year_df = df.loc[df["year"] == year]
            positions[year] = dict(zip(year_df["code"], year_df["position"]))
        return positions

    def score(self, year, week, n, agg_func=np.mean, cross_seasons=True):
        """Score all players based on the kwargs passed in the initialization of
        this object. The `available` column in the returned DataFrame shows the
        player availability in the week AFTER the year-week gameweek."""
        raw_scores = self.aggregate_index(self.scoring_ix, year, week, n,
                                          agg_func=agg_func, cross_seasons=cross_seasons)
        df = pd.DataFrame(data=raw_scores, columns=["raw_score"])
        roi_scores = self.aggregate_index(self.scoring_ix_roi, year, week, n,
                                          agg_func=agg_func, cross_seasons=cross_seasons)
        df["roi_score"] = roi_scores
        df["value"] = df.index.map(self.get_player_values(year, week))
        df["position"] = df.index.map(self.positions[year])
        minutes_percent = self.player_minutes_percent(year, week, n, cross_seasons=cross_seasons)
        df["minutes_percent"] = df.index.map(minutes_percent)
        next_y, next_w = next_week(year, week)
        df["available"] = df.index.map(self.get_players_available(next_y, next_w))
        df["name"] = df.index.map(self.__player_names)
        df["team"] = df.index.map(self.__year_player_team[year])
        return df
