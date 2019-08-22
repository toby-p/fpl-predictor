
from numbers import Number
import numpy as np
import pandas as pd

from fpl_predictor.functions import previous_week
from fpl_predictor.squad import Squad
from fpl_predictor.player_scorer import PlayerScorer
from fpl_predictor.api import ApiData
from fpl_predictor.player_information import PlayerInformation


class SquadBuilder:
    def __init__(self, **kwargs):
        """Class for building a squad with the highest possible score on the
        `scoring_metric`. The team is picked FOR the year-week, using data up to
        and including the PREVIOUS year-week. If `live` then current data from
        the live API is used for availability, price etc."""
        defaults = {"scoring_metric": "total_points", "n": 10, "agg_func": np.mean, "cross_seasons": True,
                    "percent_chance": 100, "min_minute_percent": 0.5}
        kw = {**defaults, **kwargs}
        self.scoring_metric = kw["scoring_metric"]
        self._val_metric = f"{self.scoring_metric}_per_value"
        self.n = kw["n"]
        self.cross_seasons = kw["cross_seasons"]
        self.agg_func = kw["agg_func"]
        self.percent_chance = kw["percent_chance"]
        self.min_minute_percent = kw["min_minute_percent"]
        self._scoring_data = PlayerScorer(metric=self.scoring_metric)
        self._api_data = ApiData()
        self._player_info = PlayerInformation()

        # Player availability index. Use minutes played previous week as a proxy for availability:
        df = self._scoring_data._ix_player_minutes.copy()
        df = df.shift(1)
        df[df > 0] = True
        df = df.replace(0, False)
        df = df.fillna(True)
        # For the first week of a season, use the second week's data:
        for year in df.index.levels[0]:
            try:
                s = df.loc[(year, 2), :]
                df.loc[(year, 1), :] = s
            except KeyError:
                pass
        self._ix_player_available = df.fillna(True)

        # Attributes for tracking team selections:
        self.__squad = Squad()
        self.__first_team = pd.DataFrame()

    def player_availability(self, year, week, live=False, percent_chance=100):
        """Calculate player availability for a given year-week combination.
        If `live` then live data is pulled from the API, and year-week is ignored."""
        if live:
            return self._api_data.players_available(percent_chance=percent_chance)
        else:
            s = self._ix_player_available.loc[(year, week), :].fillna(False)
            return s.to_dict()

    def players(self, year, week, live=False):
        """Get a DataFrame of all players, sorted by total score on the
        `scoring_metric` used when instantiating this object. Players are scored
        up to BUT NOT including the year-week; availability and player values
        are FOR the year-week."""
        prev_y, prev_w = previous_week(year, week)
        df = self._scoring_data.score(prev_y, prev_w, self.n, self.agg_func, self.cross_seasons)
        df = self._player_info.add_player_info_to_df(df, year=year, week=week, live=live)
        # Add in the points-per-value metric:
        df[self._val_metric] = df[self.scoring_metric] / df["value"]
        return df.set_index("code")

    def _player_pool(self, year, week, live=False, position=None,
                     min_score=None, min_score_per_value=None, max_val=None,
                     drop_unavailable=True):
        """Get a pool of players to select from which meet the criteria."""
        df = self.players(year, week, live)
        if position:
            if isinstance(position, str):
                position = [position]
            assert all([p in ["GK", "DEF", "MID", "FWD"] for p in position]), \
                f"invalid position(s): {', '.join(position)}"
            df = df.loc[df["position"].isin(position)]
        if isinstance(min_score, Number):
            df = df.loc[df[self.scoring_metric] > min_score]
        if isinstance(min_score_per_value, Number):
            df = df.loc[df[self._val_metric] > min_score_per_value]
        if isinstance(max_val, Number):
            df = df.loc[df["value"] <= max_val]
        if isinstance(self.min_minute_percent, Number):
            df = df.loc[df["minutes_percent"] >= self.min_minute_percent]
        if drop_unavailable:
            available = self.player_availability(year, week, live, self.percent_chance)
            df["available"] = df.index.map(available)
            # Drop False available (or NaN - assumed to have left league):
            df["available"].fillna(False, inplace=True)
            df = df.loc[df["available"]]

        # Remove already selected players:
        df = df.loc[~df.index.isin(self.squad.selected_list)]

        # Remove players from teams already maxed out:
        if len(self.squad.maxed_out_teams):
            df = df.loc[~df["team"].isin(self.squad.maxed_out_teams)]

        # Rename scoring columns to generic names:
        df.rename(columns={self.scoring_metric: "score", self._val_metric: "score_per_value"}, inplace=True)

        # Drop players with NaNs in these columns - they have left the league:
        df.dropna(subset=["position", "value", "team", "score_per_value"], inplace=True)

        return df

    @staticmethod
    def _select_player(pool, score_per_value=False):
        """Pick the top player from a player pool based on either `score` or
        `score_per_value`. In the event of ties the cheapest player is selected
        first; if that also ties selection is based on the other metric."""
        if not score_per_value:
            sort_by = ["score", "value", "score_per_value"]
        else:
            sort_by = ["score_per_value", "value", "score"]
        pool.sort_values(by=sort_by, ascending=[False, True, False], inplace=True)
        code = list(pool.index)[0]
        player = pool.loc[code].to_dict()
        player["code"] = code
        return player

    @property
    def squad(self):
        return self.__squad

    def squad_unavailable(self, year, week, live=False, percent_chance=100):
        """Identify members of the currently selected squad who are unavailable
        for selection in the given game-week."""
        availability = self.player_availability(year, week, live=live, percent_chance=percent_chance)
        codes = self.squad.selected_list
        return {k: v for k, v in availability.items() if k in codes and not v}

    def first_team_unavailable(self, year, week, live=False, percent_chance=100):
        """Identify members of the current first team who are unavailable for
        selection in the given game-week."""
        availability = self.player_availability(year, week, live=live, percent_chance=percent_chance)
        codes = sorted(self.squad.first_team["code"])
        return {k: v for k, v in availability.items() if k in codes and not v}

    @property
    def _test_new_squad(self):
        return Squad()

    def build_squad(self, build_function, year: int, week: int, live=False,
                    **kwargs):
        """Build a new squad using the supplied `build_function`."""
        build_function(self, year, week, live, **kwargs)

    def optimise_squad(self, optimise_function, n_iterations: int, year: int,
                       week: int, live=False, **kwargs):
        """Optimise the currently selected squad using the supplied
        `optimise_function`, attempting to run the optimising function
        `n_iterations` times."""
        optimise_function(self, n_iterations, year, week, live, **kwargs)
