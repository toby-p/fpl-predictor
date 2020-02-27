from numbers import Number
import numpy as np
import pandas as pd

from fpl_predictor.functions import previous_week
from fpl_predictor.squad import Squad
from fpl_predictor.player_scorer import PlayerScorer
from fpl_predictor.api import ApiData
from fpl_predictor.player_information import PlayerInformation

legal_formations = [{"GK": 1, "DEF": 3, "MID": 4, "FWD": 3},
                    {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2},
                    {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
                    {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1},
                    {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},
                    {"GK": 1, "DEF": 5, "MID": 4, "FWD": 1},
                    {"GK": 1, "DEF": 5, "MID": 3, "FWD": 2}]

VERBOSE = True


def verbose_print(msg: str):
    if VERBOSE:
        print(msg)


class SquadBuilder:
    def __init__(self, **kwargs):
        """Class for building a squad with the highest possible score on the
        `scoring_metric`. The team is picked FOR the year-week, using data up to
        and including the PREVIOUS year-week. If `live` then current data from
        the live API is used for availability, price etc."""
        defaults = {"scoring_metric": "total_points", "n": 10, "agg_func": np.mean, "cross_seasons": True,
                    "percent_chance": 100, "min_minute_percent": 0.5}
        kw = {**defaults, **kwargs}
        self.set_scoring_metric(kw["scoring_metric"])
        self.set_n(kw["n"])
        self.cross_seasons = kw["cross_seasons"]
        self.agg_func = kw["agg_func"]
        self.percent_chance = kw["percent_chance"]
        self.min_minute_percent = kw["min_minute_percent"]
        self.PlayerScorer = PlayerScorer(metric=self.scoring_metric)
        self.ApiData = ApiData()
        self.PlayerInformation = PlayerInformation()

        # Player availability index. Use minutes played previous week as a proxy for availability:
        df = self.PlayerScorer.minutes.copy()
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
        self.squad = Squad()
        self.__first_team = pd.DataFrame()

    @property
    def scoring_cols(self):
        return sorted(self.PlayerScorer.master.columns)

    def set_scoring_metric(self, metric):
        self.PlayerScorer = PlayerScorer(metric=metric)
        self.__scoring_metric = metric
        self.__val_metric = f"{metric}_per_value"

    @property
    def val_metric(self):
        return self.__val_metric

    @property
    def scoring_metric(self):
        return self.__scoring_metric

    def set_n(self, n: int):
        assert isinstance(n, int), f"`n` must be int."
        self.__n = n

    @property
    def n(self):
        return self.__n

    def player_availability(self, year, week, live=False, percent_chance=100):
        """Calculate player availability for a given year-week combination.
        If `live` data is pulled from the API, and year-week is ignored."""
        if live:
            return self.ApiData.players_available(percent_chance=percent_chance)
        else:
            s = self._ix_player_available.loc[(year, week), :].fillna(False)
            return s.to_dict()

    def players(self, year: int, week: int, live=False):
        """Get a DataFrame of all players, sorted by total score on the
        `scoring_metric` used when instantiating this object. Players are scored
        up to BUT NOT including the year-week; availability and player values
        are FOR the year-week."""
        prev_y, prev_w = previous_week(year, week)
        df = self.PlayerScorer.get_all_scores(year=prev_y, week=prev_w, n=self.n, agg_func=self.agg_func,
                                              cross_seasons=self.cross_seasons)
        df = self.PlayerInformation.add_player_info_to_df(df, year=year, week=week, live=live)
        # Add in the points-per-value metric:
        df[self.val_metric] = df[self.scoring_metric] / df["value"]
        return df

    def get_player_pool(self, year, week, live=False, position=None,
                        min_score=None, min_score_per_value=None, max_val=None,
                        drop_unavailable=True, drop_selected=True,
                        drop_maxed_out=True):
        """Get a pool of players to select from which meet the criteria."""
        df = self.players(year, week, live)
        n = len(df)

        if position:
            position = [position] if isinstance(position, str) else position
            assert all([p in ["GK", "DEF", "MID", "FWD"] for p in position]), \
                f"invalid position(s): {', '.join(position)}"
            df = df.loc[df["position"].isin(position)]
            verbose_print(f"Dropped {n - len(df):,} rows for `position`")
            n = len(df)
        if isinstance(min_score, Number):
            df = df.loc[df[self.scoring_metric] > min_score]
            verbose_print(f"Dropped {n - len(df):,} rows for `min_score`")
            n = len(df)
        if isinstance(min_score_per_value, Number):
            df = df.loc[df[self.val_metric] > min_score_per_value]
            verbose_print(f"Dropped {n - len(df):,} rows for `min_score_per_value`")
            n = len(df)
        if isinstance(max_val, Number):
            df = df.loc[df["value"] <= max_val]
            verbose_print(f"Dropped {n - len(df):,} rows for `max_val`")
            n = len(df)
        if isinstance(self.min_minute_percent, Number):
            df = df.loc[df["minutes_percent"] >= self.min_minute_percent]
            verbose_print(f"Dropped {n - len(df):,} rows for `self.min_minute_percent`")
            n = len(df)
        if drop_unavailable:
            available = self.player_availability(year, week, live, self.percent_chance)
            df["available"] = df.index.map(available)
            # Drop False available (or NaN - assumed to have left league):
            df["available"].fillna(False, inplace=True)
            df = df.loc[df["available"]]
            verbose_print(f"Dropped {n - len(df):,} rows for `drop_unavailable`")
            n = len(df)

        # Remove already selected players:
        if drop_selected:
            df = df.loc[~df.index.isin(self.squad.selected_list)]
            verbose_print(f"Dropped {n - len(df):,} rows for `drop_selected`")
            n = len(df)

        # Remove players from teams already maxed out:
        if drop_maxed_out:
            if len(self.squad.maxed_out_teams):
                df = df.loc[~df["team"].isin(self.squad.maxed_out_teams)]
                verbose_print(f"Dropped {n - len(df):,} rows for `drop_maxed_out`")
                n = len(df)

        # Rename scoring columns to generic names:
        df.rename(columns={self.scoring_metric: "score", self.val_metric: "score_per_value"}, inplace=True)

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

    def team_points(self, year, week, benchboost=False):
        """Get the total fantasy points the currently selected team would have
        achieved in a given gameweek."""
        selected = self.squad.selected_list
        points = self.PlayerInformation.points_scored(year=year, week=week)
        points = {k: v for k, v in points.items() if k in selected}
        minutes = self.PlayerInformation.minutes_played(year, week)
        minutes = {k: v for k, v in minutes.items() if k in selected}

        # Double captain/vice-captain's points if they played:
        captain = list(self.squad.captain.keys())[0]
        vice_captain = list(self.squad.vice_captain.keys())[0]
        if minutes[captain] > 0:
            points[captain] = points[captain] * 2
        elif minutes[vice_captain] > 0:
            points[vice_captain] = points[vice_captain] * 2

        if benchboost:
            return {k: v for k, v in points.items() if k in selected}

        to_score = self.squad.first_team

        # Sub in substitutes who played over first-teamers who didn't:
        subs = self.squad.substitutes
        subs["minutes"] = subs["code"].map(minutes)
        subs_played = subs.loc[subs["minutes"] > 0]
        first_team = self.squad.first_team
        first_team["minutes"] = first_team["code"].map(minutes)
        didnt_play = first_team.loc[first_team["minutes"] == 0]

        # Switch keepers first if possible:
        if "GK" in subs_played["position"] and "GK" in didnt_play["position"]:
            new_keeper = subs_played.loc[subs_played["position"] == "GK"].iloc[0].to_dict()
            to_score = to_score.loc[to_score["position"] != "GK"]
            to_score = to_score.append(new_keeper)
        subs_played = subs_played.loc[subs_played["position"] != "GK"]

        # Now try swapping any other subs in if legal formations:
        for sub in subs_played.iterrows():
            revert = to_score.copy()
            swap_in = sub[1].to_dict()
            for out in didnt_play.iterrows():
                swap_out = out[1].to_dict()
                to_score = to_score.loc[to_score["code"] != swap_out["code"]]
                to_score = to_score.append(swap_in, ignore_index=True)
                formation = to_score["position"].value_counts().to_dict()
                if formation in legal_formations:
                    didnt_play = didnt_play.loc[didnt_play["code"] != swap_in["code"]]
                    break
                else:
                    to_score = revert

        return {k: v for k, v in points.items() if k in list(to_score["code"])}

    def add_player(self, code: int, year: int, week: int, live=False):
        """Add a player to the current squad."""
        player = self.PlayerInformation.get_player(code, year, week, live)
        score = self.PlayerScorer.get_player_gw_score(code, year, week)
        player["score"] = score
        player["score_per_value"] = score / player["value"]
        self.squad.add_player(**player)
        print(f"Player added to squad: {player['name']}")

    def top_fill(self, year: int, week: int, live: bool = False, n_top: int = 4,
                 score_per_value: bool = False, **kwargs):
        """Build a team from scratch selecting the `n_top` absolute highest
        scoring players, then selecting from the bottom."""
        self.squad.remove_all_players()

        # Get all available players:
        df = self.get_player_pool(year, week, live=live)

        # Sort by score descending, value ascending:
        score_col = f"{self.scoring_metric}_per_value" if score_per_value else self.scoring_metric
        df.sort_values(by=[score_col, "value"], ascending=[False, True], inplace=True)

        while len(self.squad.selected) < n_top:
            df = df.loc[(df["position"].isin(self.squad.need_positions)) &
                        (~df["team"].isin(self.squad.maxed_out_teams))]
            row = df.reset_index().iloc[0].to_dict()
            self.squad.add_player(**row)
            df = df.iloc[1:]

        # Now just keep adding cheap players:
        df.sort_values(by=["value", "score"], ascending=[True, False], inplace=True)
        while not self.squad.squad_full:
            df = df.loc[(df["position"].isin(self.squad.need_positions)) &
                        (~df["team"].isin(self.squad.maxed_out_teams))]
            row = df.reset_index().iloc[0].to_dict()
            self.squad.add_player(**row)
            df = df.iloc[1:]

        return self.squad.selected
