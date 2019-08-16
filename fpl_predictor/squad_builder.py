
import itertools
from numbers import Number
import numpy as np
import pandas as pd

from fpl_predictor.functions import previous_week
from fpl_predictor.squad import Squad
from fpl_predictor.player_scorer import PlayerScorer
from fpl_predictor.api import ApiData
from fpl_predictor.player_information import PlayerInformation


class SquadBuilder:
    def __init__(self, scoring_metric="total_points"):
        """Class for building a squad with the highest possible score on the
        `scoring_metric`. The team is picked FOR the year-week, using data up to
        and including the PREVIOUS year-week. If `live` then current data from
        the live API is used for availability, price etc."""
        self._scoring_metric = scoring_metric
        self._val_metric = f"{self._scoring_metric}_per_value"
        self._scoring_data = PlayerScorer(metric=scoring_metric)
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

    def players(self, year, week, n, agg_func=np.mean, cross_seasons=True,
                live=False):
        """Get a DataFrame of all players, sorted by total score on the
        `scoring_metric` used when instantiating this object. Players are scored
        up to BUT NOT including the year-week; availability and player values
        are FOR the year-week."""
        prev_y, prev_w = previous_week(year, week)
        df = self._scoring_data.score(prev_y, prev_w, n, agg_func, cross_seasons)
        df = self._player_info.add_player_info_to_df(df, year=year, week=week, live=live)
        # Add in the points-per-value metric:
        df[self._val_metric] = df[self._scoring_metric] / df["value"]
        return df.set_index("code")

    def _player_pool(self, year, week, n, agg_func=np.mean, cross_seasons=True,
                     live=False, percent_chance=100, position=None, min_score=None,
                     min_score_per_value=None, max_val=None, min_minute_percent=0.5,
                     drop_unavailable=True):
        df = self.players(year, week, n, agg_func, cross_seasons, live)
        if position:
            if isinstance(position, str):
                position = [position]
            assert all([p in ["GK", "DEF", "MID", "FWD"] for p in position]), \
                f"invalid position(s): {', '.join(position)}"
            df = df.loc[df["position"].isin(position)]
        if isinstance(min_score, Number):
            df = df.loc[df[self._scoring_metric] > min_score]
        if isinstance(min_score_per_value, Number):
            df = df.loc[df[self._val_metric] > min_score_per_value]
        if isinstance(max_val, Number):
            df = df.loc[df["value"] <= max_val]
        if isinstance(min_minute_percent, Number):
            df = df.loc[df["minutes_percent"] >= min_minute_percent]
        if drop_unavailable:
            available = self.player_availability(year, week, live, percent_chance)
            df["available"] = df.index.map(available)
            # Drop False available (or NaN - assumed to have left league):
            df["available"].fillna(False, inplace=True)
            df = df.loc[df["available"]]

        # Remove already selected players:
        df = df.loc[~df.index.isin(self.__squad._selected_list)]

        # Remove players from teams already maxed out:
        if len(self.__squad.maxed_out_teams):
            df = df.loc[~df["team"].isin(self.__squad.maxed_out_teams)]

        # Drop players with NaNs in these columns - they have left the league:
        df.dropna(subset=["position", "value", "team", "total_points_per_value"], inplace=True)

        return df

    def _select_player(self, pool, score_per_value=False):
        """Pick the top player from a player pool based on either `raw_score` or
        `roi_score`. In the event of ties the cheapest player is selected first;
        if that also ties selection is based on the other metric."""
        if not score_per_value:
            sort_by = [self._scoring_metric, "value", self._val_metric]
        else:
            sort_by = [self._val_metric, "value", self._scoring_metric]
        pool.sort_values(by=sort_by, ascending=[False, True, False], inplace=True)
        code = list(pool.index)[0]
        player = pool.loc[code].to_dict()
        player["code"] = code
        return player

    def build_team(self, year, week, n=5, n_premium=3, min_minute_percent=0.5,
                   cross_seasons=True, agg_func=np.mean, live=False,
                   percent_chance=100):
        """Build a team, first by selecting `n_premium` players with the highest
        total scores available, then by selecting the rest on `score_per_value`.

        Args:
            year (int): game year.
            week (int): game week. Team is selected FOR this week, using data up
                to the previous week to calculate scores.
            n (int): number of weeks of historic data to use to score players.
            n_premium (int): number of players to select by highest score.
            min_minute_percent (float): minimum percent of previous `n` week's
                games a player must have played to be eligible for selection.
            cross_seasons (bool): if True, use data from the previous season
                where applicable.
            agg_func (function): aggregating function to use to score players.
            live (bool): if True, get live player availability, value etc. from
                the API instead of historic data.
            percent_chance (int): if `live` is True, minimum percent chance a
                player must have to be eligible for selection.
        """
        self.__squad.remove_all_players()

        # Choose `n_premium` players first:
        for p in range(n_premium):
            need_positions = self.__squad.need_positions
            total_budget = self.__squad.available_budget
            pool = self._player_pool(year, week, n, agg_func=agg_func, cross_seasons=cross_seasons,
                                     position=need_positions, live=live, percent_chance=percent_chance,
                                     max_val=total_budget, min_minute_percent=min_minute_percent)
            if not len(pool):  # Can't fill squad, reduce the target number of n_premium by 1.
                self.build_team(year, week, n_premium - 1, min_minute_percent=min_minute_percent,
                                cross_seasons=cross_seasons, agg_func=agg_func, live=live,
                                percent_chance=percent_chance)
                return
            player = self._select_player(pool, score_per_value=False)
            player["score_per_value"] = player[self._val_metric]
            player["score"] = player[self._scoring_metric]
            self.__squad.add_player(**player)

        # Fill rest of squad:
        for p in range(15 - len(self.__squad.selected)):
            need_positions = self.__squad.need_positions
            total_budget = self.__squad.available_budget
            pool = self._player_pool(year, week, n, agg_func=agg_func, cross_seasons=cross_seasons,
                                     position=need_positions, live=live, percent_chance=percent_chance,
                                     max_val=total_budget, min_minute_percent=min_minute_percent)
            if not len(pool):  # Can't fill squad, reduce the target number of n_premium by 1.
                self.build_team(year, week, n_premium - 1, min_minute_percent=min_minute_percent,
                                cross_seasons=cross_seasons, agg_func=agg_func, live=live,
                                percent_chance=percent_chance)
                return
            player = self._select_player(pool, score_per_value=True)
            player["score_per_value"] = player[self._val_metric]
            player["score"] = player[self._scoring_metric]
            self.__squad.add_player(**player)

        print(f"Build team with {n_premium} players by raw score, {15 - n_premium} by roi, "
              f"using {n} games data. Total score = {self.__squad.total_score}. "
              f"Remaining budget = {self.__squad.available_budget}")
        self.pick_first_team()
        return self.squad

    def pick_first_team(self):
        """Pick the highest scoring 11 players from the team squad."""
        df = self.__squad.selected.copy()
        df.sort_values(by=["score", "score_per_value"], ascending=False, inplace=True)
        max_pos_picks = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 5}
        pos_filled = {k: 0 for k in max_pos_picks}
        first_team = pd.DataFrame(columns=df.columns)
        for row in df.iterrows():
            player = row[1].to_dict()
            position = player["position"]
            position_count = len(first_team.loc[first_team["position"] == position])
            if position_count < max_pos_picks[position]:
                first_team = first_team.append(player, ignore_index=True)
                pos_filled[position] += 1
            if len(first_team) == 11:
                break

        # If a position has no players, drop the lowest player from the other positions:
        missing = [k for k, v in pos_filled.items() if v == 0]
        if len(missing):
            missing_position = missing[0]
            to_remove = first_team.loc[first_team["position"] != "GK"].copy()
            to_remove.sort_values(by=["score", "score_per_value"], ascending=False, inplace=True)
            to_remove = list(to_remove["code"])[-1]
            first_team = first_team.loc[first_team["code"] != to_remove]
            to_add = df.loc[df["position"] == missing_position].copy()
            to_add.sort_values(by=["score", "score_per_value"], ascending=False, inplace=True)
            for row in to_add.iterrows():
                player = row[1].to_dict()
                first_team = first_team.append(player, ignore_index=True)
                break

        self.__first_team = first_team

    @property
    def first_team(self):
        return self.__first_team

    @property
    def squad(self):
        return self.__squad

    @property
    def captain(self):
        if not len(self.first_team):
            return dict()
        else:
            df = self.first_team.sort_values(by="score")
            captain = df.loc[list(df.index)[-1]]
        return {captain["code"]: captain["name"]}

    @property
    def vice_captain(self):
        if not len(self.first_team):
            return dict()
        else:
            df = self.first_team.sort_values(by="score")
            captain = df.loc[list(df.index)[-2]]
        return {captain["code"]: captain["name"]}

    def _team_perms(self, r=2):
        """Make all permutations of `r` length combinations of players in the
        currently selected squad."""
        team = self.squad.selected.copy()
        codes = list(team["code"])
        perms = list(itertools.combinations(codes, r=r))
        df = pd.DataFrame(data=perms, columns=[f"p{i}" for i in range(r)])
        scores = dict(zip(team["code"], team["score"]))
        positions = dict(zip(team["code"], team["position"]))
        values = dict(zip(team["code"], team["value"]))
        for i in range(r):
            df[f"p{i}_val"] = df[f"p{i}"].map(values)
            df[f"p{i}_score"] = df[f"p{i}"].map(scores)
            df[f"p{i}_position"] = df[f"p{i}"].map(positions)
        score_cols = [f"p{i}_score" for i in range(r)]
        val_cols = [f"p{i}_val" for i in range(r)]
        df["score_total"] = df[score_cols].sum(axis=1)
        df["value_total"] = df[val_cols].sum(axis=1)
        position_cols = [f"p{i}_position" for i in range(r)]
        df["positions"] = tuple(zip(*[df[c] for c in position_cols]))
        df["positions"] = [tuple(sorted(p)) for p in df["positions"]]
        player_cols = [c for c in df.columns if "p" in c and len(c) == 2]
        df["players"] = tuple(zip(*[df[c] for c in player_cols]))
        df["players"] = [tuple(sorted(p)) for p in df["players"]]
        return df[["players", "positions", "score_total", "value_total"]]

    def _pool_perms(self, year, week, n, agg_func=np.mean, cross_seasons=True,
                    live=False, percent_chance=100, position=None,
                    min_score=None, min_score_per_value=None, max_val=None,
                    min_minute_percent=0.5, drop_unavailable=True, r=2):
        """Make all permutations of `r` length combinations of players in the
        pool returned by the other criteria."""
        pool = self._player_pool(year, week, n, agg_func=agg_func,
                                 cross_seasons=cross_seasons, live=live,
                                 percent_chance=percent_chance, position=position,
                                 min_score=min_score, min_score_per_value=min_score_per_value,
                                 max_val=max_val, min_minute_percent=min_minute_percent,
                                 drop_unavailable=drop_unavailable)
        codes = list(pool.index)
        perms = list(itertools.combinations(codes, r=r))
        df = pd.DataFrame(data=perms, columns=[f"p{i}" for i in range(r)])
        scores = dict(zip(pool.index, pool[self._scoring_metric]))
        positions = dict(zip(pool.index, pool["position"]))
        values = dict(zip(pool.index, pool["value"]))
        for i in range(r):
            df[f"p{i}_val"] = df[f"p{i}"].map(values)
            df[f"p{i}_score"] = df[f"p{i}"].map(scores)
            df[f"p{i}_position"] = df[f"p{i}"].map(positions)
        score_cols = [f"p{i}_score" for i in range(r)]
        val_cols = [f"p{i}_val" for i in range(r)]
        df["score_total"] = df[score_cols].sum(axis=1)
        df["value_total"] = df[val_cols].sum(axis=1)
        position_cols = [f"p{i}_position" for i in range(r)]
        df["positions"] = tuple(zip(*[df[c] for c in position_cols]))
        df["positions"] = [tuple(sorted(p)) for p in df["positions"]]
        player_cols = [c for c in df.columns if "p" in c and len(c) == 2]
        df["players"] = tuple(zip(*[df[c] for c in player_cols]))
        df["players"] = [tuple(sorted(p)) for p in df["players"]]
        return df[["players", "positions", "score_total", "value_total"]]

    @staticmethod
    def _subset_perms(df, positions: tuple, min_score: float, max_val: float):
        """Get a subset of a `perms` DataFrame meeting the criteria."""
        df = df.loc[(df["positions"] == positions) & (df["score_total"] > min_score) & (df["value_total"] < max_val)]
        return df.sort_values(by=["score_total", "value_total"], ascending=[False, True])

    @property
    def _test_new_squad(self):
        return Squad()

    def squad_optimisations(self, year, week, n, agg_func=np.mean,
                            cross_seasons=True, live=False, percent_chance=100,
                            position=None, min_score=None,
                            min_score_per_value=None, max_val=None,
                            min_minute_percent=0.5, drop_unavailable=True, r=2):
        """Get a DataFrame of all possible optimisations of the current squad,
        by removing r-length combinations of players. Note that not all
        optimisations returned may be possible, as the 3-players-per-team limit
        may be met by already selected players."""
        df = pd.DataFrame()
        team_perms = self._team_perms(r=r)
        team_perms.sort_values(by=["score_total", "value_total"], ascending=[True, False], inplace=True)
        team_perms.reset_index(drop=True, inplace=True)
        av_budget = self.__squad.available_budget
        pool_perms = self._pool_perms(year=year, week=week, n=n, agg_func=agg_func, cross_seasons=cross_seasons,
                                      live=live, percent_chance=percent_chance, position=position,
                                      min_score=min_score, min_score_per_value=min_score_per_value, max_val=max_val,
                                      min_minute_percent=min_minute_percent, drop_unavailable=drop_unavailable, r=2)

        for row_i in team_perms.iterrows():
            remove = row_i[1].to_dict()
            players_to_remove = remove["players"]
            budget = av_budget + remove["value_total"]
            better_perms = self._subset_perms(pool_perms, remove["positions"], remove["score_total"], budget)
            better_perms = better_perms.loc[better_perms["players"] != players_to_remove]
            if len(better_perms):
                s = better_perms.iloc[0].to_dict()
                s["players_to_remove"] = players_to_remove
                s["to_remove_value_total"] = remove["value_total"]
                s["to_remove_score_total"] = remove["score_total"]
                df = df.append(s, ignore_index=True)
        df = df[["positions", "players", "score_total", "value_total",
                 "players_to_remove", "to_remove_score_total", "to_remove_value_total"]]
        df["value_diff"] = df["value_total"] - df["to_remove_value_total"]
        df["score_diff"] = df["score_total"] - df["to_remove_score_total"]
        df.sort_values(by=["score_diff", "value_diff"], ascending=[False, True], inplace=True)
        return df.reset_index(drop=True)

