from numbers import Number
import numpy as np
import pandas as pd

from functions import previous_week
from team import Team
from fpl_predictor.dataloader import DataLoader


class TeamBuilder:
    def __init__(self, year, week, n=10, min_minute_percent=0.5,
                 cross_seasons=False, agg_func=np.mean, agg_col="total_points",
                 n_by_score=3, pick_first_team_by="score",
                 live_availability=False):
        """Build a team squad with the highest possible score on the given
        metrics. Note that the player scores are calculated on the WEEK PRIOR to
        the given year-week combination. So if the year-week is 2018-1, then the
        scores will be calculated on the previous season's data.

        Args:
            year (int): year to choose the team for.
            week (int): gameweek to choose the team for.
            n (int): number of weeks' of historic data to score the players on.
            min_minute_percent (float): ratio of total possible minutes of the
                previous `n` games that a player must have played in to be
                eligible for selection.
            cross_seasons (bool): if True, use historic data from the previous
                season when scoring players (True by default for the first week
                of a year).
            agg_func (function): function to use in scoring player data.
            agg_col (str): the column of gameweek data to use to score players
                to create their ranking. Effectively this is picking your team.
            n_by_score (int): number of players to select based on raw score
                when building team, before selecting on roi.
            live_availability (bool): whether or not to use live API data to
                determine player availability.
        """
        self.year, self.week = year, week
        self.scoring_year, self.scoring_week = previous_week(year, week)
        self.n = n
        self.min_minute_percent = min_minute_percent
        self.cross_seasons = cross_seasons
        self.agg_func = agg_func
        self.agg_col = agg_col
        self.pick_first_team_by = pick_first_team_by
        self.n_by_score = n_by_score
        self.live_availability = live_availability

        self.dataloader = DataLoader(make_master=False, build_player_db=False)

        # Load the gameweek data:
        self.gw = self.dataloader.gw(year, week)

        # Get the current player values:
        self.values = dict(zip(self.gw["uuid"], self.gw["value"]))

        # Load the previous week's data:
        self.prev_gw = self.dataloader.gw(self.scoring_year, self.scoring_week)

        # Store player IDs who missed previous week. This is used as a proxy for
        # availability in the historic data, as this data isn't available.
        self.missed_prev_week = sorted(self.prev_gw.loc[self.prev_gw["minutes"] == 0, "uuid"])

        # Attributes for tracking team selections:
        self.team = Team()
        self._old_team = self.team.team
        self.scores = self.score()
        self.scores["position"] = self.scores.index.map(self.team.player_positions)
        self.scores["value"] = self.scores.index.map(self.values)

        # Attribute for storing the first team selection:
        self.__first_team = pd.DataFrame()

    def score(self):
        """Create a DataFrame of player scores based on the kwargs passed in the
        initialization of this object."""
        subset = self.dataloader.subset(year=self.scoring_year, week=self.scoring_week,
                                        n=self.n, cross_seasons=self.cross_seasons)
        raw_scores = self.dataloader.score(subset=subset,
                                           min_minute_percent=self.min_minute_percent,
                                           agg_func=self.agg_func, agg_cols=[self.agg_col])
        raw_scores.rename(columns={self.agg_col: f"{self.agg_col} (raw)"}, inplace=True)
        roi_scores = self.dataloader.score(subset=subset, roi=True,
                                           min_minute_percent=self.min_minute_percent,
                                           agg_func=self.agg_func, agg_cols=[self.agg_col])
        roi_scores.rename(columns={self.agg_col: f"{self.agg_col} (roi)"}, inplace=True)
        roi_scores.drop(columns=["most_recent_value", "minutes_percent"], inplace=True)
        df = pd.merge(raw_scores, roi_scores, left_index=True, right_index=True)
        teams = subset.loc[subset["year, week"] == subset["year, week"].max()]
        teams = dict(zip(teams["player_uuid"], teams["team"]))
        df["team"] = df.index.map(teams)
        return df

    def player_pool(self, position=None, drop_unavailable=True, min_score=None,
                    min_roi=None, max_price=None):
        """Pool of players to select from that meet criteria. Excludes players
        already selected at the `team.team` attribute.
        """
        df = self.scores.copy()
        if drop_unavailable:
            available = self.available_players(live_data=self.live_availability)
            df = df.loc[df.index.isin(available)]
        if position:
            if isinstance(position, str):
                position = [position]
            assert all([p in ["GK", "DEF", "MID", "FWD"] for p in position])
            df = df.loc[df["position"].isin(position)]
        if isinstance(min_score, Number):
            raw_col = [c for c in df.columns if "(raw)" in c][0]
            df = df.loc[df[raw_col] > min_score]
            raw_col = [c for c in df.columns if "(raw)" in c][0]
            df.sort_values(by=[raw_col, "value"], ascending=[False, True], inplace=True)
        if isinstance(min_roi, Number):
            roi_col = [c for c in df.columns if "(roi)" in c][0]
            df = df.loc[df[roi_col] > min_roi]
            df.sort_values(by=[roi_col, "value"], ascending=[False, True], inplace=True)
        if isinstance(max_price, Number):
            df = df.loc[df["value"] <= max_price]

        # Remove already selected:
        selected = list(self.team.team["uuid"])
        df = df.loc[~df.index.isin(selected)]

        # Remove maxed out teams:
        if len(self.maxed_out_teams):
            df = df.loc[~df["team"].isin(self.maxed_out_teams)]

        return df

    @staticmethod
    def select_player(player_pool, select_by_roi=False):
        """Pick the top player from a player pool. If select by ROI choose
        player with best ROI score, else choose player with best total score"""
        if select_by_roi:
            col = [c for c in player_pool.columns if "(roi)" in c][0]
        else:
            col = [c for c in player_pool.columns if "(raw)" in c][0]
        player_pool.sort_values(by=[col, "value"], ascending=[False, True], inplace=True)

        new = dict()
        new["uuid"] = list(player_pool.index)[0]
        player = player_pool.loc[new["uuid"]]
        roi_ix = [i for i in player.index if "(roi)" in i][0]
        new["roi_score"] = player[roi_ix]
        score_ix = [i for i in player.index if "(raw)" in i][0]
        new["score"] = player[score_ix]
        new["value"] = player["value"]
        return new

    def build_team(self, n_by_score=None):
        """Build a team, first by selecting based purely on raw score regardless
        of price, then by selecting on roi.

        Args:
            n_by_score (int): number of players to select by raw score.
        """
        if not isinstance(n_by_score, int):
            n_by_score = self.n_by_score
        assert n_by_score <= 15, "Maximum number of players is 15"
        self.team.clear_team()
        for p in range(n_by_score):
            pool = self.player_pool(position=self.team.need_positions,
                                    drop_unavailable=True, min_score=None, min_roi=None,
                                    max_price=self.team.available_budget)
            if not len(pool):
                self.build_team(n_by_score - 1)
                return
            player = self.select_player(pool, select_by_roi=False)
            self.team.add_player(**player)

        for p in range(15 - len(self.team.team)):
            pool = self.player_pool(position=self.team.need_positions,
                                    drop_unavailable=True, min_score=None, min_roi=None,
                                    max_price=self.team.available_budget)
            if not len(pool):
                self.build_team(n_by_score - 1)
                return
            player = self.select_player(pool, select_by_roi=True)
            self.team.add_player(**player)

        print(f"Build team with {n_by_score} players chosen by raw "
              f"score, {15 - n_by_score} by roi. Total score = {self.team.total_score}. "
              f"Remaining budget = {self.team.available_budget}")
        self.pick_first_team(sort_by=self.pick_first_team_by)

    def pick_first_team(self, sort_by="score"):
        """Pick the highest scoring 11 players from the team squad, sorted by
        the columns given in the `by` argument."""
        if sort_by == "score":
            sort_by = ["score", "roi_score"]
        elif sort_by == "roi_score":
            sort_by = ["roi_score", "score"]
        else:
            raise ValueError(f"Invalid `sort_by` arg: {sort_by}")
        df = self.team.team.copy()
        df.sort_values(by=sort_by, ascending=False, inplace=True)
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
            to_remove.sort_values(by=sort_by, ascending=False, inplace=True)
            to_remove = list(to_remove["uuid"])[-1]
            first_team = first_team.loc[first_team["uuid"] != to_remove]
            to_add = df.loc[df["position"] == missing_position].copy()
            to_add.sort_values(by=sort_by, ascending=False, inplace=True)
            for row in to_add.iterrows():
                player = row[1].to_dict()
                first_team = first_team.append(player, ignore_index=True)
                break

        self.__first_team = first_team

    @property
    def first_team(self):
        return self.__first_team

    def available_players(self, live_data=False):
        """List of player UUIDs available for selection in the week. If using
        historical data, whether or not the player played the week before is
        used as a proxy as the data is not available."""
        if not live_data:
            df = self.gw.copy()
            available = df.loc[~df["uuid"].isin(self.missed_prev_week), "uuid"]
            return sorted(set(available))
        else:
            df = self.dataloader.current_api_elements.copy()
            df = df.loc[(df["chance_of_playing_next_round"].isna()) |
                        (df["chance_of_playing_next_round"] == 100)]
            return sorted(set(df["uuid"]))

    @property
    def maxed_out_teams(self):
        """List of teams from which 3 players have already been selected."""
        team = self.team.team.copy()
        if len(team) < 4:
            return list()
        team_dict = dict(zip(self.scores.index, self.scores["team"]))
        team["team"] = team["uuid"].map(team_dict)
        team = team["team"].value_counts().to_dict()
        return [k for k, v in team.items() if v == 3]
