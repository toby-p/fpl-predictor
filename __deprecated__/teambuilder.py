
from numbers import Number
import numpy as np
import pandas as pd

from fpl_predictor.functions import previous_week
from fpl_predictor.squad import Squad


class TeamBuilder:
    def __init__(self, scoring_data, year, week, n=10, min_minute_percent=0.5,
                 cross_seasons=False, agg_func=np.mean, n_by_raw_score=3,
                 pick_team_by="raw_score", live_data=False):
        """Build a team squad with the highest possible score on the given
        metrics. Note that the player scores are calculated on the WEEK PRIOR to
        the given year-week combination. So if the year-week is 2018-1, then the
        scores will be calculated on the previous season's data.

        Args:
            scoring_data (ScoringData): instance of ScoringData object.
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
            n_by_raw_score (int): number of players to select based on raw score
                when building team, before selecting on roi.
            live_data (bool): whether or not to use live API data to determine
                cost and player availability.
        """
        self._scoring_data = scoring_data
        self._player_values = self._scoring_data.get_player_values(year, week)
        self.year, self.week = year, week
        self.scoring_year, self.scoring_week = previous_week(year, week)
        self.n = n
        self.min_minute_percent = min_minute_percent
        self.cross_seasons = cross_seasons
        self.agg_func = agg_func
        self.pick_team_by = pick_team_by
        self.n_by_raw_score = n_by_raw_score
        self.live_data = live_data
        self.scores = self._scoring_data.score(self.scoring_year, self.scoring_week, self.n,
                                               agg_func=self.agg_func, cross_seasons=self.cross_seasons)

        # Attributes for tracking team selections:
        self.team = Squad()

        # Attribute for storing the first team selection:
        self.__first_team = pd.DataFrame()

    def player_pool(self, position=None, drop_unavailable=True, min_score=None,
                    min_roi=None, max_price=None):
        """Pool of players to select from that meet criteria. Excludes players
        already selected at the `team.team` attribute.
        """
        df = self.scores.copy()
        if drop_unavailable:
            df = df.loc[df["available"]]
        if position:
            if isinstance(position, str):
                position = [position]
            assert all([p in ["GK", "DEF", "MID", "FWD"] for p in position])
            df = df.loc[df["position"].isin(position)]
        if isinstance(min_score, Number):
            df = df.loc[df["raw_score"] > min_score]
        if isinstance(min_roi, Number):
            df = df.loc[df["roi_score"] > min_roi]
        if isinstance(max_price, Number):
            df = df.loc[df["value"] <= max_price]
        df = df.loc[~df.index.isin(self.team.selected_list)]
        if len(self.team.maxed_out_teams):
            df = df.loc[~df["team"].isin(self.team.maxed_out_teams)]
        return df

    @staticmethod
    def select_player(pool, select_by="raw_score"):
        """Pick the top player from a player pool based on either `raw_score` or
        `roi_score`. In the event of ties the cheapest player is selected first;
        if that also ties selection is based on the other metric."""
        if select_by == "raw_score":
            sort_by = ["raw_score", "value", "roi_score"]
        elif select_by == "roi_score":
            sort_by = ["roi_score", "value", "raw_score"]
        else:
            raise ValueError(f"Invalid `select_by` arg: {select_by}")
        pool.sort_values(by=sort_by, ascending=[False, True, False], inplace=True)
        code = list(pool.index)[0]
        player = pool.loc[code].to_dict()
        player["code"] = code
        return player

    def build_team(self, n_by_raw_score=None):
        """Build a team, first by selecting based purely on raw score regardless
        of price, then by selecting on roi.

        Args:
            n_by_raw_score (int): number of players to select by raw score.
        """
        if not isinstance(n_by_raw_score, int):
            n_by_raw_score = self.n_by_raw_score
        assert n_by_raw_score <= 15, "Maximum number of players is 15"
        self.team.remove_all_players()
        for p in range(n_by_raw_score):
            pool = self.player_pool(position=self.team.need_positions,
                                    drop_unavailable=True, min_score=None, min_roi=None,
                                    max_price=self.team.available_budget)
            if not len(pool):
                self.build_team(n_by_raw_score - 1)
                return
            player = self.select_player(pool, select_by="raw_score")
            self.team.add_player(**player)

        for p in range(15 - len(self.team.selected)):
            pool = self.player_pool(position=self.team.need_positions,
                                    drop_unavailable=True, min_score=None, min_roi=None,
                                    max_price=self.team.available_budget)
            if not len(pool):
                self.build_team(n_by_raw_score - 1)
                return
            player = self.select_player(pool, select_by="roi_score")
            self.team.add_player(**player)

        print(f"Build team with {n_by_raw_score} players by raw score, {15 - n_by_raw_score} by roi, "
              f"using {self.n} games data. "
              f"Total score = {self.team.total_score}. "
              f"Remaining budget = {self.team.available_budget}")
        self.pick_first_team(sort_by=self.pick_team_by)

    @staticmethod
    def _sort_cols(sort_by="raw_score"):
        if sort_by == "raw_score":
            sort_by = ["raw_score", "roi_score"]
        elif sort_by == "roi_score":
            sort_by = ["roi_score", "raw_score"]
        else:
            raise ValueError(f"Invalid `sort_by` arg: {sort_by}")
        return sort_by

    def pick_first_team(self, sort_by="raw_score"):
        """Pick the highest scoring 11 players from the team squad, sorted by
        the columns given in the `by` argument."""
        sort_by = self._sort_cols(sort_by)
        df = self.team.selected.copy()
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

    def evaluate_transfers(self, sort_by="raw_score"):
        """Evaluate the current team against any possible transfers which will
        improve the overall team score."""
        transfers = pd.DataFrame(columns=["old", "new", "score_differential",
                                          "roi_differential", "value_differential"])
        df = self.team.selected.copy()
        for row in df.iterrows():
            player = row[1].to_dict()
            pool = self.player_pool(position=player["position"],
                                    drop_unavailable=True,
                                    min_score=player["raw_score"],
                                    min_roi=["roi_score"],
                                    max_price=player["value"] + self.team.available_budget)
            if len(pool):
                new_player = pool.loc[list(pool.index)[0]].to_dict()
                new_player["uuid"] = list(pool.index)[0]
                transfer = {"old": player["uuid"], "new": new_player["uuid"],
                            "score_differential": new_player["total_points (raw)"] - player["raw_score"],
                            "roi_differential": new_player["total_points (roi)"] - player["roi_score"],
                            "value_differential": new_player["value"] - player["value"],
                            }
                transfers = transfers.append(transfer, ignore_index=True)

        # Remove any players transferring in twice:
        if sort_by == "raw_score":
            sort_by = "score_differential"
        elif sort_by == "score_roi":
            sort_by = "roi_differential"
        transfers.sort_values(by=sort_by, ascending=False)
        transfers.drop_duplicates(subset=["new"], keep="first")

        # Make sure no existing players in transfers:
        transfers = transfers.loc[~transfers["new"].isin(self.team.selected["uuid"].unique())]

        return transfers

    def add_player(self, code: int):
        """Shortcut to add a player to the team."""
        player = self.scores.loc[code].to_dict()
        player = {k: v for k, v in player.items() if k in self.team.selected.columns}
        player["code"] = code
        self.team.add_player(**player)

    @property
    def captain(self):
        if not len(self.first_team):
            return dict()
        else:
            df = self.first_team.sort_values(by="raw_score")
            captain = df.loc[list(df.index)[-1]]
        return {captain["uuid"]: captain["name"]}

    @property
    def vice_captain(self):
        if not len(self.first_team):
            return dict()
        else:
            df = self.first_team.sort_values(by="raw_score")
            captain = df.loc[list(df.index)[-2]]
        return {captain["uuid"]: captain["name"]}
