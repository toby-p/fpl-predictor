from numbers import Number
import numpy as np
import pandas as pd
import random

from functions import previous_week
from team import Team
from fpl_predictor.dataloader import DataLoader


class TeamBuilder:
    def __init__(self, year, week, n=10, min_minute_percent=0.5,
                 cross_seasons=False, agg_func=np.mean, agg_col="total_points"):
        self.year = year
        self.week = week
        self.n = n
        self.min_minute_percent = min_minute_percent
        self.cross_seasons = cross_seasons if week > 1 else True
        self.agg_func = agg_func
        self.agg_col = agg_col

        self.dataloader = DataLoader(make_master=False, build_player_db=False)

        # Load the gameweek data:
        self.gw = DataLoader.gw(year, week)

        # Map in the UUIDs
        uuids = self.dataloader.player_uuids
        year_ids = uuids.loc[uuids["year"] == year]
        year_ids = dict(zip(year_ids["id"], year_ids["cross_season_id"]))
        self.gw["uuid"] = self.gw["element"].map(year_ids)

        # Get the current player values:
        self.values = dict(zip(self.gw["uuid"], self.gw["value"]))

        # Load the previous week's data:
        prev_year, prev_week = previous_week(year, week)
        self.prev_gw = DataLoader.gw(prev_year, prev_week)

        year_ids = uuids.loc[uuids["year"] == prev_year]
        year_ids = dict(zip(year_ids["id"], year_ids["cross_season_id"]))
        self.prev_gw["uuid"] = self.prev_gw["element"].map(year_ids)

        # Store player IDs who missed previous week:
        self.missed_prev_week = sorted(self.prev_gw.loc[self.prev_gw["minutes"] == 0, "uuid"])

        # Attributes for tracking team selections:
        self.team = Team()
        self._old_team = self.team.team

        self.scores = self.score()
        self.scores["position"] = self.scores.index.map(self.team.player_positions)
        self.scores["value"] = self.scores.index.map(self.values)

    def revert(self):
        self.team.team = self._old_team

    def score(self):
        subset = self.dataloader.subset(year=self.year, week=self.week, n=self.n,
                                        cross_seasons=self.cross_seasons)

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
        return df

    def player_pool(self, position=None, drop_missed_prev_week=True,
                    drop_unavailable=True, min_score=None, min_roi=None, max_price=None):
        """Pool of players to select from that meet criteria. Excludes players
        already selected at the `team.team` attribute.
        """
        df = self.scores.copy()
        if drop_missed_prev_week:
            df = df.loc[~df.index.isin(self.missed_prev_week)]
        if position:
            if isinstance(position, str):
                position = [position]
            assert all([p in ["GK", "DEF", "MID", "FWD"] for p in position])
            df = df.loc[df["position"].isin(position)]
        if drop_unavailable:
            # TODO!
            pass
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

        return df

    @staticmethod
    def select_player(player_pool, select_by_roi=False):
        """Pick the top player from a player pool. If select by ROI 
        choose player with best ROI score, else choose player with best total score"""
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

    def build_team(self, n_premium=3):
        """Build a team, first by selecting `n_premium` based purely on points 
        return regardless of price, then by selecting on roi.
        """
        assert n_premium <= 15, "Maximum number of players is 15"
        self.team.clear_team()
        for p in range(n_premium):
            pool = self.player_pool(position=self.team.need_positions, drop_missed_prev_week=True,
                                    drop_unavailable=True, min_score=None, min_roi=None,
                                    max_price=self.team.available_budget)
            player = self.select_player(pool, select_by_roi=False)
            self.team.add_player(**player)

        for p in range(15 - n_premium):
            pool = self.player_pool(position=self.team.need_positions, drop_missed_prev_week=True,
                                    drop_unavailable=True, min_score=None, min_roi=None,
                                    max_price=self.team.available_budget)
            player = self.select_player(pool, select_by_roi=True)
            self.team.add_player(**player)

        print(f"Build team with total score {self.team.total_score}, "
              f"remaining budget {self.team.available_budget}")

    def optimise_1(self, n_top_score=5, x_bottom_roi=5):
        """Select a player from the top n number of players ranked by score, and
        replace them with a cheaper player with a higher roi score. Then drop a
        player with one of the bottom x roi scores and replace them with a
        player with a higher total score. If the total score of the team
        increases keep the changes, else revert to the previous team.
        """
        if not self.team.filled:
            print("Team must be filled before being optimized.")
            return

        self._old_team, old_score = self.team.team.copy(), self.team.total_score
        team = self.team.team.copy()
        n_new_player, x_new_player = None, None
        n_remove_player, x_remove_player = None, None

        # Randomly choose a player with a top n score:
        team.sort_values(by="score", ascending=False, inplace=True)
        team.reset_index(drop=True, inplace=True)
        top_n = list(team.loc[:n_top_score, "uuid"])
        random.shuffle(top_n)
        for n in top_n:
            position = self.team.player_positions[n]
            roi = team.loc[team["uuid"] == n, "roi_score"].values[0]
            max_price = 1000 - (self.team.total_value - team.loc[team["uuid"] == n, "value"].values[0])
            print(max_price)
            pool = self.player_pool(position=position, drop_missed_prev_week=True,
                                    drop_unavailable=True, min_score=None, min_roi=roi,
                                    max_price=max_price)
            if len(pool):
                n_new_player = self.select_player(pool)
                n_remove_player = n

        # Randomly choose a player with a bottom x roi:
        team.sort_values(by="roi_score", ascending=True, inplace=True)
        team.reset_index(drop=True, inplace=True)
        bottom_x = list(team.loc[:x_bottom_roi, "uuid"])
        random.shuffle(bottom_x)
        for x in bottom_x:
            position = self.team.player_positions[x]
            score = self.team.team.loc[team["uuid"] == x, "score"].values[0]
            max_price = 1000 - (self.team.total_value - team.loc[team["uuid"] == x, "value"].values[0])
            print(max_price)
            pool = self.player_pool(position=position, drop_missed_prev_week=True,
                                    drop_unavailable=True, min_score=score, min_roi=None,
                                    max_price=max_price)
            if len(pool):
                x_new_player = self.select_player(pool)
                x_remove_player = x

        if n_new_player and x_new_player:
            self.team.remove_player(n_remove_player)
            print(f"Removed player {self.team.player_names[n_remove_player]} ", end="")
            self.team.add_player(**n_new_player)
            print(f"added player {self.team.player_names[n_new_player['uuid']]}")
            self.team.remove_player(x_remove_player)
            print(f"Removed player {self.team.player_names[x_remove_player]} ", end="")
            self.team.add_player(**x_new_player)
            print(f"added player {self.team.player_names[x_new_player['uuid']]}")

        else:
            print("Couldn't find players.")
            self.revert()
            return

        if self.team.total_score > old_score:
            print(f"Increased score by {self.team.total_score - old_score:,}")
        else:
            print("No score improvement.")
            self.revert()

    def optimize(self, iterations=10):
        """Needs some work."""
        for i in range(iterations):
            df = self.team.team.sort_values(by=["roi_score"], ascending=True)
            for row in df.iterrows():
                player = row[1]
                budget = self.team.available_budget + player["value"]
                print(budget)
                position = player["position"]
                score = player["score"]
                roi_score = player["roi_score"]
                pool = self.player_pool(position=position, drop_missed_prev_week=True,
                                        drop_unavailable=True, min_score=None,
                                        min_roi=roi_score, max_price=budget)

                if len(pool):
                    remove_uuid = player["uuid"]
                    r_name = self.team.player_names[remove_uuid]
                    self.team.remove_player(remove_uuid)
                    new = self.select_player(pool, select_by_roi=True)
                    self.team.add_player(**new)
                    a_name = self.team.player_names[new["uuid"]]
                    print(f"Found improvement - removed {r_name} for {a_name}.")
