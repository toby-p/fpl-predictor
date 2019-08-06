import numpy as np
import pandas as pd

from .team import Team
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
        uuids = self.dataloader.player_uuids
        uuids = uuids.loc[uuids["year"] == year]
        uuids = dict(zip(uuids["id"], uuids["cross_season_id"]))
        self.gw = DataLoader.gw(year, week)
        self.gw["uuid"] = self.gw["element"].map(uuids)
        self.values = dict(zip(self.gw["uuid"], self.gw["value"]))

        self.team = Team()

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
        return pd.merge(raw_scores, roi_scores, left_index=True, right_index=True)

    def build(self):

        df = self.score()
        df["value"] = df.index.map(self.values)
        while not self.team.filled:
            for uuid in df.index:
                value = df.loc[uuid, "value"]
                score = df.loc[uuid, "total_points (raw)"]
                roi_score = df.loc[uuid, "total_points (roi)"]
                try:
                    self.team.add_player(uuid, value, score, roi_score)
                except:
                    continue
