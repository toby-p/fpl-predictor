
import itertools
import os
import pandas as pd
import random
import string
import warnings

from nav import DIR_STRUCTURED_DATA, DIR_ML_DATA


def random_str(length=15):
    return "".join([random.choice(string.ascii_uppercase) for _ in range(length)])


FEATURE_COLS = [
    "assists", "attempted_passes", "big_chances_created", "big_chances_missed", "bonus", "bps",  "clean_sheets",
    "clearances_blocks_interceptions", "completed_passes", "creativity", "dribbles", "errors_leading_to_goal",
    "errors_leading_to_goal_attempt", "fouls", "goals_conceded", "goals_scored", "ict_index", "influence", "key_passes",
    "offside", "open_play_crosses", "opponent_team_score", "own_goals", "penalties_conceded", "penalties_missed",
    "penalties_saved", "player_team_goal_difference", "player_team_score", "recoveries", "red_cards", "saves",
    "selected", "tackled", "tackles", "target_missed", "threat", "total_points", "transfers_balance", "transfers_in",
    "transfers_out", "value", "was_home", "winning_goals", "yellow_cards"
]
DUMMY_COLS = ["player_team_name", "opponent_team_name"]


class FeatureEngineer:
    def __init__(self):
        fp = os.path.join(DIR_STRUCTURED_DATA, "master.csv")
        self.__master = pd.read_csv(fp, encoding="latin-1")

        # Create the DataFrame which will store the features:
        codes = self.master["code"].unique()
        gws = self.master["GW"].unique()
        seasons = self.master["Season"].unique()
        df = pd.DataFrame(list(itertools.product(seasons, gws, codes)), columns=["Season", "GW", "code"])
        assert len(df) == len(codes) * len(gws) * len(seasons)
        self.__df = df.set_index(["Season", "GW", "code"])

    @property
    def master(self):
        return self.__master

    @property
    def dataset(self):
        return self.__df

    @property
    def target_features(self):
        return sorted([c for c in self.dataset.columns if "TARGET_" in c])

    def add_feature(self, feature="total_points", x=5):
        """Add a predictor feature aggregating the last `x` weeks total of the
        given column to the dataset."""
        if feature not in FEATURE_COLS:
            warnings.warn(f"Column not previously identified as suitable ML feature: {feature}")
        df = pd.pivot_table(self.master, index=("Season", "GW"), columns="code",
                            values=feature, aggfunc="mean").fillna(0).shift(1)
        df = df.rolling(window=x).sum().stack().reset_index()
        df = df.rename(columns={0: f"last_{x}_GW_{feature}"})
        df = df.set_index(["Season", "GW", "code"])
        self.__df = pd.merge(self.__df, df, left_index=True, right_index=True)

    def add_target(self, feature="total_points"):
        """As a target independent variable feature to the dataset."""
        series = self.master.set_index(["Season", "GW", "code"])[feature]
        series.name = f"TARGET_{feature}"
        self.__df = pd.merge(self.__df, series, left_index=True, right_index=True)

    def save_dataset(self, name=None):
        assert len(self.target_features), "No target features added to dataset, use `add_target`."
        name = f"{random_str()}.csv" if not name else f"{name}.csv"
        fp = os.path.join(DIR_ML_DATA, name)
        self.dataset.to_csv(fp, encoding="latin-1", index=True)
        print(f"Dataset saved at: {fp}")
