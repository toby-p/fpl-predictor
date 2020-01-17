import itertools
import os
import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder
import string
import warnings

from nav import DIR_STRUCTURED_DATA, DIR_ML_DATA


def random_str(length=15):
    return "".join([random.choice(string.ascii_uppercase) for _ in range(length)])


FEATURE_COLS = [
    "assists", "attempted_passes", "big_chances_created", "big_chances_missed", "bonus", "bps", "clean_sheets",
    "clearances_blocks_interceptions", "completed_passes", "creativity", "dribbles", "errors_leading_to_goal",
    "errors_leading_to_goal_attempt", "fouls", "goals_conceded", "goals_scored", "ict_index", "influence", "key_passes",
    "offside", "open_play_crosses", "opponent_team_score", "own_goals", "penalties_conceded", "penalties_missed",
    "penalties_saved", "player_team_goal_difference", "player_team_score", "recoveries", "red_cards", "saves",
    "selected", "tackled", "tackles", "target_missed", "threat", "total_points", "transfers_balance", "transfers_in",
    "transfers_out", "value", "was_home", "winning_goals", "yellow_cards"
]
DUMMY_COLS = ["player_team_name", "opponent_team_name"]


class FeatureEngineer:
    ix_cols = ["Season", "GW", "code"]

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
        self.__features_already_added = list()

    @property
    def dummy_features(self):
        return sorted(DUMMY_COLS)

    @property
    def linear_features(self):
        return sorted(FEATURE_COLS)

    @property
    def master(self):
        return self.__master

    @property
    def df(self):
        return self.__df

    @property
    def target_features(self):
        return sorted([c for c in self.df.columns if "TARGET_" in c])

    @property
    def predictors(self):
        return sorted([c for c in self.df.columns if "TARGET_" not in c])

    def add_sum_feature(self, feature: str = "total_points", x: int = 5):
        """Add a predictor feature to the `df` attribute giving the sum of
        the last `x` weeks of a linear feature.

        Args:
            feature (str): column name of a feature in the raw data.
            x (int): number of recent weeks' data to sum.
        """
        column_name = f"last_{x}_GW_{feature}_sum"
        if column_name in self.__features_already_added:
            print(f"Feature already added to df: {column_name}")
            return
        if feature not in FEATURE_COLS:
            warnings.warn(f"Column not previously identified as suitable ML feature: {feature}")
        df = pd.pivot_table(self.master, index=("Season", "GW"), columns="code",
                            values=feature, aggfunc="mean").fillna(0).shift(1)
        df = df.rolling(window=x).sum().stack().reset_index()
        df = df.rename(columns={0: column_name})
        df = df.set_index(self.ix_cols)
        self.__df = pd.merge(self.__df, df, left_index=True, right_index=True, how="outer")
        print(f"Added sum feature column: {column_name}")
        self.__features_already_added.append(column_name)

    def add_mean_feature(self, feature: str = "total_points", x: int = 5):
        """Add a predictor feature to the `df` attribute giving the mean of
        the last `x` weeks of a feature column.

        Args:
            feature (str): column name of a feature in the raw data.
            x (int): number of recent weeks' data to average.
        """
        column_name = f"last_{x}_GW_{feature}_mean"
        if column_name in self.__features_already_added:
            print(f"Feature already added to df: {column_name}")
            return
        if feature not in FEATURE_COLS:
            warnings.warn(f"Column not previously identified as suitable ML feature: {feature}")
        df = pd.pivot_table(self.master, index=("Season", "GW"), columns="code",
                            values=feature, aggfunc="mean").fillna(0).shift(1)
        df = df.rolling(window=x).mean().stack().reset_index()
        df = df.rename(columns={0: column_name})
        df = df.set_index(self.ix_cols)
        self.__df = pd.merge(self.__df, df, left_index=True, right_index=True, how="outer")
        print(f"Added mean feature column: {column_name}")
        self.__features_already_added.append(column_name)

    def add_x_y_diff_feature(self, x: int = 5, y: int = 20,
                             feature: str = "total_points"):
        """Add a column with the mean of the feature in the previous `x` weeks
        minus the mean in the previous `y` weeks. Positive score indicates
        recent score on the feature has been higher than previously.

        Args:
            x (int): short period of recent weeks data to get mean of.
            y (int): long period of recent weeks data to get mean of.
            feature (str): column name of a feature in the raw data.
        """
        if feature not in FEATURE_COLS:
            warnings.warn(f"Column not previously identified as suitable ML feature: {feature}")
        new_col_name = f"{x}_diff_{y}_{feature}"
        if new_col_name in self.__features_already_added:
            print(f"Feature already added to df: {new_col_name}")
            return
        dfs = []
        for column_name, num_weeks in [("x", x), ("y", y)]:
            df = pd.pivot_table(self.master, index=("Season", "GW"), columns="code",
                                values=feature, aggfunc="mean").fillna(0).shift(1)
            df = df.rolling(window=num_weeks).mean().stack().reset_index()
            df = df.rename(columns={0: column_name})
            df = df.set_index(self.ix_cols)
            dfs.append(df)
        df = pd.merge(dfs[0], dfs[1], left_index=True, right_index=True)
        df[new_col_name] = df["x"] - df["y"]
        df.drop(columns=["x", "y"], inplace=True)
        self.__df = pd.merge(self.__df, df, left_index=True, right_index=True, how="outer")
        print(f"Added column to `df` attribute: {new_col_name}")
        self.__features_already_added.append(new_col_name)

    def add_raw_feature(self, feature: str = "ict_index"):
        """Add a raw independent variable feature to the df.

        Args:
            feature (str): column name of a feature in the raw data.
        """
        if feature in self.__features_already_added:
            print(f"Feature already added to df: {feature}")
            return
        series = self.master.set_index(["Season", "GW", "code"])[feature]
        self.__df = pd.merge(self.__df, series, left_index=True, right_index=True, how="outer")
        print(f"Added feature to `df`: {series.name}")
        self.__features_already_added.append(feature)

    def add_target_feature(self, feature: str = "total_points"):
        """Add a target dependent variable feature to the df.

        Args:
            feature (str): column name of a feature in the raw data.
        """
        if feature in self.__features_already_added:
            print(f"Feature already added to df: {feature}")
            return
        series = self.master.set_index(["Season", "GW", "code"])[feature]
        series.name = f"TARGET_{feature}"
        self.__df = pd.merge(self.__df, series, left_index=True, right_index=True)
        print(f"Added target feature to `df`: {series.name}")
        self.__features_already_added.append(feature)

    def save_df(self, name: str = None):
        """Save the DataFrame currently at the `df` attribute as a CSV.

        Args:
            name (str): filename for the CSV; if not passed a random string is
                used.
        """
        assert len(self.target_features), "No target features added to df, use `add_target`."
        name = f"{random_str()}.csv" if not name else f"{name}.csv"
        fp = os.path.join(DIR_ML_DATA, name)
        self.df.to_csv(fp, encoding="latin-1", index=True)
        print(f"Dataset saved at: {fp}")

    def add_dummy_feature(self, feature: str = "player_team_name"):
        """Create dummy variable columns from a categorical feature.

        Args:
            feature (str): column name of a categorical feature in the raw data.
        """
        assert feature in self.dummy_features, f"Not an eligible dummy feature: {feature}"
        if feature in self.__features_already_added:
            print(f"Dummy feature already added to df: {feature}")
            return
        df = self.master[self.ix_cols + [feature]]
        enc = OneHotEncoder(handle_unknown="ignore")
        enc.fit(df[[feature]])
        dummies = pd.DataFrame(enc.fit_transform(df[[feature]]).toarray(),
                               columns=enc.categories_)
        dummies.columns = dummies.columns.levels[0]
        rename = {c: f"{feature}_{c}" for c in dummies.columns}
        dummies.rename(columns=rename, inplace=True)
        dummies = pd.merge(dummies, df[self.ix_cols], left_index=True, right_index=True)
        dummies.set_index(self.ix_cols, inplace=True)
        df = pd.merge(self.__df, dummies, left_index=True, right_index=True, how="outer")
        new_column_names = list(rename.values())
        df[new_column_names] = df[new_column_names].fillna(value=0)
        self.__df = df
        print(f"Added dummy features for categorical column: {feature}")
        self.__features_already_added.append(feature)

    def __str__(self):
        s = f"FeatureEngineer\n\nTargets:\n  {self.target_features}\n\nPredictors:\n  {self.predictors}"
        return s

    def __repr__(self):
        return self.__str__()
