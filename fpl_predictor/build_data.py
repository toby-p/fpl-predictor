
import json
import numpy as np
import os
import pandas as pd

from fpl_predictor.functions import check
from nav import DIR_RAW_DATA, DIR_STRUCTURED_DATA, DIR_STATIC


class DataBuilder:
    """Class for processing the package's raw data files and converting them to
    structured data formats for the prediction algorithms."""

    def __init__(self, build_player_files=True, build_master_csv=True):
        if build_player_files:
            self.build_player_files()

        # Create a dict of 'year-id' to player code:
        fp = os.path.join(DIR_STRUCTURED_DATA, "players_master.csv")
        df = pd.read_csv(fp, encoding="utf-8")
        year_id = tuple(zip(df["year"], df["id"]))
        self.__year_id_to_code = dict(zip(year_id, df["code"]))

        if build_master_csv:
            self.build_master_csv()

    def build_player_files(self):
        """Build master CSV of all players per year, with their unique code.
        Also builds a nested JSON dict of {year: {player code: position}}."""
        print("Making master player files ... ", end="")
        fp = os.path.join(DIR_STRUCTURED_DATA, "players_master.csv")
        df, positions, teams, year_team_codes = pd.DataFrame(), dict(), dict(), dict()

        for y in self.years:
            y_df = self.load_year_players(year=y)
            y_df["year"] = y
            y_df["Season"] = self.year_to_season(y)

            # Save the positions & team codes:
            y_df["position"] = y_df["element_type"].map({1: "GK", 2: "DEF", 3: "MID", 4: "FWD"})
            positions[y] = dict(zip(y_df["code"], y_df["position"]))
            teams[y] = dict(zip(y_df["code"], y_df["team_code"]))
            year_team_codes[y] = dict(zip(y_df["team"], y_df["team_code"]))

            df = df.append(y_df, sort=False)

        # Save the CSV file:
        df.reset_index(drop=True, inplace=True)
        df.to_csv(fp, encoding="utf-8", index=False)

        # Save JSON files of player names:
        names = dict(zip(df["code"], df["name"]))
        fp = os.path.join(DIR_STRUCTURED_DATA, "player_names.json")
        with open(fp, "w") as f:
            json.dump(names, f)

        # Save JSON file of positions/teams per year (they can change over seasons):
        fp = os.path.join(DIR_STRUCTURED_DATA, "year_player_position.json")
        with open(fp, "w") as f:
            json.dump(positions, f)

        fp = os.path.join(DIR_STRUCTURED_DATA, "year_player_team.json")
        with open(fp, "w") as f:
            json.dump(teams, f)

        fp = os.path.join(DIR_STRUCTURED_DATA, "year_team_codes.json")
        with open(fp, "w") as f:
            json.dump(year_team_codes, f)

        check()

    def load_year_players(self, year):
        """Load the raw file of player data for the given year. Drops the
        columns which are in the raw files but are actually weekly data (e.g.
        points, value, etc.). Adds `name` column."""
        fp = os.path.join(self.__season_dir(year), "players_raw.csv")
        cols = ["code", "element_type", "first_name", "id", "second_name",
                "squad_number", "team", "team_code", "web_name"]
        df = pd.read_csv(fp, encoding="utf-8")
        df["name"] = df["first_name"].str.cat(df["second_name"], sep=" ")
        return df[["name"] + cols]

    @staticmethod
    def __season_dir(year: int):
        """Absolute file path of the directory of raw season data for the given
        start year."""
        return os.path.join(DIR_RAW_DATA, f"{year}-{(year - 2000) + 1}")

    @property
    def seasons(self):
        """List of seasons currently in the raw data directory."""
        return [x for x in os.listdir(DIR_RAW_DATA) if "." not in x]

    @property
    def years(self):
        """List of years currently in the raw data directory."""
        return sorted([int(x[:4]) for x in self.seasons])

    @staticmethod
    def year_to_season(year: int):
        """Convert int year to string season."""
        return f"{year}-{(year - 2000) + 1}"

    def get_gw_data(self, year: int, week: int):
        """Return the gameweek data for the given year-week combination."""
        path = os.path.join(self.__season_dir(year), "gws")
        filename = f"gw{week}.csv"
        df = pd.read_csv(os.path.join(path, filename), encoding='latin1')
        df["year"] = year
        df["year, id"] = tuple(zip(df["year"], df["element"]))
        df["code"] = df["year, id"].map(self.__year_id_to_code)
        return df

    def make_year_gw_df(self, year: int):
        """Make master DF of all player GW data for the given year."""
        df = pd.DataFrame()
        for i in range(1, 39, 1):
            try:
                new = self.get_gw_data(year, i)
                new["GW"] = i
                df = df.append(new, sort=False)
            except FileNotFoundError:
                continue
        df["Season"] = f"{year}-{(year - 2000) + 1}"
        df["opponent_team_name"] = df["opponent_team"].map(self.year_team_codes[year])
        df["opponent_team_name"] = df["opponent_team_name"].map(self.team_codes)
        return df.reset_index(drop=True)

    def build_master_csv(self):
        """Build master CSV file of all player GW data for all years."""
        print("Making master player gameweek CSV file ... ", end="")
        df = pd.DataFrame()
        for year in self.years:
            df = df.append(self.make_year_gw_df(year), sort=False)
        for col in ["GW", "element"]:
            df[col] = df[col].astype(int)
        df.reset_index(drop=True, inplace=True)
        df["year, element"] = tuple(zip(df["year"], df["element"]))
        df["player_team_name"] = df["year, element"].map(self.team_map)

        # Add columns for player/opponent team goals scored:
        df["player_team_score"] = np.where(df["was_home"], df["team_h_score"], df["team_a_score"])
        df["opponent_team_score"] = np.where(df["was_home"], df["team_a_score"], df["team_h_score"])
        df["player_team_goal_difference"] = df["player_team_score"] - df["opponent_team_score"]

        fp = os.path.join(DIR_STRUCTURED_DATA, "master.csv")
        df.to_csv(fp, encoding="utf-8", index=False)
        check()

    @property
    def team_codes(self):
        """Maps team codes to full team names."""
        fp = os.path.join(DIR_STATIC, "team_codes.json")
        with open(fp, "r") as f:
            return {int(k): v for k, v in json.load(f).items()}

    @property
    def year_player_team(self):
        """Dictionary of player codes to team in year."""
        fp = os.path.join(DIR_STRUCTURED_DATA, "year_player_team.json")
        with open(fp, "r") as f:
            d = json.load(f)
        return {int(k1): {int(k2): v for k2, v in v1.items()} for k1, v1 in d.items()}

    @property
    def team_map(self):
        """Dict of (year, code): team for all players."""
        d = dict()
        for k, v in self.year_player_team.items():
            for k2, v2 in v.items():
                d[(k, k2)] = self.team_codes[v2]
        return d

    @property
    def year_team_codes(self):
        """Dictionary of team codes to team in year."""
        fp = os.path.join(DIR_STRUCTURED_DATA, "year_team_codes.json")
        with open(fp, "r") as f:
            d = json.load(f)
        return {int(k1): {int(k2): v for k2, v in v1.items()} for k1, v1 in d.items()}
