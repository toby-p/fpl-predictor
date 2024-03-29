
import json
import os
import pandas as pd

from fpl_predictor.api import ApiData
from fpl_predictor.functions import index_players
from fpl_predictor.dtypes import csvtypes
from nav import DIR_STRUCTURED_DATA, DIR_STATIC


class PlayerInformation:
    """Class for loading player details such as value, position, and team for a
    given year-week."""

    def __init__(self):
        self._api_data = ApiData()
        fp = os.path.join(DIR_STRUCTURED_DATA, "master.csv")
        df = pd.read_csv(fp, encoding="utf-8", dtype=csvtypes)
        self._master = df

    def _master_year_week(self, year, week):
        """Get a subset of the master DF for the given year-week."""
        assert 0 < week < 39, f"Week must be int from 1 to 38"
        df = self._master.copy()
        df = df.loc[(df["year"] == year) & (df["GW"] == week)]
        return df.reset_index(drop=True)

    @property
    def _team_codes(self):
        """Maps team codes to full team names."""
        fp = os.path.join(DIR_STATIC, "team_codes.json")
        with open(fp, "r") as f:
            return {int(k): v for k, v in json.load(f).items()}

    def player_teams(self, year: int, live=False):
        """Dictionary of player codes to team in year."""
        if live:
            df = self._api_data.elements[["code", "team_code"]].copy()
            df["team"] = df["team_code"].map(self._team_codes)
            return dict(zip(df["code"], df["team"]))
        else:
            fp = os.path.join(DIR_STRUCTURED_DATA, "year_player_team.json")
            with open(fp, "r") as f:
                d = json.load(f)[str(year)]
            d = {int(k): self._team_codes[int(v)] for k, v in d.items()}
            return d

    def player_names(self, live=False):
        """Dictionary of player codes to player names."""
        if live:
            df = self._api_data.elements[["code", "first_name", "second_name"]].copy()
            df["name"] = df["first_name"].str.cat(df["second_name"], sep=" ")
            return dict(zip(df["code"], df["name"]))
        else:
            fp = os.path.join(DIR_STRUCTURED_DATA, "player_names.json")
            with open(fp, "r") as f:
                return {int(k): v for k, v in json.load(f).items()}

    def search_players(self, *strings: str, live: bool = False):
        """Search for player names matching the supplied strings to get their
        unqiue codes."""
        strings = [s.lower() for s in strings]
        names = self.player_names(live=live)
        return {k: v for k, v in names.items() if all([s in str.lower(v) for s in strings])}

    def player_values(self, year: int = None, week: int = None, live=False):
        """Dictionary of player codes to value in year-week."""
        if live:
            df = self._api_data.elements
            return dict(zip(df["code"], df["now_cost"]))
        else:
            df = self._master_year_week(year, week)
            return dict(zip(df["code"], df["value"]))

    def player_positions(self, year: int = None, live=False):
        """Dictionary of player codes to position in year."""
        if live:
            df = self._api_data.elements[["code", "element_type"]].copy()
            df["position"] = df["element_type"].map({1: "GK", 2: "DEF", 3: "MID", 4: "FWD"})
            return dict(zip(df["code"], df["position"]))
        else:
            fp = os.path.join(DIR_STRUCTURED_DATA, "year_player_position.json")
            with open(fp, "r") as f:
                d = json.load(f)[str(year)]
            return {int(k): v for k, v in d.items()}

    def add_player_info_to_df(self, df, year: int = None, week: int = None,
                              live=False, code_col: str = None):
        """Adds all player details to a Pandas DataFrame, mapping to the
        `code_col`. If`code_col` is None maps to the index."""
        df = df.copy()
        map_col = df[code_col] if code_col else df.index
        df["position"] = map_col.map(self.player_positions(year=year, live=live))
        df["value"] = map_col.map(self.player_values(year=year, week=week, live=live))
        df["team"] = map_col.map(self.player_teams(year=year, live=live))
        df["name"] = map_col.map(self.player_names(live=live))
        return df

    def points_scored(self, year: int = None, week: int = None):
        """Dictionary of player codes to total points they earnt in a given
        week. Can only query historical data."""
        df = self._master_year_week(year, week)
        return dict(zip(df["code"], df["total_points"]))

    def minutes_played(self, year: int = None, week: int = None):
        """Dictionary of player codes to minutes played in the given year-week.
        """
        df = index_players(self._master, column="minutes")
        return df.loc[(year, week), :].to_dict()

    def get_player(self, code: int, year: int = None, week: int = None,
                   live=False):
        """Get a dict of player information from their code."""
        player = dict(code=code, name=self.player_names()[code],
                      position=self.player_positions(year=year, live=live)[code],
                      value=self.player_values(year=year, week=week, live=live)[code],
                      team=self.player_teams(year=year, live=live)[code])
        return player
