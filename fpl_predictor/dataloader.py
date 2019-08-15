
import datetime
import json
import os
import pandas as pd

from fpl_predictor.api import ApiData
from fpl_predictor.functions import check
from nav import DIR_RAW_DATA, DIR_STRUCTURED_DATA, DIR_STATIC


class DataLoader:
    """Class for loading raw historical data and converting it to structured
    data formats for the prediction algorithms."""

    def __init__(self, build_master=True, build_players=True):
        self.api_data = ApiData()

        if build_players:
            self.build_player_csv()
        self.players_master = self._players_master
        self.player_year_id_to_code = self._player_year_id_to_code

        if build_master:
            self.build_master_csv()
        self.master = pd.read_csv(os.path.join(DIR_STRUCTURED_DATA, "master.csv"), encoding="utf-8")
        self.master["code"] = self.master["code"].astype(int)
        self.master["year"] = self.master["year"].astype(int)

        if build_master or build_players:
            self._build_year_player_team()
        self.year_player_team = self._year_player_team

    @property
    def _players_master(self):
        fp = os.path.join(DIR_STRUCTURED_DATA, "players_master.csv")
        if not os.path.exists(fp):
            self.build_player_csv()
        df = pd.read_csv(fp, encoding="utf-8")
        return df

    @property
    def _master(self):
        fp = os.path.join(DIR_STRUCTURED_DATA, "master.csv")
        if not os.path.exists(fp):
            self.build_master_csv()
        df = pd.read_csv(fp, encoding="utf-8")
        return df

    @property
    def _year_player_team(self):
        fp = os.path.join(DIR_STRUCTURED_DATA, "year_player_team.json")
        if not os.path.exists(fp):
            self._build_year_player_team()
        with open(fp, "r") as f:
            d = json.load(f)
        return {int(k): {int(k2): v2 for k2, v2 in v.items()} for k, v in d.items()}

    @property
    def player_names(self):
        fp = os.path.join(DIR_STRUCTURED_DATA, "player_names.json")
        with open(fp, "r") as f:
            return {int(k): v for k, v in json.load(f).items()}

    @property
    def player_positions(self):
        fp = os.path.join(DIR_STRUCTURED_DATA, "player_positions.json")
        with open(fp, "r") as f:
            return json.load(f)

    @property
    def team_codes(self):
        fp = os.path.join(DIR_STATIC, "team_codes.json")
        with open(fp, "r") as f:
            return {int(k): v for k, v in json.load(f).items()}

    def build_player_csv(self):
        """Build master CSV of all players with their unique code.
        """
        print("Making master player database CSV file ... ", end="")
        fp = os.path.join(DIR_STRUCTURED_DATA, "players_master.csv")

        years = self.years

        # Get first year's data:
        first_year = years[0]
        df = self.load_year_players(year=first_year)
        df["year"] = first_year
        df["Season"] = self.year_to_season(first_year)

        # Merge with all the other years' data by player name:
        for y in years[1:]:
            new = self.load_year_players(year=y)
            new["year"] = y
            new["Season"] = self.year_to_season(y)
            df = df.append(new, sort=False)

        df.reset_index(drop=True, inplace=True)

        # Add in position:
        df["position"] = df["element_type"].map({1: "GK", 2: "DEF", 3: "MID", 4: "FWD"})

        # Save JSON files of positions and names for quick access:
        player_names = dict(zip(df["code"], df["name"]))
        names_fp = os.path.join(DIR_STRUCTURED_DATA, "player_names.json")
        with open(names_fp, "w") as f:
            json.dump(player_names, f)
        player_positions = dict(zip(df["code"], df["position"]))
        positions_fp = os.path.join(DIR_STRUCTURED_DATA, "player_positions.json")
        with open(positions_fp, "w") as f:
            json.dump(player_positions, f)

        # Save the CSV file:
        df.to_csv(fp, encoding="utf-8", index=False)
        check()

    def _build_year_player_team(self):
        """Build a nested JSON dict of {year: {player code: team}}."""
        print("Making year_player_team JSON file ... ", end="")
        data = dict()
        for year in self.years:
            df = self.load_year_players(year)
            data[year] = dict(zip(df["code"], df["team_code"]))
        fp = os.path.join(DIR_STRUCTURED_DATA, "year_player_team.json")
        with open(fp, "w") as f:
            json.dump(data, f)
        check()

    @staticmethod
    def _season_dir(year: int):
        """Absolute file path of the directory of raw season data for the given
        start year.
        """
        return os.path.join(DIR_RAW_DATA, f"{year}-{(year - 2000) + 1}")

    @property
    def seasons(self):
        """List of seasons currently available in the raw data directory.
        """
        return [x for x in os.listdir(DIR_RAW_DATA) if "." not in x]
    
    @property
    def years(self):
        """List of years currently available in the data directory.
        """
        return sorted([int(x[:4]) for x in self.seasons])

    @staticmethod
    def year_to_season(year: int):
        return f"{year}-{(year - 2000) + 1}"

    def _make_year_gw_df(self, year: int):
        """Make master DF of all player GW data for the given year.
        """
        df = pd.DataFrame()
        for i in range(1, 39, 1):
            try:
                new = self.gw(year, i)
                new["GW"] = i
                df = df.append(new, sort=False)
            except FileNotFoundError:
                continue
        df["Season"] = f"{year}-{(year - 2000) + 1}"
        return df.reset_index(drop=True)

    def build_master_csv(self):
        """Build master CSV file of all player GW data for all years.
        """
        print("Making master player gameweek CSV file ... ", end="")

        years = [int(y[:4]) for y in self.seasons]
        df = pd.DataFrame()
        for year in years:
            df = df.append(self._make_year_gw_df(year), sort=False)

        for col in ["GW", "element"]:
            df[col] = df[col].astype(int)
        df.reset_index(drop=True, inplace=True)
        fp = os.path.join(DIR_STRUCTURED_DATA, "master.csv")
        df.to_csv(fp, encoding="utf-8", index=False)
        check()

    def gw(self, year: int, week: int):
        """Return the gameweek data for the given year-week combination."""
        path = os.path.join(self._season_dir(year), "gws")
        filename = f"gw{week}.csv"
        df = pd.read_csv(os.path.join(path, filename), encoding='latin1')
        df["year"] = year
        df["year, id"] = tuple(zip(df["year"], df["element"]))
        df["code"] = df["year, id"].map(self.player_year_id_to_code)
        return df

    @property
    def current_api_elements(self):
        """Return the current `elements` data returned by the API."""
        df = self.api_data.elements.copy()
        curr_month = datetime.datetime.now().month
        year = datetime.datetime.now().year
        if curr_month in range(1, 8, 1):
            year -= 1
        df["year"] = year
        df["year, id"] = tuple(zip(df["year"], df["id"]))
        df["team_name"] = df["team_code"].map(self.team_codes)
        df["name"] = df["first_name"].str.cat(df["second_name"], sep=" ")
        return df

    @property
    def _player_year_id_to_code(self):
        """Dictionary of year-player ID tuple, to the unique identifying code
        for that player."""
        df = self.players_master.copy()
        year_id = tuple(zip(df["year"], df["id"]))
        return dict(zip(year_id, df["code"]))

    def load_year_players(self, year):
        """Load the raw file of player data for the given year. Drops the
        columns which are in the raw files but are actually weekly data (e.g.
        points, value, etc.). Adds `name` column."""
        fp = os.path.join(self._season_dir(year), "players_raw.csv")
        cols = ["code", "element_type", "first_name", "id", "second_name",
                "squad_number", "team", "team_code", "web_name"]
        df = pd.read_csv(fp, encoding="utf-8")
        df["name"] = df["first_name"].str.cat(df["second_name"], sep=" ")
        df["team_name"] = df["team_code"].map(self.team_codes)
        return df[["name", "team_name"] + cols]

    @property
    def current_values(self):
        df = self.api_data.elements.copy()
        df.sort_values(by=["now_cost"], ascending=False, inplace=True)
        return dict(zip(df["code"], df["now_cost"]))
